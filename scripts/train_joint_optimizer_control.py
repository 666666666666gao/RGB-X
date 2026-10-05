"""Train C/D paired controls without localization checks or trajectory data usage."""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'third_party/XTrack')]
from lib.config.xtrack.config import cfg, update_config_from_file
from lib.models.xtrack import build_xtrack
from lib.train.admin.settings import Settings
from lib.train.actors import XTrackActor
from rgbx_method.data import build_pair_loaders
from rgbx_method.losses import components
from rgbx_method.optimizer import TaskAdaptiveAdamW
from rgbx_controls.updates import normalized_joint_loss, flatten_gradients, parameter_manifest, apply_joint_adamw


def global_mean(value):
    result = value.detach().clone()
    dist.all_reduce(result)
    return result / dist.get_world_size()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--sanity-steps', type=int, default=0)
    parser.add_argument('--resume')
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
    kind = protocol['optimizer']
    assert kind in ['adamw', 'task_adaptive_only']
    prefix = 'xtrack_fixed_mixed_adamw' if kind == 'adamw' else 'xtrack_task_stats_only'
    assert protocol['run_id'] == prefix + '_s' + str(protocol['seed'])
    assert protocol['output'] == '/home/gaob/rgbx-full-method/' + protocol['run_id']
    assert protocol['modules_enabled'] == ([] if kind == 'adamw' else ['task_adaptive_statistics'])
    assert protocol['batch_per_rank'] == 8 and not protocol['amp']
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert world == protocol['world_size'] == 3
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl')
    seed = protocol['seed'] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file(str(PROJECT / protocol['base_config']))
    model = build_xtrack(cfg, training=False)
    initializer = torch.load(protocol['initial_checkpoint'], map_location='cpu')
    model.load_state_dict(initializer['net'], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('MeME' in name)
    model.to(device)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    names = [name for name, _ in named]
    if kind == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=protocol['lr'], weight_decay=protocol['weight_decay'],
                                      betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = TaskAdaptiveAdamW(named, protocol['lr'], protocol['weight_decay'], protocol['grad_clip_norm'])
    settings = Settings()
    settings.local_rank = rank
    settings.script_name = 'xtrack'
    formal_steps = protocol['samples_per_epoch'] // (8 * world)
    steps = args.sanity_steps or formal_steps
    _, loaders = build_pair_loaders(cfg, settings, steps, rank, world, 0 if args.sanity_steps else protocol['num_workers'])
    training_loader = loaders[0]
    # The existing factory also constructs a check loader; it is never iterated.
    actor = XTrackActor(model, {}, {}, settings, cfg)
    root = Path(protocol['output'] + ('_sanity' if args.sanity_steps else ''))
    assert root.parent == Path('/home/gaob/rgbx-full-method')
    if rank == 0:
        root.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()
    initial_epoch, global_step, epoch_history = 0, 0, []
    if args.resume:
        assert not args.sanity_steps
        restored = torch.load(args.resume, map_location=device)
        assert restored['protocol'] == protocol and restored['trainable_parameter_names'] == names
        assert restored['sanity_steps'] == 0
        assert restored['global_step'] == restored['epoch'] * formal_steps
        model.load_state_dict(restored['net'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        initial_epoch, global_step, epoch_history = restored['epoch'], restored['global_step'], restored['epoch_history']
        rng = restored['rng_by_rank'][rank]
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch_cpu'].cpu())
        torch.cuda.set_rng_state(rng['torch_cuda'].cpu(), device)
    receipt = dict(status='RUNNING_NOT_EVALUATED', run_id=protocol['run_id'], protocol=protocol,
                   protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                   initial_checkpoint_sha256=hashlib.sha256(Path(protocol['initial_checkpoint']).read_bytes()).hexdigest(),
                   optimizer_kind=kind, world_size=world, output=str(root), modules_enabled=protocol['modules_enabled'],
                   localization_check_calls=0, correction_calls=0, trajectory_calls=0, check_examples=0,
                   unused_check_loader_constructed_not_iterated=True, formal_tracking_scores='NOT_RUN',
                   started_at=datetime.now(timezone.utc).isoformat(), rank_pids=[None] * world)
    if args.resume:
        receipt.update(completed_epochs=initial_epoch, last_checkpoint=str(Path(args.resume).resolve(strict=True)))
    dist.all_gather_object(receipt['rank_pids'], os.getpid())
    receipt['attempt_id'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if rank == 0:
        if args.resume:
            shutil.copyfile(root / 'run.json', root / ('previous_attempt_' + receipt['attempt_id'] + '.json'))
            with (root / 'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(record_type='RESUME_BOUNDARY', attempt_id=receipt['attempt_id'],
                                             restored_epoch=initial_epoch, restored_global_step=global_step,
                                             checkpoint=str(args.resume),
                                             telemetry_note='Partial-attempt rows are retained; worker-prefetch sample replay is not bitwise guaranteed.')) + '\n')
        (root / 'parameter_manifest.json').write_text(json.dumps(parameter_manifest(named, kind != 'adamw'), indent=2) + '\n')
        (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
    started = time.perf_counter()
    epochs = 1 if args.sanity_steps else protocol['epochs']
    for epoch in range(initial_epoch + 1, epochs + 1):
        lr = protocol['lr'] * (protocol['lr_drop_factor'] if epoch > protocol['lr_drop_epoch'] else 1)
        if kind == 'adamw':
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.lr = lr
        sums = dict(total=0., balance=0., classification=0., iou=0.)
        for step, batch in enumerate(training_loader, 1):
            tick = time.perf_counter()
            global_step += 1
            batch = batch.to(device)
            batch['epoch'] = epoch
            model.train()
            actor.fix_bns()
            prediction = actor.forward_pass(batch)
            losses, _, counts, _, metrics = components(prediction, batch, cfg)
            global_counts = counts.clone()
            dist.all_reduce(global_counts)
            assert global_counts.tolist() == [8, 8, 8]
            metrics['total'] = normalized_joint_loss(losses.detach(), counts, global_counts) * world
            if kind == 'adamw':
                values = torch.autograd.grad(normalized_joint_loss(losses, counts, global_counts), parameters,
                                             allow_unused=True)
                gradient = flatten_gradients(values, parameters)
                active = torch.tensor([value is not None for value in values], dtype=torch.long, device=device)
                dist.all_reduce(gradient)
                dist.all_reduce(active, op=dist.ReduceOp.MAX)
                assert torch.isfinite(gradient).all()
                update_stats = apply_joint_adamw(optimizer, gradient, active.bool(), protocol['grad_clip_norm'])
            else:
                gradients, activity = [], []
                for task in range(3):
                    values = torch.autograd.grad(losses[task] * counts[task] / global_counts[task], parameters,
                                                retain_graph=task < 2, allow_unused=True)
                    gradients.append(flatten_gradients(values, parameters))
                    activity.append([value is not None for value in values])
                gradients = torch.stack(gradients)
                active = torch.tensor(activity, dtype=torch.long, device=device)
                dist.all_reduce(gradients)
                dist.all_reduce(active, op=dist.ReduceOp.MAX)
                assert torch.isfinite(gradients).all()
                base = optimizer.values()
                displacement, update_stats = optimizer.propose(gradients, active.bool())
                optimizer.assign(base + displacement)
            metric_vector = global_mean(torch.stack([metrics[key] for key in sums]))
            for key, value in zip(sums, metric_vector.tolist()):
                sums[key] += value
            torch.cuda.synchronize(device)
            record = dict(attempt_id=receipt['attempt_id'], epoch=epoch, step=step, global_step=global_step,
                          optimizer_kind=kind, global_task_counts=global_counts.tolist(), learning_rate=lr,
                          step_seconds=time.perf_counter() - tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          main_training_examples=24, localization_check_calls=0, correction_calls=0,
                          trajectory_calls=0, pair_check_forward_examples=0, trajectory_model_forward_examples=0,
                          optimizer_statistics_updates=1, weight_decay_applications=1, **update_stats)
            if rank == 0:
                with (root / 'updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                if step % 10 == 0 or step == steps:
                    receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), epoch=epoch, step=step,
                                   global_step=global_step, training_examples=24 * global_step, latest=record,
                                   wallclock_seconds=time.perf_counter() - started)
                    (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
                    print(json.dumps({key:receipt[key] for key in ['epoch','step','global_step','wallclock_seconds']}), flush=True)
            del prediction, losses, values
            if kind == 'adamw':
                del gradient
            else:
                del gradients, base, displacement
        assert step == steps
        epoch_history.append(dict(epoch=epoch, steps=step, means={key:value / steps for key,value in sums.items()}))
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                   torch_cuda=torch.cuda.get_rng_state(device))
        rng_by_rank = [None] * world
        dist.all_gather_object(rng_by_rank, rng)
        vector = torch.cat([parameter.detach().flatten() for parameter in parameters])
        digest = hashlib.sha256(vector.cpu().numpy().tobytes()).hexdigest()
        replicas = [None] * world
        dist.all_gather_object(replicas, digest)
        assert len(set(replicas)) == 1, replicas
        if rank == 0:
            path = root / ('XTrack_ep%04d.pth.tar' % epoch)
            temporary = root / 'checkpoint.tmp'
            torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), optimizer_kind=kind,
                            sanity_steps=args.sanity_steps,
                            trainable_parameter_names=names, epoch=epoch, global_step=global_step, protocol=protocol,
                            epoch_history=epoch_history, rng_by_rank=rng_by_rank, replicas_identical=replicas), temporary)
            os.replace(str(temporary), str(path))
            receipt.update(last_checkpoint=str(path), completed_epochs=epoch, epoch_history=epoch_history,
                           replica_parameter_sha256=digest,
                           checkpoint_retention='External first-epoch audit and owned retention activation required; no latest alias is generated.')
            (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
        dist.barrier()
    if rank == 0:
        receipt.update(status='SANITY_COMPLETE_NOT_FORMAL_RESULT' if args.sanity_steps else 'TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT',
                       ended_at=datetime.now(timezone.utc).isoformat(), epoch_history=epoch_history,
                       global_step=global_step, formal_tracking_scores='NOT_RUN')
        (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
