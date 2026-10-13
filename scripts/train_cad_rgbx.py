"""Scratch joint B0 training on exactly two GPUs; no pretrained or resume entry."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx import bootstrap
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from cad_rgbx.model import configure, build_random_model, state_digest
from cad_rgbx.data import build_loader
from cad_rgbx.losses import localization_losses
from cad_rgbx.hardware import require_gpu0_limits, CoolingGuard


def learning_rate(protocol, step, steps_per_epoch):
    warmup = protocol['warmup_epochs'] * steps_per_epoch
    if step <= warmup:
        factor = protocol['warmup_initial_factor'] + (1 - protocol['warmup_initial_factor']) * step / warmup
        return protocol['learning_rate'] * factor
    progress = (step - warmup) / (protocol['epochs'] * steps_per_epoch - warmup)
    return protocol['minimum_learning_rate'] + (protocol['learning_rate'] - protocol['minimum_learning_rate']) * (1 + math.cos(math.pi * progress)) / 2


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--sanity-steps', type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert protocol['physical_gpus'] == [0, 1] and protocol['modules_enabled'] == []
    assert protocol['optimizer'] == 'AdamW' and not protocol['amp']
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert rank == local_rank and world == protocol['world_size'] == 2
    # Admission happens before CUDA allocations on both ranks.
    hardware = require_gpu0_limits(protocol)
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    random.seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    cfg = configure(protocol)
    model = build_random_model(cfg)
    initial_digest = state_digest(model)
    assert initial_digest == protocol['initial_state_sha256'], initial_digest
    manifest = [dict(name=name, shape=list(value.shape), numel=value.numel(), requires_grad=value.requires_grad)
                for name, value in model.named_parameters()]
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl')
    cooling_guard = CoolingGuard(protocol, rank, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert protocol['sync_batch_norm']
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # Pruned source retains positional parameters not used by its forward path.
    network = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                   weight_decay=protocol['weight_decay'], betas=(0.9, 0.999), eps=1e-8)
    seed = protocol['seed'] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    global_batch = protocol['batch_per_rank'] * world
    assert global_batch == 24 and protocol['samples_per_epoch'] % global_batch == 0
    formal_steps = protocol['samples_per_epoch'] // global_batch
    steps = args.sanity_steps or formal_steps
    loader, datasets = build_loader(cfg, protocol, rank, steps, 0 if args.sanity_steps else protocol['num_workers'])
    root = Path(protocol['output'] + ('_sanity' if args.sanity_steps else ''))
    assert root.parent == Path('/home/gaob/rgbx-cadtrack')
    if rank == 0:
        root.mkdir(parents=True)
    dist.barrier()
    rank_pids = [None] * world
    dist.all_gather_object(rank_pids, os.getpid())
    receipt = dict(status='RUNNING_NOT_EVALUATED', protocol=protocol, rank_pids=rank_pids,
        started_at=datetime.now(timezone.utc).isoformat(), initial_state_sha256=initial_digest,
        pretrained_loaded=False, auto_resume=False, trainable_parameter_tensors=len(manifest),
        trainable_parameter_elements=sum(value['numel'] for value in manifest), physical_gpus=[0,1],
        gpu0_admission=hardware, formal_tracking_scores='NOT_RUN', completed_epochs=0,
        checkpoint_policy=protocol['checkpoint_policy'], sanity_steps=args.sanity_steps)
    if rank == 0:
        write_json(root / 'parameter_manifest.json', manifest)
        write_json(root / 'run.json', receipt)
    started = time.perf_counter()
    epoch_history, global_step, pauses = [], 0, 0.0
    epochs = 1 if args.sanity_steps else protocol['epochs']
    for epoch in range(1, epochs + 1):
        network.train()
        metrics_sum = torch.zeros((3, 3), device=device)
        for step, batch in enumerate(loader, 1):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            pauses += cooling_guard.pause()
            global_step += 1
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(protocol, global_step, formal_steps)
            optimizer.param_groups[0]['lr'] = lr
            outputs = network(template=list(batch['template_images']), search=list(batch['search_images']))
            assert len(outputs) == 1
            losses, metrics, counts = localization_losses(outputs[0], batch, cfg)
            assert counts.tolist() == [4, 4, 4]
            dist.all_reduce(counts)
            assert counts.tolist() == protocol['global_task_counts']
            loss = losses.mean()
            assert torch.isfinite(loss).item(), losses
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip_norm'], error_if_nonfinite=True)
            if global_step == 1:
                active = [name for name, value in model.named_parameters() if value.grad is not None]
                assert any(name.startswith('backbone.blocks.') for name in active)
                assert any(name.startswith('box_head.') for name in active)
                if rank == 0:
                    write_json(root / 'first_update_gradient_manifest.json', dict(active=active,
                        inactive=[value['name'] for value in manifest if value['name'] not in active]))
            optimizer.step()
            dist.all_reduce(metrics)
            metrics /= world
            metrics_sum += metrics
            torch.cuda.synchronize(device)
            record = dict(epoch=epoch, step=step, global_step=global_step, learning_rate=lr,
                global_task_counts=counts.tolist(), task_response_box_iou=metrics.tolist(),
                gradient_norm_before_clip=float(gradient_norm), step_seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), cooling_pause_seconds=pauses,
                main_training_examples=global_batch * global_step)
            if rank == 0 and (step % protocol['log_every_steps'] == 0 or step == steps):
                receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), latest=record,
                               wallclock_seconds=time.perf_counter()-started)
                write_json(root / 'run.json', receipt)
                with (root / 'updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            del outputs, losses, loss, batch
        assert step == steps
        epoch_history.append(dict(epoch=epoch, steps=steps, task_response_box_iou=(metrics_sum / steps).tolist()))
        digest = state_digest(model)
        replicas = [None] * world
        dist.all_gather_object(replicas, digest)
        assert len(set(replicas)) == 1, replicas
        rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                   torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device))
        rng_by_rank = [None] * world
        dist.all_gather_object(rng_by_rank, rng)
        if rank == 0:
            checkpoint = root / 'CADTrack_latest.pth.tar'
            temporary = root / 'checkpoint.tmp'
            # Replacement retires only our preceding recovery file in this new owned directory.
            torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                global_step=global_step, protocol=protocol, epoch_history=epoch_history,
                rng_by_rank=rng_by_rank, initial_state_sha256=initial_digest,
                replica_state_sha256=replicas, trainable_parameter_names=[value['name'] for value in manifest],
                sanity_steps=args.sanity_steps, model_type='CADTrackPruned'), temporary)
            os.replace(str(temporary), str(checkpoint))
            receipt.update(completed_epochs=epoch, last_checkpoint=str(checkpoint), epoch_history=epoch_history)
            write_json(root / 'run.json', receipt)
        dist.barrier()
    if rank == 0:
        receipt.update(status='SANITY_COMPLETE_NOT_FORMAL_RESULT' if args.sanity_steps else 'TRAINING_COMPLETE_NOT_EVALUATED',
            ended_at=datetime.now(timezone.utc).isoformat(), global_step=global_step,
            wallclock_seconds=time.perf_counter()-started, cooling_pause_seconds=pauses)
        write_json(root / 'run.json', receipt)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
