"""Train one RGB-X model with all three training-time modules on three GPUs."""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.distributed as dist

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'third_party/XTrack')]
from lib.config.xtrack.config import cfg, update_config_from_file
from lib.models.xtrack import build_xtrack
from lib.train.admin.settings import Settings
from lib.train.actors import XTrackActor
from rgbx_method.data import build_pair_loaders, TrainingClips
from rgbx_method.losses import components
from rgbx_method.optimizer import TaskAdaptiveAdamW
from rgbx_method.risk import localization_risk, correct_displacement
from rgbx_method.trajectory import ClosedLoopProbe


def flatten_gradients(gradients, parameters):
    return torch.cat([gradient.flatten() if gradient is not None else torch.zeros_like(parameter).flatten()
                      for gradient, parameter in zip(gradients, parameters)])


def global_mean(value):
    result = value.detach().clone()
    dist.all_reduce(result)
    return result / dist.get_world_size()


def route_changes(before, after):
    return global_mean((before != after).any(-1).float().mean(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--sanity-steps', type=int, default=0)
    parser.add_argument('--resume')
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
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
    optimizer = TaskAdaptiveAdamW(model.named_parameters(), protocol['lr'], protocol['weight_decay'], protocol['grad_clip_norm'])
    shared_mask = optimizer.shared_mask()
    settings = Settings()
    settings.local_rank = rank
    settings.script_name = 'xtrack'
    steps = args.sanity_steps or protocol['samples_per_epoch'] // (8 * world)
    datasets, loaders = build_pair_loaders(cfg, settings, steps, rank, world, 0 if args.sanity_steps else protocol['num_workers'])
    actor = XTrackActor(model, {}, {}, settings, cfg)
    clips = TrainingClips(datasets[rank], protocol['trajectory_length'])
    trajectory = ClosedLoopProbe(model, cfg, device)
    root = Path(protocol['output'] + ('_sanity' if args.sanity_steps else ''))
    assert root.parent == Path('/home/gaob/rgbx-full-method')
    if rank == 0:
        root.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()
    manifest = optimizer.parameter_manifest()
    initial_epoch, global_step, epoch_history = 0, 0, []
    if args.resume:
        restored = torch.load(args.resume, map_location=device)
        assert restored['protocol'] == protocol
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
                   world_size=world, rank_pids=[None] * world, output=str(root),
                   modules_enabled=protocol['modules_enabled'], formal_tracking_scores='NOT_RUN',
                   started_at=datetime.now(timezone.utc).isoformat(), training_examples_per_update=24,
                   check_examples_per_update=9, short_clip_source='TRAIN_ONLY', baseline_setting='AUTHOR_RELEASE_FINETUNE')
    receipt['attempt_id'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dist.all_gather_object(receipt['rank_pids'], os.getpid())
    if rank == 0:
        if args.resume:
            shutil.copyfile(root / 'run.json', root / ('previous_attempt_' + receipt['attempt_id'] + '.json'))
            with (root / 'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(record_type='RESUME_BOUNDARY', attempt_id=receipt['attempt_id'],
                                             checkpoint=str(args.resume), restored_epoch=initial_epoch,
                                             restored_global_step=global_step,
                                             telemetry_note='Prior partial epoch rows are retained attempts, not committed updates. Worker-prefetch sample replay is not bitwise guaranteed.')) + '\n')
        (root / 'parameter_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
    started = time.perf_counter()
    epochs = 1 if args.sanity_steps else protocol['epochs']
    for epoch in range(initial_epoch + 1, epochs + 1):
        optimizer.lr = protocol['lr'] * (protocol['lr_drop_factor'] if epoch > protocol['lr_drop_epoch'] else 1)
        sums = dict(total=0., balance=0., classification=0., iou=0.)
        for step, (training_batch, check_batch) in enumerate(zip(*loaders), 1):
            tick = time.perf_counter()
            global_step += 1
            training_batch, check_batch = training_batch.to(device), check_batch.to(device)
            training_batch['epoch'] = check_batch['epoch'] = epoch
            assert check_batch['dataset'] == ['lasher', 'depthtrack', 'visevent']
            model.train()
            actor.fix_bns()
            prediction = actor.forward_pass(training_batch)
            losses, _, counts, _, metrics = components(prediction, training_batch, cfg)
            global_counts = counts.clone()
            dist.all_reduce(global_counts)
            assert global_counts.tolist() == [8, 8, 8]
            metrics['total'] = (losses.detach() * counts / global_counts).sum() * world / 3
            gradients, active = [], []
            for task in range(3):
                values = torch.autograd.grad(losses[task] * counts[task] / global_counts[task], optimizer.parameters,
                                            retain_graph=task < 2, allow_unused=True)
                gradients.append(flatten_gradients(values, optimizer.parameters))
                active.append([gradient is not None for gradient in values])
            gradients = torch.stack(gradients)
            active = torch.tensor(active, dtype=torch.long, device=device)
            dist.all_reduce(gradients)
            dist.all_reduce(active, op=dist.ReduceOp.MAX)
            assert torch.isfinite(gradients).all()
            base = optimizer.values()
            candidate, update_stats = optimizer.propose(gradients, active.bool())
            del gradients, values, losses, prediction
            model.eval()
            with torch.no_grad():
                _, before_local, _, before_routes, _ = components(actor.forward_pass(check_batch), check_batch, cfg)
                before = global_mean(before_local)
                optimizer.assign(base + candidate)
                _, after_local, _, after_routes, _ = components(actor.forward_pass(check_batch), check_batch, cfg)
                after = global_mean(after_local)
            risk, scale = localization_risk(before, after, protocol['risk_relative_tolerance'])
            active_risks = torch.nonzero(risk > 0, as_tuple=False)
            corrected = candidate
            correction_stats = dict(correction_norm=0., active_risk_constraints=0)
            if len(active_risks):
                _, objectives, _, _, _ = components(actor.forward_pass(check_batch), check_batch, cfg)
                risk_gradients = []
                for number, (task, target) in enumerate(active_risks.tolist()):
                    values = torch.autograd.grad(objectives[task, target] / world, optimizer.parameters,
                                                retain_graph=number + 1 < len(active_risks), allow_unused=True)
                    flat = flatten_gradients(values, optimizer.parameters)
                    dist.all_reduce(flat)
                    risk_gradients.append(flat)
                excess = (after - before - protocol['risk_relative_tolerance'] * scale)[risk > 0]
                corrected, correction_stats = correct_displacement(candidate, torch.stack(risk_gradients), excess,
                                                                     shared_mask, protocol['max_relative_correction'])
                del objectives, risk_gradients, values, flat
            use_trajectory = bool(len(active_risks)) or global_step % protocol['trajectory_period'] == 0 or bool(args.sanity_steps)
            clip_info, clip_infos, baseline_trajectory, selected_trajectory = None, None, None, None
            baseline_details, trial_records = None, []
            if use_trajectory:
                frames, boxes, clip_info = clips.sample()
                clip_infos = [None] * world
                dist.all_gather_object(clip_infos, clip_info)
                optimizer.assign(base)
                score, local_details = trajectory.evaluate(frames, boxes)
                baseline_details = [None] * world
                dist.all_gather_object(baseline_details, local_details)
                baseline_trajectory = torch.zeros(3, device=device)
                baseline_trajectory[rank] = score
                dist.all_reduce(baseline_trajectory)
            strengths = protocol['correction_strengths'] if len(active_risks) else [0.0]
            choices = []
            for alpha in strengths:
                displacement = candidate + alpha * (corrected - candidate)
                optimizer.assign(base + displacement)
                with torch.no_grad():
                    if alpha == 0:
                        value, routes = after, after_routes
                    else:
                        _, local_value, _, routes, _ = components(actor.forward_pass(check_batch), check_batch, cfg)
                        value = global_mean(local_value)
                    candidate_risk, _ = localization_risk(before, value, protocol['risk_relative_tolerance'])
                    trajectory_value = None
                    if use_trajectory:
                        score, trajectory_details = trajectory.evaluate(frames, boxes)
                        trajectory_value = torch.zeros(3, device=device)
                        trajectory_value[rank] = score
                        dist.all_reduce(trajectory_value)
                        details_by_task = [None] * world
                        dist.all_gather_object(details_by_task, trajectory_details)
                    cost = candidate_risk.mean()
                    if use_trajectory:
                        trajectory_risk, _ = localization_risk(baseline_trajectory, trajectory_value, protocol['risk_relative_tolerance'])
                        cost = cost + protocol['trajectory_risk_weight'] * trajectory_risk.mean()
                    distance = (displacement - candidate).norm() / candidate.norm().clamp_min(1e-12)
                    cost = cost + protocol['candidate_distance_weight'] * distance
                choices.append((cost.item(), alpha, displacement, value, routes, trajectory_value))
                trial_records.append(dict(alpha=alpha, cost=cost.item(), localization=value.tolist(),
                                          trajectory=None if trajectory_value is None else trajectory_value.tolist(),
                                          trajectory_details=details_by_task if use_trajectory else None))
            selected = min(choices, key=lambda item: item[0])
            _, alpha, displacement, accepted, accepted_routes, selected_trajectory = selected
            optimizer.assign(base + displacement)
            accepted_risk, _ = localization_risk(before, accepted, protocol['risk_relative_tolerance'])
            candidate_route_change = route_changes(before_routes, after_routes)
            accepted_route_change = route_changes(before_routes, accepted_routes)
            metric_vector = global_mean(torch.stack([metrics[key] for key in sums]))
            for key, value in zip(sums, metric_vector.tolist()):
                sums[key] += value
            torch.cuda.synchronize(device)
            record = dict(attempt_id=receipt['attempt_id'], epoch=epoch, step=step, global_step=global_step, before_localization=before.tolist(),
                          candidate_localization=after.tolist(), accepted_localization=accepted.tolist(),
                          candidate_risk=risk.tolist(), accepted_risk=accepted_risk.tolist(),
                          candidate_route_change=candidate_route_change.tolist(), accepted_route_change=accepted_route_change.tolist(),
                          selected_correction_strength=alpha, trajectory_checked=use_trajectory,
                          baseline_trajectory=None if baseline_trajectory is None else baseline_trajectory.tolist(),
                          accepted_trajectory=None if selected_trajectory is None else selected_trajectory.tolist(),
                          training_clips=clip_infos, global_task_counts=global_counts.tolist(),
                          baseline_trajectory_details=baseline_details, candidate_trials=trial_records,
                          pair_check_forward_examples=9 * (2 + int(bool(len(active_risks))) + len(strengths) - 1),
                          trajectory_model_forward_examples=6 * (1 + len(strengths)) if use_trajectory else 0,
                          routing_check_mode='eval_no_noise_no_dropout_same_augmented_training_pairs',
                          optimizer_statistics_updates=1, weight_decay_applications=1,
                          step_seconds=time.perf_counter() - tick,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          **update_stats, **correction_stats)
            if rank == 0:
                with (root / 'updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                if step % 10 == 0 or step == steps:
                    receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), epoch=epoch, step=step,
                                   global_step=global_step, latest=record, training_examples=24 * global_step,
                                   check_examples=9 * global_step, wallclock_seconds=time.perf_counter() - started)
                    (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
                    print(json.dumps({key: receipt[key] for key in ['epoch', 'step', 'global_step', 'wallclock_seconds']}), flush=True)
        assert step == steps
        epoch_history.append(dict(epoch=epoch, steps=step, means={key: value / steps for key, value in sums.items()}))
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                   torch_cuda=torch.cuda.get_rng_state(device))
        rng_by_rank = [None] * world
        dist.all_gather_object(rng_by_rank, rng)
        replicas = [None] * world
        digest = hashlib.sha256(optimizer.values().cpu().numpy().tobytes()).hexdigest()
        dist.all_gather_object(replicas, digest)
        assert len(set(replicas)) == 1, replicas
        if rank == 0:
            path = root / ('XTrack_ep%04d.pth.tar' % epoch)
            temporary = root / 'checkpoint.tmp'
            torch.save(dict(net=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, global_step=global_step,
                            protocol=protocol, epoch_history=epoch_history, rng_by_rank=rng_by_rank,
                            replicas_identical=replicas), temporary)
            os.replace(str(temporary), str(path))
            shutil.copyfile(path, root / 'latest.tmp')
            os.replace(str(root / 'latest.tmp'), str(root / 'latest.pth.tar'))
            receipt.update(last_checkpoint=str(path), completed_epochs=epoch, replica_parameter_sha256=digest)
            if args.sanity_steps:
                saved = torch.load(path, map_location='cpu')
                model.load_state_dict(saved['net'], strict=True)
                frozen = [name for name in saved['net'] if 'MeME' not in name]
                assert all(torch.equal(saved['net'][name], initializer['net'][name]) for name in frozen)
                assert all(torch.isfinite(tensor).all().item() for tensor in saved['net'].values())
                assert all(torch.isfinite(state[key]).all().item() for state in saved['optimizer']['state'] for key in ['m', 'v'])
                assert saved['replicas_identical'] == replicas
                receipt['sanity_checkpoint_check'] = dict(strict_reload=True, frozen_tensors_unchanged=True,
                                                         frozen_tensor_count=len(frozen), optimizer_state_finite=True,
                                                         replicas_identical=True, rng_rank_count=len(saved['rng_by_rank']))
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
