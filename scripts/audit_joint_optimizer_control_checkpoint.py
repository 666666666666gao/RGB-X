"""Read-only CPU audit of an exact C/D sanity, epoch1 or terminal checkpoint."""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'third_party/XTrack'), str(PROJECT / 'third_party/XTrack/lib/train')]
from lib.config.xtrack.config import cfg, update_config_from_file
from lib.models.xtrack import build_xtrack
from rgbx_controls.checkpoint import verify_optimizer_state, verify_rng_states

AUTHOR_SHA = '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
PROTOCOLS = {
    'adamw': ('xtrack_fixed_mixed_adamw_s2026', 'configs/xtrack_fixed_mixed_adamw_s2026.json'),
    'task_adaptive_only': ('xtrack_task_stats_only_s2026', 'configs/xtrack_task_stats_only_s2026.json'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', choices=list(PROTOCOLS), required=True)
    parser.add_argument('--stage', choices=['sanity', 'epoch1', 'endpoint'], required=True)
    parser.add_argument('--execution')
    args = parser.parse_args()
    run_id, relative_protocol = PROTOCOLS[args.optimizer]
    protocol_path = PROJECT / relative_protocol
    protocol = json.loads(protocol_path.read_text())
    assert protocol['run_id'] == run_id and protocol['optimizer'] == args.optimizer
    assert protocol['seed'] == 2026 and protocol['epochs'] == 15 and protocol['samples_per_epoch'] == 60000
    assert protocol['world_size'] == 3 and protocol['batch_per_rank'] == 8 and not protocol['amp']
    root = Path('/home/gaob/rgbx-full-method') / (run_id + ('_sanity' if args.stage == 'sanity' else ''))
    assert Path(protocol['output']) == Path('/home/gaob/rgbx-full-method') / run_id
    epoch = 15 if args.stage == 'endpoint' else 1
    checkpoint_path = root / ('XTrack_ep%04d.pth.tar' % epoch)
    saved = torch.load(checkpoint_path, map_location='cpu')
    run_path = root / 'run.json'
    run = json.loads(run_path.read_text())
    steps = 2 if args.stage == 'sanity' else 2500
    updates = epoch * steps
    assert saved['epoch'] == epoch and saved['global_step'] == updates
    assert saved['sanity_steps'] == (2 if args.stage == 'sanity' else 0)
    assert saved['protocol'] == run['protocol'] == protocol
    assert run['run_id'] == run_id and run['optimizer_kind'] == args.optimizer and run['output'] == str(root)
    assert run['protocol_sha256'] == sha(protocol_path)
    assert run['modules_enabled'] == ([] if args.optimizer == 'adamw' else ['task_adaptive_statistics'])
    assert all(run[key] == 0 for key in ['localization_check_calls', 'correction_calls', 'trajectory_calls', 'check_examples'])
    assert run['unused_check_loader_constructed_not_iterated']
    if args.stage != 'epoch1':
        expected_status = 'SANITY_COMPLETE_NOT_FORMAL_RESULT' if args.stage == 'sanity' else 'TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT'
        assert run['status'] == expected_status and run['global_step'] == updates and run['completed_epochs'] == epoch
        assert Path(run['last_checkpoint']).resolve(strict=True) == checkpoint_path.resolve(strict=True)
        execution = json.loads(Path(args.execution).read_text())
        assert execution['run_id'] == run_id and execution['stage'] == args.stage
        assert execution['status'] == 'EXITED' and execution['exit_code'] == 0
        assert execution['protocol_sha256'] == sha(protocol_path)
    history = saved['epoch_history']
    assert [row['epoch'] for row in history] == list(range(1, epoch + 1))
    assert all(row['steps'] == steps and all(math.isfinite(value) for value in row['means'].values()) for row in history)
    update_config_from_file(str(PROJECT / protocol['base_config']))
    model = build_xtrack(cfg, training=False)
    model.load_state_dict(saved['net'], strict=True)
    assert all(torch.isfinite(value).all().item() for value in saved['net'].values())
    initial_path = Path(protocol['initial_checkpoint'])
    assert sha(initial_path) == run['initial_checkpoint_sha256'] == AUTHOR_SHA
    initial = torch.load(initial_path, map_location='cpu')['net']
    frozen = [name for name in saved['net'] if 'MeME' not in name]
    assert len(frozen) == 244 and all(torch.equal(saved['net'][name], initial[name]) for name in frozen)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('MeME' in name)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(named) == 504 and sum(parameter.numel() for _, parameter in named) == 5821440
    optimizer_check = verify_optimizer_state(named, saved, protocol)
    assert optimizer_check['always_used_witness_tensors'] == 24
    vector = torch.cat([parameter.detach().flatten() for _, parameter in named])
    digest = hashlib.sha256(vector.numpy().tobytes()).hexdigest()
    replicas = saved['replicas_identical']
    assert len(replicas) == 3 and replicas == [digest] * 3
    rng_check = verify_rng_states(saved['rng_by_rank'])
    status = {'sanity': 'PASS_CONTROL_SANITY_CHECKPOINT_NOT_FORMAL_RESULT',
              'epoch1': 'PASS_CONTROL_EPOCH1_RECOVERY_CHECKPOINT_NOT_BENCHMARK', 'endpoint': 'PASS'}[args.stage]
    report = dict(status=status, role='joint_rgbx_checkpoint_audit_not_tracking_benchmark',
                  endpoint_kind='optimizer_control', stage=args.stage, run_id=run_id,
                  checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                  checkpoint_bytes=checkpoint_path.stat().st_size, epoch=epoch, global_step=updates, seed=2026,
                  protocol=str(protocol_path), protocol_sha256=sha(protocol_path),
                  config=str(PROJECT / protocol['base_config']), config_sha256=sha(PROJECT / protocol['base_config']),
                  strict_model_load=True, all_model_tensors_finite=True, frozen_tensors_unchanged=True,
                  frozen_tensor_count=len(frozen), replicas_identical=True, saved_replica_hash_matches_loaded_parameters=True,
                  optimizer_check=optimizer_check, rng_check=rng_check, training_mutations=0,
                  cuda_initialized=torch.cuda.is_initialized(), model_forward_calls=0, formal_tracking_scores='NOT_RUN',
                  observed_at=datetime.now(timezone.utc).isoformat(), run_sha256=sha(run_path), script_sha256=sha(__file__),
                  scope='Exact saved-state CPU audit; no real CUDA restoration, model forward, training update or benchmark. Evaluation admission and actual tracking must be verified separately.')
    assert not report['cuda_initialized']
    if args.stage != 'epoch1':
        report.update(controller_exit_code=0, execution_receipt=str(args.execution), execution_receipt_sha256=sha(args.execution))
    output = PROJECT / 'reports' / (run_id + '_' + args.stage + '_checkpoint_audit.json')
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
