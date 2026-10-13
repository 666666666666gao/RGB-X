"""Strict CPU saved-state verification; no inference, CUDA restoration or benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx import bootstrap
import torch
from cad_rgbx.model import configure, build_random_model, state_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sanity','formal'], required=True)
    args = parser.parse_args()
    protocol = json.loads((PROJECT/'configs/cad_rgbx_b0_s2026.json').read_text())
    root = Path(protocol['output'] + ('_sanity' if args.stage == 'sanity' else ''))
    run = json.loads((root/'run.json').read_text())
    execution = json.loads(Path(str(root)+'_controller/execution.json').read_text())
    assert execution['status'] == 'EXITED' and execution['exit_code'] == 0
    assert run['status'] == ('SANITY_COMPLETE_NOT_FORMAL_RESULT' if args.stage == 'sanity' else 'TRAINING_COMPLETE_NOT_EVALUATED')
    checkpoint_path = root/'CADTrack_latest.pth.tar'
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    assert checkpoint['protocol'] == protocol and run['protocol'] == protocol
    expected_steps = 2 if args.stage == 'sanity' else protocol['epochs'] * protocol['samples_per_epoch'] // 24
    assert checkpoint['global_step'] == expected_steps and run['global_step'] == expected_steps
    assert checkpoint['epoch'] == (1 if args.stage == 'sanity' else protocol['epochs'])
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    model = build_random_model(configure(protocol))
    assert state_digest(model) == checkpoint['initial_state_sha256'] == protocol['initial_state_sha256']
    model.load_state_dict(checkpoint['net'], strict=True)
    assert all(torch.isfinite(value).all().item() for value in model.state_dict().values())
    assert checkpoint['replica_state_sha256'] == [state_digest(model)] * 2
    assert state_digest(model) != protocol['initial_state_sha256']
    named = list(model.named_parameters())
    assert [name for name,_ in named] == checkpoint['trainable_parameter_names']
    activity = json.loads((root/'first_update_gradient_manifest.json').read_text())
    optimizer = checkpoint['optimizer']
    identifiers = optimizer['param_groups'][0]['params']
    assert len(identifiers) == len(named) == 211
    used = []
    for identifier, (name, parameter) in zip(identifiers, named):
        if identifier in optimizer['state']:
            state = optimizer['state'][identifier]
            assert state['exp_avg'].shape == parameter.shape and state['exp_avg_sq'].shape == parameter.shape
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
            assert int(state['step'].item()) == expected_steps
            used.append(name)
    assert used == activity['active']
    assert len(checkpoint['rng_by_rank']) == 2
    for rng in checkpoint['rng_by_rank']:
        assert set(rng) == {'python','numpy','torch_cpu','torch_cuda'}
        assert rng['torch_cpu'].device.type == rng['torch_cuda'].device.type == 'cpu'
        assert rng['torch_cuda'].dtype == torch.uint8 and rng['torch_cuda'].numel() > 0
    assert not torch.cuda.is_initialized()
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    report = dict(status='PASS',role='CAD_CPU_SAVED_STATE_CHECK_NOT_BENCHMARK',stage=args.stage,
        observed_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest.hexdigest(), checkpoint_bytes=checkpoint_path.stat().st_size,
        strict_model_load=True,all_model_tensors_finite=True,replica_state_hash_matches=True,
        active_optimizer_states=len(used),trainable_parameters=211,global_step=expected_steps,
        pretrained_loaded=False,model_forward_calls=0,cuda_initialized=False,
        cuda_rng_restoration='NOT_EXECUTED_CPU_INSPECTION_ONLY',formal_tracking_scores='NOT_RUN')
    output = root/'saved_state_audit.json'
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
