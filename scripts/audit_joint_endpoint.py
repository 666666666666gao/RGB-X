"""CPU audit of an author release or a completed full-method joint checkpoint."""
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

AUTHOR_SHA = '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['author_release', 'full_method'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run')
    parser.add_argument('--controller')
    args = parser.parse_args()
    checkpoint_path, config_path = Path(args.checkpoint).resolve(strict=True), Path(args.config).resolve(strict=True)
    digest = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    update_config_from_file(str(config_path))
    model = build_xtrack(cfg, training=False)
    model.load_state_dict(checkpoint['net'], strict=True)
    assert all(torch.isfinite(value).all().item() for value in checkpoint['net'].values())
    report = dict(status='PASS', role='joint_rgbx_checkpoint_audit_not_tracking_benchmark',
                  endpoint_kind=args.kind, checkpoint=str(checkpoint_path), checkpoint_sha256=digest,
                  checkpoint_bytes=checkpoint_path.stat().st_size, config=str(config_path),
                  config_sha256=sha(config_path), strict_model_load=True, all_model_tensors_finite=True,
                  observed_at=datetime.now(timezone.utc).isoformat(), formal_tracking_scores='NOT_RUN')
    if args.kind == 'author_release':
        assert digest == AUTHOR_SHA and checkpoint_path.stat().st_size == 441674839
        report.update(source='https://huggingface.co/taryya/XTrack/resolve/main/XTrack_Base.pth.tar',
                      training_provenance='AUTHOR_RELEASE_NOT_LOCAL_REPRODUCTION', author_saved_epoch=checkpoint['epoch'])
    else:
        run = json.loads(Path(args.run).read_text())
        controller = json.loads(Path(args.controller).read_text())
        protocol = json.loads((PROJECT / 'configs/full_method_v1.json').read_text())
        assert checkpoint['protocol'] == run['protocol'] == protocol
        assert run['status'] == 'TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT'
        assert controller['status'] == 'EXITED' and controller['exit_code'] == 0
        assert controller['run_id'] == protocol['run_id'] == 'xtrack_full_method_v1_s2026'
        assert checkpoint['epoch'] == run['completed_epochs'] == protocol['epochs'] == 15
        assert checkpoint['global_step'] == run['global_step'] == 37500
        assert Path(run['last_checkpoint']).resolve(strict=True) == checkpoint_path
        history = checkpoint['epoch_history']
        assert [row['epoch'] for row in history] == list(range(1, 16))
        assert all(row['steps'] == 2500 and all(math.isfinite(value) for value in row['means'].values()) for row in history)
        replicas = checkpoint['replicas_identical']
        assert len(replicas) == 3 and len(set(replicas)) == 1
        initial_path = Path(protocol['initial_checkpoint'])
        assert sha(initial_path) == run['initial_checkpoint_sha256'] == AUTHOR_SHA
        initial = torch.load(initial_path, map_location='cpu')['net']
        frozen = [name for name in checkpoint['net'] if 'MeME' not in name]
        assert all(torch.equal(checkpoint['net'][name], initial[name]) for name in frozen)
        state = checkpoint['optimizer']
        assert len(state['names']) == len(state['state'])
        assert all(torch.isfinite(item[key]).all().item() for item in state['state'] for key in ['m', 'v'])
        assert all(0 <= step <= 37500 for item in state['state'] for step in item['steps'])
        witnesses = [index for index, name in enumerate(state['names']) if name.endswith(('MeME_attn.ffn1_1.weight', 'MeME_attn.ffn1_1.bias'))]
        assert len(witnesses) == 24
        assert all(state['state'][index]['steps'] == [37500, 37500, 37500] for index in witnesses)
        report.update(run_id=protocol['run_id'], epoch=15, global_step=37500, seed=protocol['seed'],
                      controller_exit_code=0, replicas_identical=True, frozen_tensors_unchanged=True,
                      frozen_tensor_count=len(frozen), optimizer_finite=True, always_used_witness_tensors=24,
                      run_sha256=sha(args.run), controller_sha256=sha(args.controller),
                      protocol_sha256=sha(PROJECT / 'configs/full_method_v1.json'),
                      training_provenance='AUTHOR_RELEASE_PLUS_FULL_METHOD_FINETUNE_NOT_EQUAL_TOTAL_BUDGET')
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
