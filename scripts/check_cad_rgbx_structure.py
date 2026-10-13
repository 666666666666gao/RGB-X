"""Check the identified pruned architecture and prohibit pretrained loads during construction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx.bootstrap import CAD_SOURCE
import torch
from lib.config.cadtrack.config import cfg
from lib.models.cadtrack import build_cadtrack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    cfg.MODEL.PRETRAIN_FILE = ''
    cfg.DATA.SEARCH.SIZE = 256
    cfg.DATA.TEMPLATE.SIZE = 128
    cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = True
    cfg.MODEL.BACKBONE.CROSS_LOC = [3, 6, 9]
    cfg.TRAIN.FIX_BN = False
    with patch('torch.load', side_effect=AssertionError('Pretrained construction is forbidden')) as loader:
        model = build_cadtrack(cfg, training=True)
        assert loader.call_count == 0
    state = model.state_dict()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    expected = checkpoint['net']
    assert set(state) == set(expected)
    assert {name: list(value.shape) for name, value in state.items()} == {name: list(value.shape) for name, value in expected.items()}
    assert len(state) == 250 and sum(value.numel() for value in state.values()) == 103142866
    assert all(torch.isfinite(value).all().item() for value in state.values())
    witnesses = ['backbone.patch_embed.proj.weight', 'backbone.blocks.0.attn.qkv.weight', 'box_head.conv1_ctr.0.weight']
    assert all(not torch.equal(state[name], expected[name]) for name in witnesses)
    names = list(dict(model.named_parameters()))
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert not any('MeME' in name or 'w_gate' in name for name in names)
    assert not torch.cuda.is_initialized()
    checksum = hashlib.sha256()
    for name, value in state.items():
        checksum.update(name.encode())
        checksum.update(value.numpy().tobytes())
    report = dict(status='PASS_PRUNED_CAD250_STRUCTURE_AND_RANDOM_INITIALIZATION_CPU_ONLY_NOT_TRAINING',
                  observed_at=datetime.now(timezone.utc).isoformat(), model_type=type(model).__name__,
                  model_source=str(CAD_SOURCE), reference_checkpoint=args.checkpoint, reference_epoch=checkpoint['epoch'],
                  reference_role='Architecture metadata only; no checkpoint values assigned to this model.',
                  model_state_tensors=250, model_state_elements=103142866,
                  named_parameter_tensors=len(names), named_parameter_elements=sum(p.numel() for p in model.parameters()),
                  exact_state_names_and_shapes_match=True, pretrained_load_calls_during_build=0,
                  pretrained_file='', auto_resume=False, all_parameters_require_grad=True,
                  fresh_initial_state_sha256=checksum.hexdigest(), seed=2026,
                  differing_random_weight_witnesses=witnesses,
                  model_shapes={name: list(value.shape) for name, value in state.items()},
                  cuda_initialized=False, model_forward_calls=0, training_updates=0,
                  scope='CPU structure and initialization check only; no GPU preflight/training success or tracking benchmark claim.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status','named_parameter_tensors','named_parameter_elements','fresh_initial_state_sha256','pretrained_load_calls_during_build']}))


if __name__ == '__main__':
    main()
