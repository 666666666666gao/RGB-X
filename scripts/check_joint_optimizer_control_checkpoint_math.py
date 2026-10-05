"""CPU-only round-trip checks of the newly prepared control state auditor."""
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rgbx_controls.checkpoint import verify_optimizer_state
from rgbx_controls.updates import apply_joint_adamw
from rgbx_method.optimizer import TaskAdaptiveAdamW


def rejects(saved, named, protocol):
    # These are deliberate negative checks of audit rejection, not a runtime fallback.
    try:
        verify_optimizer_state(named, saved, protocol)
    except AssertionError:
        return True
    return False


def serialize(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def main():
    outcomes = []
    for kind in ['adamw', 'task_adaptive_only']:
        protocol = dict(optimizer=kind, lr=1e-5, lr_drop_epoch=10, lr_drop_factor=0.1,
                        weight_decay=1e-4, grad_clip_norm=0.1)
        named = [('backbone.block0.MeME_attn.ffn1_1.weight', torch.nn.Parameter(torch.ones(2))),
                 ('backbone.block0.MeME_attn.experts.0.weight', torch.nn.Parameter(torch.ones(3))),
                 ('backbone.block0.MeME_attn.unused.weight', torch.nn.Parameter(torch.ones(1)))]
        if kind == 'adamw':
            optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=1e-5, weight_decay=1e-4)
            for _ in range(2):
                apply_joint_adamw(optimizer, torch.tensor([1., -2., 3., 0., -1., 0.]), torch.tensor([True, True, False]), 0.1)
        else:
            optimizer = TaskAdaptiveAdamW(named, 1e-5, 1e-4, 0.1)
            for _ in range(2):
                base = optimizer.values()
                displacement, _ = optimizer.propose(torch.tensor([[1., -2., 3., 0., -1., 0.],
                                                                  [2., -1., 0., 1., -3., 0.],
                                                                  [-3., 2., 1., 0., 1., 0.]]),
                                                     torch.tensor([[True, True, False]] * 3))
                optimizer.assign(base + displacement)
        assert torch.equal(named[2][1].detach(), torch.ones_like(named[2][1]))
        saved = dict(trainable_parameter_names=[name for name, _ in named], optimizer_kind=kind,
                     optimizer=optimizer.state_dict(), epoch=1, global_step=2)
        raw = serialize(saved)
        restored = torch.load(io.BytesIO(raw), map_location='cpu')
        vector = torch.cat([parameter.detach().flatten() for _, parameter in named]).clone()
        receipt = verify_optimizer_state(named, restored, protocol)
        assert receipt['always_used_witness_tensors'] == 1
        assert receipt['parameter_update_calls'] == 0
        assert torch.equal(vector, torch.cat([parameter.detach().flatten() for _, parameter in named]))
        assert hashlib.sha256(serialize(restored)).hexdigest() == hashlib.sha256(raw).hexdigest()
        wrong_names = copy.deepcopy(restored)
        wrong_names['trainable_parameter_names'] = list(reversed(wrong_names['trainable_parameter_names']))
        assert rejects(wrong_names, named, protocol)
        damaged = copy.deepcopy(restored)
        if kind == 'adamw':
            state = next(iter(damaged['optimizer']['state'].values()))
            state['exp_avg'].flatten()[0] = float('nan')
        else:
            damaged['optimizer']['state'][0]['m'].flatten()[0] = float('nan')
        assert rejects(damaged, named, protocol)
        wrong_budget = copy.deepcopy(restored)
        wrong_budget['global_step'] = 1
        assert rejects(wrong_budget, named, protocol)
        outcomes.append(dict(optimizer=kind, serialized_cpu_roundtrip=True, loaded_state_valid=True,
                             parameter_values_unchanged=True, serialized_input_unchanged=True,
                             wrong_parameter_order_rejected=True, nonfinite_moment_rejected=True,
                             optimizer_steps_exceeding_committed_updates_rejected=True,
                             sparse_unused_parameter_retained_without_update=True))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS_CPU_CONTROL_OPTIMIZER_STATE_ROUNDTRIP_AND_REJECTION_NOT_REAL_CHECKPOINT_AUDIT',
                          checks=outcomes, cuda_initialized=False, model_forward_calls=0,
                          dataset_reads=0, generated_checkpoint_files=0,
                          scope='In-memory synthetic CPU optimizer-state serialization/reload and rejection checks only. Exact real XTrack checkpoint/model/RNG audits, DDP and benchmarks remain unexecuted.')))


if __name__ == '__main__':
    main()
