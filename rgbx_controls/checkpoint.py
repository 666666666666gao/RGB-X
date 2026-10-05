"""CPU validation of the two saved optimizer-state formats used by C/D."""
import math
import random

import numpy as np
import torch

from rgbx_method.optimizer import TaskAdaptiveAdamW


def verify_optimizer_state(named, saved, protocol):
    names = [name for name, _ in named]
    assert saved['trainable_parameter_names'] == names
    assert saved['optimizer_kind'] == protocol['optimizer']
    updates = saved['global_step']
    expected_lr = protocol['lr'] * (protocol['lr_drop_factor'] if saved['epoch'] > protocol['lr_drop_epoch'] else 1)
    witnesses = [index for index, name in enumerate(names)
                 if name.endswith(('MeME_attn.ffn1_1.weight', 'MeME_attn.ffn1_1.bias'))]
    value = saved['optimizer']
    if protocol['optimizer'] == 'adamw':
        assert len(value['param_groups']) == 1
        group = value['param_groups'][0]
        identifiers = group['params']
        assert len(identifiers) == len(named) and len(set(identifiers)) == len(named)
        assert set(value['state']).issubset(identifiers)
        assert group['lr'] == expected_lr and group['weight_decay'] == protocol['weight_decay']
        assert group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8
        optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=expected_lr,
                                      weight_decay=protocol['weight_decay'])
        optimizer.load_state_dict(value)
        for _, parameter in named:
            if parameter not in optimizer.state:
                continue
            state = optimizer.state[parameter]
            step = state['step'].item()
            assert math.isfinite(step) and step == int(step) and 1 <= step <= updates
            for key in ['exp_avg', 'exp_avg_sq']:
                assert state[key].shape == parameter.shape and torch.isfinite(state[key]).all().item()
            assert (state['exp_avg_sq'] >= 0).all().item()
        assert all(optimizer.state[named[index][1]]['step'].item() == updates for index in witnesses)
        state_count = len(optimizer.state)
    else:
        assert protocol['optimizer'] == 'task_adaptive_only'
        optimizer = TaskAdaptiveAdamW(named, expected_lr, protocol['weight_decay'], protocol['grad_clip_norm'])
        expected_shapes = [state['m'].shape for state in optimizer.state]
        assert value['names'] == names and len(value['state']) == len(named)
        assert value['lr'] == expected_lr and value['weight_decay'] == protocol['weight_decay']
        assert value['clip_norm'] == protocol['grad_clip_norm']
        assert (value['beta1'], value['beta2'], value['eps']) == (0.9, 0.999, 1e-8)
        optimizer.load_state_dict(value)
        for state, shape in zip(optimizer.state, expected_shapes):
            assert state['m'].shape == state['v'].shape == shape and len(state['steps']) == shape[0]
            assert all(isinstance(step, int) and 0 <= step <= updates for step in state['steps'])
            assert torch.isfinite(state['m']).all().item() and torch.isfinite(state['v']).all().item()
            assert (state['v'] >= 0).all().item()
        assert all(optimizer.state[index]['steps'] == [updates, updates, updates] for index in witnesses)
        state_count = len(optimizer.state)
    return dict(optimizer_kind=protocol['optimizer'], saved_optimizer_state_count=state_count,
                trainable_parameter_tensors=len(named), always_used_witness_tensors=len(witnesses),
                learning_rate=expected_lr, names_shapes_moments_steps_valid=True,
                optimizer_reload_executed=True, parameter_update_calls=0)


def verify_rng_states(rng_by_rank):
    assert len(rng_by_rank) == 3
    for rng in rng_by_rank:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch_cpu'])
        assert rng['torch_cuda'].dtype == torch.uint8 and rng['torch_cuda'].ndim == 1
        assert rng['torch_cuda'].numel() > 0
    return dict(saved_rank_count=3, python_numpy_cpu_rng_restore_valid=True,
                cuda_rng_bytes_present=True, cuda_rng_restore_executed=False)
