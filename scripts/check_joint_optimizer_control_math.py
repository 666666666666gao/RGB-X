"""CPU-only semantic checks; no datasets, model forward, DDP or benchmark run."""
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rgbx_controls.updates import normalized_joint_loss, apply_joint_adamw
from rgbx_method.optimizer import TaskAdaptiveAdamW


def main():
    assert not torch.cuda.is_initialized()
    dtype = torch.float64
    parameter = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    inputs = torch.arange(1, 25, dtype=dtype) / 10
    targets = torch.arange(24, dtype=dtype) / 20
    task_ids = torch.arange(24) % 3
    global_counts = torch.tensor([8, 8, 8])
    per_example = (parameter * inputs - targets).square()
    local_objectives, balances = [], []
    for rank in range(3):
        indices = slice(8 * rank, 8 * (rank + 1))
        ids = task_ids[indices]
        values = per_example[indices]
        balance = (parameter * (rank + 1)).square()
        balances.append(balance)
        counts = torch.stack([(ids == task).sum() for task in range(3)])
        losses = torch.stack([values[ids == task].mean() + balance for task in range(3)])
        local_objectives.append(normalized_joint_loss(losses, counts, global_counts))
    distributed_objective = torch.stack(local_objectives).sum()
    direct_objective = per_example.mean() + torch.stack(balances).mean()
    distributed_gradient = torch.autograd.grad(distributed_objective, parameter, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(direct_objective, parameter)[0]
    assert torch.allclose(distributed_objective, direct_objective, rtol=0, atol=1e-14)
    assert torch.allclose(distributed_gradient, direct_gradient, rtol=0, atol=1e-14)

    shared = torch.nn.Parameter(torch.tensor(2., dtype=dtype))
    sparse = torch.nn.Parameter(torch.tensor(-0.5, dtype=dtype))
    optimizer = torch.optim.AdamW([shared, sparse], lr=1e-3, weight_decay=0.1,
                                  betas=(0.9, 0.999), eps=1e-8)
    stats = apply_joint_adamw(optimizer, torch.tensor([2., 0.], dtype=dtype),
                             torch.tensor([True, False]), 0.1)
    clipped = 2 * 0.1 / (2 + 1e-6)
    expected = 2 * (1 - 1e-3 * 0.1) - 1e-3 * clipped / (clipped + 1e-8)
    assert abs(shared.item() - expected) < 1e-14
    assert sparse.item() == -0.5 and sparse not in optimizer.state
    assert optimizer.state[shared]['step'].item() == 1
    previous = shared.item()
    apply_joint_adamw(optimizer, torch.zeros(2, dtype=dtype), torch.tensor([False, True]), 0.1)
    assert shared.item() == previous
    assert abs(sparse.item() - (-0.5 * (1 - 1e-3 * 0.1))) < 1e-14
    assert optimizer.state[sparse]['step'].item() == 1

    parameters = [torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for value in [2., -0.5, 1.]]
    adaptive = TaskAdaptiveAdamW(zip(['MeME.shared', 'MeME.experts.0.weight', 'MeME.unused'], parameters),
                               lr=1e-3, weight_decay=0.1, clip_norm=0.1)
    gradients = torch.tensor([[1., 1., 0.], [1., 1., 0.], [-2., -2., 0.]], dtype=torch.float32)
    active = torch.tensor([[True, True, False]] * 3)
    base = adaptive.values()
    displacement, adaptive_stats = adaptive.propose(gradients, active)
    direction = (2 / (1 + 1e-8) - 2 / (2 + 1e-8)) / 3
    expected_displacement = torch.tensor([-1e-3 * direction - 1e-3 * 0.1 * 2,
                                         -1e-3 * 0.1 * -0.5, 0], dtype=torch.float32)
    assert torch.allclose(displacement, expected_displacement, rtol=0, atol=1e-9)
    assert adaptive.state[0]['steps'] == [1, 1, 1]
    assert adaptive.state[1]['steps'] == [1]
    assert adaptive.state[2]['steps'] == [0, 0, 0]
    adaptive.assign(base + displacement)
    assert parameters[2].item() == 1.
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS_CPU_CONTROL_ALGEBRA_AND_OPTIMIZER_SEMANTICS_NOT_DDP_OR_BENCHMARK',
        torch_version=torch.__version__, dtype=dict(loss_and_adamw='float64', task_stats='production_float32'), model_forward_calls=0, dataset_reads=0,
        cuda_initialized=False, mixed_rank_gradient_error=abs(distributed_gradient.item() - direct_gradient.item()),
        full_mixed_batch_balance_gradient_preserved=True, ordinary_adamw_first_step=shared.item(),
        sparse_none_gradient_skips_state_and_decay=True, sparse_zero_gradient_commits_one_state_and_decay=True,
        adaptive_shared_delta=displacement[0].item(), adaptive_routed_expert_delta=displacement[1].item(),
        adaptive_three_shared_states_one_expert_state=True, inactive_adaptive_parameter_skips_state_and_decay=True,
        scope='Synthetic CPU checks of global normalization, ordinary AdamW, sparse activation and existing task statistics. No GPU/model/real-data preflight, saved-state resume, evaluation, training result or canonical AdaTask acceptance.')))


if __name__ == '__main__':
    main()
