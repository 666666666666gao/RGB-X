"""Joint-loss normalization and ordinary AdamW on the same mixed-task objective."""
import torch


def normalized_joint_loss(task_losses, local_counts, global_counts):
    return (task_losses * local_counts / global_counts).sum() / 3


def flatten_gradients(gradients, parameters):
    return torch.cat([gradient.flatten() if gradient is not None else torch.zeros_like(parameter).flatten()
                      for gradient, parameter in zip(gradients, parameters)])


def parameter_manifest(named_parameters, task_adaptive):
    result = []
    for name, parameter in named_parameters:
        shared = '.experts.' not in name
        category = 'router' if name.endswith(('w_gate', 'w_noise')) else ('shared_content' if shared else 'routed_expert')
        result.append(dict(name=name, shape=list(parameter.shape), numel=parameter.numel(), category=category,
                           task_states=3 if task_adaptive and shared else 1, risk_corrected=False))
    return result


def apply_joint_adamw(optimizer, joint_gradient, active, clip_norm):
    assert len(optimizer.param_groups) == 1
    parameters = optimizer.param_groups[0]['params']
    norm = joint_gradient.norm().item()
    coefficient = min(1.0, clip_norm / (norm + 1e-6))
    gradient = joint_gradient * coefficient
    offset = 0
    for index, parameter in enumerate(parameters):
        end = offset + parameter.numel()
        parameter.grad = gradient[offset:end].view_as(parameter) if active[index].item() else None
        offset = end
    assert offset == joint_gradient.numel()
    optimizer.step()
    return dict(joint_grad_norm_before_clip=norm, gradient_clip_coefficient=coefficient)
