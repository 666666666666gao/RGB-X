"""Risk measurements and a bounded correction in active risk-gradient space."""
import torch


def localization_risk(before, after, tolerance):
    scale = before.detach().abs().clamp_min(1e-8)
    return ((after - before) / scale - tolerance).clamp_min(0), scale


@torch.no_grad()
def correct_displacement(candidate, gradients, excess, shared_mask, max_relative_correction):
    restricted = gradients * shared_mask
    gram = restricted.double() @ restricted.double().T
    ridge = gram.trace() / len(gradients) * 1e-6 + 1e-12
    coefficients = torch.linalg.solve(gram + ridge * torch.eye(len(gradients), device=gram.device), excess.double())
    correction = -(coefficients @ restricted.double()).to(candidate.dtype)
    limit = max_relative_correction * candidate[shared_mask].norm()
    norm = correction.norm()
    correction.mul_(torch.minimum(torch.ones_like(norm), limit / norm.clamp_min(1e-12)))
    assert torch.isfinite(correction).all()
    return candidate + correction, dict(correction_norm=correction.norm().item(),
                                       candidate_shared_norm=candidate[shared_mask].norm().item(),
                                       active_risk_constraints=len(gradients))
