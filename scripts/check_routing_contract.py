"""Exercise the pinned routing implementation on a deterministic CPU fixture."""
import argparse
import ast
import json
import sys
from pathlib import Path

import torch

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / 'third_party/XTrack'))
from lib.models.layers.moe_lora import MoE_lora

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
module = MoE_lora(4, 4, 6, 4, patch_num=6)
x = torch.ones(2, 4, 4)
z = torch.ones(2, 2, 4)
clean = torch.tensor([1.5, 1.2, 0.2, -0.3, -0.6, -1.0])
with torch.no_grad():
    module.w_gate.copy_(clean.expand(6, -1) / 6)
torch.manual_seed(2026)
rng = torch.get_rng_state()
gates_before, load_before, supervised_before, _ = module.noisy_top_k_gating(x, z, True)
torch.set_rng_state(rng)
noise = torch.randn(2, 6) * (torch.nn.functional.softplus(torch.zeros(2, 6)) + 0.01)
expected_probabilities = torch.softmax(clean.expand(2, -1) + noise, dim=1)
with torch.no_grad():
    module.w_gate.add_(10 / 6)
torch.set_rng_state(rng)
gates_after, load_after, _, _ = module.noisy_top_k_gating(x, z, True)
tree = ast.parse((project / 'third_party/XTrack/lib/models/xtrack/vit_ce_adapter.py').read_text())
appended = [node.args[0].id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'logits_prompt' and node.func.attr == 'append']
checks = {
    'top_k_and_fusion_weights_translation_invariant': torch.allclose(gates_before, gates_after, atol=1e-6),
    'estimated_load_translation_invariant': torch.allclose(load_before, load_after, atol=1e-5),
    'router_supervision_uses_one_softmax': torch.allclose(supervised_before, expected_probabilities, atol=1e-6),
    'attention_and_ffn_supervision_both_included_once': appended == ['logits_attn', 'logits_ffn'],
}
report = dict(status='PASS' if all(checks.values()) else 'FAIL_ROUTING_CONTRACT', checks=checks,
              load_before=load_before.tolist(), load_after=load_after.tolist(), append_arguments=appended,
              role='CPU_SYNTHETIC_IMPLEMENTATION_CHECK_NOT_TRACKING_ACCURACY')
Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
sys.exit(0 if all(checks.values()) else 1)
