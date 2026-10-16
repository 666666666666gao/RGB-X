"""CPU key/shape audit only; this is not a GPU training readiness check."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx import bootstrap
import torch
from cad_rgbx.model import build_random_model, state_digest
from cad_rgbx.sot_baseline import configure_sot, inspect_initializer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    model = build_random_model(configure_sot(protocol))
    source, manifest = inspect_initializer(model, protocol['pretrained_file'])
    assert manifest['source_sha256'] == protocol['pretrained_sha256']
    assert not manifest['shape_mismatches'], manifest['shape_mismatches']
    before = state_digest(model)
    state = model.state_dict()
    state.update({name: source[name] for name in manifest['loaded_keys']})
    model.load_state_dict(state, strict=True)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='CPU_INITIALIZER_COVERAGE_ONLY_NOT_GPU_READY',
        initializer_manifest=manifest, random_state_sha256=before,
        initial_state_sha256=state_digest(model), pretrained_scope='backbone_only',
        new_fusion_and_head_random=True, cuda_initialized=False, model_forward_calls=0)), flush=True)


if __name__ == '__main__':
    main()
