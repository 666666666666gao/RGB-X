"""One real balanced batch from training data only; no model, CUDA or updates."""
import argparse
import json
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx import bootstrap
import numpy as np
import torch
from cad_rgbx.model import configure
from cad_rgbx.data import build_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    protocol = json.loads((PROJECT / 'configs/cad_rgbx_b0_s2026.json').read_text())
    random.seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed'])
    loader, datasets = build_loader(configure(protocol), protocol, rank=0, steps=1, workers=0)
    batch = next(iter(loader))
    assert list(batch['template_images'].shape) == [1,12,6,128,128]
    assert list(batch['search_images'].shape) == [1,12,6,256,256]
    assert all(torch.isfinite(batch[key]).all() for key in ['template_images','search_images','search_anno'])
    counts = {name: batch['dataset'].count(name) for name in ['lasher','depthtrack','visevent']}
    assert list(counts.values()) == [4,4,4]
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS_REAL_THREE_TASK_TRAINING_DATA_BATCH_CPU_ONLY',
        observed_at=datetime.now(timezone.utc).isoformat(), train_sequences=[value.get_num_sequences() for value in datasets],
        local_task_counts=counts, template_shape=list(batch['template_images'].shape),
        search_shape=list(batch['search_images'].shape), cuda_initialized=False,
        model_forward_calls=0, training_updates=0, formal_tracking_scores='NOT_RUN')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
