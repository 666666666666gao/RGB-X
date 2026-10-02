"""TraX entry point using the joint checkpoint and the author's RGB-D preprocessing."""
import argparse
import json
import random
import sys

import numpy as np
import torch
from trax import Server, TraxStatus
from trax.region import Rectangle

from lib.test.tracker.xtrack import XTrack
from lib.train.dataset.depth_utils import get_rgbd_frame
from xtrack_eval_parameters import parameters

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--expected-sha256', required=True)
parser.add_argument('--terminal-audit', required=True)
parser.add_argument('--seed', type=int, default=2026)
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_num_threads(1)
assert args.seed == 2026
params = parameters(args.config, args.checkpoint, args.expected_sha256, args.terminal_audit)
tracker = XTrack(params)
print(json.dumps(dict(**params.evaluation_identity, seed=args.seed, strict_load=True)),
      file=sys.stderr, flush=True)
with Server(['rectangle'], ['path'], image_channels=['color', 'depth']) as handle:
    while True:
        request = handle.wait()
        if request.type == TraxStatus.QUIT:
            break
        assert set(request.image) == {'color', 'depth'}
        image = get_rgbd_frame(request.image['color'].path(), request.image['depth'].path(),
                               dtype='rgbcolormap', depth_clip=True)
        if request.type == TraxStatus.INITIALIZE:
            torch.cuda.reset_peak_memory_stats()
            assert len(request.objects) == 1
            region, _ = request.objects[0]
            assert region.type == 'rectangle'
            box = region.bounds()
            tracker.initialize(image, {'init_bbox': list(box)})
            properties = {}
        else:
            assert request.type == TraxStatus.FRAME
            output = tracker.track(image)
            box, confidence = output['target_bbox'], output['best_score']
            assert np.isfinite(box).all() and np.isfinite(confidence)
            assert box[2] > 0 and box[3] > 0
            properties = {'confidence': float(confidence)}
        torch.cuda.synchronize()
        properties.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        handle.status([(Rectangle.create(*box), properties)])
