"""Run every assigned OPE sequence from an audited explicit frame manifest."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from lib.test.tracker.xtrack import XTrack
from lib.train.dataset.depth_utils import get_x_frame
from xtrack_eval_parameters import parameters

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--expected-sha256', required=True)
parser.add_argument('--terminal-audit', required=True)
parser.add_argument('--manifest', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--shard', type=int, required=True)
parser.add_argument('--shards', type=int, default=3)
parser.add_argument('--seed', type=int, default=2026)
args = parser.parse_args()
manifest_path = Path(args.manifest).resolve(strict=True)
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'READY'
assert len(manifest['sequences']) == manifest['expected_sequences']
assert 0 <= args.shard < args.shards
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_num_threads(1)
assert args.seed == 2026
params = parameters(args.config, args.checkpoint, args.expected_sha256, args.terminal_audit)
tracker = XTrack(params)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
assigned = manifest['sequences'][args.shard::args.shards]
assert not any((output / (row['name'] + suffix)).exists()
               for row in assigned for suffix in ('.txt', '.json', '_confidence.txt', '_time.txt'))
for row in assigned:
    gt_path = Path(row['groundtruth'])
    assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == row['groundtruth_sha256']
    gt = np.loadtxt(gt_path, delimiter=',', ndmin=2)
    frames = row['frames']
    assert len(frames) == len(gt) == row['frame_count']
    assert gt.shape[1] == 4 and np.isfinite(gt).all()
    boxes, scores, times = [], [], []
    for index, (rgb_path, auxiliary_path) in enumerate(frames):
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = get_x_frame(rgb_path, auxiliary_path, dtype='rgbrgb')
        assert image.ndim == 3 and image.shape[2] == 6
        if index == 0:
            tracker.initialize(image, {'init_bbox': gt[0].tolist()})
            box, score = gt[0].tolist(), 1.0
        else:
            prediction = tracker.track(image)
            box, score = prediction['target_bbox'], prediction['best_score']
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        assert np.isfinite(box).all() and np.isfinite(score)
        assert box[2] > 0 and box[3] > 0
        boxes.append(box)
        scores.append(score)
    predictions_path = output / (row['name'] + '.txt')
    np.savetxt(predictions_path, boxes, fmt='%.14f', delimiter=',')
    np.savetxt(output / (row['name'] + '_confidence.txt'), scores, fmt='%.14f')
    np.savetxt(output / (row['name'] + '_time.txt'), times, fmt='%.14f')
    sequence_receipt = dict(sequence=row['name'], dataset=manifest['dataset'],
                            checkpoint=params.checkpoint, checkpoint_sha256=args.expected_sha256,
                            config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                            terminal_audit_sha256=params.evaluation_identity['terminal_audit_sha256'],
                            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            seed=args.seed, shard=args.shard, frame_count=len(boxes),
                            expected_frame_count=row['frame_count'], groundtruth_sha256=row['groundtruth_sha256'],
                            predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                            elapsed_seconds=sum(times), timing='synchronized_image_read_preprocessing_and_tracking',
                            closed_loop_search=True, initial_confidence_is_protocol_marker=True,
                            status='COMPLETE')
    (output / (row['name'] + '.json')).write_text(json.dumps(sequence_receipt, indent=2) + '\n')
    print(json.dumps(sequence_receipt), flush=True)
