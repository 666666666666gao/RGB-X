"""Three real frames through the official DepthTrack TraX runtime, no score."""
import argparse
import json
import math
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
args = parser.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == str(args.gpu)
from vot.workspace import Workspace
from vot.tracker import ObjectStatus

root = Path(args.workspace)
identity = json.loads((root/'identity.json').read_text())
assert identity['model'] == 'CADTrackPruned' and identity['shards'] == 2
workspace = Workspace.load(str(root))
tracker, = workspace.registry.resolve(identity['tracker_id'], storage=workspace.storage.substorage('results'), skip_unknown=False)
experiment = workspace.stack[identity['experiment_id']]
data = json.loads(Path(identity['dataset_audit']).read_text())
sequence = workspace.dataset[data['sequences'][args.gpu]['name']]
observations = []
with experiment._get_runtime(tracker, sequence) as runtime:
    _, elapsed = runtime.initialize(sequence.frame(0), [ObjectStatus(experiment._get_initialization(sequence, 0), {})])
    for index in [1, 2]:
        state, elapsed = runtime.update(sequence.frame(index), [])
        region = state.region
        confidence = float(state.properties['confidence'])
        allocated = float(state.properties['gpu_peak_allocated_bytes'])
        assert all(math.isfinite(value) for value in region.bounds())
        assert region.width > 0 and region.height > 0 and math.isfinite(confidence)
        assert allocated > 0
        observations.append(dict(frame=index, box=list(region.bounds()), confidence=confidence,
                                 raw_trax_confidence=state.properties['confidence'], peak_allocated_bytes=allocated, seconds=elapsed))
report = dict(status='PASS_NATIVE_THREE_FRAME_RUNTIME_NOT_BENCHMARK', physical_gpu=args.gpu,
              sequence=sequence.name, frames=3, predictions=observations, optimizer_steps=0,
              checkpoint_sha256=identity['checkpoint_sha256'])
(root/('native_gpu{}.json'.format(args.gpu))).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
