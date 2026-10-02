"""Require full RGB-D trajectory coverage before running every official analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from cachetools import LRUCache
from vot.analysis.processor import AnalysisProcessor, process_stack_analyses
from vot.experiment.multistart import find_anchors
from vot.region import RegionType
from vot.tracker import Trajectory
from vot.utilities import ThreadPoolExecutor
from vot.utilities.io import JSONEncoder
from vot.workspace import Workspace
from xtrack_eval_parameters import verified_endpoint

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
args = parser.parse_args()
root = Path(args.workspace).resolve(strict=True)
identity_path = root/'identity.json'
identity = json.loads(identity_path.read_text())
endpoint = verified_endpoint(identity['config'], identity['checkpoint'], identity['checkpoint_sha256'], identity['terminal_audit'])
assert endpoint['config_sha256'] == identity['config_sha256']
assert endpoint['terminal_audit_sha256'] == identity['terminal_audit_sha256']
for filename, key in [('stack.yaml','workspace_stack_sha256'), ('trackers.ini','trackers_ini_sha256')]:
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == identity[key]
data_path = Path(identity['dataset_audit'])
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == identity['dataset_audit_sha256']
data = json.loads(data_path.read_text())
assert {p.name for p in root.glob('shard_*.json')} == {'shard_0.json','shard_1.json','shard_2.json'}
for shard in range(3):
    receipt = json.loads((root/('shard_{}.json'.format(shard))).read_text())
    assert receipt['status'] == 'TRAJECTORIES_WRITTEN_NOT_SCORED'
    assert receipt['seed'] == 2026 and receipt['shard'] == shard
    assert receipt['identity_sha256'] == hashlib.sha256(identity_path.read_bytes()).hexdigest()
    assert receipt['completed_sequences'] == [row['name'] for row in data['sequences'][shard::3]]
workspace = Workspace.load(str(root))
tracker, = workspace.registry.resolve(identity['tracker_id'], storage=workspace.storage.substorage('results'), skip_unknown=False)
experiment = workspace.stack[identity['experiment_id']]
assert set(workspace.dataset.keys()) == {row['name'] for row in data['sequences']}
coverage = []
for row in data['sequences']:
    assert hashlib.sha256(Path(row['groundtruth']).read_bytes()).hexdigest() == row['groundtruth_sha256']
    sequence = workspace.dataset[row['name']]
    assert len(sequence) == row['frame_count']
    for channel_name in ['color','depth']:
        channel = sequence.channel(channel_name)
        filenames = [channel.filename(i) for i in range(len(sequence))]
        assert hashlib.sha256('\n'.join(filenames).encode()).hexdigest() == row['channels'][channel_name]['filenames_sha256']
    if identity['dataset'] == 'DepthTrack':
        expected = [(sequence.name+'_001', len(sequence))]
    else:
        forward, backward = find_anchors(sequence)
        assert forward == row['anchors']['forward'] and backward == row['anchors']['backward']
        expected = [(sequence.name+'_%08d'%i, len(sequence)-i) for i in forward]
        expected += [(sequence.name+'_%08d'%i, i+1) for i in backward]
    results = experiment.results(tracker, sequence)
    assert set(results.find('*.txt')) | set(results.find('*.bin')) == {name+'.bin' for name,_ in expected}
    for name, n in expected:
        trajectory = Trajectory.read(results, name)
        assert len(trajectory) == n
        assert trajectory.region(0).type == RegionType.SPECIAL and trajectory.region(0).code == 1
        for property_name in ['confidence','time','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes']:
            assert results.exists(name+'_'+property_name+'.value')
            with results.read(name+'_'+property_name+'.value') as source:
                assert len(source.read().splitlines()) == n
        seconds = 0.0
        allocated = reserved = 0
        for i in range(n):
            values = trajectory.properties(i)
            assert math.isfinite(values['time']) and values['time'] > 0
            seconds += values['time']
            if i:
                assert values['gpu_peak_allocated_bytes'] > 0 and values['gpu_peak_reserved_bytes'] > 0
                allocated = max(allocated,values['gpu_peak_allocated_bytes'])
                reserved = max(reserved,values['gpu_peak_reserved_bytes'])
                region = trajectory.region(i)
                assert region.type == RegionType.RECTANGLE
                assert all(math.isfinite(v) for v in [region.x,region.y,region.width,region.height])
                assert region.width > 0 and region.height > 0
                assert math.isfinite(values['confidence'])
        hashes = {}
        for filename in Trajectory.gather(results, name):
            with results.read(filename) as source:
                contents = source.read()
            hashes[filename] = hashlib.sha256(contents.encode() if isinstance(contents,str) else contents).hexdigest()
        coverage.append(dict(sequence=sequence.name, trajectory=name, frames=n,
                             rpc_tracking_seconds=seconds, hashes=hashes,
                             peak_allocated_bytes=allocated,peak_reserved_bytes=reserved))
coverage_report = dict(status='PASS', role='full_trajectory_coverage_not_metric_scoring',
                       identity=identity, sequences=len(data['sequences']),
                       trajectories=len(coverage), frames=sum(row['frames'] for row in coverage),
                       missing_sequences=0, missing_trajectories=0, per_trajectory=coverage)
coverage_report['rpc_tracking_seconds'] = sum(row['rpc_tracking_seconds'] for row in coverage)
coverage_report['fps'] = coverage_report['frames']/coverage_report['rpc_tracking_seconds']
coverage_report['gpu_memory'] = dict(peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in coverage),
                                    peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in coverage),
                                    scope='maximum_of_PyTorch_peaks_saved_in_native_trajectory_properties')
(root/'coverage.json').write_text(json.dumps(coverage_report,indent=2)+'\n')
assert not (root/'official_metrics.json').exists()
with ThreadPoolExecutor(1) as executor, AnalysisProcessor(executor, LRUCache(1000)):
    official = process_stack_analyses(workspace,[tracker])
assert official is not None and len(official) == 1
analyses = official[experiment]
assert len(analyses) == len(experiment.analyses) and all(grid is not None for grid in analyses.values())
headline = {}
raw = []
for analysis, grid in analyses.items():
    kind = type(analysis).__name__
    raw.append(dict(type=kind, parameters=analysis.dump(), values=grid))
    if kind == 'PrecisionRecall':
        pr, recall, fscore = grid[0,0]
        headline.update(Precision=pr, Recall=recall, F_score=fscore)
    elif kind == 'EAOScore':
        headline['EAO'] = grid[0,0][0]
    elif kind == 'AverageAccuracyRobustness':
        headline.update(Accuracy=grid[0,0][0], Robustness=grid[0,0][1])
assert set(headline) == ({'Precision','Recall','F_score'} if identity['dataset']=='DepthTrack' else {'EAO','Accuracy','Robustness'})
assert all(math.isfinite(value) for value in headline.values())
report = dict(status='OFFICIAL_ANALYSES_COMPLETE', identity=identity, headline_metrics=headline,
              sequence_order=[sequence.name for sequence in workspace.dataset],
              official_analyses=raw, coverage_sha256=hashlib.sha256((root/'coverage.json').read_bytes()).hexdigest(),
              scorer='vot-toolkit==0.7.1', fresh_in_memory_analysis_cache=True,
              metric_units='fractions_0_to_1', evaluation_type='real_gt')
(root/'official_metrics.json').write_text(json.dumps(report,indent=2,cls=JSONEncoder)+'\n')
print(json.dumps(dict(status=report['status'],dataset=identity['dataset'],headline_metrics=headline)))
