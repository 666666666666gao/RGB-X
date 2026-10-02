"""Export named PR/F curves and explicitly labeled DepthTrack tag diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from vot.region import RegionType, calculate_overlaps
from vot.tracker import Trajectory
from vot.workspace import Workspace
from xtrack_eval_parameters import verified_endpoint

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--attributes', required=True)
args = parser.parse_args()
root = Path(args.workspace).resolve(strict=True)
identity = json.loads((root/'identity.json').read_text())
endpoint = verified_endpoint(identity['config'],identity['checkpoint'],identity['checkpoint_sha256'],identity['terminal_audit'])
assert all(endpoint[key]==identity[key] for key in endpoint)
assert identity['dataset']=='DepthTrack'
metrics_path = root/'official_metrics.json'
metrics = json.loads(metrics_path.read_text())
coverage_path = root/'coverage.json'
coverage = json.loads(coverage_path.read_text())
assert metrics['status']=='OFFICIAL_ANALYSES_COMPLETE' and metrics['identity']==identity
assert metrics['coverage_sha256']==hashlib.sha256(coverage_path.read_bytes()).hexdigest()
assert coverage['status']=='PASS' and coverage['identity']==identity and coverage['sequences']==50
attribute_path = Path(args.attributes)
attributes = json.loads(attribute_path.read_text())
assert attributes['status']=='READY' and attributes['dataset']=='DepthTrack'
assert attributes['dataset_audit_sha256']==identity['dataset_audit_sha256']
workspace = Workspace.load(str(root))
names = metrics['sequence_order']
assert names==[sequence.name for sequence in workspace.dataset]
assert set(names)=={row['sequence'] for row in attributes['per_sequence']} and len(names)==50
tracker, = workspace.registry.resolve(identity['tracker_id'],storage=workspace.storage.substorage('results'),skip_unknown=False)
experiment = workspace.stack[identity['experiment_id']]
analysis, = [row for row in metrics['official_analyses'] if row['type']=='PrecisionRecallCurves']
assert len(analysis['values'])==50
thresholds = np.asarray(analysis['values'][0][1],dtype=float)
points = []
for curves, sequence_thresholds in analysis['values']:
    assert sequence_thresholds==thresholds.tolist()
    curves = np.asarray(curves,dtype=float)
    assert curves.shape==(len(thresholds),2) and np.isfinite(curves).all()
    points.append(curves)
points = np.stack(points)

def summarize(precision,recall):
    denominator = precision+recall
    fcurve = np.divide(2*precision*recall,denominator,out=np.zeros_like(denominator),where=denominator>0)
    best = int(fcurve.argmax())
    return dict(Precision=float(precision[best]),Recall=float(recall[best]),F_score=float(fcurve[best]),
                best_threshold_index=best,precision_curve=precision.tolist(),recall_curve=recall.tolist(),F_curve=fcurve.tolist())

overall = summarize(points[:,:,0].mean(axis=0),points[:,:,1].mean(axis=0))
assert all(abs(overall[key]-metrics['headline_metrics'][key])<1e-12 for key in ['Precision','Recall','F_score'])
per_sequence = [dict(sequence=name,**summarize(curves[:,0],curves[:,1])) for name,curves in zip(names,points)]
sequence_groups = {}
for label, group in attributes['groups'].items():
    indices = [i for i,name in enumerate(names) if name in group['sequences']]
    assert indices
    selected = points[indices].mean(axis=0)
    sequence_groups[label] = dict(sequences=[names[i] for i in indices],
                                  rule='whole_sequences_with_at_least_one_positive_tag_frame_equal_sequence_mean_global_thresholds',
                                  **summarize(selected[:,0],selected[:,1]))

frame_data = {}
for row in attributes['per_sequence']:
    sequence = workspace.dataset[row['sequence']]
    results = experiment.results(tracker,sequence)
    original, = [item for item in coverage['per_trajectory'] if item['sequence']==sequence.name]
    for filename,digest in original['hashes'].items():
        with results.read(filename) as source:
            content = source.read()
        assert hashlib.sha256(content.encode() if isinstance(content,str) else content).hexdigest()==digest
    trajectory = Trajectory.read(results,sequence.name+'_001')
    overlaps = np.asarray(calculate_overlaps(trajectory.regions(),sequence.groundtruth(),sequence.size,
                                             ignore=sequence.object('_ignore')),dtype=float)
    assert np.isfinite(overlaps).all()
    confidence = np.asarray([trajectory.properties(i).get('confidence',0) for i in range(len(sequence))])
    visible = np.asarray([region.type!=RegionType.SPECIAL for region in sequence.groundtruth()])
    tags = {}
    for label,tag in row['tags'].items():
        path = Path(tag['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==tag['sha256']
        mask = np.asarray([line=='1' for line in path.read_text().splitlines()])
        assert len(mask)==len(sequence) and int(mask.sum())==tag['positive_frames']
        tags[label] = mask
    frame_data[sequence.name] = (overlaps,confidence,visible,tags)
frame_groups = {}
for label,group in attributes['groups'].items():
    sums = np.zeros(len(thresholds))
    counts = np.zeros(len(thresholds))
    for overlaps,confidence,visible,tags in frame_data.values():
        for i,threshold in enumerate(thresholds):
            selected = tags[label] & (confidence>=threshold)
            sums[i] += overlaps[selected].sum()
            counts[i] += selected.sum()
    visible_count = group['positive_visible_gt_frames']
    assert visible_count>0
    precision = np.divide(sums,counts,out=np.ones_like(sums),where=counts>0)
    recall = sums/visible_count
    frame_groups[label] = dict(rule='pooled_tagged_frames_original_native_overlaps_global_confidence_thresholds',
                               tagged_frames=group['positive_frames'],visible_gt_frames=visible_count,
                               **summarize(precision,recall))
report = dict(status='ATTRIBUTE_DIAGNOSTICS_COMPLETE',role='additional_diagnostics_not_official_overall_headline',
              dataset='DepthTrack',identity=identity,official_headline_metrics=metrics['headline_metrics'],
              evaluation_type='real_gt',official_metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
              attributes_sha256=hashlib.sha256(attribute_path.read_bytes()).hexdigest(),
              sequence_order=names,confidence_thresholds=thresholds.tolist(),per_sequence=per_sequence,
              sequence_groups=sequence_groups,frame_groups=frame_groups,
              zero_precision_and_recall_F_definition=0,initial_missing_confidence_uses_original_VOT_default=0,
              overall_PR_F_coordinates_from_official_PR_analysis=overall,
              raw_official_F_curve_metadata_preserved_in_official_metrics=True,
              note='Sequence groups include all their frames; frame groups use only tagged frames. These differ in aggregation and do not replace official overall scores.')
output = root/'attribute_diagnostics.json'
assert not output.exists()
output.write_text(json.dumps(report,indent=2)+'\n')
with (root/'per_sequence_metrics.csv').open('w',newline='') as destination:
    fields = ['sequence','Precision','Recall','F_score','best_threshold_index']
    writer = csv.DictWriter(destination,fieldnames=fields)
    writer.writeheader()
    writer.writerows({key:row[key] for key in fields} for row in per_sequence)
figure,axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot(overall['recall_curve'],overall['precision_curve'])
axes[0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
axes[1].plot(range(len(thresholds)),overall['F_curve'])
axes[1].set(xlabel='Official confidence threshold index',ylabel='F-score',ylim=(0,1))
for axis in axes:
    axis.grid(alpha=.3)
figure.tight_layout()
figure.savefig(root/'full_PR_F_curves.png',dpi=180)
plt.close(figure)
print(json.dumps(dict(status=report['status'],dataset='DepthTrack',attributes=len(frame_groups),sequences=len(names))))
