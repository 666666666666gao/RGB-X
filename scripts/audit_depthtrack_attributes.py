"""Audit every existing DepthTrack frame tag, preserving unknown GT frames."""
import argparse
import hashlib
import json
from pathlib import Path

from vot.dataset import load_dataset
from vot.region import RegionType

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-audit', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
audit_path = Path(args.dataset_audit)
data = json.loads(audit_path.read_text())
assert data['status']=='READY' and data['dataset']=='DepthTrack' and len(data['sequences'])==50
dataset = load_dataset(data['root'])
rows = []
labels = sorted(p.stem for p in Path(data['sequences'][0]['groundtruth']).parent.glob('*.tag'))
assert labels
for row in data['sequences']:
    sequence = dataset[row['name']]
    gt = Path(row['groundtruth'])
    assert hashlib.sha256(gt.read_bytes()).hexdigest()==row['groundtruth_sha256']
    assert len(gt.read_text().splitlines())==len(sequence)==row['frame_count']
    assert sorted(p.stem for p in gt.parent.glob('*.tag'))==labels
    visible = [region.type!=RegionType.SPECIAL for region in sequence.groundtruth()]
    attributes = {}
    for label in labels:
        path = gt.parent/(label+'.tag')
        bits = path.read_text().splitlines()
        assert len(bits)==row['frame_count'] and set(bits)<={'0','1'}
        mask = [bit=='1' for bit in bits]
        assert mask==[label in sequence.tags(i) for i in range(len(sequence))]
        attributes[label] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 positive_frames=sum(mask),positive_visible_gt_frames=sum(m and v for m,v in zip(mask,visible)))
    rows.append(dict(sequence=row['name'],frames=row['frame_count'],tags=attributes))
groups = {label:dict(sequences=[row['sequence'] for row in rows if row['tags'][label]['positive_frames']],
                     positive_frames=sum(row['tags'][label]['positive_frames'] for row in rows),
                     positive_visible_gt_frames=sum(row['tags'][label]['positive_visible_gt_frames'] for row in rows))
          for label in labels}
report = dict(status='READY',role='frame_attribute_provenance_not_tracking_benchmark',dataset='DepthTrack',
              dataset_root=data['root'],sequences=50,frame_pairs=sum(row['frames'] for row in rows),
              dataset_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),labels=labels,
              raw_tag_lengths_equal_frame_counts=True,raw_tag_bits_match_vot_parser=True,groups=groups,per_sequence=rows)
Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key!='per_sequence'}))
