"""Bind all 245 sequences to dataset GT verified against the pinned toolkit mirror."""
import hashlib
import json
import re
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')
ROOT = PROJECT / 'data/lasher/testingset'
DEST = PROJECT / 'reports/lasher_eval_manifest.json'
audit = json.loads((PROJECT / 'reports/lasher_test_annotation_audit.json').read_text())
assert audit['all_lengths_aligned'] and audit['all_dataset_init_matches_mirror']
sequences = []


def frame_id(path):
    match = re.fullmatch(r'[vi]?(\d+)', path.stem)
    assert match is not None, str(path)
    return int(match.group(1))


for row in audit['sequences']:
    folder = ROOT / row['sequence']
    rgb = sorted(folder.joinpath('visible').glob('*.jpg'), key=frame_id)
    infrared = sorted(folder.joinpath('infrared').glob('*.jpg'), key=frame_id)
    assert [frame_id(p) for p in rgb] == [frame_id(p) for p in infrared]
    assert len(set(frame_id(p) for p in rgb)) == len(rgb)
    assert len(rgb) == len(infrared) == row['dataset_init_rows']
    gt_path = folder / 'init.txt'
    assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == row['dataset_init_sha256']
    sequences.append(dict(name=row['sequence'], frame_count=len(rgb), groundtruth=str(gt_path),
                          groundtruth_sha256=row['dataset_init_sha256'],
                          frames=[[str(a), str(b)] for a, b in zip(rgb, infrared)]))
assert len(sequences) == len(set(row['name'] for row in sequences)) == 245
assert sum(row['frame_count'] for row in sequences) == 220703
manifest = dict(status='READY', dataset='LasHeR', expected_sequences=245,
                expected_frame_pairs=220703, groundtruth_source='dataset_init_txt',
                annotation_audit_sha256=hashlib.sha256((PROJECT / 'reports/lasher_test_annotation_audit.json').read_bytes()).hexdigest(),
                sequences=sequences)
DEST.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(status='READY', sequences=245, frame_pairs=220703,
                      manifest_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest())))
