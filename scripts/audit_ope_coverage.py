"""Verify complete OPE predictions before allowing any benchmark score claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from xtrack_eval_parameters import verified_endpoint

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--expected-sha256', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--terminal-audit', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
identity = verified_endpoint(args.config, args.checkpoint, args.expected_sha256, args.terminal_audit)
manifest_path = Path(args.manifest)
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'READY'
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
config_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
root = Path(args.output)
names = [row['name'] for row in manifest['sequences']]
assert len(names) == len(set(names)) == manifest['expected_sequences']
assert {p.stem for p in root.glob('*.json')} == set(names)
rows = []
for row in manifest['sequences']:
    receipt = json.loads((root / (row['name'] + '.json')).read_text())
    prediction_path = root / (row['name'] + '.txt')
    boxes = np.loadtxt(prediction_path, delimiter=',', ndmin=2)
    confidence = np.loadtxt(root / (row['name'] + '_confidence.txt'), ndmin=1)
    times = np.loadtxt(root / (row['name'] + '_time.txt'), ndmin=1)
    n = row['frame_count']
    assert boxes.shape == (n, 4) and confidence.shape == times.shape == (n,)
    assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    assert np.isfinite(confidence).all() and np.isfinite(times).all() and (times > 0).all()
    assert receipt['status'] == 'COMPLETE' and receipt['sequence'] == row['name']
    assert receipt['dataset'] == manifest['dataset'] and receipt['seed'] == 2026
    assert receipt['checkpoint_sha256'] == args.expected_sha256
    assert receipt['config_sha256'] == config_sha and receipt['manifest_sha256'] == manifest_sha
    assert receipt['terminal_audit_sha256'] == identity['terminal_audit_sha256']
    assert receipt['groundtruth_sha256'] == row['groundtruth_sha256']
    assert hashlib.sha256(Path(row['groundtruth']).read_bytes()).hexdigest() == row['groundtruth_sha256']
    assert receipt['frame_count'] == receipt['expected_frame_count'] == n
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == receipt['predictions_sha256']
    gt = np.loadtxt(row['groundtruth'], delimiter=',', ndmin=2)
    assert np.allclose(boxes[0], gt[0], atol=1e-12, rtol=0)
    rows.append(dict(sequence=row['name'], frames=n, seconds=float(times.sum()),
                     fps=n / float(times.sum()), predictions_sha256=receipt['predictions_sha256'],
                     physical_gpu=receipt['physical_gpu'],gpu=receipt['gpu'],
                     peak_allocated_bytes=receipt['peak_allocated_bytes'],peak_reserved_bytes=receipt['peak_reserved_bytes']))
report = dict(status='PASS', role='full_trajectory_coverage_not_metric_scoring',
              dataset=manifest['dataset'], sequences=len(rows),
              frames=sum(row['frames'] for row in rows), expected_sequences=manifest['expected_sequences'],
              expected_frames=manifest['expected_frame_pairs'], missing_sequences=0,
              checkpoint_sha256=args.expected_sha256, config_sha256=config_sha,
              terminal_audit_sha256=identity['terminal_audit_sha256'],
              manifest_sha256=manifest_sha, elapsed_seconds=sum(row['seconds'] for row in rows),
              timing='synchronized_image_read_preprocessing_and_tracking', per_sequence=rows)
assert report['frames'] == report['expected_frames']
report['fps'] = report['frames'] / report['elapsed_seconds']
report['gpu_memory'] = dict(peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows),
                          peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in rows),
                          scope='maximum_of_per_sequence_PyTorch_peaks_including_resident_model')
Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'per_sequence'}))
