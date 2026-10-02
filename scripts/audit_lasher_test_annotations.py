"""Check dataset init.txt against the pinned public LasHeR toolkit annotation mirror."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/data/gb/rgbx-risk/data/lasher/testingset')
ARCHIVE = Path('/home/gaob/.cache/rgbx_lasher_mirror_annos.tar.gz')
MIRROR = Path('/home/gaob/rgbx-eval-data/LasHeR_toolkit_annotation_mirror')
REPORT = Path('/data/gb/rgbx-risk/reports/lasher_test_annotation_audit.json')
assert not MIRROR.exists()
MIRROR.mkdir()
with tarfile.open(ARCHIVE) as source:
    source.extractall(MIRROR)
names = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
assert len(names) == 245
rows = []
for name in names:
    folder = ROOT / name
    init = np.loadtxt(folder / 'init.txt', delimiter=',', ndmin=2)
    mirror_file = MIRROR / 'annos' / (name + '.txt')
    mirror = np.loadtxt(mirror_file, delimiter=',', ndmin=2)
    rgb_count = len(list(folder.joinpath('visible').glob('*.jpg')))
    ir_count = len(list(folder.joinpath('infrared').glob('*.jpg')))
    rows.append(dict(sequence=name, rgb_frames=rgb_count, infrared_frames=ir_count,
                     dataset_init_rows=len(init), mirror_gt_rows=len(mirror),
                     dataset_init_exactly_matches_mirror=init.shape == mirror.shape and np.array_equal(init, mirror),
                     aligned_lengths=rgb_count == ir_count == len(init) == len(mirror),
                     dataset_init_sha256=hashlib.sha256((folder / 'init.txt').read_bytes()).hexdigest(),
                     mirror_gt_sha256=hashlib.sha256(mirror_file.read_bytes()).hexdigest()))
report = dict(observed_at=datetime.now(timezone.utc).isoformat(),
              source_type='dataset_provided_groundtruth_compared_with_third_party_toolkit_mirror',
              mirror_repository='https://github.com/xuboyue1999/RGBT-Tracking',
              mirror_commit='3613a3cbc9d0626a54c7755a549241f658ef6d46',
              sequences=rows, actual_sequences=len(rows),
              actual_frame_pairs=sum(x['rgb_frames'] for x in rows),
              all_lengths_aligned=all(x['aligned_lengths'] for x in rows),
              all_dataset_init_matches_mirror=all(x['dataset_init_exactly_matches_mirror'] for x in rows))
REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'sequences'}))
assert report['all_lengths_aligned'] and report['all_dataset_init_matches_mirror']
