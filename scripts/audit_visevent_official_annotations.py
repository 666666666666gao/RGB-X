"""Compare the complete local test split against the authors' scorer annotations."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/data/wangwj/dataset/VisEvent/test_subset')
ZIP = Path('/home/gaob/.cache/rgbx_visevent_official_annos.zip')
OFFICIAL = Path('/home/gaob/rgbx-eval-data/VisEvent_official_annotations')
REPORT = Path('/data/gb/rgbx-risk/reports/visevent_official_annotations_audit.json')

assert not OFFICIAL.exists()
with zipfile.ZipFile(ZIP) as source:
    source.extractall(OFFICIAL)
names = ROOT.joinpath('list_full.txt').read_text().split()
assert len(names) == len(set(names)) == 320
assert set(names) == {p.stem for p in OFFICIAL.joinpath('gt_rect').glob('*.txt')}
rows = []
for name in names:
    local = ROOT / name
    rgb = sorted(local.joinpath('vis_imgs').glob('*.bmp'))
    event = sorted(local.joinpath('event_imgs').glob('*.bmp'))
    gt = np.loadtxt(local / 'groundtruth.txt', delimiter=',', ndmin=2)
    official_gt = np.loadtxt(OFFICIAL / 'gt_rect' / (name + '.txt'), delimiter=',', ndmin=2)
    absent = np.loadtxt(OFFICIAL / 'absent' / (name + '.txt'), ndmin=1)
    raw_absent = np.loadtxt(local / 'absent_label.txt', ndmin=1)
    same_gt = gt.shape == official_gt.shape and np.array_equal(gt, official_gt)
    first_present = int(raw_absent.argmax())
    raw_aligned = raw_absent[first_present:]
    absent_match = raw_aligned.shape == absent.shape and np.array_equal(1 - raw_aligned, absent)
    rows.append(dict(sequence=name, rgb_frames=len(rgb), event_frames=len(event),
                     local_gt_rows=len(gt), official_gt_rows=len(official_gt),
                     official_absent_rows=len(absent), raw_absent_rows=len(raw_absent),
                     first_present_in_raw=first_present, groundtruth_exact=same_gt,
                     official_absence_equals_inverse_aligned_raw=absent_match,
                     official_absent_frames=int((absent == 1).sum()),
                     aligned_lengths=len(rgb) == len(event) == len(gt) == len(absent)))
report = dict(observed_at=datetime.now(timezone.utc).isoformat(),
              official_repository='https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark',
              official_commit='c30908779d58404ce932abdbdb68a4dfcd07e684',
              official_annotations_zip_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              sequence_list=str(ROOT / 'list_full.txt'), expected_sequences=320,
              actual_sequences=len(rows), actual_frame_pairs=sum(x['rgb_frames'] for x in rows),
              all_lengths_aligned=all(x['aligned_lengths'] for x in rows),
              all_groundtruth_exact=all(x['groundtruth_exact'] for x in rows),
              all_absence_labels_match=all(x['official_absence_equals_inverse_aligned_raw'] for x in rows),
              sequences=rows)
REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'sequences'}))
assert report['all_lengths_aligned'] and report['all_groundtruth_exact'] and report['all_absence_labels_match']
