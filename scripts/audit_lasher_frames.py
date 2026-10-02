"""Verify annotation count and paired frame ordering in the existing LasHeR training pool."""
import csv
import json
import re
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')
ROOT = Path('/data/wangwj/dataset/LasHeR/trainingset')


def main():
    rows = []
    for sequence in sorted(ROOT.iterdir()):
        if not sequence.is_dir():
            continue
        rgb = sorted((sequence / 'visible').glob('*.jpg'))
        thermal = sorted((sequence / 'infrared').glob('*.jpg'))
        rgb_ids = [int(re.findall(r'\d+', p.stem)[-1]) for p in rgb]
        thermal_ids = [int(re.findall(r'\d+', p.stem)[-1]) for p in thermal]
        with (sequence / 'init.txt').open() as source:
            annotations = sum(bool(row) for row in csv.reader(source))
        rows.append({'sequence': sequence.name, 'annotations': annotations, 'rgb_frames': len(rgb),
                     'thermal_frames': len(thermal), 'pair_ids_equal': rgb_ids == thermal_ids,
                     'lexical_order_is_temporal': rgb_ids == sorted(rgb_ids) and thermal_ids == sorted(thermal_ids),
                     'counts_equal': annotations == len(rgb) == len(thermal)})
    issues = [row for row in rows if not (row['lexical_order_is_temporal'] and row['counts_equal'])]
    result = {'sequences_checked': len(rows), 'issues': issues, 'sequences': rows,
              'pairing_protocol': 'Pair by sorted ordinal position, as stored in the existing dataset.',
              'limits': 'Filename IDs differ in two sequences; this audit does not independently verify sensor time synchronization.'}
    (PROJECT / 'reports/lasher_frame_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('LasHeR sequences checked:', len(rows), 'issues:', len(issues), issues[:8], flush=True)
    assert not issues, 'Existing frame layout needs correction; see lasher_frame_audit.json.'


if __name__ == '__main__':
    main()
