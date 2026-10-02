"""Inspect existing data without copying or changing any dataset."""
import hashlib
import json
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')
SPECS = PROJECT / 'third_party/XTrack/lib/train/data_specs'
SOURCE = Path('/data/wangwj/dataset')


def lines(path):
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def inspect(name, root, spec, annotation):
    listed = lines(spec)
    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    missing = [s for s in listed if not (root / s / annotation).is_file()]
    return {
        'name': name, 'root': str(root), 'directory_count': len(folders),
        'split_file': str(spec), 'split_sha256': hashlib.sha256(spec.read_bytes()).hexdigest(),
        'split_count': len(listed), 'unique_split_count': len(set(listed)),
        'missing_annotations': missing,
        'directories_outside_split': sorted(set(folders) - set(listed)),
    }


def main():
    entries = [
        inspect('LasHeR_train', SOURCE / 'LasHeR/trainingset', SPECS / 'lasher_train.txt', 'init.txt'),
        inspect('LasHeR_val', SOURCE / 'LasHeR/trainingset', SPECS / 'lasher_val.txt', 'init.txt'),
        inspect('DepthTrack_train', PROJECT / 'data/depthtrack/train', SPECS / 'depthtrack_train.txt', 'groundtruth.txt'),
        inspect('DepthTrack_val', PROJECT / 'data/depthtrack/train', SPECS / 'depthtrack_val.txt', 'groundtruth.txt'),
        inspect('VisEvent_train', SOURCE / 'VisEvent/train_subset', SOURCE / 'VisEvent/train_subset/trainlist.txt', 'groundtruth.txt'),
    ]
    tests = {name: {'root': str(SOURCE / path), 'sequences': sum(p.is_dir() for p in (SOURCE / path).iterdir())}
             for name, path in [('LasHeR', 'LasHeR/testingset'), ('DepthTrack', 'DepthTrack/Dep_test'), ('VisEvent', 'VisEvent/test_subset')]}
    result = {'train_splits': entries, 'test_roots': tests,
              'deferred_evaluations': ['RGBT234', 'VOT-RGBD2022'],
              'annotations_present': all(not e['missing_annotations'] for e in entries)}
    (PROJECT / 'reports/data_inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    for e in entries:
        print(e['name'], 'listed=', e['split_count'], 'folders=', e['directory_count'], 'missing=', e['missing_annotations'])
    print('tests:', tests)
    assert result['annotations_present'], 'Author split references missing sequences; inspect data_inventory.json.'


if __name__ == '__main__':
    main()
