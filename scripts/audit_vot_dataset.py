"""Audit the actual RGB-D dataset through the pinned VOT parser, without inference."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from vot.dataset import load_dataset
from vot.experiment.multistart import find_anchors
from vot.region import RegionType

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['DepthTrack', 'VOT-RGBD2022'], required=True)
parser.add_argument('--root', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
assert version('vot-toolkit') == '0.7.1'
root = Path(args.root).resolve(strict=True)
names = [line.strip() for line in (root / 'list.txt').read_text().splitlines()]
expected = 50 if args.dataset == 'DepthTrack' else 127
assert len(names) == len(set(names)) == expected
dataset = load_dataset(str(root))
assert set(dataset.keys()) == set(names)
rows = []
for sequence in dataset:
    assert set(sequence.channels()) == {'color', 'depth'}
    n = len(sequence)
    assert n > 0
    assert len(sequence.groundtruth()) == n
    channels = {}
    for channel_name in ['color', 'depth']:
        channel = sequence.channel(channel_name)
        assert len(channel) == n
        filenames = [channel.filename(i) for i in range(n)]
        assert len(filenames) == len(set(filenames))
        assert all(Path(p).is_file() for p in filenames)
        channels[channel_name] = dict(first=filenames[0], last=filenames[-1], frames=n,
                                      filenames_sha256=hashlib.sha256('\n'.join(filenames).encode()).hexdigest())
    gt = root / sequence.name / 'groundtruth.txt'
    config = root / sequence.name / 'sequence'
    assert gt.is_file() and config.is_file()
    anchors = None
    if args.dataset == 'VOT-RGBD2022':
        forward, backward = find_anchors(sequence)
        assert forward or backward
        anchors = dict(forward=forward, backward=backward,
                       total_trajectory_frames=sum(n-i for i in forward)+sum(i+1 for i in backward))
    rows.append(dict(name=sequence.name, frame_count=n, groundtruth=str(gt),
                     groundtruth_sha256=hashlib.sha256(gt.read_bytes()).hexdigest(),
                     sequence_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                     special_groundtruth_frames=sum(r.type == RegionType.SPECIAL for r in sequence.groundtruth()),
                     channels=channels, anchors=anchors))
report = dict(status='READY', role='dataset_and_protocol_parser_audit_not_tracking_benchmark',
              dataset=args.dataset, root=str(root), vot_toolkit='0.7.1',
              expected_sequences=expected, expected_frame_pairs=sum(row['frame_count'] for row in rows),
              dataset_list_sha256=hashlib.sha256((root/'list.txt').read_bytes()).hexdigest(), sequences=rows)
Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key != 'sequences'}))
