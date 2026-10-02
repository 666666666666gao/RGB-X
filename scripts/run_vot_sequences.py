"""Run one GPU shard of all official RGB-D sequences, without cached-result skipping."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--shard', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()
assert 0 <= args.shard < 3 and 0 <= args.gpu < 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from vot.workspace import Workspace
from vot import config
from xtrack_eval_parameters import verified_endpoint

root = Path(args.workspace).resolve(strict=True)
assert config.results_binary is True
identity = json.loads((root/'identity.json').read_text())
endpoint = verified_endpoint(identity['config'], identity['checkpoint'], identity['checkpoint_sha256'], identity['terminal_audit'])
assert endpoint['config_sha256'] == identity['config_sha256']
assert endpoint['terminal_audit_sha256'] == identity['terminal_audit_sha256']
assert hashlib.sha256((root/'stack.yaml').read_bytes()).hexdigest() == identity['workspace_stack_sha256']
assert hashlib.sha256((root/'trackers.ini').read_bytes()).hexdigest() == identity['trackers_ini_sha256']
data_path = Path(identity['dataset_audit'])
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == identity['dataset_audit_sha256']
data = json.loads(data_path.read_text())
workspace = Workspace.load(str(root))
tracker, = workspace.registry.resolve(identity['tracker_id'], storage=workspace.storage.substorage('results'), skip_unknown=False)
experiment = workspace.stack[identity['experiment_id']]
assigned = data['sequences'][args.shard::3]
receipt_path = root/('shard_{}.json'.format(args.shard))
assert not receipt_path.exists()
completed = []
started = time.perf_counter()
for row in assigned:
    assert hashlib.sha256(Path(row['groundtruth']).read_bytes()).hexdigest() == row['groundtruth_sha256']
    sequence = workspace.dataset[row['name']]
    assert len(sequence) == row['frame_count']
    assert not experiment.results(tracker, sequence).find('*')
    experiment.execute(tracker, sequence, force=True)
    complete, files, _ = experiment.scan(tracker, sequence)
    assert complete and files
    completed.append(row['name'])
    print(json.dumps(dict(sequence=row['name'], shard=args.shard, trajectory_files=files)), flush=True)
receipt_path.write_text(json.dumps(dict(status='TRAJECTORIES_WRITTEN_NOT_SCORED', shard=args.shard,
                                      gpu=args.gpu, seed=2026, completed_sequences=completed,
                                      shard_wallclock_seconds=time.perf_counter()-started,
                                      identity_sha256=hashlib.sha256((root/'identity.json').read_bytes()).hexdigest()), indent=2)+'\n')
