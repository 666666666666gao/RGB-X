"""Create a fresh, endpoint-bound workspace using the official RGB-D experiment."""
import argparse
import hashlib
import json
import shlex
from importlib.metadata import version
from pathlib import Path

import vot
import yaml
from vot.workspace import Workspace
from xtrack_eval_parameters import verified_endpoint

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--expected-sha256', required=True)
parser.add_argument('--terminal-audit', required=True)
parser.add_argument('--dataset-audit', required=True)
parser.add_argument('--workspace', required=True)
args = parser.parse_args()
assert version('vot-toolkit') == '0.7.1' and version('vot-trax') == '4.0.2'
assert vot.config.results_binary is True
identity = verified_endpoint(args.config, args.checkpoint, args.expected_sha256, args.terminal_audit)
data_path = Path(args.dataset_audit).resolve(strict=True)
data = json.loads(data_path.read_text())
assert data['status'] == 'READY' and data['dataset'] in ['DepthTrack','VOT-RGBD2022']
root = Path(args.workspace).resolve()
assert not root.exists(), 'Use a fresh workspace; previous trajectories and caches must not be reused.'
year = 'vot2021' if data['dataset'] == 'DepthTrack' else 'vot2022'
stack_path = Path(vot.__file__).parent / 'stack' / year / 'rgbd.yaml'
stack = yaml.safe_load(stack_path.read_text())
stack.pop('dataset')
experiment_id, = stack['experiments']
analyses = stack['experiments'][experiment_id]['analyses']
if data['dataset'] == 'DepthTrack':
    analyses.append(dict(type='pr_curves', name='per_sequence_pr_curves'))
else:
    analyses.extend([dict(type='multistart_ar', name='per_sequence_ar'),
                     dict(type='multistart_eao_curves', name='per_sequence_eao', high=755)])
tracker_id = 'xtrack_joint_s2026_' + args.expected_sha256[:12]
Workspace.initialize(str(root), dict(stack='stack.yaml', sequences=data['root'],
                                    registry=['trackers.ini']), download=False)
(root/'stack.yaml').write_text(yaml.safe_dump(stack, sort_keys=False))
project = Path('/data/gb/rgbx-risk')
command = ['/home/gaob/conda/envs/rgbx-eval/bin/python', str(project/'scripts/evaluate_xtrack_vot.py'),
           '--config', identity['config'], '--checkpoint', identity['checkpoint'],
           '--expected-sha256', args.expected_sha256, '--terminal-audit', str(Path(args.terminal_audit).resolve()),
           '--seed', '2026']
(root/'trackers.ini').write_text('[{}]\nlabel = XTrack-B joint checkpoint, inference seed2026\nprotocol = trax\ncommand = {}\n'
                              'timeout = 120\nenv_PYTHONPATH = {}:{}\n'.format(
                                  tracker_id, shlex.join(command), project/'third_party/XTrack', project/'scripts'))
identity.update(dataset=data['dataset'], dataset_audit=str(data_path),
                dataset_audit_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                terminal_audit=str(Path(args.terminal_audit).resolve()),
                source_stack_sha256=hashlib.sha256(stack_path.read_bytes()).hexdigest(),
                workspace_stack_sha256=hashlib.sha256((root/'stack.yaml').read_bytes()).hexdigest(),
                trackers_ini_sha256=hashlib.sha256((root/'trackers.ini').read_bytes()).hexdigest(),
                tracker_id=tracker_id, experiment_id=experiment_id, seed=2026,
                expected_sequences=data['expected_sequences'],
                expected_frame_pairs=data['expected_frame_pairs'])
(root/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
workspace = Workspace.load(str(root))
assert workspace.stack.dataset is None
assert set(workspace.dataset.keys()) == {row['name'] for row in data['sequences']}
assert len(workspace.stack) == 1
print(json.dumps(identity))
