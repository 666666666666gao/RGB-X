import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path('C:/Users/gb/projects/rgbx-risk')
TEMP = Path('C:/Users/gb/.codex_tmp')
PREFIX = TEMP / 'rgbx_d_documented_sanity_20261007'
RUN_ID = 'xtrack_task_stats_only_s2026'
REPORT = PROJECT / ('reports/' + RUN_ID + '_documented_sanity_20261007.json')
EVIDENCE = PROJECT / ('reports/' + RUN_ID + '_sanity_evidence_20261007')
launch_lines = Path(str(PREFIX) + '.ssh.stdout.log').read_text(encoding='utf-8').splitlines()
launch = json.loads(launch_lines[0])
assert datetime.now(timezone.utc) >= datetime.fromisoformat(launch['first_observation_not_before'])
assert not REPORT.exists() and not EVIDENCE.exists()

REMOTE = r'''
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
project = Path('/data/gb/rgbx-risk')
sys.path.insert(0, str(project))
from rgbx_controls.execution import verify_completed_sanity
run_id = 'xtrack_task_stats_only_s2026'
base = Path('/home/gaob/rgbx-full-method')
root = base / (run_id + '_sanity')
controller = base / (run_id + '_sanity_controller')
paths = {
    'training_exit': controller / 'training_exit.json',
    'controller_run': controller / 'run.json',
    'train_log': controller / 'train.log',
    'saved_state_audit_log': controller / 'saved_state_audit.log',
    'sanity_run': root / 'run.json',
    'updates': root / 'updates.jsonl',
    'parameter_manifest': root / 'parameter_manifest.json',
    'checkpoint_audit': project / 'reports' / (run_id + '_sanity_checkpoint_audit.json'),
    'launcher_log': base / (run_id + '_sanity_controller_launcher.log'),
    'wrapper_exit': base / (run_id + '_sanity_controller_launcher_exit.json'),
}
artifacts = {}
payloads = {}
for key, path in paths.items():
    if path.exists():
        data = path.read_bytes()
        payloads[key] = data
        artifacts[key] = dict(path=str(path), exists=True, bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest(), base64=base64.b64encode(data).decode('ascii'))
    else:
        artifacts[key] = dict(path=str(path), exists=False)
receipts = {key: json.loads(payloads[key].decode('utf-8'))
            for key in ['training_exit', 'controller_run', 'sanity_run', 'checkpoint_audit', 'wrapper_exit'] if key in payloads}
owner_pids = [3756736, 3756735]
if 'training_exit' in receipts:
    owner_pids.append(receipts['training_exit']['worker_pid'])
if 'sanity_run' in receipts:
    owner_pids.extend(receipts['sanity_run']['rank_pids'])
processes = []
for pid in owner_pids:
    proc = Path('/proc', str(pid))
    item = dict(pid=pid, exists=proc.exists())
    if proc.exists():
        item.update(stat=(proc / 'stat').read_text(encoding='utf-8'),
                    argv=(proc / 'cmdline').read_bytes().decode('utf-8').split('\x00'))
    processes.append(item)
preparation = json.loads((project / 'reports/joint_optimizer_controls_execution_preparation_20261005.json').read_text(encoding='utf-8'))
source_sha = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in preparation['source_sha256']}
spec = json.loads((project / 'configs/env-spec.json').read_text(encoding='utf-8'))
packet = dict(observed_at=datetime.now(timezone.utc).isoformat(), artifacts=artifacts, receipts=receipts,
              processes=processes, source_sha256=source_sha, source_matches_preparation=source_sha == preparation['source_sha256'],
              entry_sha256=hashlib.sha256((project / 'scripts/xtrack_eval_parameters.py').read_bytes()).hexdigest(),
              env_spec_canonical_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
              runtime=dict(python=sys.version, executable=sys.executable,
                           packages={name:importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy']},
                           torch_imported='torch' in sys.modules),
              formal_output_exists=Path('/home/gaob/rgbx-full-method/xtrack_task_stats_only_s2026').exists(),
              formal_controller_exists=Path('/home/gaob/rgbx-full-method/xtrack_task_stats_only_s2026_controller').exists())
print(json.dumps(packet), flush=True)
assert all(item['exists'] for item in artifacts.values())
assert receipts['wrapper_exit']['status'] == 'ACTUAL_SANITY_WRAPPER_EXITED' and receipts['wrapper_exit']['exit_code'] == 0
protocol = json.loads((project / 'configs/xtrack_task_stats_only_s2026.json').read_text(encoding='utf-8'))
run, audit, execution = receipts['sanity_run'], receipts['checkpoint_audit'], receipts['training_exit']
verify_completed_sanity(protocol, run, audit, execution)
assert receipts['controller_run']['saved_state_audit_exit_code'] == 0
assert audit['execution_receipt_sha256'] == artifacts['training_exit']['sha256']
assert audit['run_sha256'] == artifacts['sanity_run']['sha256']
assert audit['all_model_tensors_finite'] and not audit['cuda_initialized']
assert audit['frozen_tensor_count'] == 244
assert audit['optimizer_check']['optimizer_kind'] == 'task_adaptive_only'
assert audit['optimizer_check']['trainable_parameter_tensors'] == 504
assert audit['optimizer_check']['always_used_witness_tensors'] == 24
assert audit['rng_check']['saved_rank_count'] == 3
assert audit['rng_check']['python_numpy_cpu_rng_restore_valid'] and audit['rng_check']['cuda_rng_bytes_present']
assert not audit['rng_check']['cuda_rng_restore_executed']
assert run['initial_checkpoint_sha256'] == '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
assert len(run['rank_pids']) == run['world_size'] == 3
updates = [json.loads(line) for line in payloads['updates'].decode('utf-8').splitlines()]
assert [row['global_step'] for row in updates] == [1, 2]
for row in updates:
    assert row['global_task_counts'] == [8, 8, 8]
    assert row['optimizer_statistics_updates'] == row['weight_decay_applications'] == 1
    assert all(row[key] == 0 for key in ['localization_check_calls', 'correction_calls', 'trajectory_calls', 'pair_check_forward_examples', 'trajectory_model_forward_examples'])
    assert math.isfinite(row['step_seconds']) and row['step_seconds'] > 0
assert len(run['epoch_history']) == 1 and run['epoch_history'][0]['steps'] == 2
assert all(math.isfinite(value) for value in run['epoch_history'][0]['means'].values())
assert not any(item['exists'] for item in processes)
assert packet['source_matches_preparation']
assert packet['entry_sha256'] == preparation['prepared_entry_sha256']
assert packet['env_spec_canonical_sha256'] == '433a922a61d09bbf39d671e9ee1636c4ab0cac3ec6865b32baac9e7ce7e12fa4'
assert not packet['runtime']['torch_imported']
assert not packet['formal_output_exists'] and not packet['formal_controller_exists']
print(json.dumps(dict(status='PASS_EXISTING_COMPLETED_SANITY_CONTRACT_ON_ACTUAL_RECEIPTS',
                      verify_completed_sanity_invocations=1, model_forward_calls=0,
                      torch_imported=False, observed_at=datetime.now(timezone.utc).isoformat(),
                      updates=updates)), flush=True)
'''

ssh_argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=20',
            '2028', '/data/gb/conda/envs/rgbx-risk/bin/python', '-u', '-']
result = subprocess.run(ssh_argv, input=REMOTE.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
with Path(str(PREFIX) + '.collection.stdout.log').open('xb') as stream:
    stream.write(result.stdout)
with Path(str(PREFIX) + '.collection.stderr.log').open('xb') as stream:
    stream.write(result.stderr)
lines = result.stdout.decode('utf-8').splitlines()
packet = json.loads(lines[0])
EVIDENCE.mkdir()
file_names = {
    'training_exit': 'training_exit.json', 'controller_run': 'controller_run.json',
    'train_log': 'train.log', 'saved_state_audit_log': 'saved_state_audit.log',
    'sanity_run': 'sanity_run.json', 'updates': 'updates.jsonl',
    'parameter_manifest': 'parameter_manifest.json', 'checkpoint_audit': 'checkpoint_audit.json',
    'launcher_log': 'launcher.log', 'wrapper_exit': 'wrapper_exit.json',
}
for key, artifact in packet['artifacts'].items():
    if artifact['exists']:
        data = base64.b64decode(artifact.pop('base64'))
        assert hashlib.sha256(data).hexdigest() == artifact['sha256'] and len(data) == artifact['bytes']
        local_path = EVIDENCE / file_names[key]
        with local_path.open('xb') as stream:
            stream.write(data)
        artifact['local_path'] = str(local_path)
        assert hashlib.sha256(local_path.read_bytes()).hexdigest() == artifact['sha256']
passed = result.returncode == 0
validation = json.loads(lines[1]) if passed else dict(status='FAILED_READ_OR_VALIDATION', stderr=result.stderr.decode('utf-8'))
report = dict(status='PASS_DOCUMENTED_TWO_STEP_D_TASK_STATISTICS_SANITY_NOT_FORMAL_RESULT' if passed else 'FAIL_DOCUMENTED_D_TASK_STATISTICS_SANITY',
              recorded_at=datetime.now(timezone.utc).isoformat(), run_id=RUN_ID,
              agent='/root/rgbx_d_documented_sanity_20261007', review_independence='same-family',
              acceptance_status='provisional', model_backend_independently_attested=False,
              runtime_attested=passed, experiment_execution_runtime_attested=passed,
              skill='C:/Users/gb/.codex/skills/run-experiment/SKILL.md',
              compute_contract='C:/Users/gb/.codex/skills/shared-references/compute-env-contract.md section 4.3',
              provider_ledger='docs/RGB-X_研究与实验完整交接_2026-10-02.md sections 2/4 and C/D execution section near line 1890',
              launch=launch, native_exec_session_id=5342, local_waiter_pid=42248,
              invocation_count=1, first_observation_not_before=launch['first_observation_not_before'],
              collection_ssh_exit_code=result.returncode, raw_observation=packet, completion_contract=validation,
              local_source_sha256={name:hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in packet['source_sha256']},
              protocol_scope=dict(documented_epochs=15, actual_sanity_steps=2, actual_sanity_epochs=1,
                                  sanity_does_not_satisfy_formal_budget=True),
              doc_vs_reality=dict(invocation_changed=False, runtime_divergences=[],
                                  historical_statuses_require_parent_update=['The dated preparation section still says admission unactivated and D sanity not executed; activation and this actual run supersede those historical states.'],
                                  saved_cuda_rng_restore_executed=False),
              formal_training_started=False, formal_tracking_scores='NOT_RUN', benchmark_success_claimed=False,
              model_source_edits=0, environment_rebuilt=False, synthetic_tests_repeated=0,
              checkpoint_deletions=0, external_notifications=0, commits_or_pushes=0,
              intended_next_action='Return verified D sanity evidence to parent; parent owns any separately authorized formal D launch and sole handoff publication.' if passed else 'Stop and report original launcher/worker logs and receipts to parent; no rerun or fix.')
assert report['local_source_sha256'] == packet['source_sha256']
with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(status=report['status'], report=str(REPORT), report_sha256=hashlib.sha256(REPORT.read_bytes()).hexdigest(),
                      observation=packet['observed_at'], owner_processes=packet['processes'],
                      wrapper_exit=packet['receipts'].get('wrapper_exit'),
                      checkpoint_audit=packet['receipts'].get('checkpoint_audit'),
                      validation=validation), ensure_ascii=False), flush=True)
raise SystemExit(result.returncode)
