"""Resume the observed pre-tracker directory failure without replaying OPE."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT = Path('/data/gb/rgbx-risk')
PYTHON = '/data/gb/conda/envs/rgbx-risk/bin/python'
EVAL_PYTHON = '/home/gaob/conda/envs/rgbx-eval/bin/python'
TRAINING = Path('/home/gaob/rgbx-full-method/xtrack_full_method_v1_s2026')
ORIGINAL = TRAINING.with_name(TRAINING.name + '_controller')
REFERENCE = Path('/home/gaob/rgbx-eval-results/xtrack_author_release_s2026')
METHOD = Path('/home/gaob/rgbx-eval-results/xtrack_full_method_v1_s2026')
ROOT = TRAINING.with_name(TRAINING.name + '_evaluation_recovery_20261005')
REPORT = PROJECT / 'reports/full_method_v1_evaluation_recovery_20261005.json'
original = json.loads((ORIGINAL / 'run.json').read_text())
reference = json.loads((REFERENCE / 'run.json').read_text())
assert original['status'] == 'FAILED' and original['current_stage'] == 'reference-evaluation' and original['exit_code'] == 1
assert reference['completed_datasets'] == ['LasHeR', 'VisEvent'] and reference['worker_exit_codes'] == [1, 1, 1]
assert not Path('/proc/' + str(original['controller_pid'])).exists()
assert not Path('/proc/' + str(reference['controller_pid'])).exists()
assert not METHOD.exists() and not ROOT.exists() and not REPORT.exists()
assert not list((REFERENCE / 'DepthTrack/results').rglob('*'))
assert not list((REFERENCE / 'DepthTrack').glob('shard_*.json'))
training_exit = json.loads((ORIGINAL / 'training_exit.json').read_text())
assert training_exit['status'] == 'EXITED' and training_exit['exit_code'] == 0
assert {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in training_exit['source_hashes']} == training_exit['source_hashes']
for name in ['author_release_endpoint_audit.json', 'full_method_v1_endpoint_audit.json']:
    audit = json.loads((PROJECT / 'reports' / name).read_text())
    assert audit['status'] == 'PASS'
    assert hashlib.sha256(Path(audit['checkpoint']).read_bytes()).hexdigest() == audit['checkpoint_sha256']
regression = json.loads((PROJECT / 'reports/vot_empty_results_directory_regression_20261005.json').read_text())
assert regression['status'] == 'ACTUAL_INSTALLED_VOT_EMPTY_DIRECTORY_REGRESSION_PASS_NOT_TRACKING'
ROOT.mkdir()
snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
cards = [tuple(int(x.strip()) for x in line.split(',')) for line in snapshot.splitlines()]
assert [index for index, _ in cards] == [0, 1, 2] and all(memory < 500 for _, memory in cards)
sources = ['scripts/run_vot_sequences.py', 'scripts/check_vot_empty_results.py', 'scripts/resume_reference_depthtrack_and_method_v1.py']
receipt = dict(status='RUNNING_RECOVERY_NOT_COMPLETE', controller_pid=os.getpid(), run_id=TRAINING.name,
    started_at=datetime.now(timezone.utc).isoformat(), poll_seconds=240, original_controller_status=original['status'],
    original_controller_sha256=hashlib.sha256((ORIGINAL / 'run.json').read_bytes()).hexdigest(),
    original_failed_reference_sha256=hashlib.sha256((REFERENCE / 'run.json').read_bytes()).hexdigest(),
    original_completed_reference_datasets=reference['completed_datasets'], completed_stages=[], gpu_preflight=snapshot,
    source_hashes={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in sources},
    training_source_hashes=training_exit['source_hashes'], benchmark_goal='NOT_YET_JUDGED',
    scope='Continue reference DepthTrack after an actual missing-directory failure before any tracker execution; preserve completed reference OPE and old failures. Then run complete method evaluation/comparison/conditional sequence uncertainty. Training and checkpoint do not change.')
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PYTHONPATH=str(PROJECT/'third_party/XTrack')+':'+str(PROJECT/'scripts'))
started = time.perf_counter()
(ROOT / 'original_controller_failure.json').write_text(json.dumps(original, indent=2) + '\n')
(ROOT / 'original_reference_failure.json').write_text(json.dumps(reference, indent=2) + '\n')


def save():
    receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), wallclock_seconds=time.perf_counter()-started)
    raw = json.dumps(receipt, indent=2) + '\n'
    (ROOT / 'run.json').write_text(raw)
    REPORT.write_text(raw)


def save_reference():
    now = datetime.now(timezone.utc)
    reference.update(observed_at=now.isoformat(), recovery_controller_pid=os.getpid(),
        wallclock_seconds=(now-datetime.fromisoformat(reference['started_at'])).total_seconds(),
        wallclock_scope='Includes original evaluation, observed directory failure, diagnosis gap and recovery; per-frame inference timing remains in coverage.')
    (REFERENCE / 'run.json').write_text(json.dumps(reference, indent=2) + '\n')


def stage(label, argv, visible=''):
    receipt.update(current_stage=label, argv=argv)
    with (ROOT / (label + '.log')).open('x') as log:
        process = subprocess.Popen(argv, cwd=str(PROJECT), env=dict(environment, CUDA_VISIBLE_DEVICES=visible), stdout=log, stderr=subprocess.STDOUT)
    receipt['worker_pid'] = process.pid
    save()
    while process.poll() is None:
        time.sleep(240)
        save()
    code = process.wait()
    receipt.update(last_stage_exit_code=code)
    if code != 0:
        receipt.update(status='FAILED', exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
        save()
        raise SystemExit(code)
    receipt['completed_stages'].append(label)
    save()


save()
reference.update(status='RUNNING_REFERENCE_DEPTHTRACK_RECOVERY_NOT_SCORED', original_controller_pid=reference['controller_pid'],
    controller_pid=os.getpid(), recovery_receipt=str(REPORT))
processes = []
for gpu in range(3):
    argv = [EVAL_PYTHON, str(PROJECT/'scripts/run_vot_sequences.py'), '--workspace', str(REFERENCE/'DepthTrack'), '--shard', str(gpu), '--gpu', str(gpu)]
    with (ROOT / ('reference-depthtrack-shard'+str(gpu)+'.log')).open('x') as log:
        process = subprocess.Popen(argv, cwd=str(PROJECT), env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
    processes.append(process)
    reference['worker_processes'].append(dict(dataset='DepthTrack', shard=gpu, pid=process.pid, attempt='directory_fix_recovery'))
    reference['commands'].append(dict(label='DepthTrack-directory-fix-recovery', argv=argv, CUDA_VISIBLE_DEVICES=str(gpu)))
receipt.update(current_stage='reference-depthtrack', worker_pids=[p.pid for p in processes])
save_reference()
save()
while any(process.poll() is None for process in processes):
    reference['worker_exit_codes'] = [process.poll() for process in processes]
    save_reference()
    save()
    time.sleep(240)
codes = [process.wait() for process in processes]
reference['worker_exit_codes'] = codes
save_reference()
if codes != [0, 0, 0]:
    receipt.update(status='FAILED', exit_code=1, worker_exit_codes=codes)
    save()
    raise SystemExit(1)
receipt['completed_stages'].append('reference-depthtrack-trajectories')
save()
stage('reference-depthtrack-scoring', [EVAL_PYTHON,str(PROJECT/'scripts/audit_and_score_vot.py'),'--workspace',str(REFERENCE/'DepthTrack')])
stage('reference-depthtrack-attributes', [EVAL_PYTHON,str(PROJECT/'scripts/score_depthtrack_attributes.py'),'--workspace',str(REFERENCE/'DepthTrack'),
    '--attributes',str(PROJECT/'reports/depthtrack_test_attributes.json')])
reference['completed_datasets'].append('DepthTrack')
reference.update(status='ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT', recovery_ended_at=datetime.now(timezone.utc).isoformat())
save_reference()
stage('method-evaluation', [PYTHON,str(PROJECT/'scripts/evaluate_existing_three.py'),'--terminal-audit',str(PROJECT/'reports/full_method_v1_endpoint_audit.json'),
    '--output-root',str(METHOD)],visible='0,1,2')
comparison_path = PROJECT/'reports/full_method_v1_complete_comparison.json'
assert not comparison_path.exists()
stage('complete-comparison', [PYTHON,str(PROJECT/'scripts/compare_complete_results.py'),'--reference',str(REFERENCE),'--method',str(METHOD),
    '--training-run',str(TRAINING/'run.json'),'--training-updates',str(TRAINING/'updates.jsonl'),'--output',str(comparison_path)])
comparison = json.loads(comparison_path.read_text())
receipt.update(benchmark_goal=comparison['status'], comparison=str(comparison_path))
save()
stage('paired-sequence-uncertainty', [EVAL_PYTHON,str(PROJECT/'scripts/analyze_paired_sequence_uncertainty.py'),
    '--comparison',str(comparison_path),'--output',str(PROJECT/'reports/full_method_v1_paired_sequence_uncertainty.json'),'--draws','10000','--seed','2026'])
receipt.update(status=comparison['status'], exit_code=0, ended_at=datetime.now(timezone.utc).isoformat())
save()
print(json.dumps(receipt))
