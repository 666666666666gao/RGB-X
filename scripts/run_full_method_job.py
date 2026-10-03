"""Run the full method, then both complete reference and method evaluations."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')
PYTHON = '/data/gb/conda/envs/rgbx-risk/bin/python'
PROTOCOL = PROJECT / 'configs/full_method_v1.json'
protocol = json.loads(PROTOCOL.read_text())
run_id = protocol['run_id']
root = Path('/home/gaob/rgbx-full-method') / (run_id + '_controller')
root.mkdir(parents=True)
snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
cards = [tuple(int(value.strip()) for value in line.split(',')) for line in snapshot.splitlines()]
assert [index for index, _ in cards] == [0, 1, 2] and all(memory < 500 for _, memory in cards)
assert shutil.disk_usage(root).free > 20 * 1024 ** 3
sanity = json.loads(Path(protocol['output'] + '_sanity/run.json').read_text())
assert sanity['status'] == 'SANITY_COMPLETE_NOT_FORMAL_RESULT' and sanity['global_step'] == 2
assert sanity['sanity_checkpoint_check']['strict_reload'] and sanity['sanity_checkpoint_check']['frozen_tensors_unchanged']
assert sanity['sanity_checkpoint_check']['replicas_identical'] and sanity['modules_enabled'] == protocol['modules_enabled']
assert sanity['protocol'] == protocol
review = json.loads((PROJECT / 'reports/full_method_code_review.json').read_text())
assert review['status'] == 'SOURCE_REVIEW_PASS' and review['blocking_issues'] == []
source_hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [PROTOCOL, PROJECT / 'scripts/train_full_method.py', *sorted((PROJECT / 'rgbx_method').glob('*.py'))]}
receipt = dict(status='RUNNING', run_id=run_id, controller_pid=os.getpid(), exit_code=None,
               started_at=datetime.now(timezone.utc).isoformat(), gpu_preflight=snapshot,
               poll_seconds=240, protocol=protocol, source_hashes=source_hashes, completed_stages=[])
receipt_path = root / 'run.json'
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(PROJECT) + ':' + str(PROJECT / 'third_party/XTrack') + ':' + str(PROJECT / 'third_party/XTrack/lib/train'))
started = time.perf_counter()


def save():
    receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), wallclock_seconds=time.perf_counter() - started)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


def run_stage(label, argv, cpu=False):
    receipt.update(current_stage=label, argv=argv)
    with (root / (label + '.log')).open('x') as log:
        process = subprocess.Popen(argv, env=dict(environment, CUDA_VISIBLE_DEVICES='' if cpu else '0,1,2'),
                                   stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT))
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
run_stage('train', [PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=3', '--master_port=29529',
                    str(PROJECT / 'scripts/train_full_method.py'), '--protocol', str(PROTOCOL)])
# The endpoint audit requires this actual, completed training-controller receipt.
training_receipt = dict(receipt, status='EXITED', exit_code=0, ended_at=datetime.now(timezone.utc).isoformat())
training_receipt_path = root / 'training_exit.json'
training_receipt_path.write_text(json.dumps(training_receipt, indent=2) + '\n')
config = str(PROJECT / protocol['base_config'])
method_audit = PROJECT / 'reports/full_method_v1_endpoint_audit.json'
run_stage('method-endpoint-audit', [PYTHON, str(PROJECT / 'scripts/audit_joint_endpoint.py'), '--kind', 'full_method',
                                  '--checkpoint', str(Path(protocol['output']) / 'XTrack_ep0015.pth.tar'), '--config', config,
                                  '--run', str(Path(protocol['output']) / 'run.json'), '--controller', str(training_receipt_path),
                                  '--output', str(method_audit)], cpu=True)
reference_audit = PROJECT / 'reports/author_release_endpoint_audit.json'
for label, audit, output in [('reference-evaluation', reference_audit, 'xtrack_author_release_s2026'),
                             ('method-evaluation', method_audit, 'xtrack_full_method_v1_s2026')]:
    run_stage(label, [PYTHON, str(PROJECT / 'scripts/evaluate_existing_three.py'), '--terminal-audit', str(audit),
                      '--output-root', '/home/gaob/rgbx-eval-results/' + output])
receipt.update(status='BOTH_ENDPOINTS_THREE_DATASETS_SCORED_REQUIRES_RESULT_COMPARISON', exit_code=0,
               ended_at=datetime.now(timezone.utc).isoformat(), benchmark_goal='NOT_YET_JUDGED')
save()
comparison_path = PROJECT / 'reports/full_method_v1_complete_comparison.json'
run_stage('complete-comparison', [PYTHON, str(PROJECT / 'scripts/compare_complete_results.py'),
                                 '--reference', '/home/gaob/rgbx-eval-results/xtrack_author_release_s2026',
                                 '--method', '/home/gaob/rgbx-eval-results/xtrack_full_method_v1_s2026',
                                 '--output', str(comparison_path)], cpu=True)
comparison = json.loads(comparison_path.read_text())
receipt.update(status=comparison['status'], comparison=str(comparison_path),
               benchmark_goal=comparison['status'], ended_at=datetime.now(timezone.utc).isoformat())
save()
print(json.dumps(receipt))
