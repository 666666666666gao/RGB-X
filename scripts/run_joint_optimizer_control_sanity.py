"""Run one exact real two-step C/D preflight after distance0 releases the GPUs."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rgbx_controls.execution import RUNS, launch_inputs, sha

PYTHON = '/data/gb/conda/envs/rgbx-risk/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', choices=list(RUNS), required=True)
    args = parser.parse_args()
    run_id, protocol_path, protocol = launch_inputs(PROJECT, args.optimizer)
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    cards = [tuple(int(value.strip()) for value in line.split(',')) for line in snapshot.splitlines()]
    assert [index for index, _ in cards] == [0, 1, 2] and all(memory < 500 for _, memory in cards)
    assert shutil.disk_usage('/home/gaob/rgbx-full-method').free > 20 * 1024 ** 3
    root = Path('/home/gaob/rgbx-full-method') / (run_id + '_sanity_controller')
    root.mkdir()
    assert not Path(protocol['output'] + '_sanity').exists()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(PROJECT) + ':' + str(PROJECT / 'third_party/XTrack') + ':' + str(PROJECT / 'third_party/XTrack/lib/train'))
    port = '29531' if args.optimizer == 'adamw' else '29532'
    argv = [PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=3', '--master_port=' + port,
            str(PROJECT / 'scripts/train_joint_optimizer_control.py'), '--protocol', str(protocol_path), '--sanity-steps', '2']
    with (root / 'train.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=str(PROJECT), env=environment, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(status='RUNNING', run_id=run_id, stage='sanity', controller_pid=os.getpid(), worker_pid=process.pid,
                   started_at=datetime.now(timezone.utc).isoformat(), exit_code=None, argv=argv,
                   protocol_sha256=sha(protocol_path), gpu_preflight=snapshot, poll_seconds=240)
    output = root / 'training_exit.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    while process.poll() is None:
        time.sleep(240)
    code = process.wait()
    receipt.update(status='EXITED', exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    if code != 0:
        raise SystemExit(code)
    audit_argv = [PYTHON, str(PROJECT / 'scripts/audit_joint_optimizer_control_checkpoint.py'),
                  '--optimizer', args.optimizer, '--stage', 'sanity', '--execution', str(output)]
    with (root / 'saved_state_audit.log').open('x') as log:
        audit_code = subprocess.call(audit_argv, cwd=str(PROJECT), env=dict(environment, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    receipt.update(saved_state_audit_argv=audit_argv, saved_state_audit_exit_code=audit_code,
                   formal_training_started=False, formal_tracking_scores='NOT_RUN')
    (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if audit_code != 0:
        raise SystemExit(audit_code)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
