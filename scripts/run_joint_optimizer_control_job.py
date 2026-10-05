"""Run one audited C/D control and complete cached-reference three-dataset comparison."""
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rgbx_controls.execution import RUNS, launch_inputs, sha, verify_completed_sanity

PYTHON = '/data/gb/conda/envs/rgbx-risk/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', choices=list(RUNS), required=True)
    args = parser.parse_args()
    run_id, protocol_path, protocol = launch_inputs(PROJECT, args.optimizer)
    sanity_root = Path(protocol['output'] + '_sanity')
    sanity = json.loads((sanity_root / 'run.json').read_text())
    sanity_audit_path = PROJECT / 'reports' / (run_id + '_sanity_checkpoint_audit.json')
    sanity_audit = json.loads(sanity_audit_path.read_text())
    sanity_execution_path = Path('/home/gaob/rgbx-full-method') / (run_id + '_sanity_controller') / 'training_exit.json'
    execution = json.loads(sanity_execution_path.read_text())
    verify_completed_sanity(protocol, sanity, sanity_audit, execution)
    assert sanity['protocol_sha256'] == sha(protocol_path)
    assert sanity_audit['execution_receipt_sha256'] == sha(sanity_execution_path)
    assert not Path('/proc', str(execution['worker_pid'])).exists()
    checkpoint = sanity_root / 'XTrack_ep0001.pth.tar'
    assert checkpoint.resolve(strict=True) == checkpoint and Path(sanity_audit['checkpoint']) == checkpoint
    metadata = checkpoint.lstat()
    assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
    assert sha(checkpoint) == sanity_audit['checkpoint_sha256']
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    cards = [tuple(int(value.strip()) for value in line.split(',')) for line in snapshot.splitlines()]
    assert [index for index, _ in cards] == [0, 1, 2] and all(memory < 500 for _, memory in cards)
    assert shutil.disk_usage('/home/gaob/rgbx-full-method').free > 20 * 1024 ** 3
    assert not Path(protocol['output']).exists()
    root = Path('/home/gaob/rgbx-full-method') / (run_id + '_controller')
    root.mkdir()
    receipt = dict(status='RUNNING', run_id=run_id, controller_pid=os.getpid(), exit_code=None,
                   started_at=datetime.now(timezone.utc).isoformat(), gpu_preflight=snapshot, poll_seconds=240,
                   protocol=protocol, protocol_sha256=sha(protocol_path), completed_stages=[],
                   first_epoch_action='External scheduled observer: after actual epoch1 save execute exact CPU epoch1 audit once, then launch retain_joint_optimizer_control_checkpoints once. Do not reuse another run receipt.',
                   first_epoch_audit_and_retention_activation='PENDING_ACTUAL_SAVED_EPOCH1', benchmark_goal='NOT_YET_JUDGED')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(PROJECT) + ':' + str(PROJECT / 'third_party/XTrack') + ':' + str(PROJECT / 'third_party/XTrack/lib/train'))
    started = time.perf_counter()

    def save():
        receipt.update(observed_at=datetime.now(timezone.utc).isoformat(), wallclock_seconds=time.perf_counter() - started)
        (root / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def stage(label, argv, cpu=False):
        receipt.update(current_stage=label, argv=argv)
        with (root / (label + '.log')).open('x') as log:
            process = subprocess.Popen(argv, cwd=str(PROJECT), env=dict(environment, CUDA_VISIBLE_DEVICES='' if cpu else '0,1,2'), stdout=log, stderr=subprocess.STDOUT)
        receipt['worker_pid'] = process.pid
        save()
        if label == 'train':
            # Formal training starts from the public release, not from this sanity weight.
            receipt['sanity_weight_retirement'] = dict(path=str(checkpoint), bytes=metadata.st_size, sha256=sanity_audit['checkpoint_sha256'],
                                                        audit_sha256=sha(sanity_audit_path), removed_at=datetime.now(timezone.utc).isoformat())
            checkpoint.unlink()
            save()
        while process.poll() is None:
            time.sleep(240)
            save()
        code = process.wait()
        receipt['last_stage_exit_code'] = code
        if code != 0:
            receipt.update(status='FAILED', exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
            save()
            raise SystemExit(code)
        receipt['completed_stages'].append(label)
        save()

    save()
    port = '29531' if args.optimizer == 'adamw' else '29532'
    stage('train', [PYTHON, '-m', 'torch.distributed.run', '--nproc_per_node=3', '--master_port=' + port,
                    str(PROJECT / 'scripts/train_joint_optimizer_control.py'), '--protocol', str(protocol_path)])
    execution_path = root / 'training_exit.json'
    execution = dict(receipt, status='EXITED', exit_code=0, stage='endpoint', ended_at=datetime.now(timezone.utc).isoformat())
    execution_path.write_text(json.dumps(execution, indent=2) + '\n')
    audit = PROJECT / 'reports' / (run_id + '_endpoint_checkpoint_audit.json')
    stage('endpoint-audit', [PYTHON, str(PROJECT / 'scripts/audit_joint_optimizer_control_checkpoint.py'), '--optimizer', args.optimizer,
                             '--stage', 'endpoint', '--execution', str(execution_path)], cpu=True)
    evaluation_root = '/home/gaob/rgbx-eval-results/' + run_id
    stage('evaluation', [PYTHON, str(PROJECT / 'scripts/evaluate_existing_three.py'), '--terminal-audit', str(audit), '--output-root', evaluation_root])
    author_comparison = PROJECT / 'reports' / (run_id + '_author_comparison.json')
    stage('author-comparison', [PYTHON, str(PROJECT / 'scripts/compare_joint_optimizer_control_results.py'),
                               '--reference', '/home/gaob/rgbx-eval-results/xtrack_author_release_s2026', '--method', evaluation_root,
                               '--training-run', protocol['output'] + '/run.json', '--training-updates', protocol['output'] + '/updates.jsonl',
                               '--output', str(author_comparison)], cpu=True)
    author_uncertainty = PROJECT / 'reports' / (run_id + '_author_paired_sequence_uncertainty.json')
    stage('author-uncertainty', [PYTHON, str(PROJECT / 'scripts/analyze_paired_sequence_uncertainty.py'),
                                '--comparison', str(author_comparison), '--output', str(author_uncertainty), '--draws', '10000', '--seed', '2026'], cpu=True)
    if args.optimizer == 'task_adaptive_only':
        component_comparison = PROJECT / 'reports' / (run_id + '_vs_fixed_adamw_comparison.json')
        stage('component-comparison', [PYTHON, str(PROJECT / 'scripts/compare_joint_optimizer_control_results.py'),
                                      '--reference', '/home/gaob/rgbx-eval-results/xtrack_fixed_mixed_adamw_s2026', '--method', evaluation_root,
                                      '--training-run', protocol['output'] + '/run.json', '--training-updates', protocol['output'] + '/updates.jsonl',
                                      '--output', str(component_comparison)], cpu=True)
        stage('component-uncertainty', [PYTHON, str(PROJECT / 'scripts/analyze_paired_sequence_uncertainty.py'),
                                       '--comparison', str(component_comparison), '--output', str(PROJECT / 'reports' / (run_id + '_vs_fixed_adamw_paired_sequence_uncertainty.json')),
                                       '--draws', '10000', '--seed', '2026'], cpu=True)
        receipt['component_comparison'] = str(component_comparison)
    result = json.loads(author_comparison.read_text())
    receipt.update(status=result['status'], exit_code=0, ended_at=datetime.now(timezone.utc).isoformat(),
                   author_comparison=str(author_comparison), author_paired_sequence_uncertainty=str(author_uncertainty),
                   benchmark_goal=result['status'])
    save()
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
