"""Evaluate the completed scratch CAD B0 on the three frozen full test sets.

Sanity is deliberately separate and cannot create formal coverage or scores.
Formal workers run once per physical GPU0/1, without periodic status polling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

from cad_eval_parameters import verified_endpoint

PROJECT = Path('/data/gb/rgbx-risk')
PYTHON = '/home/gaob/conda/envs/rgbx-eval/bin/python'
CONFIG = PROJECT/'configs/cad_rgbx_b0_s2026.json'
CHECKPOINT = Path('/home/gaob/rgbx-cadtrack/cad_pruned_rgbx_b0_s2026/CADTrack_latest.pth.tar')
AUDIT = PROJECT/'reports/cad_rgbx_final_checkpoint_audit_20261016.json'
SHA = '4f113dfd0fd3f5ffb13add1fc2c7b9c52f9e063d29ef8d20470b976bfb13aa20'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sanity', 'formal'], required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--sanity-report')
    parser.add_argument('--review-report')
    args = parser.parse_args()
    expected_versions = json.loads((PROJECT/'configs/cad_eval_environment_spec.json').read_text())['versions']
    actual_versions = {name: version(name) for name in expected_versions}
    assert actual_versions == expected_versions, actual_versions
    identity = verified_endpoint(CONFIG, CHECKPOINT, SHA, AUDIT)
    assert Path(sys.executable).resolve() == Path(PYTHON).resolve()
    snapshot = subprocess.check_output(['nvidia-smi', '-i', '0,1',
        '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    cards = [tuple(int(part.strip()) for part in line.split(',')) for line in snapshot.splitlines()]
    assert [card[0] for card in cards] == [0, 1] and all(card[1] < 500 for card in cards), snapshot
    root = Path(args.output_root)
    assert root.parent == Path('/home/gaob/rgbx-eval-results') and not root.exists()
    if args.stage == 'formal':
        sanity = json.loads(Path(args.sanity_report).read_text())
        review = json.loads(Path(args.review_report).read_text())
        assert sanity['status'] == 'SANITY_PASS_NOT_BENCHMARK' and sanity['identity'] == identity
        assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
        assert review['acceptance_status'] == 'provisional'
        assert review['sanity_report_sha256'] == hashlib.sha256(Path(args.sanity_report).read_bytes()).hexdigest()
        for name, digest in review['source_sha256'].items():
            assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() == digest, name
    (root/'logs').mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(PROJECT)+':'+str(PROJECT/'scripts'),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    receipt = dict(status='RUNNING_NOT_SCORED', stage=args.stage, identity=identity,
                   started_at=datetime.now(timezone.utc).isoformat(), controller_pid=os.getpid(),
                   physical_gpus=[0, 1], gpu_preflight=snapshot, commands=[], workers=[], completed_datasets=[],
                   output_root=str(root), sanity_report=args.sanity_report, review_report=args.review_report)
    receipt['environment_versions'] = actual_versions
    started = time.perf_counter()

    def save():
        receipt['observed_at'] = datetime.now(timezone.utc).isoformat()
        receipt['wallclock_seconds'] = time.perf_counter()-started
        temporary = root/'run.tmp'
        temporary.write_text(json.dumps(receipt, indent=2)+'\n')
        os.replace(str(temporary), str(root/'run.json'))

    def run(label, script, arguments, gpu=''):
        argv = [PYTHON, str(PROJECT/'scripts'/script), *arguments]
        receipt['commands'].append(dict(label=label, argv=argv, CUDA_VISIBLE_DEVICES=gpu))
        save()
        with (root/'logs'/(label+'.log')).open('x') as log:
            result = subprocess.run(argv, env=dict(env, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            receipt.update(status='FAILED', failed_command=label, actual_exit_code=result.returncode)
            save()
            raise SystemExit(result.returncode)

    def shards(label, script, arguments, ope=False):
        processes = []
        for gpu in [0, 1]:
            extra = ['--shard', str(gpu), '--shards', '2'] if ope else ['--shard', str(gpu), '--gpu', str(gpu)]
            argv = [PYTHON, str(PROJECT/'scripts'/script), *arguments, *extra]
            with (root/'logs'/('{}-gpu{}.log'.format(label, gpu))).open('x') as log:
                process = subprocess.Popen(argv, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
            receipt['commands'].append(dict(label=label, argv=argv, CUDA_VISIBLE_DEVICES=str(gpu)))
            receipt['workers'].append(dict(dataset=label, physical_gpu=gpu, pid=process.pid))
            processes.append(process)
        save()
        # Wait for the original workers, not a repeated remote poll loop.
        codes = [process.wait() for process in processes]
        receipt['last_worker_exit_codes'] = codes
        save()
        if codes != [0, 0]:
            receipt.update(status='FAILED', failed_command=label, actual_exit_code=1)
            save()
            raise SystemExit(1)

    common = ['--config', str(CONFIG), '--checkpoint', str(CHECKPOINT), '--expected-sha256', SHA, '--terminal-audit', str(AUDIT)]
    workspace = str(root/'DepthTrack')
    save()
    run('DepthTrack-workspace', 'prepare_vot_workspace.py', common+['--dataset-audit', str(PROJECT/'reports/depthtrack_vot_dataset_audit.json'),
        '--workspace', workspace, '--shards', '2'])
    if args.stage == 'sanity':
        for gpu in [0, 1]:
            for dataset, manifest in [('LasHeR', 'lasher_eval_manifest.json'), ('VisEvent', 'visevent_eval_manifest.json')]:
                run('{}-native-gpu{}'.format(dataset, gpu), 'evaluate_xtrack_ope.py', common+[
                    '--manifest', str(PROJECT/'reports'/manifest), '--output', str(root/dataset/('gpu'+str(gpu))),
                    '--shard', str(gpu), '--shards', '2', '--sanity-frames', '3'], str(gpu))
            run('DepthTrack-native-gpu'+str(gpu), 'check_cad_vot_runtime.py', ['--workspace', workspace, '--gpu', str(gpu)], str(gpu))
        receipt.update(status='SANITY_PASS_NOT_BENCHMARK', optimizer_steps=0, benchmark_datasets_completed=0)
        save()
        print(json.dumps(receipt))
        return
    for dataset, manifest, attributes, scorer, toolkit in [
        ('LasHeR', 'lasher_eval_manifest.json', 'lasher_test_attributes.json', 'score_lasher.py', '/home/gaob/.cache/rgbx_lasher_toolkit_src'),
        ('VisEvent', 'visevent_eval_manifest.json', 'visevent_test_attributes.json', 'score_visevent.py', '/home/gaob/.cache/rgbx_official_visevent_source')]:
        directory = root/dataset
        predictions = str(directory/'predictions')
        shards(dataset, 'evaluate_xtrack_ope.py', common+['--manifest', str(PROJECT/'reports'/manifest), '--output', predictions], ope=True)
        coverage = str(directory/'coverage.json')
        run(dataset+'-coverage', 'audit_ope_coverage.py', common+['--manifest', str(PROJECT/'reports'/manifest), '--output', predictions, '--report', coverage])
        run(dataset+'-scoring', scorer, common+['--manifest', str(PROJECT/'reports'/manifest), '--attributes', str(PROJECT/'reports'/attributes),
            '--predictions', predictions, '--coverage', coverage, '--toolkit', toolkit, '--output', str(directory/'scores')])
        receipt['completed_datasets'].append(dataset)
        save()
    shards('DepthTrack', 'run_vot_sequences.py', ['--workspace', workspace])
    run('DepthTrack-official-scoring', 'audit_and_score_vot.py', ['--workspace', workspace])
    run('DepthTrack-attributes', 'score_depthtrack_attributes.py', ['--workspace', workspace, '--attributes', str(PROJECT/'reports/depthtrack_test_attributes.json')])
    receipt['completed_datasets'].append('DepthTrack')
    receipt.update(status='ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT', actual_exit_code=0)
    save()
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
