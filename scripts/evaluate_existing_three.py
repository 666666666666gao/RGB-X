"""Run all existing test sets once from a verified joint-training endpoint."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from xtrack_eval_parameters import verified_endpoint

PROJECT = Path('/data/gb/rgbx-risk')
PYTHON = '/home/gaob/conda/envs/rgbx-eval/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('--terminal-audit', required=True)
parser.add_argument('--output-root', required=True)
args = parser.parse_args()
terminal_path = Path(args.terminal_audit).resolve(strict=True)
terminal = json.loads(terminal_path.read_text())
config = terminal['config'] if terminal['role'] == 'joint_rgbx_checkpoint_audit_not_tracking_benchmark' else str(PROJECT/'configs/xtrack_b_adamw_3gpu.yaml')
identity = verified_endpoint(config,terminal['checkpoint'],terminal['checkpoint_sha256'],str(terminal_path))
gpu_snapshot = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
cards = [tuple(int(value.strip()) for value in line.split(',')) for line in gpu_snapshot.splitlines()]
assert [index for index,_ in cards]==[0,1,2] and all(memory<500 for _,memory in cards)
root = Path(args.output_root).resolve()
assert root.parent==Path('/home/gaob/rgbx-eval-results') and not root.exists()
(root/'logs').mkdir(parents=True)
environment = dict(os.environ,PYTHONPATH=str(PROJECT/'third_party/XTrack')+':'+str(PROJECT/'scripts'),
                   OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
receipt = dict(status='RUNNING_NOT_SCORED',identity=identity,output_root=str(root),controller_pid=os.getpid(),
               started_at=datetime.now(timezone.utc).isoformat(),gpu_preflight=gpu_snapshot,
               terminal_audit_sha256=hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
               poll_seconds=240,completed_datasets=[],worker_processes=[],commands=[])
started = time.perf_counter()
receipt_path = root/'run.json'

def save():
    receipt['observed_at']=datetime.now(timezone.utc).isoformat()
    receipt['wallclock_seconds']=time.perf_counter()-started
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')

def command(script,arguments):
    return [PYTHON,str(PROJECT/'scripts'/script),*arguments]

def run(label,script,arguments,visible=''):
    argv = command(script,arguments)
    receipt['commands'].append(dict(label=label,argv=argv,CUDA_VISIBLE_DEVICES=visible))
    save()
    with (root/'logs'/(label+'.log')).open('x') as log:
        subprocess.run(argv,env=dict(environment,CUDA_VISIBLE_DEVICES=visible),stdout=log,stderr=subprocess.STDOUT,check=True)

def shards(label,script,arguments,ope=False):
    processes = []
    for gpu in range(3):
        extra=['--shard',str(gpu),'--shards','3'] if ope else ['--shard',str(gpu),'--gpu',str(gpu)]
        argv = command(script,arguments+extra)
        with (root/'logs'/('{}-shard{}.log'.format(label,gpu))).open('x') as log:
            process = subprocess.Popen(argv,env=dict(environment,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        processes.append(process)
        receipt['commands'].append(dict(label=label,argv=argv,CUDA_VISIBLE_DEVICES=str(gpu)))
        receipt['worker_processes'].append(dict(dataset=label,shard=gpu,pid=process.pid))
    save()
    while any(process.poll() is None for process in processes):
        receipt['worker_exit_codes']=[process.poll() for process in processes]
        save()
        time.sleep(240)
    codes = [process.wait() for process in processes]
    receipt['worker_exit_codes']=codes
    save()
    assert codes==[0,0,0],codes

common=['--config',config,'--checkpoint',identity['checkpoint'],'--expected-sha256',identity['checkpoint_sha256'],
        '--terminal-audit',str(terminal_path)]
save()
run('cuda-witness','gpu_witness.py',[],visible='0,1,2')
for dataset,manifest_name,attribute_name,scorer,toolkit in [
    ('LasHeR','lasher_eval_manifest.json','lasher_test_attributes.json','score_lasher.py','/home/gaob/.cache/rgbx_lasher_toolkit_src'),
    ('VisEvent','visevent_eval_manifest.json','visevent_test_attributes.json','score_visevent.py','/home/gaob/.cache/rgbx_official_visevent_source')]:
    directory=root/dataset
    directory.mkdir()
    manifest=str(PROJECT/'reports'/manifest_name)
    predictions=str(directory/'predictions')
    shards(dataset,'evaluate_xtrack_ope.py',common+['--manifest',manifest,'--output',predictions,'--seed','2026'],ope=True)
    coverage=str(directory/'coverage.json')
    run(dataset+'-coverage','audit_ope_coverage.py',common+['--manifest',manifest,'--output',predictions,'--report',coverage])
    run(dataset+'-scoring',scorer,common+['--manifest',manifest,'--attributes',str(PROJECT/'reports'/attribute_name),
                                        '--predictions',predictions,'--coverage',coverage,'--toolkit',toolkit,'--output',str(directory/'scores')])
    receipt['completed_datasets'].append(dataset)
    save()
workspace=str(root/'DepthTrack')
run('DepthTrack-workspace','prepare_vot_workspace.py',common+['--dataset-audit',str(PROJECT/'reports/depthtrack_vot_dataset_audit.json'),'--workspace',workspace])
shards('DepthTrack','run_vot_sequences.py',['--workspace',workspace])
run('DepthTrack-official-scoring','audit_and_score_vot.py',['--workspace',workspace])
run('DepthTrack-attributes','score_depthtrack_attributes.py',['--workspace',workspace,'--attributes',str(PROJECT/'reports/depthtrack_test_attributes.json')])
receipt['completed_datasets'].append('DepthTrack')
receipt['status']='ALL_THREE_SCORED_REQUIRES_RESULT_AUDIT'
save()
print(json.dumps(receipt))
