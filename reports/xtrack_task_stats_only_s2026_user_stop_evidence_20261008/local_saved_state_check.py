"""Verify the completed saved D run on CPU, then end only its held automatic controller."""
import json
from pathlib import Path
import subprocess

project = Path('C:/Users/gb/projects/rgbx-risk')
private = Path('C:/Users/gb/.codex_tmp')
report_name = 'reports/xtrack_task_stats_only_s2026_user_stop_saved_state_audit_20261008.json'
assert not (project / report_name).exists()
code = r'''
import datetime,hashlib,json,math,os,pathlib,signal,subprocess,sys,time
project=pathlib.Path('/data/gb/rgbx-risk')
sys.path[:0]=[str(project),str(project/'third_party/XTrack'),str(project/'third_party/XTrack/lib/train')]
import torch
from lib.config.xtrack.config import cfg,update_config_from_file
from lib.models.xtrack import build_xtrack
from rgbx_controls.checkpoint import verify_optimizer_state,verify_rng_states
def sha(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
root=pathlib.Path('/home/gaob/rgbx-full-method/xtrack_task_stats_only_s2026')
controller_root=root.with_name(root.name+'_controller')
run=json.loads((root/'run.json').read_text())
controller=json.loads((controller_root/'run.json').read_text())
assert run['status']=='TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT'
assert run['run_id']=='xtrack_task_stats_only_s2026' and run['completed_epochs']==15 and run['global_step']==37500
assert controller['controller_pid']==3766982 and controller['worker_pid']==3767016 and controller['completed_stages']==[]
assert pathlib.Path('/proc/3766982/stat').read_text().rsplit(')',1)[1].split()[0]=='T'
worker_stat=pathlib.Path('/proc/3767016/stat').read_text()
worker_fields=worker_stat.rsplit(')',1)[1].split()
assert worker_fields[0]=='Z' and int(worker_fields[49])==0
assert all(not pathlib.Path('/proc',str(pid)).exists() for pid in run['rank_pids'])
assert not (controller_root/'evaluation.log').exists()
protocol_path=project/'configs/xtrack_task_stats_only_s2026.json'
protocol=json.loads(protocol_path.read_text())
assert run['protocol']==protocol and run['protocol_sha256']==sha(protocol_path)=='7156203dd350c70484e6e60c3f187d2d78401cfd23abfb3086c633772b757381'
checkpoint=root/'XTrack_ep0015.pth.tar'
assert pathlib.Path(run['last_checkpoint'])==checkpoint
saved=torch.load(checkpoint,map_location='cpu')
assert saved['epoch']==15 and saved['global_step']==37500 and saved['sanity_steps']==0 and saved['protocol']==protocol
assert [row['epoch'] for row in saved['epoch_history']]==list(range(1,16))
assert all(row['steps']==2500 and all(math.isfinite(value) for value in row['means'].values()) for row in saved['epoch_history'])
update_config_from_file(str(project/protocol['base_config']))
model=build_xtrack(cfg,training=False)
model.load_state_dict(saved['net'],strict=True)
assert all(torch.isfinite(value).all().item() for value in saved['net'].values())
initial_path=pathlib.Path(protocol['initial_checkpoint'])
assert sha(initial_path)==run['initial_checkpoint_sha256']=='3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
initial=torch.load(initial_path,map_location='cpu')['net']
frozen=[name for name in saved['net'] if 'MeME' not in name]
assert len(frozen)==244 and all(torch.equal(saved['net'][name],initial[name]) for name in frozen)
for name,parameter in model.named_parameters():
    parameter.requires_grad_('MeME' in name)
named=[(name,parameter) for name,parameter in model.named_parameters() if parameter.requires_grad]
assert len(named)==504 and sum(parameter.numel() for _,parameter in named)==5821440
optimizer_check=verify_optimizer_state(named,saved,protocol)
assert optimizer_check['always_used_witness_tensors']==24
digest=hashlib.sha256(torch.cat([parameter.detach().flatten() for _,parameter in named]).numpy().tobytes()).hexdigest()
assert saved['replicas_identical']==[digest]*3
rng_check=verify_rng_states(saved['rng_by_rank'])
assert not torch.cuda.is_initialized()
rows=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
assert len(rows)==37500 and [row['global_step'] for row in rows]==list(range(1,37501))
assert all(row['global_task_counts']==[8,8,8] and row['main_training_examples']==24 for row in rows)
assert all(row['optimizer_statistics_updates']==row['weight_decay_applications']==1 for row in rows)
assert all(row['localization_check_calls']==row['correction_calls']==row['trajectory_calls']==row['pair_check_forward_examples']==row['trajectory_model_forward_examples']==0 for row in rows)
assert all(math.isfinite(row['step_seconds']) and math.isfinite(row['joint_grad_norm_before_clip']) for row in rows)
value=dict(status='PASS_SAVED_D15_STATE_ACTUAL_TRAIN_EXIT0_USER_STOP_NO_BENCHMARK',run_id=run['run_id'],
    audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_run=run,
    checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
    exact_protocol_sha256=sha(protocol_path),training_updates=37500,main_training_examples=900000,
    entire_update_prefix_checked=True,fixed_global_8T8D8E=True,extra_module_calls=0,
    update_loop_seconds=sum(row['step_seconds'] for row in rows),mean_step_seconds=sum(row['step_seconds'] for row in rows)/37500,
    strict_model_load=True,all_model_tensors_finite=True,frozen_tensors_unchanged=True,frozen_tensor_count=244,
    replicas_identical=True,loaded_trainable_replica_sha256=digest,optimizer_check=optimizer_check,rng_check=rng_check,
    actual_training_worker_pid=3767016,actual_training_worker_exit_code=0,exit_evidence='Linux proc pid stat field52 wait status0 for unreaped torchrun child while parent controller is SIGSTOP-held.',
    raw_training_worker_terminal_proc_stat=worker_stat,training_rank_processes_absent=True,
    automatic_controller_was_held=True,automatic_controller_original_receipt_status=controller['status'],
    automatic_controller_completed_stages=controller['completed_stages'],actual_controller_exit_code=None,
    automatic_controller_stop_reason='Explicit user request: finish current training then stop. Not a training failure and not a controller exit0 claim.',
    model_forward_calls=0,cuda_initialized=False,actual_cuda_restoration='NOT_EXECUTED',formal_tracking_scores='NOT_RUN_BY_EXPLICIT_USER_STOP',
    standard_automatic_endpoint_auditor='NOT_EXECUTED_CONTROLLER_HELD_THIS_IS_SEPARATE_CPU_SAVED_STATE_VERIFICATION',goal_achieved=False)
assert pathlib.Path('/proc/3766982/cmdline').read_bytes().split(b'\0')[:-1]==[b'/data/gb/conda/envs/rgbx-risk/bin/python',b'/data/gb/rgbx-risk/scripts/run_joint_optimizer_control_job.py',b'--optimizer',b'task_adaptive_only']
os.kill(3766982,signal.SIGTERM)
os.kill(3766982,signal.SIGCONT)
time.sleep(1)
assert not pathlib.Path('/proc/3766982').exists()
assert not pathlib.Path('/proc/3767016').exists()
assert not (controller_root/'evaluation.log').exists()
value.update(controller_termination_signals=['SIGTERM','SIGCONT'],automatic_controller_actual_process_absent=True,
    training_worker_actual_process_absent_after_parent_termination=True,stop_enforced_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    final_checkpoint_preserved=True,unscored_final_checkpoint_is_not_called_best=True,
    raw_file_sha256={str(path):sha(path) for path in [root/'run.json',root/'updates.jsonl',root/'parameter_manifest.json',controller_root/'run.json',controller_root/'train.log']})
output=project/'reports/xtrack_task_stats_only_s2026_user_stop_saved_state_audit_20261008.json'
assert not output.exists()
output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(value,ensure_ascii=True))
'''
raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=20','2028','/data/gb/conda/envs/rgbx-risk/bin/python','-'],input=code,text=True,encoding='utf-8')
value=json.loads(raw)
(project/report_name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({key:value[key] for key in ['status','training_updates','main_training_examples','checkpoint','checkpoint_sha256','checkpoint_bytes','actual_training_worker_exit_code','automatic_controller_actual_process_absent','stop_enforced_at','formal_tracking_scores']},ensure_ascii=True),flush=True)
