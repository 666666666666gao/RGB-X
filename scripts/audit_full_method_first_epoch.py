"""Read-only CPU verification of the first saved full-method epoch."""
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

project = Path('/data/gb/rgbx-risk')
sys.path[:0] = [str(project), str(project/'third_party/XTrack'), str(project/'third_party/XTrack/lib/train')]
from lib.config.xtrack.config import cfg, update_config_from_file
from lib.models.xtrack import build_xtrack
from rgbx_method.optimizer import TaskAdaptiveAdamW


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


root = Path('/home/gaob/rgbx-full-method/xtrack_full_method_v1_s2026')
checkpoint_path = root/'XTrack_ep0001.pth.tar'
protocol_path = project/'configs/full_method_v1.json'
assert sha(protocol_path) == '452d8ea219af641f7a7b237eb654df9c1dfb88c551ea98ee43b600927ac6dded'
protocol = json.loads(protocol_path.read_text())
saved = torch.load(checkpoint_path, map_location='cpu')
assert saved['epoch'] == 1 and saved['global_step'] == 2500
assert saved['protocol'] == protocol
assert len(saved['epoch_history']) == 1
assert saved['epoch_history'][0]['epoch'] == 1 and saved['epoch_history'][0]['steps'] == 2500
assert all(math.isfinite(value) for value in saved['epoch_history'][0]['means'].values())
update_config_from_file(str(project/protocol['base_config']))
model = build_xtrack(cfg, training=False)
model.load_state_dict(saved['net'], strict=True)
assert all(torch.isfinite(value).all().item() for value in saved['net'].values())
initial_path = Path(protocol['initial_checkpoint'])
assert sha(initial_path) == '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
initial = torch.load(initial_path, map_location='cpu')['net']
frozen = [name for name in saved['net'] if 'MeME' not in name]
assert len(frozen) == 244
assert all(torch.equal(saved['net'][name], initial[name]) for name in frozen)
for name, parameter in model.named_parameters():
    parameter.requires_grad_('MeME' in name)
optimizer = TaskAdaptiveAdamW(model.named_parameters(), protocol['lr'], protocol['weight_decay'], protocol['grad_clip_norm'])
expected_shapes = [tuple(item['m'].shape) for item in optimizer.state]
optimizer.load_state_dict(saved['optimizer'])
assert optimizer.lr == protocol['lr']
for state, shape in zip(optimizer.state, expected_shapes):
    assert tuple(state['m'].shape) == tuple(state['v'].shape) == shape
    assert len(state['steps']) == shape[0]
    assert all(0 <= step <= 2500 for step in state['steps'])
    assert torch.isfinite(state['m']).all().item() and torch.isfinite(state['v']).all().item()
assert len(optimizer.state) == len(expected_shapes)
witnesses = [i for i, name in enumerate(optimizer.names) if name.endswith(('MeME_attn.ffn1_1.weight','MeME_attn.ffn1_1.bias'))]
assert len(witnesses) == 24
assert all(optimizer.state[i]['steps'] == [2500,2500,2500] for i in witnesses)
replicas = saved['replicas_identical']
assert len(replicas) == 3 and len(set(replicas)) == 1
assert hashlib.sha256(optimizer.values().numpy().tobytes()).hexdigest() == replicas[0]
assert len(saved['rng_by_rank']) == 3
rng_records = []
for rank, rng in enumerate(saved['rng_by_rank']):
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    assert rng['torch_cuda'].dtype == torch.uint8 and rng['torch_cuda'].ndim == 1 and rng['torch_cuda'].numel() > 0
    rng_records.append({'rank':rank,'python_numpy_cpu_rng_restore_valid':True,
                        'cuda_rng_bytes':rng['torch_cuda'].numel(),
                        'cuda_rng_sha256':hashlib.sha256(rng['torch_cuda'].numpy().tobytes()).hexdigest()})
run = json.loads((root/'run.json').read_text())
controller = json.loads((root.parent/'xtrack_full_method_v1_s2026_controller'/'run.json').read_text())
pids = [controller['controller_pid'],controller['worker_pid'],*run['rank_pids']]
alive = {str(pid):Path('/proc',str(pid)).exists() for pid in pids}
assert all(alive.values())
report = {'status':'PASS_FIRST_EPOCH_CHECKPOINT_NOT_FINAL_OR_BENCHMARK',
          'observed_at':datetime.now(timezone.utc).isoformat(),'run_id':protocol['run_id'],
          'checkpoint':str(checkpoint_path),'checkpoint_bytes':checkpoint_path.stat().st_size,
          'checkpoint_sha256':sha(checkpoint_path),'epoch':1,'global_step':2500,
          'strict_model_reload':True,'all_model_tensors_finite':True,
          'frozen_tensor_count':len(frozen),'frozen_tensors_unchanged':True,
          'optimizer_reload_names_shapes_and_moments_valid':True,'always_used_three_task_step_witnesses':24,
          'saved_replica_hashes_identical_and_match_loaded_parameters':True,
          'rng_by_rank':rng_records,'cuda_rng_restore_executed':False,
          'latest_training_epoch':run['epoch'],'latest_training_step':run['global_step'],
          'current_training_pids_alive':alive,'epoch1_history':saved['epoch_history'],
          'training_mutations':0,'formal_tracking_scores':'NOT_RUN',
          'script_sha256':sha(__file__),
          'scope':'CPU audit of epoch1 saved state. RNG restore uses this isolated CPU audit process; no live trainer state or GPU is changed. CUDA RNG byte states are present but GPU restoration was not executed. Fifteen-epoch endpoint and complete benchmark comparison remain pending.'}
output = project/'reports/full_method_v1_epoch1_checkpoint_audit.json'
assert not output.exists()
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
