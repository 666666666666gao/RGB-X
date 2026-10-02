"""CPU audit required before evaluating the formal joint-training endpoint."""
import contextlib
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from lib.config.xtrack.config import cfg, update_config_from_file
from lib.models.xtrack import build_xtrack
from lib.train.base_functions import get_optimizer_scheduler

PROJECT = Path('/data/gb/rgbx-risk')
RUN_ID = 'xtrack_b_adamw_3gpu_s2026_20261002'
CHECKPOINT = PROJECT / 'outputs' / RUN_ID / 'checkpoints/train/xtrack/rgbx_b_adamw_3gpu/XTrack_ep0065.pth.tar'
REPORT = PROJECT / 'reports/terminal_xtrack_audit.json'
receipt = json.loads((PROJECT / 'reports' / (RUN_ID + '.json')).read_text())
assert receipt['exit_code'] == 0 and receipt['status'] == 'COMPLETED'
assert receipt['run_id'] == RUN_ID and receipt['config'] == 'rgbx_b_adamw_3gpu'
assert receipt['seed'] == 2026 and receipt['world_size'] == 3
assert CHECKPOINT.is_file()
checkpoint = torch.load(CHECKPOINT, map_location='cpu')
assert checkpoint['epoch'] == 65
assert all(torch.isfinite(tensor).all().item() for tensor in checkpoint['net'].values())
histories = {}
for split, expected_epochs in [('train', 65), ('val', 13)]:
    histories[split] = {}
    for name in ['Loss/total', 'Loss/giou', 'Loss/l1', 'Loss/location', 'balance', 'classification', 'IoU']:
        values = list(checkpoint['stats'][split][name].history)
        assert len(values) == expected_epochs, (split, name, len(values))
        assert all(math.isfinite(float(value)) for value in values), (split, name)
        histories[split][name] = values
config = PROJECT / 'configs/xtrack_b_adamw_3gpu.yaml'
assert hashlib.sha256(config.read_bytes()).hexdigest() == receipt['config_sha256']
update_config_from_file(str(config))
with (PROJECT / 'logs/terminal_xtrack_cpu_model_build.log').open('w') as log, contextlib.redirect_stdout(log):
    model = build_xtrack(cfg, training=False)
    model.load_state_dict(checkpoint['net'], strict=True)
    optimizer, _ = get_optimizer_scheduler(model, cfg)
names = [name for name, value in model.named_parameters() if value.requires_grad]
assert len(names) == 504
saved_groups = checkpoint['optimizer']['param_groups']
assert len(saved_groups) == len(optimizer.param_groups) == 1
ids = saved_groups[0]['params']
assert len(ids) == len(set(ids)) == len(names)
named_ids = dict(zip(names, ids))
witnesses = ['backbone.blocks.{}.MeME_attn.ffn1_1.{}'.format(block, suffix)
             for block in range(12) for suffix in ['weight', 'bias']]
witness_steps = {}
for name in witnesses:
    state = checkpoint['optimizer']['state'][named_ids[name]]
    step = int(state['step'])
    assert step == 162500, (name, step)
    assert torch.isfinite(state['exp_avg']).all().item()
    assert torch.isfinite(state['exp_avg_sq']).all().item()
    witness_steps[name] = step
initializer = PROJECT / 'pretrained/OSTrack_ep0300.pth.tar'
initializer_hash = hashlib.sha256(initializer.read_bytes()).hexdigest()
assert initializer_hash == receipt['initializer_sha256']
initial_state = torch.load(initializer, map_location='cpu')['net']
assert len(initial_state) == 242
assert all(torch.equal(checkpoint['net'][name], tensor) for name, tensor in initial_state.items())
digest = hashlib.sha256()
with CHECKPOINT.open('rb') as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
        digest.update(block)
report = dict(status='PASS', role='formal_terminal_checkpoint_audit_not_tracking_benchmark',
              audited_at=datetime.now(timezone.utc).isoformat(), run_id=RUN_ID,
              checkpoint=str(CHECKPOINT), checkpoint_sha256=digest.hexdigest(),
              checkpoint_bytes=CHECKPOINT.stat().st_size, epoch=65,
              config_sha256=receipt['config_sha256'], seed=2026,
              launcher_exit_code=0, strict_cpu_model_load=True, model_tensors_finite=True,
              train_epochs=65, validation_epochs=13, metric_histories=histories,
              trainable_tensors=504, always_used_optimizer_witness_steps=witness_steps,
              expected_optimizer_updates=162500, unchanged_initializer_tensors=242,
              initializer_sha256=initializer_hash)
REPORT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'metric_histories'}), flush=True)
