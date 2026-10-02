"""Run the author trainer for two real three-GPU updates and verify synchronization."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist

from lib.train.actors import XTrackActor
from lib.train.run_training import main
from lib.train.trainers import BaseTrainer

PROJECT = Path(__file__).resolve().parents[1]
original_train = BaseTrainer.train
original_call = XTrackActor.__call__
task_counts = Counter()
losses = []


def checked_call(self, data):
    loss, stats = original_call(self, data)
    assert torch.isfinite(loss), stats
    assert all(math.isfinite(float(value)) for value in stats.values()), stats
    if torch.is_grad_enabled():
        task_counts.update(data['dataset'])
        losses.append(float(loss.detach()))
    return loss, stats


def checked_train(self, max_epochs, **kwargs):
    assert max_epochs == 1 and len(self.loaders[0]) == 2
    net = self.actor.net.module
    before = {name: p.detach().clone() for name, p in net.named_parameters() if p.requires_grad}
    frozen_versions = {name: p._version for name, p in net.named_parameters() if not p.requires_grad}
    original_step = self.optimizer.step
    step_count = 0

    def checked_step(*args, **step_kwargs):
        nonlocal step_count
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        result = original_step(*args, **step_kwargs)
        step_count += 1
        return result

    self.optimizer.step = checked_step
    original_train(self, max_epochs, **kwargs)
    assert self.epoch == 1 and step_count == 2
    changed = [name for name, p in net.named_parameters() if p.requires_grad and not torch.equal(before[name], p)]
    assert changed
    assert all(p._version == frozen_versions[name] for name, p in net.named_parameters() if not p.requires_grad)
    digest = hashlib.sha256()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p).all(), name
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
    row = {'rank': dist.get_rank(), 'optimizer_steps': step_count, 'tasks': dict(task_counts),
           'losses': losses, 'changed_tensors': len(changed), 'trainable_sha256': digest.hexdigest(),
           'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
           'frozen_parameters_unchanged': True}
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, row)
    assert len(rows) == 3
    assert len({r['trainable_sha256'] for r in rows}) == 1, rows
    observed = Counter()
    for rank_row in rows:
        observed.update(rank_row['tasks'])
    assert set(observed) == {'lasher', 'depthtrack', 'visevent'}, observed
    if dist.get_rank() == 0:
        result = {'status': 'PASS', 'formal_training': False, 'world_size': 3,
                  'global_batch_size': 24, 'total_training_examples': sum(observed.values()),
                  'task_counts': dict(observed), 'synchronized_trainable_parameters': True, 'ranks': rows}
        (PROJECT / 'reports/preflight_xtrack_ddp.json').write_text(json.dumps(result, indent=2) + '\n')
        print('DDP_PREFLIGHT_PASS', json.dumps(result), flush=True)


XTrackActor.__call__ = checked_call
BaseTrainer.train = checked_train

if __name__ == '__main__':
    main()
