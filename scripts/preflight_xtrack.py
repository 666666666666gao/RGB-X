"""One disposable joint update from actual T/D/E batches; no formal checkpoint."""
import contextlib
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss

from lib.config.xtrack.config import cfg, update_config_from_file
from lib.models.xtrack import build_xtrack
from lib.train.actors import XTrackActor
from lib.train.admin.settings import Settings
from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler, update_settings
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss

PROJECT = Path('/data/gb/rgbx-risk')


def main():
    started = time.monotonic()
    cv2.setNumThreads(0)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    update_config_from_file(str(PROJECT / 'configs/xtrack_b_adamw.yaml'))
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.NUM_WORKER = 0
    cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 2
    cfg.DATA.VAL.DATASETS_NAME = [None]

    with (PROJECT / 'logs/preflight-build.log').open('w') as log, contextlib.redirect_stdout(log):
        net = build_xtrack(cfg).cuda().train()
        optimizer, _ = get_optimizer_scheduler(net, cfg)
    checkpoint = PROJECT / 'pretrained/OSTrack_ep0300.pth.tar'
    author_state = torch.load(checkpoint, map_location='cpu')['net']
    model_state = net.state_dict()
    unexpected = sorted(set(author_state) - set(model_state))
    missing = sorted(set(model_state) - set(author_state))
    assert not unexpected, unexpected
    new_frozen_keys = ['backbone.temporal_pos_embed_x', 'backbone.temporal_pos_embed_z']
    assert [name for name in missing if 'MeME' not in name] == new_frozen_keys
    assert all(torch.equal(model_state[name].cpu(), tensor) for name, tensor in author_state.items())
    initialized_tensors = len(author_state)
    del author_state, model_state
    trainable = {name: p for name, p in net.named_parameters() if p.requires_grad}
    frozen_versions = {name: p._version for name, p in net.named_parameters() if not p.requires_grad}
    before = {name: p.detach().clone() for name, p in trainable.items()}
    accumulated = {name: torch.zeros_like(p) for name, p in trainable.items()}
    task_rows = []
    grad_names = {}
    gradient_received = set()

    for task in ['LasHeR_train', 'DepthTrack_train', 'VisEvent']:
        task_cfg = copy.deepcopy(cfg)
        task_cfg.DATA.TRAIN.DATASETS_NAME = [task]
        task_cfg.DATA.TRAIN.DATASETS_RATIO = [1]
        settings = Settings()
        settings.local_rank = -1
        settings.use_lmdb = False
        update_settings(settings, task_cfg)
        loader, _ = build_dataloaders(task_cfg, settings)
        batch = next(iter(loader)).to('cuda')
        batch['epoch'] = 1
        assert batch['template_images'].shape == (1, 2, 6, 128, 128)
        assert batch['search_images'].shape == (1, 2, 6, 256, 256)
        actor = XTrackActor(net, {'giou': giou_loss, 'l1': l1_loss, 'focal': FocalLoss(), 'cls': BCEWithLogitsLoss()},
                            {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.},
                            settings, task_cfg)
        actor.fix_bns()
        optimizer.zero_grad(set_to_none=True)
        loss, status = actor(batch)
        assert torch.isfinite(loss), task
        loss.backward()
        active = []
        for name, p in trainable.items():
            if p.grad is not None:
                gradient_received.add(name)
                assert torch.isfinite(p.grad).all(), name
                accumulated[name].add_(p.grad / 3.)
                if torch.count_nonzero(p.grad):
                    active.append(name)
        assert active, task
        grad_names[task] = active
        task_rows.append({'task': task, 'batch_datasets': list(batch['dataset']), 'losses': status,
                          'nonzero_gradient_tensors': len(active),
                          'template_shape': list(batch['template_images'].shape),
                          'search_shape': list(batch['search_images'].shape)})
        print(task, status, flush=True)

    for name, p in trainable.items():
        p.grad = accumulated[name] if name in gradient_received else None
    torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg.TRAIN.GRAD_CLIP_NORM)
    optimizer.step()
    changed = [name for name, p in trainable.items() if not torch.equal(before[name], p)]
    assert changed
    assert all(p.grad is None and p._version == frozen_versions[name]
               for name, p in net.named_parameters() if not p.requires_grad)
    assert all(torch.isfinite(p).all() for p in trainable.values())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        'status': 'PASS', 'formal_training': False, 'optimizer_steps': 1,
        'sampling': 'Two actual examples per task; averaged gradient for disposable joint AdamW update.',
        'seed': 2026, 'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'total_parameters': sum(p.numel() for p in net.parameters()),
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'trainable_tensors': len(trainable), 'changed_trainable_tensors': len(changed),
        'frozen_parameters_unchanged': True, 'tasks': task_rows,
        'gradient_support': grad_names,
        'note': 'Gradient participation does not prove task-shared semantics; parameter scope still requires source analysis.',
        'checkpoint_sha256': checkpoint_sha, 'elapsed_seconds': time.monotonic() - started,
        'author_tensors_loaded_exactly': initialized_tensors,
        'new_model_keys': missing, 'unexpected_author_keys': unexpected,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
    }
    (PROJECT / 'reports/preflight_xtrack.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: one disposable joint update; no trained checkpoint saved.', flush=True)


if __name__ == '__main__':
    main()
