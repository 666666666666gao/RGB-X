"""Explicit OSTrack visual-backbone initialization, with random fusion and head."""
import hashlib
from pathlib import Path

import torch
from cad_rgbx.model import configure, build_random_model


def configure_sot(protocol):
    assert protocol['initialization'] == 'sot_backbone_only'
    assert not protocol['auto_resume'] and protocol['modules_enabled'] == []
    cfg = configure(dict(protocol, initialization='random_all_parameters', pretrained_file=''))
    cfg.DATA.SEARCH.CENTER_JITTER = protocol['search_center_jitter']
    cfg.DATA.SEARCH.SCALE_JITTER = protocol['search_scale_jitter']
    cfg.TRAIN.BACKBONE_MULTIPLIER = protocol['backbone_learning_rate_multiplier']
    return cfg


def inspect_initializer(model, path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = torch.load(str(path), map_location='cpu')
    source = checkpoint['net']
    target = model.state_dict()
    source_keys = sorted(name for name in source if name.startswith('backbone.'))
    target_keys = sorted(name for name in target if name.startswith('backbone.'))
    matched = sorted(set(source_keys) & set(target_keys))
    mismatches = [dict(name=name, source=list(source[name].shape), target=list(target[name].shape))
                  for name in matched if source[name].shape != target[name].shape]
    report = dict(source_sha256=digest, source_bytes=path.stat().st_size,
        source_checkpoint_fields=sorted(checkpoint), loaded_keys=matched,
        missing_backbone_keys=sorted(set(target_keys)-set(source_keys)),
        unused_source_backbone_keys=sorted(set(source_keys)-set(target_keys)),
        shape_mismatches=mismatches,
        excluded_non_backbone_source_keys=sorted(set(source)-set(source_keys)),
        loaded_parameter_elements=sum(target[name].numel() for name in matched),
        model_non_backbone_keys_kept_random=sorted(set(target)-set(target_keys)))
    return source, report


def build_sot_model(cfg, protocol):
    model = build_random_model(cfg)
    source, report = inspect_initializer(model, protocol['pretrained_file'])
    assert report == protocol['initializer_manifest'], report
    assert report['source_sha256'] == protocol['pretrained_sha256']
    assert not report['shape_mismatches']
    assert any(name.startswith('backbone.blocks.11.') for name in report['loaded_keys'])
    state = model.state_dict()
    state.update({name: source[name] for name in report['loaded_keys']})
    model.load_state_dict(state, strict=True)
    return model


def plateau_decision(history, settings):
    """A significant new best resets patience; test scores never enter this rule."""
    losses = [sum(row[0]+row[1] for row in epoch['task_response_box_iou'])/3
              for epoch in history]
    window = settings['rolling_epochs']
    best, best_epoch, stale = None, None, 0
    current = None
    for end in range(window, len(losses)+1):
        current = sum(losses[end-window:end])/window
        if best is None or current < best*(1-settings['relative_min_improvement']):
            best, best_epoch, stale = current, end, 0
        else:
            stale += 1
    stop = len(losses) >= settings['minimum_epochs'] and stale >= settings['patience_epochs']
    return dict(stop=stop, monitor='mean_TDE_response_plus_weighted_box',
                rolling_loss=current, significant_best_loss=best,
                significant_best_epoch=best_epoch, epochs_without_significant_improvement=stale,
                test_metrics_used=False)
