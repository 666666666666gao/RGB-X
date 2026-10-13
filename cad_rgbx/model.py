"""Explicit scratch configuration; checkpoint files are never model inputs."""
import hashlib
from unittest.mock import patch
from lib.config.cadtrack.config import cfg
from lib.models.cadtrack import build_cadtrack


def configure(protocol):
    assert protocol['initialization'] == 'random_all_parameters'
    assert protocol['pretrained_file'] == '' and not protocol['auto_resume']
    cfg.MODEL.PRETRAIN_FILE = ''
    cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = True
    cfg.MODEL.BACKBONE.CROSS_LOC = [3, 6, 9]
    cfg.DATA.TEMPLATE.SIZE = protocol['template_size']
    cfg.DATA.SEARCH.SIZE = protocol['search_size']
    cfg.DATA.TEMPLATE.FACTOR = protocol['template_factor']
    cfg.DATA.SEARCH.FACTOR = protocol['search_factor']
    cfg.DATA.TEMPLATE.NUMBER = cfg.DATA.SEARCH.NUMBER = 1
    cfg.DATA.TRAIN.DATASETS_NAME = protocol['train_datasets']
    cfg.TRAIN.BATCH_SIZE = protocol['batch_per_rank']
    cfg.TRAIN.FIX_BN = False
    cfg.TRAIN.FREEZE_LAYERS = []
    cfg.TRAIN.BACKBONE_MULTIPLIER = 1.0
    return cfg


def build_random_model(cfg):
    # Source builders expose pretrained hooks; the check makes the user's requirement explicit.
    with patch('torch.load', side_effect=AssertionError('Pretrained loading is forbidden')):
        model = build_cadtrack(cfg, training=True)
    assert all(value.requires_grad for value in model.parameters())
    return model


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
