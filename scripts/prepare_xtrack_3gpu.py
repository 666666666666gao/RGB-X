"""Prepare the three-GPU author baseline and its disposable training check."""
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / 'third_party/XTrack'
cfg = yaml.safe_load((PROJECT / 'configs/xtrack_b_adamw_lowdisk.yaml').read_text())
cfg['MODEL']['PRETRAIN_FILE'] = str(PROJECT / 'pretrained/OSTrack_ep0300.pth.tar')
cfg['TRAIN']['BATCH_SIZE'] = 8
for path in [PROJECT / 'configs/xtrack_b_adamw_3gpu.yaml', REPO / 'experiments/xtrack/rgbx_b_adamw_3gpu.yaml']:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

cfg['TRAIN']['EPOCH'] = 1
cfg['TRAIN']['SAVE_EPOCH_INTERVAL'] = 1
cfg['TRAIN']['VAL_EPOCH_INTERVAL'] = 1
cfg['TRAIN']['PRINT_INTERVAL'] = 1
cfg['TEST']['EPOCH'] = 1
cfg['DATA']['TRAIN']['SAMPLE_PER_EPOCH'] = 48
cfg['DATA']['VAL']['SAMPLE_PER_EPOCH'] = 48
for path in [PROJECT / 'configs/xtrack_b_adamw_3gpu_smoke.yaml', REPO / 'experiments/xtrack/rgbx_b_adamw_3gpu_smoke.yaml']:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))

script = REPO / 'lib/train/train_script.py'
text = script.read_text()
assert text.count('trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)') == 1
script.write_text(text.replace('trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)',
                               'trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=False)'))
