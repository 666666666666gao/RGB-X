"""Set local paths while retaining the author model and training recipe."""
import json
import subprocess
from pathlib import Path
import yaml

PROJECT = Path('/data/gb/rgbx-risk')
REPO = PROJECT / 'third_party/XTrack'
SOURCE = Path('/data/wangwj/dataset')


def main():
    links = {
        'lasher/trainingset': SOURCE / 'LasHeR/trainingset',
        'lasher/testingset': SOURCE / 'LasHeR/testingset',
        'depthtrack/test': SOURCE / 'DepthTrack/Dep_test',
        'visevent/train': SOURCE / 'VisEvent/train_subset',
        'visevent/test': SOURCE / 'VisEvent/test_subset',
    }
    for relative, target in links.items():
        assert target.is_dir(), target
        link = PROJECT / 'data' / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)

    depth_train = PROJECT / 'data/depthtrack/train'
    depth_train.mkdir()
    for sequence in sorted((SOURCE / 'DepthTrack/Dep_train').iterdir()):
        if sequence.is_dir():
            (depth_train / sequence.name).symlink_to(sequence, target_is_directory=True)
    (depth_train / 'toy07_indoor_320').symlink_to(SOURCE / 'DepthTrack/toy07_indoor_320', target_is_directory=True)

    checkpoint = PROJECT / 'pretrained/OSTrack_ep0300.pth.tar'
    checkpoint.symlink_to('/data/qianmj/BASE/pretrained/OSTrack_ep0300.pth.tar')
    assert checkpoint.is_file(), checkpoint

    from lib.train.admin.environment import create_default_local_file_train
    create_default_local_file_train(str(PROJECT / 'outputs'), str(PROJECT / 'data'))

    loader = REPO / 'lib/train/dataset/lasher.py'
    source = loader.read_text()
    old_rgb = "os.path.join(seq_path, 'visible', '{:06d}.jpg'.format(frame_id))"
    old_ir = "os.path.join(seq_path, 'infrared', '{:06d}.jpg'.format(frame_id))"
    assert source.count(old_rgb) == source.count(old_ir) == 1
    source = source.replace(old_rgb, "sorted(glob(os.path.join(seq_path, 'visible', '*.jpg')))[frame_id]")
    source = source.replace(old_ir, "sorted(glob(os.path.join(seq_path, 'infrared', '*.jpg')))[frame_id]")
    source = source.replace('import os\n', 'import os\nfrom glob import glob\n', 1)
    loader.write_text(source)

    functions = REPO / 'lib/train/base_functions.py'
    source = functions.read_text()
    assert source.count('processing.BATProcessing(') == 1
    functions.write_text(source.replace('processing.BATProcessing(', 'processing.XTrackProcessing('))

    cfg = yaml.safe_load((REPO / 'experiments/xtrack/rgbt.yaml').read_text())
    cfg['MODEL']['PRETRAIN_FILE'] = str(checkpoint)
    assert cfg['DATA']['TRAIN']['DATASETS_NAME'] == ['LasHeR_train', 'DepthTrack_train', 'VisEvent']
    assert cfg['DATA']['TRAIN']['DATASETS_RATIO'] == [1, 1, 1]
    for path in [PROJECT / 'configs/xtrack_b_adamw.yaml', REPO / 'experiments/xtrack/rgbx_b_adamw.yaml']:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    lowdisk = yaml.safe_load(yaml.safe_dump(cfg))
    lowdisk['TRAIN']['SAVE_EPOCH_INTERVAL'] = lowdisk['TRAIN']['EPOCH']
    lowdisk['TRAIN']['SAVE_LAST_N_EPOCH'] = 1
    lowdisk['TEST']['EPOCH'] = lowdisk['TRAIN']['EPOCH']
    for path in [PROJECT / 'configs/xtrack_b_adamw_lowdisk.yaml', REPO / 'experiments/xtrack/rgbx_b_adamw_lowdisk.yaml']:
        path.write_text(yaml.safe_dump(lowdisk, sort_keys=False))

    receipt = {
        'author_repository': 'https://github.com/supertyd/XTrack',
        'author_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'local_configuration': 'rgbx_b_adamw', 'recipe_changes': ['MODEL.PRETRAIN_FILE (local path only)'],
        'data_links': {k: str(v) for k, v in links.items()},
        'depthtrack_train_overlay': str(depth_train),
        'data_adapter_change': 'LasHeR sorted visible/infrared JPG paths; existing filenames vary by sequence.',
        'upstream_entry_fix': 'Use existing XTrackProcessing for validation; BATProcessing does not exist in this author commit.',
        'checkpoint': str(checkpoint.resolve()),
        'optimizer': 'AdamW', 'sampling': 'Author stochastic sampling with probability 1/3 per task',
        'formal_training_launched': False,
        'note': 'The controlled fixed-task-batch optimizer comparisons are pending; this is the author recipe.',
        'lowdisk_configuration': 'rgbx_b_adamw_lowdisk',
        'lowdisk_changes': {'SAVE_EPOCH_INTERVAL': 65, 'SAVE_LAST_N_EPOCH': 1, 'TEST.EPOCH': 65},
        'lowdisk_limit': 'Only final checkpoint is saved; no intermediate checkpoint for resume or model selection.',
    }
    (PROJECT / 'reports/source_configuration.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
