"""Record the completed setup using existing receipts; no model execution."""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')


def main():
    preflight = json.loads((PROJECT / 'reports/preflight_xtrack.json').read_text())
    assert preflight['status'] == 'PASS' and preflight['optimizer_steps'] == 1
    hf = json.loads((PROJECT / 'reports/author_hf_files.json').read_text())
    author = next(row for row in hf if row['path'] == 'OSTrack_ep0300.pth.tar')
    assert author['lfs']['oid'] == preflight['checkpoint_sha256']
    path = PROJECT / 'reports/source_configuration.json'
    configuration = json.loads(path.read_text())
    configuration['upstream_entry_fix'] = 'Use existing XTrackProcessing for validation; BATProcessing is absent.'
    configuration['lowdisk_configuration'] = 'rgbx_b_adamw_lowdisk'
    configuration['lowdisk_changes'] = {'SAVE_EPOCH_INTERVAL': 65, 'SAVE_LAST_N_EPOCH': 1, 'TEST.EPOCH': 65}
    configuration['checkpoint_matches_author_lfs'] = True
    path.write_text(json.dumps(configuration, indent=2) + '\n')
    receipt = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'project': str(PROJECT), 'conda_environment': '/data/gb/conda/envs/rgbx-risk',
        'xtrack_commit': configuration['author_commit'],
        'sutrack_commit': subprocess.check_output(['git', '-C', str(PROJECT / 'third_party/SUTrack'), 'rev-parse', 'HEAD'], text=True).strip(),
        'preflight_status': 'PASS', 'formal_training_launched': False,
        'remaining_data_disk_bytes': shutil.disk_usage(PROJECT).free,
        'local_changes_sha256': hashlib.sha256((PROJECT / 'reports/XTrack_local_changes.patch').read_bytes()).hexdigest(),
        'deferred_evaluation': ['RGBT234', 'VOT-RGBD2022'],
    }
    (PROJECT / 'reports/setup_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
