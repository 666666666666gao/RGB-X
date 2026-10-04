"""Exercise the installed VOT empty-results seam without a tracker/model."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
assert version('vot-toolkit') == '0.7.1'
from vot.tracker.results import Results
from vot.workspace.storage import LocalStorage

with tempfile.TemporaryDirectory(prefix='rgbx-vot-results-directory-') as temporary:
    missing = Path(temporary) / 'tracker/experiment/sequence'
    code = 'from vot.tracker.results import Results; from vot.workspace.storage import LocalStorage; assert not Results(LocalStorage(PATH)).find("*")'
    original = subprocess.run([sys.executable, '-c', code.replace('PATH', repr(str(missing)))], capture_output=True, text=True)
    assert original.returncode != 0 and 'FileNotFoundError' in original.stderr
    missing.mkdir(parents=True, exist_ok=True)
    assert Results(LocalStorage(str(missing))).find('*') == []
    corrected = subprocess.run([sys.executable, '-c', code.replace('PATH', repr(str(missing)))], capture_output=True, text=True)
    assert corrected.returncode == 0
    (missing / 'existing_trajectory.bin').write_bytes(b'fixture-not-a-real-trajectory')
    nonempty = subprocess.run([sys.executable, '-c', code.replace('PATH', repr(str(missing)))], capture_output=True, text=True)
    assert nonempty.returncode != 0 and 'AssertionError' in nonempty.stderr
    report = dict(status='ACTUAL_INSTALLED_VOT_EMPTY_DIRECTORY_REGRESSION_PASS_NOT_TRACKING',
        observed_at=datetime.now(timezone.utc).isoformat(),vot_toolkit=version('vot-toolkit'),
        original_missing_directory_exit_code=original.returncode,original_error=original.stderr,
        corrected_empty_directory_exit_code=corrected.returncode,nonempty_cache_rejected_exit_code=nonempty.returncode,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Actual installed Results(LocalStorage).find seam: missing directory fails, explicit creation permits empty-cache check, and a pre-existing file is still rejected. Temporary fixture contains no real trajectories; no model, image, GPU, training or benchmark scoring.')
output = Path(args.output)
assert not output.exists()
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
