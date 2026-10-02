"""Download the XTrack authors' VOT-RGBD2022 archive to the home partition."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from download_archive_ranges import download_ranges

ROOT = Path('/home/gaob/rgbx-eval-data')
REPORT = Path('/data/gb/rgbx-risk/reports/vot2022_download.json')
URL = 'https://huggingface.co/datasets/taryya/VOT-RGBD202/resolve/main/VOT-RGBD2022.zip'
SIZE = 20525895229
SHA256 = '013df6f54bacbf68bf1f6bd66e1fd6ddeecbfb3b11068856ed7be737a44a8226'
ARCHIVE = ROOT / 'VOT-RGBD2022.zip'
PART = ROOT / 'VOT-RGBD2022.zip.part'
EXTRACT = ROOT / 'VOT-RGBD2022'


def write_report(state, **fields):
    report.update(state=state, observed_at=datetime.now(timezone.utc).isoformat(), **fields)
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
parser.add_argument('--proxy-port', type=int, required=True)
parser.add_argument('--chunk-mib', type=int, required=True)
args = parser.parse_args()
ROOT.mkdir(parents=True, exist_ok=True)
assert not ARCHIVE.exists() and not EXTRACT.exists()
assert PART.exists() == args.resume
report = dict(url=URL, expected_bytes=SIZE, expected_sha256=SHA256,
              archive=str(ARCHIVE), extracted_path=str(EXTRACT),
              resume_from_bytes=PART.stat().st_size if args.resume else 0,
              proxy_port=args.proxy_port, low_speed_timeout_seconds=180,
              range_bytes=args.chunk_mib * 1024 * 1024,
              started_at=datetime.now(timezone.utc).isoformat())
assert args.chunk_mib > 0
download_ranges(URL, PART, SIZE, args.proxy_port, write_report, args.chunk_mib * 1024 * 1024)
actual_size = PART.stat().st_size
write_report('VERIFYING', actual_bytes=actual_size)
assert actual_size == SIZE, (actual_size, SIZE)
digest = hashlib.sha256()
with PART.open('rb') as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
        digest.update(block)
actual_sha256 = digest.hexdigest()
write_report('VERIFIED_ARCHIVE', actual_sha256=actual_sha256)
assert actual_sha256 == SHA256, actual_sha256
PART.rename(ARCHIVE)
write_report('EXTRACTING')
with zipfile.ZipFile(ARCHIVE) as archive:
    archive.extractall(EXTRACT)
    entries = len(archive.infolist())
write_report('COMPLETE', archive_entries=entries)
