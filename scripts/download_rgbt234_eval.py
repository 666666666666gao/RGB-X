"""Fetch a pinned public RGBT234 mirror; validate dataset contents separately."""
import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from download_archive_ranges import download_ranges

ROOT = Path('/home/gaob/rgbx-eval-data')
REPORT = Path('/data/gb/rgbx-risk/reports/rgbt234_download.json')
URL = 'https://huggingface.co/datasets/xche32/rgbt234/resolve/main/rgbt234.tar.gz'
SIZE = 7665568241
SHA256 = '137f45fa9860e2ce7c5f882142703c6176ffce604cfcc1cebced3364b9cbc1ee'
ARCHIVE = ROOT / 'rgbt234.tar.gz'
PART = ROOT / 'rgbt234.tar.gz.part'
EXTRACT = ROOT / 'RGBT234'


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
report = dict(url=URL, source_type='third_party_public_mirror',
              expected_bytes=SIZE, expected_sha256=SHA256,
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
EXTRACT.mkdir()
with tarfile.open(ARCHIVE) as archive:
    archive.extractall(EXTRACT)
write_report('COMPLETE')
