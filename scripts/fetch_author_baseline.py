"""Fetch the published XTrack-B reference without retraining the baseline."""
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

directory = Path('/home/gaob/rgbx-weights')
directory.mkdir(exist_ok=True)
path = directory / 'XTrack_Base.pth.tar'
partial = path.with_suffix(path.suffix + '.part')
url = 'https://huggingface.co/taryya/XTrack/resolve/main/XTrack_Base.pth.tar'
expected_bytes = 441674839
expected_sha256 = '3aeead46ab80de95a226e9ce406a2f8b84bdeab91c4b07be188f8cb0a9902e36'
print(json.dumps({'status': 'DOWNLOADING', 'source': url, 'destination': str(path)}), flush=True)
digest = hashlib.sha256()
size = 0
with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as output:
    while True:
        block = response.read(8 * 1024 * 1024)
        if not block:
            break
        output.write(block)
        digest.update(block)
        size += len(block)
        if size % (64 * 1024 * 1024) == 0:
            print(json.dumps({'downloaded_bytes': size}), flush=True)
assert size == expected_bytes, size
assert digest.hexdigest() == expected_sha256, digest.hexdigest()
os.replace(str(partial), str(path))
report = dict(status='DOWNLOADED_CONTENT_VERIFIED_NOT_EVALUATED', source=url, checkpoint=str(path),
              checkpoint_bytes=size, checkpoint_sha256=digest.hexdigest(), verified_at=datetime.now(timezone.utc).isoformat())
Path('/data/gb/rgbx-risk/reports/author_release_checkpoint_download.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
