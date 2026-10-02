"""Bind all test sequences to the original 19 LasHeR challenge attributes."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--attributes-zip', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
manifest_path = Path(args.manifest)
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'READY' and manifest['dataset'] == 'LasHeR'
archive_path = Path(args.attributes_zip)
with zipfile.ZipFile(archive_path) as archive:
    labels = [name.strip() for name in archive.read('Attributes_order.txt').decode().strip().split(',')]
    assert len(labels) == len(set(labels)) == 19
    attributes = {}
    for row in manifest['sequences']:
        bits = [int(bit) for bit in archive.read('AttriSeqsTxt/'+row['name']+'.txt').decode().split(',')]
        assert len(bits) == 19 and set(bits) <= {0,1}
        attributes[row['name']] = dict(zip(labels,bits))
report = dict(status='READY',dataset='LasHeR',sequences=len(attributes),labels=labels,
              source_repository='https://github.com/BUGPLEASEOUT/LasHeR',
              source_commit='c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399',
              archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),attributes=attributes)
Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key!='attributes'}))
