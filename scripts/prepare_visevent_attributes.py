"""Bind the frozen VisEvent test set to the author's 17 challenge attributes."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--official-repository', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
manifest_path = Path(args.manifest)
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'READY' and manifest['dataset'] == 'VisEvent'
repository = Path(args.official_repository)
entry = repository/'Evaluate_VisEvent_SOT_benchmark.m'
assert hashlib.sha256(entry.read_bytes()).hexdigest() == 'c7b63621e386702aec73de3625c9f56e73ef31e713e983219efb32b793faeef5'
sequence_list = repository/'sequence_evaluation_config/VisEvent_testing_subset.txt'
names = [row['name'] for row in manifest['sequences']]
official_names = sequence_list.read_text().split()
assert len(official_names) == len(set(official_names)) == 320 and set(names) == set(official_names)
labels = ['CM','ROT','DEF','FOC','LI','OV','POC','VC','SV','BC','MB','ARC','FM','NMO','IV','OE','BOM']
archive_path = repository/'annos/annos.zip'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == 'db8a029d4a1bc77eb2054d166f191dbe35a61ba684d05591b7660d00841cfc2c'
attributes = {}
with zipfile.ZipFile(archive_path) as archive:
    for name in names:
        bits = [int(value) for value in archive.read('att/'+name+'.txt').decode().split()]
        assert len(bits) == 17 and set(bits) <= {0,1}
        attributes[name] = dict(zip(labels,bits))
report = dict(status='READY',dataset='VisEvent',sequences=320,labels=labels,
              source_repository='https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark',
              source_commit='c30908779d58404ce932abdbdb68a4dfcd07e684',
              attribute_order_source='Evaluate_VisEvent_SOT_benchmark.m:att_fig_name',
              attribute_order_source_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
              archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              annotation_version=manifest['annotation_version'],attributes=attributes)
Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key!='attributes'}))
