"""Freeze all existing VisEvent sequences for a paired, explicitly versioned comparison."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path('/data/wangwj/dataset/VisEvent/test_subset')
PROJECT = Path('/data/gb/rgbx-risk')
archive_audit = PROJECT/'reports/visevent_person1_author_archive_audit.json'
assert json.loads(archive_audit.read_text())['status']=='AUTHOR_ARCHIVE_SEQUENCE_MATCH'
names = (ROOT/'list_full.txt').read_text().split()
assert len(names)==len(set(names))==320
labels_root = Path('/home/gaob/rgbx-eval-data/VisEvent_paired_local_annotations')
assert not labels_root.exists()
(labels_root/'gt_rect').mkdir(parents=True)
(labels_root/'absent').mkdir()
rows = []
for name in names:
    directory = ROOT/name
    def frames(channel):
        paths = list((directory/channel).glob('*.bmp'))
        identifiers = [(int(re.fullmatch(r'frame(\d+)',p.stem).group(1)),p) for p in paths]
        identifiers.sort()
        ids = [i for i,_ in identifiers]
        assert len(ids)==len(set(ids))
        return ids,[p for _,p in identifiers]
    rgb_ids,rgb = frames('vis_imgs')
    event_ids,event = frames('event_imgs')
    assert rgb_ids==event_ids and rgb
    gt_path = directory/'groundtruth.txt'
    gt = np.loadtxt(gt_path,delimiter=',',ndmin=2)
    raw_visibility = np.loadtxt(directory/'absent_label.txt',ndmin=1)
    assert set(raw_visibility)<= {0,1}
    first_visible = int(raw_visibility.argmax())
    visibility = raw_visibility[first_visible:]
    assert visibility[0]==1 and len(visibility)==len(rgb)==len(gt)
    assert gt.shape==(len(rgb),4) and np.isfinite(gt).all() and (gt[0,2:]>0).all()
    gt_link = labels_root/'gt_rect'/(name+'.txt')
    gt_link.symlink_to(gt_path)
    absence_path = labels_root/'absent'/(name+'.txt')
    np.savetxt(absence_path,1-visibility,fmt='%d')
    rows.append(dict(name=name,frame_count=len(rgb),groundtruth=str(gt_path),
                     groundtruth_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                     raw_visibility_sha256=hashlib.sha256((directory/'absent_label.txt').read_bytes()).hexdigest(),
                     raw_visibility_start=first_visible,absence=str(absence_path),
                     absence_sha256=hashlib.sha256(absence_path.read_bytes()).hexdigest(),
                     frames=[[str(r),str(e)] for r,e in zip(rgb,event)]))
report = dict(status='READY',dataset='VisEvent',expected_sequences=320,
              expected_frame_pairs=sum(row['frame_count'] for row in rows),
              annotation_version='existing_dataset_GT_author_archive_verified_person1',
              annotation_root=str(labels_root),repository_annos_zip_equivalence=False,
              comparison='same_frozen_data_and_annotations_for_baseline_and_method',
              published_metric_comparability='not_verified_against_annos_zip_306_row_person1_version',
              sequence_list_sha256=hashlib.sha256((ROOT/'list_full.txt').read_bytes()).hexdigest(),
              person1_archive_audit_sha256=hashlib.sha256(archive_audit.read_bytes()).hexdigest(),
              sequences=rows)
assert report['expected_frame_pairs']==157421
(PROJECT/'reports/visevent_eval_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key!='sequences'}))
