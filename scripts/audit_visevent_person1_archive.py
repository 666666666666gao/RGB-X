"""Check the disputed sequence against the authors' original split-ZIP metadata."""
import hashlib
import json
import zlib
from pathlib import Path

NAME = 'dvSave-2021_02_06_09_23_50_person1'
ROOT = Path('/data/wangwj/dataset/VisEvent/test_subset')/NAME
CACHE = Path('/home/gaob/.cache')
entries_path = CACHE/'rgbx_vis_person1_archive_entries.json'
entries = json.loads(entries_path.read_text())
rows = []
for entry in entries:
    relative = Path(entry['name']).relative_to(Path('test_subset')/NAME)
    if relative.suffix != '.bmp':
        continue
    path = ROOT/relative
    assert path.is_file()
    contents = path.read_bytes()
    assert len(contents) == entry['size'] and zlib.crc32(contents) == entry['crc32'], str(relative)
    rows.append(dict(file=str(relative),bytes=len(contents),zip_crc32=entry['crc32'],
                     local_sha256=hashlib.sha256(contents).hexdigest()))
assert len(rows) == 664
assert {row['file'] for row in rows} == {str(p.relative_to(ROOT)) for channel in ['vis_imgs','event_imgs'] for p in (ROOT/channel).glob('*.bmp')}
annotations = {}
for name in ['groundtruth.txt','absent_label.txt']:
    official = CACHE/('rgbx_vis_person1_official_archive_'+name)
    local = ROOT/name
    assert official.read_bytes() == local.read_bytes()
    annotations[name] = dict(rows=len(official.read_text().splitlines()),
                             sha256=hashlib.sha256(official.read_bytes()).hexdigest())
    assert annotations[name]['rows'] == 332
report = dict(status='AUTHOR_ARCHIVE_SEQUENCE_MATCH',role='dataset_provenance_check_not_tracking_benchmark',
              sequence=NAME,author_archive='VisEvent_test.z02',archive_disk=1,
              source_folder='https://www.dropbox.com/scl/fo/r406wsgll56fy0hhhwu62/AFo3cjXjSI4Dzjn5nlnXNW0?rlkey=ecgyd26j1ycfl1jbm4pwc3vbn&dl=0',
              source_metadata='ZIP64 central directory from final VisEvent_test.zip via HTTP byte ranges',
              metadata_sha256=hashlib.sha256(entries_path.read_bytes()).hexdigest(),
              raw_gt_download_range=[263091735,263096931],raw_absence_download_range=[149689476,149690499],
              rgb_event_pairs=332,validated_image_files=664,annotations=annotations,images=rows,
              current_repository_annos_zip_rows=306,
              repository_annos_zip_mapping_status='UNRESOLVED_OFFICIAL_MATERIAL_VERSION_CONFLICT',
              permitted_comparison='same_author_archive_annotation_version_for_baseline_and_method',
              published_score_comparability='not_yet_verified_against_repository_annos_zip_version')
Path('/data/gb/rgbx-risk/reports/visevent_person1_author_archive_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key!='images'}))
