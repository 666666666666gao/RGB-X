"""Keep one completed recovery checkpoint without changing the live trainer."""
import hashlib
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/data/gb/rgbx-risk')
ROOT = Path('/home/gaob/rgbx-full-method/xtrack_full_method_v2_distance0_s2026')
AUDIT = PROJECT/'reports/full_method_v2_distance0_epoch1_checkpoint_audit.json'
OUTPUT = PROJECT/'reports/full_method_v2_distance0_checkpoint_retention.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def describe(path):
    assert path.resolve(strict=True) == path and ROOT in path.parents
    metadata = path.lstat()
    assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
    return {'path':str(path),'bytes':metadata.st_size,'sha256':sha(path)}


def main():
    assert not OUTPUT.exists()
    assert ROOT.resolve(strict=True) == ROOT
    audit = json.loads(AUDIT.read_text())
    assert audit['status'] == 'PASS_FIRST_EPOCH_CHECKPOINT_NOT_FINAL_OR_BENCHMARK'
    assert audit['run_id'] == ROOT.name and audit['epoch'] == 1 and audit['global_step'] == 2500
    report = {'status':'ACTIVE_KEEP_ONE_COMPLETED_RECOVERY_CHECKPOINT',
              'retention_worker_pid':os.getpid(),
              'started_at':datetime.now(timezone.utc).isoformat(),'run_id':ROOT.name,'poll_seconds':240,
              'policy':'During active training retain only the most recently completed named checkpoint. It includes model, optimizer and three-rank RNG. Retire older completed epochs and the byte-identical latest alias.',
              'best_by_benchmark':'NOT_DETERMINED_NO_FORMAL_SCORES',
              'after_training':'Retain the final checkpoint until complete same-protocol evaluation and diagnosis; then preserve only the best measured method checkpoint across completed candidates.',
              'reference_and_initialization_inputs':'Outside this cleanup scope and retained.',
              'deleted':[],'removed_file_bytes':0,'handled_completed_epoch':0,
              'training_parameters_modified':False,'script_sha256':sha(__file__)}
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    while True:
        run = json.loads((ROOT/'run.json').read_text())
        assert run['run_id'] == ROOT.name
        completed = run['completed_epochs']
        if completed > report['handled_completed_epoch']:
            keep = ROOT/('XTrack_ep%04d.pth.tar' % completed)
            assert Path(run['last_checkpoint']) == keep
            current = describe(keep)
            targets = []
            for path in sorted(ROOT.glob('XTrack_ep????.pth.tar')):
                epoch = int(path.name[9:13])
                if epoch < completed:
                    targets.append(describe(path))
            latest = ROOT/'latest.pth.tar'
            alias = describe(latest)
            # The trainer writes the next alias before updating completed_epochs.
            # A different alias can therefore be a newer save in progress.
            if alias['sha256'] == current['sha256']:
                targets.append(alias)
            for item in targets:
                Path(item['path']).unlink()
                report['deleted'].append(item)
                report['removed_file_bytes'] += item['bytes']
                OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
            report.update(handled_completed_epoch=completed,retained_checkpoint=current,
                          alias_retired=alias['sha256'] == current['sha256'],
                          observed_at=datetime.now(timezone.utc).isoformat())
            OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'completed_epoch':completed,'retained':current['path'],
                              'removed_files':len(targets),'cumulative_removed_file_bytes':report['removed_file_bytes']}),flush=True)
        if run['status'] == 'TRAINING_COMPLETE_REQUIRES_ENDPOINT_AND_RESULT_AUDIT':
            report.update(status='FINAL_CHECKPOINT_RETAINED_AWAITING_COMPLETE_METRICS',
                          ended_at=datetime.now(timezone.utc).isoformat())
            OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
            break
        time.sleep(240)


if __name__ == '__main__':
    main()
