import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

PROJECT = Path('C:/Users/gb/projects/rgbx-risk')
TEMP = Path('C:/Users/gb/.codex_tmp')
PREFIX = TEMP / 'rgbx_d_documented_sanity_20261007'
DOC = PROJECT / 'docs/RGB-X_研究与实验完整交接_2026-10-02.md'
DOC_SHA = '9bfd68b0b32993e621d90a996d585d779ad8015ff237e2370a462fef724e7a06'
REPORT = PROJECT / 'reports/xtrack_task_stats_only_s2026_documented_sanity_20261007.json'

assert hashlib.sha256(DOC.read_bytes()).hexdigest() == DOC_SHA
spec = json.loads((PROJECT / 'configs/env-spec.json').read_text(encoding='utf-8'))
spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
assert spec_sha[:8] == '433a922a'
prepared = json.loads((PROJECT / 'reports/joint_optimizer_controls_execution_preparation_20261005.json').read_text(encoding='utf-8'))
assert {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in prepared['source_sha256']} == prepared['source_sha256']
assert hashlib.sha256((PROJECT / 'scripts/xtrack_eval_parameters.py').read_bytes()).hexdigest() == prepared['prepared_entry_sha256']
for name in ['source', 'checkpoint_source', 'execution_source']:
    path = PROJECT / ('reports/joint_optimizer_controls_' + name + '_review_revised_20261005.json')
    review = json.loads(path.read_text(encoding='utf-8'))
    assert review['status'] == 'SOURCE_REVIEW_PASS' and review['blocking_issues'] == []
assert not REPORT.exists()

REMOTE = r'''
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone

project = Path('/data/gb/rgbx-risk')
python = '/data/gb/conda/envs/rgbx-risk/bin/python'
run_id = 'xtrack_task_stats_only_s2026'
root = Path('/home/gaob/rgbx-full-method')
log_path = root / (run_id + '_sanity_controller_launcher.log')
exit_path = root / (run_id + '_sanity_controller_launcher_exit.json')
assert not exit_path.exists()
doc = project / 'docs/RGB-X_研究与实验完整交接_2026-10-02.md'
assert hashlib.sha256(doc.read_bytes()).hexdigest() == '9bfd68b0b32993e621d90a996d585d779ad8015ff237e2370a462fef724e7a06'
spec = json.loads((project / 'configs/env-spec.json').read_text(encoding='utf-8'))
spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
assert spec_sha[:8] == '433a922a'
argv = [python, str(project / 'scripts/run_joint_optimizer_control_sanity.py'), '--optimizer', 'task_adaptive_only']
with log_path.open('xb') as log:
    process = subprocess.Popen(argv, cwd=str(project), stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
started = datetime.now(timezone.utc)
receipt = dict(status='ACTUAL_SANITY_WRAPPER_LAUNCHED', argv=argv, cwd=str(project),
               controller_pid=process.pid, linux_session_id=os.getsid(process.pid),
               parent_waiter_pid=os.getpid(), started_at=started.isoformat(),
               first_observation_not_before=(started + timedelta(seconds=300)).isoformat(),
               launcher_log=str(log_path), wrapper_exit_receipt=str(exit_path),
               documented_handoff_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),
               env_spec_canonical_sha256=spec_sha, environment_rebuilt=False,
               sanity_invocations=1, formal_training_started=False)
print(json.dumps(receipt), flush=True)
code = process.wait()
receipt.update(status='ACTUAL_SANITY_WRAPPER_EXITED', exit_code=code,
               ended_at=datetime.now(timezone.utc).isoformat())
with exit_path.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
'''

ssh_argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=20',
            '2028', '/data/gb/conda/envs/rgbx-risk/bin/python', '-u', '-']
started_wall = time.time()
deadline = started_wall + 300
print(json.dumps(dict(event='LOCAL_NATIVE_WAITER_STARTED', local_pid=os.getpid(), ssh_argv=ssh_argv,
                      env_spec_canonical_sha256=spec_sha)), flush=True)
with Path(str(PREFIX) + '.ssh.stdout.log').open('xb') as stdout, Path(str(PREFIX) + '.ssh.stderr.log').open('xb') as stderr:
    ssh = subprocess.Popen(ssh_argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
    ssh.stdin.write(REMOTE.encode('utf-8'))
    ssh.stdin.close()
    first = True
    for line in ssh.stdout:
        stdout.write(line)
        stdout.flush()
        if first:
            deadline = datetime.fromisoformat(json.loads(line.decode('utf-8'))['first_observation_not_before']).timestamp()
            print(line.decode('utf-8').rstrip(), flush=True)
            first = False
    code = ssh.wait()
remaining = deadline - time.time()
if remaining > 0:
    time.sleep(remaining)
print(json.dumps(dict(event='FIRST_OBSERVATION_NODE_REACHED', ssh_exit_code=code,
                      local_pid=os.getpid(), elapsed_seconds=time.time() - started_wall)), flush=True)
raise SystemExit(code)
