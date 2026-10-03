"""Suspend this already-running job until its requested GPU limits are met."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PROJECT = Path('/data/gb/rgbx-risk')
ROOT = Path('/home/gaob/rgbx-full-method/xtrack_full_method_v1_s2026')
RECEIPT = PROJECT / 'reports/full_method_v1_gpu_guard.json'
PIDS = [1603676, 1603680, 1603687, 1603688, 1603689]
EXPECTED_COMMANDS = {
    1603676: '/data/gb/conda/envs/rgbx-risk/bin/python scripts/run_full_method_job.py ',
    1603680: '/data/gb/conda/envs/rgbx-risk/bin/python -m torch.distributed.run --nproc_per_node=3 --master_port=29529 /data/gb/rgbx-risk/scripts/train_full_method.py --protocol /data/gb/rgbx-risk/configs/full_method_v1.json ',
    **{pid: '/data/gb/conda/envs/rgbx-risk/bin/python -u /data/gb/rgbx-risk/scripts/train_full_method.py --protocol /data/gb/rgbx-risk/configs/full_method_v1.json ' for pid in [1603687, 1603688, 1603689]},
}
UUIDS = ['GPU-3a80b2a4-4eec-b17f-5a41-88df48bd673b', 'GPU-f5c818cb-d7b0-bcd0-5544-2d172079bc40', 'GPU-e087b67f-9495-4385-6bde-335c5760d96e']


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def identity(pid):
    proc = Path('/proc') / str(pid)
    assert proc.stat().st_uid == os.getuid()
    assert proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode() == EXPECTED_COMMANDS[pid]
    return proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]


identities = {pid: identity(pid) for pid in PIDS}
state = {'started_at': now(), 'guard_pid': os.getpid(), 'training_pids': PIDS,
         'poll_seconds': 30, 'pause_temperature_c': 75, 'resume_temperature_c': 70,
         'required_power_limit_w': 250, 'events': [], 'paused_seconds': 0.0,
         'scope': 'Only five existing owned training/controller processes; no trainer, optimizer, checkpoint or hardware setting mutation.'}
paused = False
pause_started = None
while True:
    if not all(Path('/proc', str(pid)).exists() for pid in PIDS):
        state['status'] = 'TRAINING_PROCESS_ENDED_INSPECT_CONTROLLER_STAGE'
        state['observed_at'] = now()
        RECEIPT.write_text(json.dumps(state, indent=2) + '\n')
        break
    assert all(identity(pid) == identities[pid] for pid in PIDS)
    raw = subprocess.check_output(['nvidia-smi', '-i', '0,1,2', '--query-gpu=uuid,temperature.gpu,power.draw,power.limit', '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(',') for line in raw.strip().splitlines()]
    assert [row[0].strip() for row in rows] == UUIDS
    readings = [{'gpu': i, 'temperature_c': int(row[1]), 'power_w': float(row[2]), 'power_limit_w': float(row[3])} for i, row in enumerate(rows)]
    caps_met = all(row['power_limit_w'] <= 250 for row in readings)
    maximum = max(row['temperature_c'] for row in readings)
    if not paused and (not caps_met or maximum >= 75):
        for pid in PIDS:
            os.kill(pid, signal.SIGSTOP)
        paused = True
        pause_started = time.monotonic()
        event = {'at': now(), 'action': 'SIGSTOP_OWNED_TRAINING', 'readings': readings,
                 'training_snapshot': json.loads((ROOT / 'run.json').read_text())}
        state['events'].append(event)
        print(json.dumps({'at': event['at'], 'action': event['action'], 'readings': readings}), flush=True)
    elif paused and caps_met and maximum <= 70:
        for pid in reversed(PIDS):
            os.kill(pid, signal.SIGCONT)
        state['paused_seconds'] += time.monotonic() - pause_started
        paused = False
        state['events'].append({'at': now(), 'action': 'SIGCONT_OWNED_TRAINING', 'readings': readings})
        print(json.dumps(state['events'][-1]), flush=True)
    state.update(status='PAUSED_FOR_GPU_LIMITS' if paused else 'RUNNING_GUARDED', observed_at=now(), readings=readings,
                 power_limits_verified=caps_met, current_pause_seconds=time.monotonic() - pause_started if paused else 0)
    temporary = RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(RECEIPT)
    time.sleep(30)
