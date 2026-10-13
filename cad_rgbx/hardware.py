"""Readback admission and a coordinated cooling pause for physical GPU0 only."""
import subprocess
import time
import torch
import torch.distributed as dist


def gpu0_status():
    raw = subprocess.check_output(['nvidia-smi','-i','0',
        '--query-gpu=uuid,power.limit,temperature.gpu','--format=csv,noheader,nounits'], text=True)
    uuid, power, temperature = [value.strip() for value in raw.strip().split(',')]
    details = subprocess.check_output(['nvidia-smi','-i','0','-q','-d','TEMPERATURE'], text=True)
    target = [line for line in details.splitlines() if 'GPU Target Temperature' in line]
    assert len(target) == 1
    return dict(uuid=uuid, power_limit_w=float(power), temperature_c=int(temperature),
                target_temperature_c=int(target[0].split(':')[1].strip().split()[0]))


def require_gpu0_limits(protocol):
    value = gpu0_status()
    assert value['uuid'] == protocol['gpu0_uuid'], value
    assert value['power_limit_w'] <= protocol['gpu0_power_limit_w'], value
    assert value['target_temperature_c'] <= protocol['gpu0_target_temperature_c'], value
    return value


class CoolingGuard:
    def __init__(self, protocol, rank, device):
        self.protocol, self.rank, self.device = protocol, rank, device
        self.next_check = 0.0

    def pause(self):
        protocol, rank, device = self.protocol, self.rank, self.device
        value = None
        if rank == 0 and time.perf_counter() >= self.next_check:
            raw = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=temperature.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            value = dict(temperature_c=int(raw.strip()))
            self.next_check = time.perf_counter() + protocol['thermal_cooldown_check_seconds']
        flag = torch.tensor(int(value is not None and value['temperature_c'] >= protocol['gpu0_pause_temperature_c']),
                            device=device)
        dist.broadcast(flag, src=0)
        if not flag.item():
            return 0.0
        started = time.perf_counter()
        if rank == 0:
            print('GPU0_COOLING_PAUSE ' + str(value), flush=True)
        while flag.item():
            time.sleep(protocol['thermal_cooldown_check_seconds'])
            if rank == 0:
                value = require_gpu0_limits(protocol)
                flag.fill_(int(value['temperature_c'] > protocol['gpu0_resume_temperature_c']))
            dist.broadcast(flag, src=0)
        seconds = time.perf_counter() - started
        if rank == 0:
            print('GPU0_COOLING_RESUMED seconds=' + str(seconds), flush=True)
        return seconds
