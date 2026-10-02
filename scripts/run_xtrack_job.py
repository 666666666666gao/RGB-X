"""Keep one authorized run and its 240-second local monitoring alive after SSH disconnects."""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('run_id')
args = parser.parse_args()
output = PROJECT / 'outputs' / args.run_id
output.mkdir()
log_path = PROJECT / 'logs' / (args.run_id + '.log')
receipt_path = PROJECT / 'reports' / (args.run_id + '.json')
config = 'rgbx_b_adamw_3gpu'
command = ['bash', str(PROJECT / 'scripts/train_xtrack_3gpu.sh'), config, str(output)]
started = time.time()
with log_path.open('x') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = {'run_id': args.run_id, 'server': '2028', 'project': str(PROJECT),
               'monitor_pid': os.getpid(), 'launcher_pid': process.pid,
               'start_time': datetime.datetime.now().astimezone().isoformat(),
               'command': command, 'config': config, 'seed': 2026, 'world_size': 3,
               'batch_per_rank': 8, 'global_batch': 24, 'epochs': 65,
               'samples_per_epoch': 60000, 'train_steps_per_epoch': 2500,
               'output': str(output), 'stdout_log': str(log_path), 'poll_seconds': 240,
               'author_commit': subprocess.check_output(['git', '-C', str(PROJECT / 'third_party/XTrack'),
                                                          'rev-parse', 'HEAD'], text=True).strip(),
               'config_sha256': hashlib.sha256((PROJECT / 'configs/xtrack_b_adamw_3gpu.yaml').read_bytes()).hexdigest(),
               'source_patch_sha256': hashlib.sha256((PROJECT / 'reports/XTrack_local_changes.patch').read_bytes()).hexdigest(),
               'initializer_sha256': json.loads((PROJECT / 'reports/preflight_xtrack.json').read_text())['checkpoint_sha256']}
    while True:
        exit_code = process.poll()
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True)
        gpu_processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                        '--format=csv,noheader'], capture_output=True, text=True)
        metric_path = output / 'logs' / ('xtrack-' + config + '.log')
        latest = ''
        if metric_path.is_file():
            with metric_path.open('rb') as metrics:
                metrics.seek(max(0, metric_path.stat().st_size - 65536))
                latest = metrics.read().decode().splitlines()[-1]
        receipt.update({'observed_at': datetime.datetime.now().astimezone().isoformat(),
                        'elapsed_seconds': time.time() - started, 'exit_code': exit_code,
                        'state': 'running' if exit_code is None else 'exited',
                        'gpu_snapshot': gpu.stdout.strip(), 'gpu_query_exit_code': gpu.returncode,
                        'gpu_processes': gpu_processes.stdout.strip(),
                        'latest_metric': latest,
                        'nonfinite_metric': bool(re.search(r':\s*(?:nan|[+-]?inf)\b', latest, re.I))})
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        if exit_code is not None:
            break
        time.sleep(240)

checkpoint_path = output / 'checkpoints/train/xtrack' / config / 'XTrack_ep0065.pth.tar'
receipt['final_checkpoint'] = str(checkpoint_path)
receipt['status'] = 'FAILED'
if exit_code == 0 and checkpoint_path.is_file():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    finite = all(torch.isfinite(value).all().item() for value in checkpoint['net'].values())
    losses = checkpoint['stats']['train']['Loss/total'].history
    valid = checkpoint['epoch'] == 65 and finite and len(losses) == 65 and all(math.isfinite(v) for v in losses)
    receipt.update({'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_bytes': checkpoint_path.stat().st_size,
                    'checkpoint_parameters_finite': finite, 'training_loss_epochs': len(losses),
                    'training_losses_finite': all(math.isfinite(v) for v in losses),
                    'status': 'COMPLETED' if valid else 'FAILED'})
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
