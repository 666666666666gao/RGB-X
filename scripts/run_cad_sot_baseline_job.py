"""Exactly two GPUs; caps and a successful documented sanity are admission gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cad_rgbx.hardware import require_gpu0_limits

PYTHON = '/data/gb/conda/envs/rgbx-risk/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sanity', 'formal'], required=True)
    args = parser.parse_args()
    protocol_path = PROJECT / 'configs/cad_rgbx_sot_b0_s2026.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['physical_gpus'] == [0, 1]
    hardware = require_gpu0_limits(protocol)
    cards = []
    for index in protocol['physical_gpus']:
        raw = subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=index,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
        card, memory = [int(value.strip()) for value in raw.strip().split(',')]
        assert card == index and memory < 500, raw
        cards.append(dict(index=card, memory_used_mib=memory))
    assert shutil.disk_usage('/home/gaob').free > 6 * 1024 ** 3
    run_root = Path(protocol['output'] + ('_sanity' if args.stage == 'sanity' else ''))
    assert not run_root.exists(), str(run_root)
    controller_root = Path(str(run_root) + '_controller')
    controller_root.mkdir(parents=True)
    if args.stage == 'formal':
        sanity = json.loads(Path(protocol['output'] + '_sanity/run.json').read_text())
        assert sanity['status'] == 'SANITY_COMPLETE_NOT_FORMAL_RESULT' and sanity['protocol'] == protocol
        assert sanity['global_step'] == 2
        saved = json.loads(Path(protocol['output'] + '_sanity/saved_state_audit.json').read_text())
        assert saved['status'] == 'PASS' and saved['stage'] == 'sanity'
        review = json.loads((PROJECT / 'reports/cad_rgbx_sot_documented_sanity_review_20261016.json').read_text())
        assert review['status'] == 'PASS' and review['documented_command_executed_verbatim']
        assert review['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(PROJECT))
    argv = [PYTHON,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29543',
            str(PROJECT / 'scripts/train_cad_sot_baseline.py'),'--protocol',str(protocol_path)]
    if args.stage == 'sanity':
        argv += ['--sanity-steps','2']
    receipt = dict(status='RUNNING', stage=args.stage, controller_pid=os.getpid(), argv=argv,
                   physical_gpus=[0,1], gpu0_admission=hardware, gpu_preflight=cards,
                   started_at=datetime.now(timezone.utc).isoformat(), exit_code=None)
    with (controller_root / 'train.log').open('x') as stream:
        process = subprocess.Popen(argv, cwd=str(PROJECT), env=environment, stdout=stream, stderr=subprocess.STDOUT)
        receipt['torchrun_pid'] = process.pid
        (controller_root / 'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
        # Blocking on process completion emits no repeated status polling.
        code = process.wait()
    receipt.update(status='EXITED', exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
    (controller_root / 'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
