import datetime
import json
import pathlib
import subprocess
import sys
import time

evidence = pathlib.Path(__file__).resolve().parent
command = "cd /data/gb/rgbx-risk\nCUDA_VISIBLE_DEVICES='' PYTHONPATH=/data/gb/rgbx-risk:/data/gb/rgbx-risk/scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/gaob/conda/envs/rgbx-eval/bin/python scripts/evaluate_cad_existing_three.py --stage sanity --output-root /home/gaob/rgbx-eval-results/cad_pruned_rgbx_b0_s2026_eval_sanity_r2_20261016\n"
(evidence / 'documented_invocation.sh').write_bytes(command.encode('utf-8'))
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
started_monotonic = time.monotonic()
with (evidence / 'actual.stdout.txt').open('xb') as stdout, (evidence / 'actual.stderr.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '2028', 'bash', '-s'], input=command.encode('utf-8'), stdout=stdout, stderr=stderr)
receipt = {
    'command': command,
    'started_at_utc': started,
    'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'elapsed_seconds': time.monotonic() - started_monotonic,
    'actual_exit_code': result.returncode,
    'documented_invocation_execution_count': 1,
}
(evidence / 'execution_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2), flush=True)
sys.exit(result.returncode)
