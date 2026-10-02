#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
ENVIRONMENT=$(dirname "$PROJECT")/conda/envs/rgbx-risk
CONFIG=${1:?config name required}
SAVE_DIR=${2:?save directory required}
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="$PROJECT/third_party/XTrack:$PROJECT/third_party/XTrack/lib/train"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$PROJECT/third_party/XTrack"
ENTRY=lib/train/run_training.py
if [ "$CONFIG" = rgbx_b_adamw_3gpu_smoke ]; then
  ENTRY="$PROJECT/scripts/preflight_xtrack_ddp.py"
fi
exec "$ENVIRONMENT/bin/python" -m torch.distributed.launch \
  --nproc_per_node=3 --master_port=29528 "$ENTRY" \
  --script xtrack --config "$CONFIG" --save_dir "$SAVE_DIR" --use_wandb 0 --seed 2026
