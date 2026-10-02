#!/usr/bin/env bash
set -euo pipefail
source /data/liangds/anaconda3/etc/profile.d/conda.sh
conda activate /data/gb/conda/envs/rgbx-risk
export PYTHONPATH=/data/gb/rgbx-risk/third_party/XTrack
cd /data/gb/rgbx-risk/third_party/XTrack
exec python lib/train/run_training.py \
  --script xtrack --config rgbx_b_adamw_lowdisk \
  --save_dir /data/gb/rgbx-risk/outputs/author_adamw_lowdisk --use_wandb 0 --seed 2026
