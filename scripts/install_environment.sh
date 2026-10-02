#!/usr/bin/env bash
set -euo pipefail
PROJECT=/data/gb/rgbx-risk
ENVIRONMENT=/data/gb/conda/envs/rgbx-risk
export CONDA_PKGS_DIRS=/home/gaob/.cache/rgbx-conda
export TMPDIR=/home/gaob/.cache/rgbx-install
mkdir -p "$TMPDIR"
/data/liangds/anaconda3/bin/conda create -y -p "$ENVIRONMENT" \
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main python=3.8 pip
"$ENVIRONMENT/bin/python" -m pip install --no-cache-dir \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$PROJECT/requirements.txt"
