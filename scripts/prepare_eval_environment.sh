#!/usr/bin/env bash
set -euo pipefail
PROJECT=/data/gb/rgbx-risk
EVAL_ENV=/home/gaob/conda/envs/rgbx-eval
export CONDA_PKGS_DIRS=/home/gaob/.cache/rgbx-conda
export TMPDIR=/home/gaob/.cache/rgbx-install
mkdir -p "$TMPDIR"
/data/liangds/anaconda3/bin/conda create -y --offline \
  --clone /data/gb/conda/envs/rgbx-risk -p "$EVAL_ENV"
"$EVAL_ENV/bin/python" -m pip install --no-cache-dir \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple vot-toolkit==0.7.1 numpy==1.23.5
"$EVAL_ENV/bin/python" -m pip check
"$EVAL_ENV/bin/python" -c 'import torch, vot, trax; from importlib.metadata import version; print(torch.__version__, version("vot-toolkit"), version("vot-trax"))'
"$EVAL_ENV/bin/python" -m pip freeze > "$PROJECT/reports/evaluation_environment_freeze.txt"
