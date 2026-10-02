#!/usr/bin/env bash
set -euo pipefail
SCORER_ENV=/home/gaob/conda/envs/rgbx-matlab-scorer
export CONDA_PKGS_DIRS=/home/gaob/.cache/rgbx-conda
export TMPDIR=/home/gaob/.cache/rgbx-install
mkdir -p "$TMPDIR"
/data/liangds/anaconda3/bin/conda create -y --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  -p "$SCORER_ENV" octave=10.3.0
OCTAVE_HOME="$SCORER_ENV" "$SCORER_ENV/bin/octave" --no-gui --quiet --eval 'disp(version)'
/data/liangds/anaconda3/bin/conda list --explicit -p "$SCORER_ENV" \
  > /data/gb/rgbx-risk/reports/matlab_scorer_environment_explicit.txt
