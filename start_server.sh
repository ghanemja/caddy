#!/bin/bash

source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate vlm_optimizer
export DYLD_LIBRARY_PATH=/opt/homebrew/anaconda3/envs/vlm_optimizer/lib:$DYLD_LIBRARY_PATH
CONDA_PYTHON="/opt/homebrew/anaconda3/envs/vlm_optimizer/bin/python"
cd "$(dirname "$0")/backend"

if [ -f "run.py" ]; then
    "$CONDA_PYTHON" run.py
else
    echo "Error: run.py not found"
    exit 1
fi

