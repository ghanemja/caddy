#!/bin/bash

# Initialize conda
CONDA_BASE="/home/ec2-user/miniforge3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

# Activate conda environment
conda activate freecad #vlm_optimizer
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate vlm_optimizer environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Set library path (Linux uses LD_LIBRARY_PATH, not DYLD_LIBRARY_PATH)
export LD_LIBRARY_PATH="$CONDA_BASE/envs/vlm_optimizer/lib:$LD_LIBRARY_PATH"
CONDA_PYTHON="$CONDA_BASE/envs/freecad/bin/python"
cd "$(dirname "$0")/backend"

if [ -f "run.py" ]; then
    "$CONDA_PYTHON" run.py
else
    echo "Error: run.py not found"
    exit 1
fi

