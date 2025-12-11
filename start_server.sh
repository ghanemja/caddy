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
conda activate freecad 
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate freecad environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Set library path
export LD_LIBRARY_PATH="$CONDA_BASE/envs/freecad/lib:$LD_LIBRARY_PATH"

# --- MEMORY OPTIMIZATIONS (CRITICAL) ---

# 1. Reduce Fragmentation
# This helps PyTorch use the "Reserved" but "Unallocated" memory seen in your logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Enable Mixed Precision
export TORCH_CUDNN_V8_API_ENABLED=1
export P3SAM_USE_AUTOCAST=1

# 3. Reduce Model Resolution (The Main Fix)
# Lowering point counts drastically reduces the VRAM tensor size.
# Changed from 15000 -> 10000
export P3SAM_POINT_NUM=10000            
export P3SAM_INFERENCE_POINT_NUM=10000  

# 4. Reduce Complexity
# Lowering prompt counts reduces the number of masks the model tries to generate at once.
# Changed from 75 -> 50
export P3SAM_PROMPT_NUM=50              
export P3SAM_INFERENCE_PROMPT_NUM=50    

# 5. Reduce Batch Size
# Processing fewer items at once frees up working memory.
# Changed from 4 -> 2
export P3SAM_PROMPT_BS=2                

# ---------------------------------------

CONDA_PYTHON="$CONDA_BASE/envs/freecad/bin/python"
cd "$(dirname "$0")/backend"

if [ -f "run.py" ]; then
    "$CONDA_PYTHON" run.py
else
    echo "Error: run.py not found"
    exit 1
fi