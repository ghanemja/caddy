#!/bin/bash
# Startup script for the CAD Optimizer server

# Set conda environment path for FreeCAD (use vlm_optimizer where FreeCAD is installed)
export CONDA_PREFIX=/opt/homebrew/anaconda3/envs/vlm_optimizer

# Set library path for FreeCAD dependencies (macOS)
export DYLD_LIBRARY_PATH=/opt/homebrew/anaconda3/envs/vlm_optimizer/lib:$DYLD_LIBRARY_PATH

# Add conda site-packages to Python path for compatible numpy/scipy
export PYTHONPATH=/opt/homebrew/anaconda3/envs/vlm_optimizer/lib/python3.11/site-packages:$PYTHONPATH

# Activate conda environment (vlm_optimizer has FreeCAD)
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate vlm_optimizer

# Use the conda Python explicitly (not pyenv Python)
CONDA_PYTHON="/opt/homebrew/anaconda3/envs/vlm_optimizer/bin/python"

# Change to backend directory
cd "$(dirname "$0")/backend"

# Add backend and root to Python path
export PYTHONPATH="$PYTHONPATH:$(dirname "$0")/backend:$(dirname "$0")"

# Add cqparts_bucket to Python path for cqparts libraries
export PYTHONPATH="$PYTHONPATH:$(dirname "$0")/backend/cqparts_bucket"

# Run the server using the new structure
# Falls back to optim.py if run.py doesn't exist
# Use conda Python explicitly to avoid pyenv conflicts
if [ -f "run.py" ]; then
    "$CONDA_PYTHON" run.py
else
    echo "Warning: run.py not found, using legacy optim.py"
    "$CONDA_PYTHON" optim.py
fi

