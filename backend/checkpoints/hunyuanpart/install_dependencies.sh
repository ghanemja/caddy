#!/bin/bash
# Installation script for P3-SAM and Hunyuan3D-Part dependencies

set -e

echo "=========================================="
echo "Installing P3-SAM Dependencies"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Detect PyTorch version and CUDA version
echo "Detecting PyTorch and CUDA versions..."
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")

echo "PyTorch version: $PYTORCH_VERSION"
echo "CUDA version: $CUDA_VERSION"

# Install base dependencies
echo ""
echo "Installing base dependencies..."
pip install -q trimesh fpsample numba scipy scikit-learn scikit-image tqdm addict omegaconf einops timm viser ninja huggingface-hub safetensors

# Install spconv based on CUDA version
echo ""
echo "Installing spconv for CUDA $CUDA_VERSION..."
if [[ "$CUDA_VERSION" == "11.8"* ]] || [[ "$CUDA_VERSION" == "" ]]; then
    echo "Installing spconv-cu118..."
    pip install -q spconv-cu118
elif [[ "$CUDA_VERSION" == "12.1"* ]]; then
    echo "Installing spconv-cu121..."
    pip install -q spconv-cu121
elif [[ "$CUDA_VERSION" == "12.4"* ]]; then
    echo "Installing spconv-cu124..."
    pip install -q spconv-cu124
else
    echo "Warning: Unknown CUDA version, trying spconv-cu118..."
    pip install -q spconv-cu118 || echo "Failed to install spconv, you may need to install manually"
fi

# Install torch-scatter (compatible with PyTorch version)
echo ""
echo "Installing torch-scatter..."
# Extract major.minor version from PyTorch (e.g., "2.7.1" -> "2.7.0")
PYTORCH_MAJOR_MINOR=$(echo "$PYTORCH_VERSION" | cut -d. -f1,2)
PYTORCH_MAJOR_MINOR="${PYTORCH_MAJOR_MINOR}.0"

# Determine CUDA suffix for torch-scatter
if [[ "$CUDA_VERSION" == "11.8"* ]] || [[ "$CUDA_VERSION" == "" ]]; then
    CUDA_SUFFIX="cu118"
elif [[ "$CUDA_VERSION" == "12.1"* ]]; then
    CUDA_SUFFIX="cu121"
elif [[ "$CUDA_VERSION" == "12.4"* ]]; then
    CUDA_SUFFIX="cu124"
else
    CUDA_SUFFIX="cu118"  # Default
fi

echo "Installing torch-scatter for PyTorch $PYTORCH_MAJOR_MINOR + CUDA $CUDA_SUFFIX..."
pip install -q torch-scatter -f "https://data.pyg.org/whl/torch-${PYTORCH_MAJOR_MINOR}+${CUDA_SUFFIX}.html" || {
    echo "Warning: Failed to install torch-scatter from PyG, trying default..."
    pip install -q torch-scatter || echo "Warning: torch-scatter installation failed, you may need to install manually"
}

# Try to install flash-attn (optional, can fail)
echo ""
echo "Attempting to install flash-attn (optional)..."
pip install -q flash-attn || echo "Note: flash-attn not installed (optional, will be disabled automatically)"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "If you encounter import errors, you may need to:"
echo "1. Verify your PyTorch and CUDA versions match"
echo "2. Install torch-scatter manually:"
echo "   pip install torch-scatter -f https://data.pyg.org/whl/torch-{VERSION}+{CUDA}.html"
echo "3. For flash-attn issues, it's optional and will be disabled automatically"
