#!/bin/bash
# Installation script for VLM CAD Optimizer environment
# This script sets up a conda environment with all necessary dependencies

set -e  # Exit on error

echo "=========================================="
echo "VLM CAD Optimizer - Installation Script"
echo "=========================================="
echo ""

# Configuration
ENV_NAME="vlm_optimizer"
PYTHON_VERSION="3.11"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Step 1: Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists. Removing it first..."
    conda env remove -n "$ENV_NAME" -y
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo ""
echo "Step 2: Activating environment and installing packages..."
echo ""

# Activate environment and install packages
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install conda packages (for better compatibility)
echo "  Installing conda packages..."
conda install -y -c conda-forge \
    numpy \
    scipy \
    pillow \
    requests \
    flask \
    jinja2

# Install FreeCAD (optional but recommended for mesh rendering)
echo ""
echo "  Installing FreeCAD (for mesh rendering)..."
echo "    Note: FreeCAD installation may take several minutes..."
if conda install -y -c conda-forge freecad 2>&1 | tee /tmp/freecad_install.log; then
    echo "    ✓ FreeCAD installed successfully"
else
    echo "    ⚠ FreeCAD installation had issues (check /tmp/freecad_install.log)"
    echo "    The system will work without FreeCAD, but mesh rendering may be limited."
    echo "    You can try installing it manually later with:"
    echo "      conda install -c conda-forge freecad"
fi

# Install PyTorch (CPU or CUDA based on system)
echo ""
echo "  Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "12.1")
    echo "    CUDA detected (version $CUDA_VERSION) - installing PyTorch with CUDA support..."
    echo "    Note: Installing CUDA 12.1 version. Adjust if needed for your CUDA version."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
else
    echo "    No CUDA detected - installing PyTorch CPU version..."
    echo "    For CUDA support later, run:"
    echo "      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
    conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

# Install pip packages
echo ""
echo "  Installing pip packages..."
pip install --upgrade pip

pip install \
    cadquery>=2.4.0 \
    trimesh \
    transformers>=4.45.0 \
    peft>=0.17.0 \
    accelerate>=0.34.0 \
    safetensors \
    sentencepiece \
    six \
    tinydb \
    rectpack \
    anytree

# Optional: Install bitsandbytes for quantization (if needed)
# pip install bitsandbytes

echo ""
echo "Step 3: Setting up PointNet++ model..."
echo ""

# Create models directory
MODELS_DIR="models/pointnet2"
mkdir -p "$MODELS_DIR"

# Check if model already exists
if [ -f "$MODELS_DIR/pointnet2_part_seg_msg.pth" ]; then
    echo "  ✓ PointNet++ model already exists at $MODELS_DIR/pointnet2_part_seg_msg.pth"
else
    echo "  ⚠ PointNet++ model not found. You'll need to download it manually:"
    echo ""
    echo "    1. Clone the repository:"
    echo "       git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git"
    echo ""
    echo "    2. Find the model in log/ directory:"
    echo "       cp Pointnet_Pointnet2_pytorch/log/pointnet2_part_seg_msg.pth $MODELS_DIR/"
    echo ""
    echo "    Or set POINTNET2_CHECKPOINT environment variable to point to your model."
fi

echo ""
echo "Step 4: Setting up VLM model (optional)..."
echo ""

# Check for fine-tuned VLM model
VLM_MODEL_DIR="runs/onevision_lora_small/checkpoint-4"
if [ -d "$VLM_MODEL_DIR" ]; then
    echo "  ✓ Fine-tuned VLM model found at $VLM_MODEL_DIR"
else
    echo "  ⚠ Fine-tuned VLM model not found at $VLM_MODEL_DIR"
    echo "     The system will use Ollama or DummyVLMClient as fallback."
    echo "     To use your own fine-tuned model, place it at: $VLM_MODEL_DIR"
fi

echo ""
echo "Step 5: Setting up Ollama (optional, for faster CPU inference)..."
echo ""

if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama is installed"
    echo "     Make sure it's running: ollama serve"
    echo "     Pull the model: ollama pull llava:latest"
else
    echo "  ⚠ Ollama is not installed (optional but recommended for CPU)"
    echo "     Install from: https://ollama.ai"
    echo "     Or the system will use the fine-tuned model (slower on CPU)"
fi

echo ""
echo "Step 6: Verifying installation..."
echo ""

# Test imports
python -c "
import sys
errors = []

try:
    import torch
    print('  ✓ PyTorch:', torch.__version__)
    print('    CUDA available:', torch.cuda.is_available())
except Exception as e:
    errors.append(f'PyTorch: {e}')

try:
    import transformers
    print('  ✓ Transformers:', transformers.__version__)
except Exception as e:
    errors.append(f'Transformers: {e}')

try:
    import trimesh
    print('  ✓ Trimesh:', trimesh.__version__)
except Exception as e:
    errors.append(f'Trimesh: {e}')

try:
    import cadquery
    print('  ✓ CadQuery:', cadquery.__version__)
except Exception as e:
    errors.append(f'CadQuery: {e}')

try:
    import numpy
    print('  ✓ NumPy:', numpy.__version__)
except Exception as e:
    errors.append(f'NumPy: {e}')

try:
    import flask
    print('  ✓ Flask:', flask.__version__)
except Exception as e:
    errors.append(f'Flask: {e}')

try:
    import FreeCAD
    print('  ✓ FreeCAD: available')
    # Try to get version if possible
    try:
        if hasattr(FreeCAD, 'Version'):
            print(f'    Version: {FreeCAD.Version()}')
    except:
        pass
except Exception as e:
    print('  ⚠ FreeCAD: not available (optional, for mesh rendering)')
    print('    Error:', str(e))
    print('    Install with: conda install -c conda-forge freecad')
    print('    Note: FreeCAD is used for mesh rendering in the ingestion pipeline.')
    print('    The system will work without it, but you may need to provide pre-rendered images.')

if errors:
    print('')
    print('  ✗ Some packages failed to import:')
    for err in errors:
        print(f'    - {err}')
    sys.exit(1)
else:
    print('')
    print('  ✓ All core packages imported successfully!')
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the GUI server:"
echo "  cd cqparts_bucket"
echo "  python optim.py"
echo ""
echo "To test mesh analysis:"
echo "  python test_analyze_mesh.py"
echo ""
echo "Environment variables (optional):"
echo "  export POINTNET2_CHECKPOINT=/path/to/pointnet2_part_seg_msg.pth"
echo "  export FINETUNED_MODEL_PATH=/path/to/checkpoint-4"
echo "  export OLLAMA_URL=http://127.0.0.1:11434"
echo "  export USE_FINETUNED_MODEL=1  # Set to 0 to disable fine-tuned model"
echo ""
echo "For troubleshooting, see:"
echo "  - setup.txt (PointNet++ model setup)"
echo "  - README.md (general documentation)"
echo ""

