#!/bin/bash
# Setup script for CAD Optimizer conda environment

set -e  # Exit on error

echo "========================================"
echo "CAD Optimizer Environment Setup"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Detect GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    echo ""
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
    USE_GPU=true
    ENV_FILE="environment.yml"
else
    echo "⚠ No NVIDIA GPU detected or nvidia-smi not available"
    echo "Will create CPU-only environment"
    echo ""
    USE_GPU=false
    ENV_FILE="environment-cpu.yml"
fi

# Ask user for confirmation
echo "About to create environment using: $ENV_FILE"
echo ""
echo "This will:"
echo "  - Create a new conda environment named 'cad-optimizer'"
echo "  - Install Python 3.10 (best for CadQuery compatibility)"
echo "  - Install CadQuery 2.4.0 with OpenCASCADE backend"
echo "  - Install PyTorch $([ "$USE_GPU" = true ] && echo "with GPU support" || echo "(CPU-only)")"
echo "  - Install Transformers, PEFT for fine-tuned VLM"
echo "  - Install Flask, Trimesh, and other dependencies"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Remove existing environment if it exists
if conda env list | grep -q "^cad-optimizer "; then
    echo ""
    echo "⚠ Environment 'cad-optimizer' already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n cad-optimizer -y
    else
        echo "Aborted."
        exit 0
    fi
fi

# Create environment
echo ""
echo "Creating conda environment..."
echo "This may take several minutes..."
echo ""

conda env create -f "$ENV_FILE"

echo ""
echo "========================================"
echo "✓ Environment created successfully!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo ""
echo "    conda activate cad-optimizer"
echo ""
echo "Then start the CAD optimizer server:"
echo ""
echo "    cd cqparts_bucket"
echo "    python optim.py"
echo ""
echo "The server will run on: http://0.0.0.0:5160"
echo ""

if [ "$USE_GPU" = true ]; then
    echo "GPU Notes:"
    echo "  - PyTorch with CUDA support is installed"
    echo "  - Fine-tuned VLM will use GPU automatically"
    echo "  - Expect ~15-20GB GPU memory usage"
    echo ""
else
    echo "CPU Notes:"
    echo "  - Running on CPU will be slower for VLM inference"
    echo "  - Consider using smaller models or increasing timeout"
    echo "  - To disable fine-tuned model: USE_FINETUNED_MODEL=0 python optim.py"
    echo ""
fi

echo "Configuration:"
echo "  - Python 3.10 (CadQuery compatible)"
echo "  - CadQuery 2.4.0 with OCP 7.7.2"
echo "  - Transformers + PEFT for fine-tuned models"
echo ""
echo "To verify installation:"
echo "    conda activate cad-optimizer"
echo "    python -c 'import cadquery; import torch; import transformers; print(\"✓ All imports successful\")'"
echo ""


