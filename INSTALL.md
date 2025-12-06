# Installation Guide

This guide will help you set up the VLM CAD Optimizer environment on a new computer.

## Quick Start

### Option 1: Automated Installation Script (Recommended)

```bash
# Make the script executable (if not already)
chmod +x install.sh

# Run the installation script
./install.sh
```

The script will:
- Create a conda environment named `vlm_optimizer`
- Install all required packages
- Set up directory structure
- Verify the installation

### Option 2: Manual Installation with Conda Environment File

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate vlm_optimizer

# Install additional pip packages (if needed)
pip install -r requirements.txt
```

### Option 3: Manual Step-by-Step Installation

1. **Create conda environment:**
   ```bash
   conda create -n vlm_optimizer python=3.11 -y
   conda activate vlm_optimizer
   ```

2. **Install conda packages:**
   ```bash
   conda install -y -c conda-forge \
       numpy scipy pillow requests flask jinja2 freecad
   ```

3. **Install PyTorch:**
   ```bash
   # For CUDA (if you have NVIDIA GPU):
   conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   
   # For CPU only:
   conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
   ```

4. **Install pip packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Required Models

### PointNet++ Model

The PointNet++ part segmentation model is required for mesh analysis.

**Download and setup:**
```bash
# Clone the repository
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git

# Copy the model to the expected location
mkdir -p models/pointnet2
cp Pointnet_Pointnet2_pytorch/log/pointnet2_part_seg_msg.pth models/pointnet2/
```

**Or set environment variable:**
```bash
export POINTNET2_CHECKPOINT=/path/to/your/pointnet2_part_seg_msg.pth
```

### VLM Model (Optional)

The fine-tuned VLM model is optional but recommended for better semantic parameter extraction.

**Default location:**
```
runs/onevision_lora_small/checkpoint-4/
```

**Or set environment variable:**
```bash
export FINETUNED_MODEL_PATH=/path/to/your/checkpoint-4
```

If the fine-tuned model is not available, the system will:
1. Try to use Ollama (if installed and running)
2. Fall back to DummyVLMClient (for testing)

## Optional: Ollama Setup (Recommended for CPU)

Ollama provides much faster VLM inference on CPU (10-30 seconds vs 2-5 minutes).

**Install Ollama:**
- Download from: https://ollama.ai
- Or install via Homebrew (macOS): `brew install ollama`

**Start Ollama and pull model:**
```bash
# Start Ollama server
ollama serve

# In another terminal, pull the LLaVA model
ollama pull llava:latest
```

**Configure (optional):**
```bash
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llava:latest
```

## Verification

After installation, verify everything works:

```bash
# Activate environment
conda activate vlm_optimizer

# Test imports
python -c "
import torch
import transformers
import trimesh
import cadquery
print('✓ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test mesh analysis
python test_analyze_mesh.py
```

## Troubleshooting

### OpenMP Error (macOS)

If you see:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Solution:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or add to your `~/.zshrc` or `~/.bashrc`:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### FreeCAD Not Found

FreeCAD is **optional but recommended** for mesh rendering in the ingestion pipeline. The system will work without it, but mesh rendering may be limited.

**If FreeCAD installation fails:**
```bash
# Try installing separately
conda install -c conda-forge freecad

# Or if that fails, the system will use fallback rendering methods
# Check the logs for which rendering method is being used
```

**Note:** FreeCAD is used by the mesh ingestion pipeline to render mesh views for VLM analysis. If FreeCAD is not available, you may need to provide pre-rendered images or the system will attempt alternative rendering methods.

### CUDA Issues

If you have CUDA installed but PyTorch doesn't detect it:
1. Check CUDA version: `nvidia-smi`
2. Install matching PyTorch version from [pytorch.org](https://pytorch.org)

### Model Not Found

If PointNet++ model is not found:
1. Check the file exists: `ls -lh models/pointnet2/pointnet2_part_seg_msg.pth`
2. Or set environment variable: `export POINTNET2_CHECKPOINT=/path/to/model.pth`

### VLM Model Issues

If VLM generation is slow:
- **On CPU**: Use Ollama (much faster, 10-30 seconds)
- **On GPU**: Ensure fine-tuned model is loaded correctly
- Check logs for which VLM client is being used

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POINTNET2_CHECKPOINT` | Path to PointNet++ model | `models/pointnet2/pointnet2_part_seg_msg.pth` |
| `FINETUNED_MODEL_PATH` | Path to fine-tuned VLM checkpoint | `runs/onevision_lora_small/checkpoint-4` |
| `USE_FINETUNED_MODEL` | Use fine-tuned VLM (1) or not (0) | `1` |
| `OLLAMA_URL` | Ollama server URL | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llava:latest` |
| `PORT` | Flask server port | `5160` |
| `KMP_DUPLICATE_LIB_OK` | Fix OpenMP conflicts (macOS) | Not set |

## System Requirements

### Minimum
- **OS**: macOS, Linux, or Windows (WSL)
- **Python**: 3.11
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space

### Recommended
- **GPU**: NVIDIA GPU with CUDA support (for faster inference)
- **RAM**: 16GB+
- **CPU**: Multi-core processor

## Next Steps

After installation:

1. **Test the installation:**
   ```bash
   python test_analyze_mesh.py
   ```

2. **Run the GUI server:**
   ```bash
   cd cqparts_bucket
   python optim.py
   ```

3. **Read the documentation:**
   - `README.md` - General overview
   - `DEV_NOTES_POINTNET_VLM.md` - Technical details
   - `setup.txt` - PointNet++ model setup

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages in the terminal
3. Check that all models are in the correct locations
4. Verify environment variables are set correctly

