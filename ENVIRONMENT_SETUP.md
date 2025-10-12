# CAD Optimizer Environment Setup

This guide helps you create a conda environment that works with the CAD optimizer codebase, taking into account CadQuery compatibility issues.

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd /home/ec2-user/Documents/cad-optimizer
./setup_env.sh
```

This script will:
- Detect if you have a GPU
- Choose the appropriate environment file
- Create the conda environment
- Install all dependencies

### Option 2: Manual Setup

**For systems with NVIDIA GPU:**
```bash
conda env create -f environment.yml
```

**For CPU-only systems:**
```bash
conda env create -f environment-cpu.yml
```

## Why Python 3.10?

CadQuery has known compatibility issues with Python 3.11+ due to OpenCASCADE dependencies. Python 3.10 provides:
- ✅ Full CadQuery 2.4.0 compatibility
- ✅ Stable OCP (OpenCASCADE) 7.7.2 support
- ✅ Compatible with modern PyTorch and Transformers
- ✅ No dependency conflicts

## Key Dependencies

### CAD & 3D Modeling
- **CadQuery 2.4.0**: Parametric CAD framework
- **OCP 7.7.2**: OpenCASCADE Python bindings (CadQuery's backend)
- **Trimesh**: Mesh processing and GLB export
- **NumPy/SciPy**: Numerical computations

### Machine Learning
- **PyTorch 2.0+**: Deep learning framework (GPU or CPU)
- **Transformers 4.45+**: Hugging Face transformers library
- **PEFT 0.17+**: Parameter-Efficient Fine-Tuning (LoRA adapter)
- **Accelerate**: Distributed training and inference
- **Pillow**: Image processing

### Web & Utilities
- **Flask**: Web server framework
- **Requests**: HTTP library
- **rectpack/anytree**: Project-specific utilities

## Known Issues & Solutions

### Issue 1: CadQuery Version Conflicts

**Problem**: CadQuery 2.3.x vs 2.4.x have different APIs

**Solution**: This environment pins CadQuery to 2.4.0, which is tested with this codebase.

```bash
# Verify version
conda activate cad-optimizer
python -c "import cadquery as cq; print(cq.__version__)"
# Should output: 2.4.0
```

### Issue 2: OpenCASCADE (OCP) Compatibility

**Problem**: OCP needs to match CadQuery version exactly

**Solution**: Environment file specifies OCP 7.7.2 to match CadQuery 2.4.0

### Issue 3: PyTorch CUDA Version Mismatch

**Problem**: CUDA version on system doesn't match PyTorch

**Solution**: 
- GPU environment uses `pytorch-cuda=11.8` (most compatible)
- To change: edit `environment.yml` and change to `12.1` if needed
- Check your CUDA version: `nvidia-smi`

### Issue 4: Transformers Cache Takes Too Much Space

**Problem**: Hugging Face models download to `~/.cache/huggingface/`

**Solution**:
```bash
# Set custom cache location
export HF_HOME=/path/to/large/disk/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

## Activation & Usage

### Activate Environment
```bash
conda activate cad-optimizer
```

### Run the Server
```bash
cd /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
python optim.py
```

### Deactivate Environment
```bash
conda deactivate
```

## Testing the Installation

```bash
conda activate cad-optimizer

# Test all imports
python << 'EOF'
import sys
print(f"Python: {sys.version}")

import cadquery as cq
print(f"✓ CadQuery: {cq.__version__}")

import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

import transformers
print(f"✓ Transformers: {transformers.__version__}")

import peft
print(f"✓ PEFT: {peft.__version__}")

import flask
print(f"✓ Flask: {flask.__version__}")

import trimesh
print(f"✓ Trimesh: {trimesh.__version__}")

print("\n✅ All imports successful!")
EOF
```

## Environment Files

### `environment.yml` (GPU)
- PyTorch with CUDA support
- Best for training and inference
- Requires NVIDIA GPU with CUDA 11.8+

### `environment-cpu.yml` (CPU)
- PyTorch CPU-only
- Works on any system
- Slower for VLM inference (~10-20x slower)

## Troubleshooting

### "Solving environment: failed"

**Try these in order:**

1. **Update conda:**
   ```bash
   conda update -n base conda
   ```

2. **Clear conda cache:**
   ```bash
   conda clean --all
   ```

3. **Use mamba (faster solver):**
   ```bash
   conda install -n base mamba
   mamba env create -f environment.yml
   ```

### "No module named 'cadquery'"

**After activation, if imports fail:**
```bash
conda activate cad-optimizer
conda install -c conda-forge cadquery=2.4.0 ocp=7.7.2 --force-reinstall
```

### "CUDA out of memory"

**Fine-tuned model needs ~15-20GB GPU memory:**

**Solutions:**
1. Use smaller batch size (already set to 1 in code)
2. Use CPU instead: `USE_FINETUNED_MODEL=0 python optim.py`
3. Or use Ollama: Edit `optim.py` and set `USE_FINETUNED_MODEL = False`

### "Illegal instruction (core dumped)"

**CPU doesn't support required instructions (AVX2):**
```bash
# Install CPU-specific PyTorch build
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Updating Dependencies

### Update specific package:
```bash
conda activate cad-optimizer
conda update <package-name>
```

### Recreate environment from scratch:
```bash
conda env remove -n cad-optimizer
./setup_env.sh
```

## Alternative: Manual Installation

If conda environments don't work, you can install in base Python:

```bash
# CadQuery (needs conda - no pip package)
conda install -c conda-forge cadquery=2.4.0

# Everything else via pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate
pip install flask trimesh pillow numpy requests
pip install rectpack anytree
```

⚠️ **Warning**: This may cause conflicts with other packages in base environment.

## Memory Requirements

### Minimum:
- **RAM**: 8GB (CPU mode)
- **GPU**: 8GB VRAM (for base inference)
- **Disk**: 50GB (models + dependencies)

### Recommended:
- **RAM**: 32GB
- **GPU**: 24GB VRAM (A100/A6000/RTX 3090+)
- **Disk**: 100GB (multiple models)

## Support

If you encounter issues:
1. Check Python version: `python --version` (should be 3.10.x)
2. Check CadQuery: `python -c "import cadquery; print(cadquery.__version__)"`
3. Check PyTorch: `python -c "import torch; print(torch.__version__)"`
4. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## See Also

- [CadQuery Documentation](https://cadquery.readthedocs.io/)
- [Fine-Tuned Model Integration](FINETUNED_MODEL_INTEGRATION.md)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)


