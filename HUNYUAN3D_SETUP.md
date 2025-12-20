# Hunyuan3D Model Setup Guide

This document outlines the steps taken to properly get the Hunyuan3D-Part P3-SAM model running correctly.

## Overview

The Hunyuan3D-Part model is used for 3D part segmentation. It consists of:
- **P3-SAM**: Native 3D part segmentation model
- **XPart**: High-fidelity shape decomposition (code structure, used by P3-SAM)

Repository: https://github.com/Tencent-Hunyuan/Hunyuan3D-Part

---

## Step 1: Clone the Hunyuan3D-Part Repository

Clone the repository into the checkpoints directory:

```bash
cd backend/checkpoints
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git hunyuanpart
```

**Expected structure:**
```
backend/checkpoints/hunyuanpart/
├── P3-SAM/
│   ├── model.py
│   ├── demo/
│   │   └── auto_mask_no_postprocess.py  # Main AutoMask class
│   └── utils/
│       └── chamfer3D/  # CUDA extensions
└── XPart/
    └── partgen/
        ├── utils/
        │   └── misc.py  # Contains smart_load_model
        └── models/
```

---

## Step 2: Download the Model Checkpoint

Download the P3-SAM model weights from HuggingFace:

**Source:** https://huggingface.co/tencent/Hunyuan3D-Part

**Required file:** `p3sam/p3sam.safetensors` (~431 MB)

**Place it at:**
```
backend/checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors
```

You can either:
1. Download directly from HuggingFace and place it manually
2. Or clone the full HuggingFace repository:
   ```bash
   cd backend/checkpoints
   git clone https://huggingface.co/tencent/Hunyuan3D-Part
   ```

**Note:** The large `model/model.safetensors` (6.3GB) is NOT needed for segmentation - only `p3sam/p3sam.safetensors` is required.

---

## Step 3: Install Dependencies

Navigate to the cloned repository and run the installation script:

```bash
cd backend/checkpoints/hunyuanpart
bash install_dependencies.sh
```

**Or install manually:**

```bash
# Base dependencies
pip install trimesh fpsample numba scipy scikit-learn scikit-image tqdm
pip install addict omegaconf einops timm viser ninja huggingface-hub safetensors

# spconv (CUDA version-specific)
# For CUDA 11.8:
pip install spconv-cu118

# For CUDA 12.1:
pip install spconv-cu121

# For CUDA 12.4:
pip install spconv-cu124

# torch-scatter (PyTorch version-specific)
pip install torch-scatter -f https://data.pyg.org/whl/torch-{VERSION}+{CUDA}.html

# Optional: flash-attn (for performance, can fail without issues)
pip install flash-attn
```

**The installation script automatically:**
- Detects your PyTorch and CUDA versions
- Installs the correct `spconv` variant
- Installs compatible `torch-scatter`
- Attempts to install `flash-attn` (optional)

---

## Step 4: Compile CUDA Extensions

The `chamfer3D` CUDA extension needs to be compiled:

```bash
cd backend/checkpoints/hunyuanpart/P3-SAM/utils/chamfer3D
pip install -e .
```

Or it will auto-compile when first imported.

---

## Step 5: Configure Environment Variables

Set the following environment variables (in `start_server.sh` or your shell):

```bash
# Memory optimization for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# P3-SAM model initialization parameters (reduces GPU memory usage)
export P3SAM_POINT_NUM=10000           # Number of points (lower = less memory)
export P3SAM_INFERENCE_POINT_NUM=10000 # For inference
export P3SAM_PROMPT_NUM=50             # Number of prompts (lower = less memory)
export P3SAM_INFERENCE_PROMPT_NUM=50   # For inference
export P3SAM_PROMPT_BS=2               # Batch size (lower = less memory)

# Mixed precision (optional, for performance)
export TORCH_CUDNN_V8_API_ENABLED=1
export P3SAM_USE_AUTOCAST=1
```

**Key settings:**
- `P3SAM_POINT_NUM`: Controls initial GPU buffer size. Lower values = less memory but potentially less detail.
- `P3SAM_PROMPT_NUM`: Number of prompt points to generate. Lower = less memory.
- `P3SAM_PROMPT_BS`: Batch size for prompt processing. Lower = less memory.

---

## Step 6: Configure the Application

Set the segmentation backend to use Hunyuan3D:

```bash
export SEGMENTATION_BACKEND=hunyuan3d
```

Or in your code:
```python
os.environ["SEGMENTATION_BACKEND"] = "hunyuan3d"
```

---

## Step 7: Critical Path Setup (Handled Automatically)

The code automatically handles complex Python import path setup:

**The Challenge:**
- P3-SAM's `model.py` imports `utils.misc` from `XPart/partgen/utils/`
- P3-SAM's `auto_mask_no_postprocess.py` imports `utils.chamfer3D` from `P3-SAM/utils/`
- These are **two different `utils` packages** in different locations

**The Solution (in `backends.py`):**
1. Add `P3-SAM/` to `sys.path` at position 0 (for `utils.chamfer3D`)
2. Add `XPart/partgen/` to `sys.path` at position 1 (for `utils.misc`)
3. Manually import `model.py` using `importlib` with correct path ordering
4. Clear `sys.modules` cache to avoid stale imports
5. Import `auto_mask_no_postprocess.AutoMask` after paths are set

This is all handled automatically by `Hunyuan3DPartSegmentationBackend._load_model()`.

---

## Step 8: Verify Installation

The backend will check for:
1. ✅ P3-SAM code exists at `backend/checkpoints/hunyuanpart/P3-SAM/`
2. ✅ XPart/partgen exists at `backend/checkpoints/hunyuanpart/XPart/partgen/`
3. ✅ `utils/misc.py` exists at `XPart/partgen/utils/misc.py`
4. ✅ Checkpoint exists at `backend/checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors`
5. ✅ All dependencies are importable

**If any checks fail, you'll see clear error messages with instructions.**

---

## Directory Structure Summary

```
backend/checkpoints/
├── hunyuanpart/                    # Cloned repository (code)
│   ├── P3-SAM/
│   │   ├── model.py
│   │   ├── demo/
│   │   │   └── auto_mask_no_postprocess.py
│   │   └── utils/
│   │       └── chamfer3D/
│   └── XPart/
│       └── partgen/
│           └── utils/
│               └── misc.py
│
└── Hunyuan3D-Part/                 # Model weights (from HuggingFace)
    └── p3sam/
        └── p3sam.safetensors       # ~431 MB checkpoint (REQUIRED)
```

---

## Troubleshooting

### Import Errors

**Error: `No module named 'utils.misc'`**
- Ensure `XPart/partgen/utils/misc.py` exists
- Verify the repository was fully cloned (not a shallow clone)

**Error: `No module named 'utils.chamfer3D'`**
- Ensure `P3-SAM/utils/chamfer3D/` exists
- Try compiling: `cd P3-SAM/utils/chamfer3D && pip install -e .`

**Error: `No module named 'spconv'`**
- Install the correct `spconv` variant for your CUDA version:
  ```bash
  pip install spconv-cu118  # or cu121, cu124
  ```

### CUDA Out of Memory

Reduce the environment variables:
```bash
export P3SAM_POINT_NUM=5000        # Even lower
export P3SAM_PROMPT_NUM=25         # Even lower
export P3SAM_PROMPT_BS=1           # Minimum batch size
```

### Checkpoint Not Found

**Error: `P3-SAM checkpoint not found`**
- Verify `p3sam.safetensors` exists at:
  - `backend/checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors`
- Download from: https://huggingface.co/tencent/Hunyuan3D-Part

---

## Configuration in Code

The backend is configured in `backend/meshml/segmentation/backends.py`:

```python
# Default checkpoint location
backend/checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors

# Default code location
backend/checkpoints/hunyuanpart/P3-SAM/
```

These can be overridden by passing `model_ckpt_dir` to the constructor.

---

## Summary Checklist

- [ ] Cloned Hunyuan3D-Part repository to `backend/checkpoints/hunyuanpart/`
- [ ] Downloaded `p3sam.safetensors` to `backend/checkpoints/Hunyuan3D-Part/p3sam/`
- [ ] Installed all dependencies (`install_dependencies.sh` or manually)
- [ ] Compiled CUDA extensions (auto-compiles on first use)
- [ ] Set environment variables for memory optimization
- [ ] Set `SEGMENTATION_BACKEND=hunyuan3d`
- [ ] Verified all paths and imports work

---

## Key Files Modified

- `backend/meshml/segmentation/backends.py` - Main backend implementation
- `start_server.sh` - Environment variable configuration
- `backend/checkpoints/hunyuanpart/install_dependencies.sh` - Dependency installer

---

## Notes

- The large 6.3GB `model.safetensors` is **NOT required** for segmentation
- Only `p3sam.safetensors` (431MB) is needed
- All imports are handled locally - no HuggingFace API calls during runtime
- The code automatically handles complex Python path setup for the two `utils` packages