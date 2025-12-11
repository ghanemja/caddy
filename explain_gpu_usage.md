# What's Eating Your GPU Memory?

## Current Situation

From your error message:
- **Total GPU**: 22.07 GB
- **Used by your Python server**: 19.67 GB (18.69 GB allocated by PyTorch)
- **Free**: 1.55 GB
- **Other processes**: ~300 MB (processes 4838, 5282)

## The Culprit: P3-SAM Model Initialization

The **18.69 GB is consumed by the P3-SAM model** when it's loaded into memory. Here's why:

### Model Initialization Memory Usage

When `AutoMask` is initialized with:
```python
AutoMask(
    point_num=30000,  # Creates buffers for 30k points
    prompt_num=150,   # Creates buffers for 150 prompts
    ...
)
```

The model allocates GPU memory for:
1. **Base P3-SAM model weights**: ~8-10 GB
2. **Sonata sub-model weights**: ~4-6 GB  
3. **Point cloud buffers**: ~2-4 GB (based on `point_num`)
4. **Prompt buffers**: ~1-2 GB (based on `prompt_num`)
5. **Intermediate activation buffers**: ~1-2 GB

**Total: ~16-24 GB** depending on `point_num` and `prompt_num`

### The Problem

- `P3SAM_POINT_NUM` and `P3SAM_PROMPT_NUM` control **model initialization memory** (currently 30000 and 150)
- Once the model loads, these buffers stay in GPU memory for the entire server lifetime
- Even if you reduce inference parameters, the base model still occupies ~18.69 GB
- When inference runs, it needs additional memory (1.53 GB in your case), but only 1.55 GB is free → **OOM**

## Solution

### Option 1: Reduce Model Initialization Parameters (RECOMMENDED)

The model initialization parameters (`P3SAM_POINT_NUM`, `P3SAM_PROMPT_NUM`) are now set to:
- `P3SAM_POINT_NUM=20000` (reduced from 30000)
- `P3SAM_PROMPT_NUM=100` (reduced from 150)

This should reduce base model memory from ~18.69 GB to ~12-15 GB, leaving more room for inference.

**Action**: Kill your server and restart:
```bash
./kill_all_gpu_processes.sh
./start_server.sh
```

### Option 2: Ultra-Low Memory Mode

If Option 1 isn't enough, edit `start_server.sh` and uncomment the ultra-low settings:
```bash
export P3SAM_POINT_NUM=15000      # Very aggressive
export P3SAM_PROMPT_NUM=75        # Very aggressive
export P3SAM_INFERENCE_POINT_NUM=15000
export P3SAM_INFERENCE_PROMPT_NUM=75
export P3SAM_PROMPT_BS=4
```

This should reduce model memory to ~10-12 GB, leaving ~10 GB free for inference.

### Option 3: Dynamic Model Loading/Unloading (Future)

For even better memory management, we could implement:
- Load model only when needed for segmentation
- Unload model after each segmentation
- Trade-off: Slower (model loading takes time) but uses less memory

## Diagnostic Tools

Run this to see detailed GPU memory breakdown:
```bash
python diagnose_gpu_memory.py
```

Or check current status:
```bash
./check_gpu_memory.sh
```

## Memory Breakdown Summary

| Component | Memory Usage | Controlled By |
|-----------|--------------|---------------|
| Base P3-SAM model | ~8-10 GB | Model architecture |
| Sonata sub-model | ~4-6 GB | Model architecture |
| Point buffers | ~2-4 GB | `P3SAM_POINT_NUM` |
| Prompt buffers | ~1-2 GB | `P3SAM_PROMPT_NUM` |
| **TOTAL (Initialization)** | **~15-22 GB** | **P3SAM_POINT_NUM, P3SAM_PROMPT_NUM** |
| Inference buffers | +1-3 GB | `P3SAM_INFERENCE_POINT_NUM`, `P3SAM_PROMPT_BS` |

**Bottom line**: The model initialization (`P3SAM_POINT_NUM`, `P3SAM_PROMPT_NUM`) is what's consuming your GPU memory. Reducing these will free up memory for inference.
