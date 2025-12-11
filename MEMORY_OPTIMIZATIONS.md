# Memory Optimizations Applied

This document describes the memory optimizations implemented to reduce GPU and system RAM usage.

## 1. FP16 (Half Precision) Loading ✅

**What it does**: Converts model weights from FP32 (32-bit) to FP16 (16-bit), reducing memory by ~50%.

**Implementation**:
- Model is automatically converted to FP16 after loading
- Uses `model.half()` or `model.to(torch.float16)`
- Falls back to FP32 if conversion fails

**Expected savings**: ~9 GB (from ~18 GB to ~9 GB for model weights)

**Status**: ✅ Implemented in `backends.py` after model initialization

## 2. Automatic Mixed Precision (AMP) During Inference ✅

**What it does**: Uses FP16 for computations during inference where safe, reducing peak memory.

**Implementation**:
- Wraps inference in `torch.autocast(device_type="cuda", dtype=torch.float16)`
- Automatically uses FP16 for operations that support it
- Falls back to FP32 if autocast fails

**Expected savings**: ~1-2 GB during inference

**Status**: ✅ Implemented in `segment()` method

## 3. Aggressive Garbage Collection ✅

**What it does**: Forces Python to release memory more aggressively.

**Implementation**:
- Multiple rounds of `gc.collect()` (3 rounds) before and after inference
- Called after model loading, before inference, and after inference
- Combined with `torch.cuda.empty_cache()`

**Expected savings**: Helps prevent memory leaks and fragmentation

**Status**: ✅ Implemented throughout `backends.py`

## 4. CPU Offloading (Partial) ⚠️

**What it does**: Moves model to CPU when not in use, loads to GPU only when needed.

**Implementation**:
- `clear_gpu_memory()` method moves model to CPU
- Model is moved back to GPU when `_load_model()` is called
- Currently manual - could be automated between inferences

**Limitation**: P3-SAM doesn't natively support CPU offloading like diffusion models. 
Full offloading would require reloading the model each time (slow).

**Status**: ⚠️ Partial - available via `clear_gpu_memory()` API, but not automatic

## 5. Swap File Setup ✅

**What it does**: Creates virtual RAM on disk to supplement 16GB physical RAM.

**Implementation**:
- Script: `setup_swap_file.sh`
- Creates a 32GB swap file by default (configurable)
- Uses `fallocate` or `dd` to create the file
- Formats and enables swap

**Usage**:
```bash
# Create 32GB swap (default)
./setup_swap_file.sh

# Create custom size (e.g., 64GB)
./setup_swap_file.sh 64
```

**Expected benefit**: Prevents OOM crashes when loading large models into system RAM

**Status**: ✅ Script created, run manually when needed

## 6. Ultra-Low Memory Parameters ✅

**What it does**: Reduces buffer sizes to minimize memory allocation.

**Current defaults**:
- `P3SAM_POINT_NUM=15000` (was 30000)
- `P3SAM_PROMPT_NUM=75` (was 150)
- `P3SAM_INFERENCE_POINT_NUM=15000`
- `P3SAM_INFERENCE_PROMPT_NUM=75`
- `P3SAM_PROMPT_BS=4` (was 16)

**Expected savings**: ~6-8 GB (from ~18 GB to ~10-12 GB)

**Status**: ✅ Implemented in `start_server.sh` and `backends.py`

## Expected Total Memory Reduction

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Model weights (FP32→FP16) | ~18 GB | ~9 GB | ~9 GB |
| Buffers (reduced params) | ~6 GB | ~3 GB | ~3 GB |
| Inference overhead | ~3 GB | ~1.5 GB | ~1.5 GB |
| **TOTAL** | **~27 GB** | **~13.5 GB** | **~13.5 GB** |

## Usage Instructions

1. **Setup swap file** (one-time, for system RAM):
   ```bash
   ./setup_swap_file.sh
   ```

2. **Restart server** with new optimizations:
   ```bash
   ./kill_all_gpu_processes.sh
   ./start_server.sh
   ```

3. **Monitor memory**:
   ```bash
   ./check_gpu_memory.sh
   python diagnose_gpu_memory.py
   ```

4. **Clear GPU memory** if needed:
   ```bash
   curl -X POST http://localhost:5000/api/mesh/clear_gpu_memory
   ```

## Troubleshooting

### If FP16 conversion fails:
- Model may not support FP16 (unlikely for modern models)
- Check logs for conversion errors
- Falls back to FP32 automatically

### If still OOM:
1. Verify swap file is active: `free -h`
2. Check if FP16 conversion worked: Look for "✓ Converted" in logs
3. Reduce parameters further in `start_server.sh`
4. Use `clear_gpu_memory` API between inferences

### If inference is slower:
- FP16 can sometimes be slower on older GPUs
- Disable autocast if needed (remove the `with torch.autocast()` block)
- Trade-off: More memory vs. faster inference
