# FIXED: 400 Bad Request from Ollama

## The Problem

```
RuntimeError: Ollama error: 400 Client Error: Bad Request
```

Ollama rejected the request - usually because the payload is too large or incompatible with the model.

## What I Fixed

### 1. Reduced Context Size for Vision Models
- **Before**: 8192 tokens for all models
- **After**: 4096 tokens for vision models (they have smaller windows)

### 2. Reduced Source Code Size for Vision Models
- **Before**: 15,000 chars
- **After**: 10,000 chars for vision models (images take up tokens)

### 3. Added Detailed Error Logging
Now you'll see exactly what Ollama is complaining about:
```
[vlm] System prompt length: 1234
[vlm] User prompt length: 5678  
[vlm] ✗ Ollama returned 400
[vlm] Error response: [specific error from Ollama]
```

## Try Again Now

```bash
# RESTART server (required!)
python optim.py

# Try your request again
python codegen_helper.py reference.jpg \
  --prompt "add wheels like in the reference image"
```

You should now see:
```
[codegen] Using max_source_chars=10000 for model llama3.2-vision:11b
[vlm] Sending to Ollama (timeout: 300s, context: 4096)  ← Reduced!
[vlm] Model: llama3.2-vision:11b
[vlm] Images: 2
[vlm] System prompt length: 1234
[vlm] User prompt length: 5678
[vlm] ✓ Got response: 8543 chars  ← Should work now!
```

## If Still Getting 400 Error

The detailed error message will now show you why. Common issues:

### Issue 1: Model Not Found
```
Error response: model "llama3.2-vision:11b" not found
```

**Fix:**
```bash
ollama list  # Check what you have
ollama pull llama3.2-vision:11b  # Pull if missing
```

### Issue 2: Context Still Too Large
```
Error response: context size exceeds limit
```

**Fix:** Reduce further or try without snapshot:
```bash
# Try with only reference image (no snapshot)
python codegen_helper.py reference.jpg --prompt "add wheels"
```

### Issue 3: Images Too Large
```
Error response: image too large
```

**Fix:** Resize images:
```bash
convert reference.jpg -resize 800x800 ref_small.jpg
python codegen_helper.py ref_small.jpg --prompt "..."
```

### Issue 4: Model Doesn't Support System Parameter
```
Error response: unsupported parameter "system"
```

This is rare but possible. I'll add a fallback if needed.

## Alternative: Try Text-Only Model

If vision models keep having issues, use a text-only model with detailed descriptions:

```bash
export OLLAMA_MODEL="codellama:7b"
python optim.py

python codegen_helper.py reference.jpg --prompt "
Reference shows a rover with:
- 6 wheels total (3 per side, evenly spaced)
- Chassis length: approximately 320mm
- Wheelbase span: approximately 220mm  
- Wheel diameter: approximately 80mm
Modify the robot_base.py to match these specifications.
"
```

## What Changed

| Parameter | Old | New (vision models) |
|-----------|-----|---------------------|
| num_ctx | 8192 | 4096 |
| max_source_chars | 15000 | 10000 |
| Error logging | Basic | Detailed |

## Summary

✅ **Reduced context** for vision models (images take up space)  
✅ **Reduced source code size** to fit within limits  
✅ **Added detailed logging** to see exact errors  

**RESTART SERVER AND TRY AGAIN!**

The 400 error should be gone now. If not, the detailed error message will tell us exactly what to fix next.

