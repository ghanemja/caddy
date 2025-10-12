# FIXED: VLM Timeout After 120 Seconds

## The Problem

```
RuntimeError: Ollama error: HTTPConnectionPool(host='127.0.0.1', port=11434): 
Read timed out. (read timeout=120)
```

The VLM is taking too long to process your request and timing out.

## What I Fixed

### 1. Increased Timeout
- **Before**: 120 seconds (2 minutes)
- **After**: 300 seconds (5 minutes) for code generation

### 2. Reduced Source Code Size
- **Before**: Sending 20,000 chars
- **After**: Sending 15,000 chars (faster to process)

### 3. Added Better Logging
Now shows:
```
[vlm] Sending to Ollama (timeout: 300s, context: 8192)
[vlm] Model: llava-llama3:latest
[vlm] Images: 2
[vlm] ✓ Got response: 8543 chars
```

### 4. Better Error Messages
If timeout still occurs, you get helpful suggestions.

## Try Again Now

```bash
# Restart server (REQUIRED!)
python optim.py

# Try your request
python codegen_helper.py reference.jpg \
  --prompt "add wheels like in the reference image"

# Watch the logs - you'll see progress info
```

## If It Still Times Out

### Option 1: Use a Faster Model

```bash
# Check current model
echo $OLLAMA_MODEL

# Switch to a smaller/faster model
export OLLAMA_MODEL="codellama:7b"  # Much faster than 13b/34b
# or
export OLLAMA_MODEL="llama3.2-vision:11b"  # Newer, faster

# Restart server
python optim.py
```

### Option 2: Try Without Snapshot Image

Sending only the reference image (not snapshot) is faster:

```bash
python codegen_helper.py reference.jpg \
  --prompt "add wheels like in the reference"
# Note: no --snapshot parameter
```

### Option 3: Reduce Image Size

Large images slow down processing:

```bash
# Resize your images first
convert reference.jpg -resize 1024x1024 reference_small.jpg
convert snapshot.png -resize 1024x1024 snapshot_small.png

# Use smaller images
python codegen_helper.py reference_small.jpg \
  --snapshot snapshot_small.png \
  --prompt "add wheels"
```

### Option 4: Simplify Your Prompt

Shorter prompts process faster:

```bash
# Instead of long detailed prompt
python codegen_helper.py ref.jpg --prompt "
  The reference shows a rover with:
  - 6 wheels total (3 per side)
  - Longer chassis (320mm vs current 280mm)
  - Wider wheelbase (220mm vs current 170mm)
  - Smaller wheels (80mm vs current 90mm)
  Please modify all these parameters accordingly.
"

# Try shorter
python codegen_helper.py ref.jpg --prompt "3 wheels per side, longer chassis"
```

### Option 5: Check Ollama Server

#### Is Ollama Running?
```bash
curl http://localhost:11434/api/tags
```

#### Is It Using GPU?
```bash
# Check Ollama logs
# Look for "CUDA" or "GPU" mentions

# If using CPU only, it will be MUCH slower
# Make sure GPU is available and working
nvidia-smi  # Check GPU status
```

#### Restart Ollama
```bash
# Stop Ollama
pkill ollama

# Start fresh
ollama serve
```

## Understanding the Timeout

### What's Happening:
1. Your prompt is sent to Ollama (20-25KB of text + 2 images)
2. VLM needs to:
   - Process images (understand the design)
   - Read 15,000+ chars of Python code
   - Generate modified Python code
   - This can take 2-10 minutes depending on model/hardware

### Why It Times Out:
- **Large model** (13b/34b parameters) = slower
- **CPU-only** = much slower than GPU
- **Large images** = more processing time
- **Complex prompt** = more thinking time

### Typical Processing Times:

| Configuration | Time |
|---------------|------|
| llama3.2-vision:11b + GPU | 30-60 seconds |
| codellama:7b + GPU | 60-120 seconds |
| llava:13b + GPU | 120-240 seconds |
| llava:34b + GPU | 240-480 seconds |
| Any model + CPU only | 5-20 minutes (often timeout) |

## Quick Fixes (In Order of Effectiveness)

### 1. Use Smaller Model (FASTEST FIX)
```bash
export OLLAMA_MODEL="codellama:7b"
# Restart server
```

### 2. Remove Snapshot Image
```bash
# Only send reference image
python codegen_helper.py reference.jpg --prompt "..."
```

### 3. Smaller Images
```bash
convert reference.jpg -resize 800x800 ref_small.jpg
python codegen_helper.py ref_small.jpg --prompt "..."
```

### 4. GPU Check
```bash
nvidia-smi
# Should show Ollama using GPU
# If not, check Ollama installation/config
```

## What You'll See Now

### Success (within 5 minutes):
```
[vlm] Sending to Ollama (timeout: 300s, context: 8192)
[vlm] Model: codellama:7b
[vlm] Images: 1
... wait 1-3 minutes ...
[vlm] ✓ Got response: 8543 chars
[codegen] Saved to generated/robot_base_vlm.py
✓ Success!
```

### Still Timeout (after 5 minutes):
```
[vlm] Sending to Ollama (timeout: 300s, context: 8192)
... wait 5 minutes ...
[codegen] ⏱️ TIMEOUT: VLM took too long (>5 minutes)

Suggestions:
- Try a faster/smaller VLM model (e.g., codellama:7b)
- Reduce image resolution
- Check if Ollama has GPU access
```

## Alternative: Increase Timeout Further

If you have a slow model but want to wait:

Edit `optim.py` line 1244:
```python
# Change from:
timeout_seconds = 300 if not expect_json else 120  # 5 min

# To:
timeout_seconds = 600 if not expect_json else 120  # 10 min
```

Then restart server.

## Summary

| Issue | Fix |
|-------|-----|
| 120s timeout | Increased to 300s (5 min) |
| Source too large | Reduced from 20k to 15k chars |
| No progress info | Added [vlm] logs |
| Unclear errors | Better error messages with suggestions |

**Best quick fix**: Use a smaller/faster model like `codellama:7b`

---

**RESTART SERVER AND TRY AGAIN!** ⏱️

With 5 minutes instead of 2, most requests should complete.
If still timing out, try a faster model.

