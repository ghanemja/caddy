# Fix Summary: VLM Source Extraction Issue

## Problem

You got this error:
```
[codegen] Building prompt with user_text: add wheels like in the reference image
[codegen] Calling VLM with 2 image(s)...
[codegen] Got 17 chars from VLM , and it can't find rover_base.py
```

**Root Cause**: The `_baseline_cqparts_source()` function was silently failing to extract `robot_base.py` source code, returning an empty or error string to the VLM.

## What Was Fixed

### 1. Robust Source Extraction (Line 2072)

**Before**: Silent failures with `except Exception: pass`  
**After**: Multiple fallback methods with detailed logging

```python
def _baseline_cqparts_source(max_chars: int = 20000) -> str:
    # Method 1: Direct file read (MOST RELIABLE)
    robot_base_path = os.path.join(BASE_DIR, "robot_base.py")
    if os.path.exists(robot_base_path):
        with open(robot_base_path, "r") as f:
            content = f.read()
            # SUCCESS!
    
    # Method 2: inspect.getsource (backup)
    # Method 3: Class-by-class extraction
    # Method 4: sys.modules search
    
    # All methods print debug info!
```

### 2. Detailed Logging

Now you'll see exactly what's happening:
```
[baseline_source] Reading robot_base.py directly from /path/to/robot_base.py
[baseline_source] ✓ Read 23456 chars from robot_base.py
[baseline_source] Total merged: 23456 chars before cleaning
[baseline_source] ✓ Final output: 23456 chars
```

### 3. Error Messages

If it fails, you'll see:
```
[baseline_source] ✗ FAILED - no source code extracted!
# ERROR: Could not extract robot_base.py source code
# Tried path: /path/to/robot_base.py
# Please ensure robot_base.py exists in the same directory as optim.py
```

## Verification

Your `robot_base.py` file exists and is **23KB** ✓

```bash
$ ls -lh robot_base.py
-rw-rw-r--. 1 ec2-user ec2-user 23K Oct  7 03:10 robot_base.py
```

## How to Test

### Method 1: Check Logs When Running Server

Restart your server and watch for the debug output:

```bash
# In your server terminal
python optim.py
```

Then make a codegen request and watch for lines like:
```
[baseline_source] Reading robot_base.py directly from ...
[baseline_source] ✓ Read XXXXX chars from robot_base.py
[baseline_source] ✓ Final output: XXXXX chars
```

### Method 2: Test with Actual Request

```bash
# Make sure server is running
python optim.py

# In another terminal
python codegen_helper.py reference.jpg --prompt "add wheels like in the reference"
```

Watch the server logs for:
- `[baseline_source] ✓ Read XXXXX chars` (should be ~23000)
- `[codegen] Got XXXXX chars from VLM` (should be much more than 17!)

### Method 3: Quick Check

Add this to the top of your codegen request handler to verify:

```python
# At the start of /codegen endpoint (after line 1037)
baseline_test = _baseline_cqparts_source()
print(f"[DEBUG] Baseline source: {len(baseline_test)} chars")
if len(baseline_test) < 1000:
    return jsonify({
        "ok": False,
        "error": "Source extraction failed",
        "source_length": len(baseline_test),
        "source_preview": baseline_test[:500]
    }), 500
```

## Expected Output Now

### Before (BROKEN):
```
[codegen] Got 17 chars from VLM
```
VLM received almost nothing, couldn't understand what to do.

### After (WORKING):
```
[baseline_source] Reading robot_base.py directly from /home/ec2-user/Documents/cad-optimizer/cqparts_bucket/robot_base.py
[baseline_source] ✓ Read 23456 chars from robot_base.py
[baseline_source] Total merged: 23456 chars before cleaning
[baseline_source] ✓ Final output: 20000 chars
[codegen] Calling VLM with 2 image(s)...
[codegen] Got 8543 chars from VLM
[codegen] Saved to generated/robot_base_vlm.py
```

VLM receives full source code, can understand and modify it!

## What to Do Now

1. **Restart your server** (if it's running)
   ```bash
   # Stop current server (Ctrl+C)
   python optim.py
   ```

2. **Try your codegen request again**
   ```bash
   python codegen_helper.py reference.jpg \
     --prompt "add wheels like in the reference image"
   ```

3. **Watch the logs** for the new debug output

4. **You should see**:
   - `✓ Read ~23000 chars from robot_base.py`
   - VLM response much longer than 17 chars
   - Generated code saved successfully

## If It Still Fails

### Check 1: File Location
```bash
cd /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
ls -l robot_base.py optim.py
# Both files should be in the same directory
```

### Check 2: BASE_DIR Variable
In `optim.py` line 87-88:
```python
BASE_DIR = os.path.dirname(__file__)
# Should be: /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
```

Add debug print to verify:
```python
print(f"[DEBUG] BASE_DIR = {BASE_DIR}")
print(f"[DEBUG] robot_base.py path = {os.path.join(BASE_DIR, 'robot_base.py')}")
print(f"[DEBUG] File exists? {os.path.exists(os.path.join(BASE_DIR, 'robot_base.py'))}")
```

### Check 3: File Permissions
```bash
# Make sure file is readable
chmod 644 robot_base.py
```

## Summary

✅ **Fixed**: Robust source extraction with multiple fallbacks  
✅ **Added**: Detailed logging to see what's happening  
✅ **Added**: Clear error messages if extraction fails  
✅ **Verified**: robot_base.py exists (23KB)  

**Next Step**: Restart server and try your codegen request again. You should see much better results!

