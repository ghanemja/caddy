# Debugging VLM Code Generation Output

## Current Issue: "Generated code missing Rover or RobotBase class definition"

This means:
- ✅ Source extraction is working (fixed!)
- ✅ VLM is responding 
- ❌ VLM output doesn't contain expected classes

## What I Just Added

Enhanced logging to show you exactly what's happening:

```python
[codegen] Final prompt length: 23456 chars      ← How much context sent to VLM
[codegen] Calling VLM with 2 image(s)...
[codegen] Got 5432 chars from VLM               ← VLM response length
[codegen] Raw VLM output (first 500 chars):     ← What VLM actually said
[shows the actual output...]
[codegen] Extracted code length: 4321 chars     ← After cleaning
[codegen] First 300 chars: [preview...]         ← What we extracted
[codegen] Validation checks:                    ← What's in the code
  - Has 'class Rover': True/False
  - Has 'class RobotBase': True/False
  - Has imports: True/False
  - Has function defs: True/False
```

**Now the code is SAVED even if validation fails** so you can see what the VLM generated!

## Next Steps

### 1. Try Your Request Again

```bash
# Restart server to get new logging
python optim.py

# In another terminal
python codegen_helper.py reference.jpg --prompt "add wheels like in the reference image"
```

### 2. Watch the Server Logs

Look for the new detailed output. You'll see:

**A) What the VLM actually generated:**
```
[codegen] Raw VLM output (first 500 chars):
Here's the modified code:

```python
import cadquery...
```
```

**B) Validation results:**
```
[codegen] Validation checks:
  - Has 'class Rover': False  ← AH HA! Missing!
  - Has 'class RobotBase': False
```

### 3. Check the Saved File

**The code IS saved** even if validation fails:
```bash
cat generated/robot_base_vlm.py
```

This will show you exactly what the VLM generated so you can see the problem.

## Common Problems & Solutions

### Problem 1: VLM is Adding Explanations

**What you'll see:**
```python
# VLM output:
Here's the modified code with more wheels:

import cadquery as cq
...
```

**Solution:** The VLM is ignoring "output ONLY Python code". This happens with some models.

**Fix:** Add to your prompt:
```bash
python codegen_helper.py ref.jpg --prompt "
IMPORTANT: Output pure Python code only. No explanations before or after.
Start with imports, end with class definitions.

Now: add wheels like in the reference image
"
```

### Problem 2: VLM is Outputting Partial Code

**What you'll see:**
```python
# Only shows changes, not full classes
# Example:
class Rover(cqparts.Assembly):
    wheels_per_side = PositiveFloat(3)  # CHANGED from 2
    # ... rest of changes ...
```

**Solution:** VLM is showing diff-style output instead of complete code.

**Fix:** The prompt needs to emphasize "complete module". Already in the prompt, but you can reinforce:
```bash
--prompt "Output COMPLETE Python module with ALL classes. Add wheels..."
```

### Problem 3: VLM Context Too Long

**What you'll see:**
```
[codegen] Final prompt length: 45000 chars  ← Very long
[codegen] Got 234 chars from VLM  ← Very short response
```

**Solution:** VLM context limit exceeded, it's truncating or giving up.

**Fix:** Reduce max_chars in source extraction:
```python
# In optim.py, line 2072
def _baseline_cqparts_source(max_chars: int = 10000):  # Reduced from 20000
```

### Problem 4: Wrong VLM Model

**What you'll see:**
```
[codegen] Got [lots of chars] but no valid Python
```

**Solution:** Some VLM models are better at code than others.

**Fix:** Check your VLM model:
```bash
# Check current
echo $OLLAMA_MODEL

# Try a code-focused model
export OLLAMA_MODEL="codellama:13b"
# or
export OLLAMA_MODEL="llava:34b"
```

### Problem 5: Images Not Helpful

**What you'll see:**
Code that doesn't match reference at all.

**Solution:** VLM can't extract enough information from images alone.

**Fix:** Be VERY explicit in text:
```bash
--prompt "
Reference image shows:
- 6 wheels total (3 per side)
- Wheel diameter approximately 80mm
- Longer chassis, approximately 320mm
- Wheelbase span approximately 220mm

Modify the code to match these specifications.
"
```

## Diagnostic Checklist

Run through this when debugging:

```bash
# 1. Check server logs for prompt length
grep "Final prompt length" [server output]
# Should be 15,000-25,000 chars (includes source code)

# 2. Check VLM response length
grep "Got .* chars from VLM" [server output]
# Should be 2,000+ chars for full code

# 3. Check what VLM output (look in logs)
# Look for "Raw VLM output (first 500 chars):"

# 4. Check saved file
cat generated/robot_base_vlm.py
# What's actually in there?

# 5. Check validation
# Look for "Validation checks:" in logs
# Which checks failed?
```

## Manual Inspection

The file is saved at `generated/robot_base_vlm.py` even if validation fails.

```bash
# View it
cat generated/robot_base_vlm.py

# Check what's in it
grep "class " generated/robot_base_vlm.py
grep "def " generated/robot_base_vlm.py
grep "import" generated/robot_base_vlm.py

# If it looks good despite validation failure, you can use it!
cp generated/robot_base_vlm.py robot_base.py
```

## Quick Fix: Accept Partial Results

If the VLM is generating useful code but failing validation, you can:

1. **Comment out the validation** (line 1122-1141 in optim.py):
```python
# if not (has_rover or has_robotbase):
#     return jsonify({...})
```

2. **Or manually copy the useful parts**:
```bash
# VLM might generate just the Rover class modifications
# You can manually merge them into your robot_base.py
```

## What to Send Me for Help

If still stuck, send:

1. **Server logs** (the lines with [codegen] and [baseline_source])
2. **First 50 lines of generated code:**
   ```bash
   head -50 generated/robot_base_vlm.py
   ```
3. **Your VLM model:**
   ```bash
   echo $OLLAMA_MODEL
   ```
4. **Your prompt:**
   What you passed to `--prompt`

## Summary

With the new logging:
- ✅ You can see what VLM outputs (first 500 chars)
- ✅ Code is saved even if validation fails
- ✅ Detailed validation tells you what's missing
- ✅ You can inspect `generated/robot_base_vlm.py` directly

**Try again and watch the logs carefully!**

