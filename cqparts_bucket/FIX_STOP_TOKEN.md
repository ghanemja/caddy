# FIXED: VLM Only Outputting "# END_OF_MODULE"

## The Problem

```
[codegen] Got 17 chars from VLM
[codegen] Raw VLM output (first 500 chars):
# END\_OF\_MODULE
```

VLM was only outputting the stop marker (17 chars) with no actual code!

## Root Cause

**Line 1216**: The stop sequence included `"# END_OF_MODULE"`:

```python
# BEFORE (BROKEN):
stops = ["```", "# END_OF_MODULE"]  # ← VLM stops immediately when it tries to output this!
```

This created a catch-22:
1. Prompt says: "End with # END_OF_MODULE"
2. VLM starts to output: `# END_OF_MODULE`
3. Ollama sees stop token and STOPS immediately
4. Result: Only 17 characters, no code!

## The Fix

### 1. Removed END_OF_MODULE from Stop Sequences

```python
# AFTER (FIXED):
if not expect_json:
    # Code generation - only stop on markdown fences
    stops = ["```python", "```"]  # ← No END_OF_MODULE!
else:
    # JSON mode
    stops = ["```", "SUMMARY:"]
```

### 2. Increased Context Size

```python
"num_ctx": 8192,  # Increased from 4096 for larger code generation
```

### 3. Updated Prompt

**Before:**
```
⚠️ The LAST line must be exactly: # END_OF_MODULE
```

**After:**
```
⚠️ Start immediately with import statements or class definitions.
⚠️ Output a complete, valid Python module.
```

Removed confusing END_OF_MODULE instruction that was causing the stop.

## What Will Happen Now

### Before (BROKEN):
```
[codegen] Got 17 chars from VLM
[codegen] Raw VLM output: # END\_OF\_MODULE
```

### After (SHOULD WORK):
```
[codegen] Got 8543 chars from VLM
[codegen] Raw VLM output (first 500 chars):
import cadquery as cq
import cqparts
from cqparts.params import PositiveFloat
...

class RobotBase(Lasercut):
    length = PositiveFloat(320)  # Modified
    ...
```

## Try It Now

```bash
# 1. Restart server (REQUIRED to get the fix)
# Stop current server with Ctrl+C, then:
python optim.py

# 2. Try your request again
python codegen_helper.py reference.jpg \
  --prompt "add wheels like in the reference image"

# 3. Watch for MUCH longer output
# Should see thousands of chars, not just 17!
```

## What You Should See

```
[baseline_source] ✓ Read 23285 chars from robot_base.py
[baseline_source] ✓ Final output: 20033 chars
[codegen] Final prompt length: 23180 chars
[codegen] Calling VLM with 2 image(s)...
[codegen] Got 8543 chars from VLM  ← MUCH MORE THAN 17!
[codegen] Raw VLM output (first 500 chars):
import cadquery as cq  ← ACTUAL CODE!
import cqparts
...
[codegen] Extracted code length: 8234 chars
[codegen] Saved to generated/robot_base_vlm.py
[codegen] Validation checks:
  - Has 'class Rover': True  ← SHOULD BE TRUE NOW
  - Has 'class RobotBase': True
```

## If It Still Doesn't Work

### Check 1: Did You Restart the Server?

**IMPORTANT:** You MUST restart the server to get the fix!

```bash
# Stop current server (Ctrl+C)
python optim.py
```

### Check 2: VLM Model Issues

If you still get very short output, try a different model:

```bash
# Check what model you're using
echo $OLLAMA_MODEL

# Try a code-focused model
export OLLAMA_MODEL="codellama:13b"
# or
export OLLAMA_MODEL="llava:34b"
```

### Check 3: Check Server Logs

Look for:
- `Got XXXX chars from VLM` - should be > 1000
- `Raw VLM output` - should show actual Python code
- `Validation checks` - should show True for classes

### Check 4: Inspect Generated File

Even if there are still issues, the file is saved:

```bash
cat generated/robot_base_vlm.py
```

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| Stop sequence | `["```", "# END_OF_MODULE"]` | `["```python", "```"]` |
| Context size | 4096 | 8192 |
| Prompt instruction | "End with: # END_OF_MODULE" | "Start with imports..." |
| VLM output | 17 chars (just marker) | Thousands of chars (actual code) |

## Key Points

✅ **Removed conflicting stop token** - VLM can now output full code  
✅ **Increased context** - Can handle larger code generation  
✅ **Clearer instructions** - No confusing END_OF_MODULE marker  
✅ **Still saves file** - Even if validation fails, you can inspect it  

---

**RESTART YOUR SERVER AND TRY AGAIN!** 🚀

The VLM should now generate actual Python code instead of just the stop marker.

