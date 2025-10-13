# Wheel Generation Fix

## Problem

When trying to add wheels via code generation, the VLM was generating incomplete/broken code that didn't actually create wheels.

**Symptoms:**
- Generated code had `wheels_per_side = PositiveFloat(6)` ✓
- But `make_components()` method was broken/incomplete ✗
- Missing imports and class definitions ✗
- Syntax errors (e.g., `wheelbase_span-mm` instead of `wheelbase_span_mm`) ✗
- 3D model didn't update because code couldn't be executed ✗

## Root Cause

The VLM was **ignoring** the instruction to "COPY the baseline source". Instead of copying the entire 200+ line `robot_base.py` file and changing only parameter values, it was generating a minimal stub that was syntactically invalid.

## Solution

### 1. Strengthened VLM Prompt (Lines 1041-1176)

**Added explicit instructions:**
```
⚠️ COPY THE ENTIRE BASELINE SOURCE CODE EXACTLY - every import, every class, every method
⚠️ ONLY change parameter VALUES (numbers) in class parameter definitions
⚠️ DO NOT REWRITE make_components(), make_constraints(), or any other methods
```

**Added section: "WHAT COPY THE ENTIRE BASELINE MEANS":**
```
If the baseline has 200 lines, your output should have ~200 lines
If make_components() has 30 lines, copy all 30 lines EXACTLY
DO NOT summarize, DO NOT simplify, DO NOT abbreviate with "..."
```

**Added concrete before/after example** (Lines 1095-1139):
- Shows BASELINE SOURCE (complete code)
- Shows YOUR OUTPUT (same code with one number changed)
- Shows WRONG OUTPUT (abbreviated/broken code)

**Added critical reminder at end** (Lines 1231-1237):
```
⚠️ CRITICAL REMINDER:
1. COPY the ENTIRE baseline source above
2. ONLY change parameter VALUES
3. DO NOT modify imports, class structures, or methods
4. Output must be 100+ lines if baseline is 100+ lines
5. Start with #!/usr/bin/env python3
6. NO markdown code fences
```

### 2. Improved Generation Parameters (Lines 1669-1683)

**Before:**
```python
max_new_tokens=2048
temperature=0.1
```

**After:**
```python
max_new_tokens=4096  # Doubled for longer code
temperature=0.05  # Lower for more faithful copying
top_p=0.95
repetition_penalty=1.05  # Avoid getting stuck in loops
```

**Reasoning:**
- More tokens = can output complete 200+ line files
- Lower temperature = more deterministic, faithful copying
- Repetition penalty = prevents infinite loops in generated code

## How Wheels Work

### In the Original Code (`robot_base.py`):

```python
class Rover(cqparts.Assembly):
    wheels_per_side = PositiveFloat(6)  # 6 per side = 12 total
    axle_spacing_mm = PositiveFloat(70)
    
    def make_components(self):
        # ... creates base ...
        comps = {"base": base}
        
        # Create wheel+motor pairs based on wheels_per_side
        offsets = self._axle_offsets()  # Calculates positions
        for i, off in enumerate(offsets):
            comps[f"Ldrive_{i}"] = MountedStepper(...)  # Left wheel
            comps[f"Rdrive_{i}"] = MountedStepper(...)  # Right wheel
        
        return comps  # Dict with all components
```

The `wheels_per_side` parameter controls:
1. How many wheel pairs are created in `make_components()`
2. Spacing calculation in `_axle_offsets()`
3. Wheel rendering in `_emit_parametric_wheels()`

### What VLM Should Do:

**User says**: "Add more wheels" or "6 wheels per side"

**VLM should**:
1. Copy ALL of `robot_base.py` (imports, classes, methods - everything)
2. Find the line: `wheels_per_side = PositiveFloat(2)`
3. Change to: `wheels_per_side = PositiveFloat(6)`
4. Leave everything else IDENTICAL
5. Output the complete 200+ line file

## Testing the Fix

### Before Testing:
```bash
conda activate cad-optimizer
cd cqparts_bucket
```

### Test 1: Check Baseline Source Extraction
```bash
python -c "
import sys
sys.path.insert(0, '.')
from optim import _baseline_cqparts_source

src = _baseline_cqparts_source()
print(f'Baseline source length: {len(src)} chars')
print(f'Has make_components: {\"make_components\" in src}')
print(f'Has wheels_per_side: {\"wheels_per_side\" in src}')
print(f'Has MountedStepper: {\"MountedStepper\" in src}')
print('---')
print('First 500 chars:')
print(src[:500])
"
```

**Expected:** Should show ~5000-15000 chars with complete classes

### Test 2: Generate Code with Wheels
1. Start server: `python optim.py`
2. Upload a reference image with wheels visible
3. Add prompt: "add 4 wheels per side"
4. Click "Generate Code"
5. **Check the generated code** in the textarea:
   - Should have 100+ lines ✓
   - Should have all imports ✓
   - Should have complete `make_components()` method ✓
   - Should have `wheels_per_side = PositiveFloat(4)` ✓

### Test 3: Verify 3D Model Updates
After generation:
1. Check console - should show: "✓ 3D model updated with new code"
2. Count wheels in 3D view - should match `wheels_per_side * 2`
3. Model should rebuild with new wheel count

## Debugging

### If Generated Code is Still Incomplete:

**Check what the VLM is receiving:**
```python
# Add this to optim.py temporarily after line 1242:
print("[DEBUG] Prompt length:", len("".join(parts)))
print("[DEBUG] Baseline in prompt:", "BASELINE_PYTHON_SOURCE" in "".join(parts))
with open("/tmp/vlm_prompt.txt", "w") as f:
    f.write("".join(parts))
print("[DEBUG] Full prompt saved to /tmp/vlm_prompt.txt")
```

Then check `/tmp/vlm_prompt.txt` to see if the full baseline source is included.

### If VLM Output is Truncated:

The `max_new_tokens=4096` allows ~1500 lines of code. If your `robot_base.py` is longer:

```python
# Increase further:
max_tokens = 8192 if not expect_json else 1024
```

### If Wheels Still Don't Appear:

Check the generated code manually:
```bash
cat generated/robot_base_vlm.py
```

Look for:
1. `wheels_per_side = PositiveFloat(N)` where N > 0
2. Complete `make_components()` with `MountedStepper` loop
3. Complete `_axle_offsets()` method
4. All necessary imports

If missing, the VLM is still not copying properly.

## Alternative: Use Ollama

If the fine-tuned model isn't copying code well, try Ollama with a code-focused model:

```bash
# Disable fine-tuned model
USE_FINETUNED_MODEL=0 python optim.py

# Or use a better code model with Ollama:
OLLAMA_MODEL=codellama:34b python optim.py
```

## Current Status

### Prompt Improvements:
✅ **Explicit "COPY ENTIRE BASELINE" instruction**
✅ **Clear before/after examples**  
✅ **Critical reminder at end of prompt**
✅ **Better validation checklist**

### Generation Improvements:
✅ **4096 tokens** (2x more for longer code)
✅ **Temperature 0.05** (more deterministic)
✅ **Repetition penalty** (avoid loops)

### Next Steps:

1. **Test with new prompt** - try generating code again
2. **Inspect generated code** - verify it's complete
3. **Check console logs** - look for reload messages
4. **Verify 3D update** - count wheels in viewer

If it still doesn't work, we may need to:
- Switch to a different VLM model
- Add few-shot examples to your fine-tuning dataset
- Use a specialized code generation model (CodeLlama, StarCoder, etc.)

## Files Modified

1. `optim.py` (Lines 1041-1240):
   - Strengthened VLM_CODEGEN_PROMPT
   - Added explicit copying instructions
   - Added concrete examples
   - Added critical reminders

2. `optim.py` (Lines 1669-1683):
   - Increased max_new_tokens: 2048 → 4096
   - Decreased temperature: 0.1 → 0.05
   - Added repetition_penalty: 1.05
   - Added debug logging

Try generating code again with these improvements - the VLM should now copy the baseline properly and wheels should appear! 🎯

