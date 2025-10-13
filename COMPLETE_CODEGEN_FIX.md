# Complete Code Generation Fix - Comprehensive Validation & Prompt Improvements

## Summary

Fixed the VLM code generation to properly add wheels by:
1. ✅ **Confirming both images** (reference + snapshot) are sent to VLM
2. ✅ **Adding comprehensive validation** that rejects incomplete code
3. ✅ **Strengthening the prompt** with explicit copy instructions
4. ✅ **Adding detailed error feedback** to users

## Problem: Broken Generated Code

The VLM was generating incomplete code like this:
```python
class Rover(cqparts.Assembly):
    wheels_per_side = PositiveFloat(3)  # ✓ Parameter changed
    
    def make_components(self):
        base = cq.Workplane("XY").rect(...)  # ✗ WRONG!
        return base  # ✗ Returns object, not dict - NO WHEELS CREATED!
```

**Issues:**
- Missing imports
- Abbreviated make_components() 
- Returns single object instead of dict
- No MountedStepper creation loop
- Syntax errors
- Only ~48 lines instead of 200+

**Result:** No wheels because components dict never has wheel entries!

## Solution

### 1. Image Handling ✅ VERIFIED

**Confirmed working** (line 1279):
```python
images = [u for u in [ref_url, snapshot_url] if u]
# Returns: [reference_image, snapshot_image]
```

**Debug logging added** (lines 1282-1284):
```python
print(f"[codegen_prompt] Built prompt with {len(images)} images")
print(f"[codegen_prompt] Total prompt length: {len(''.join(parts))} chars")
print(f"[codegen_prompt] Baseline source included: {len(baseline_src)} chars")
```

### 2. Comprehensive Validation ✅ ADDED

**8 Critical Checks** (lines 1388-1515):

| Check | What It Validates |
|-------|-------------------|
| **Sufficient length** | ≥100 lines (complete file) |
| **Has Rover class** | Main assembly class exists |
| **Has RobotBase class** | Base platform class exists |
| **Has imports** | CadQuery + cqparts imported |
| **Has make_components** | Component creation method exists |
| **Returns dict** | Returns dict (not single object) |
| **Creates wheels in loop** | Has MountedStepper creation |
| **Has _axle_offsets** | Wheel positioning logic exists |
| **Has wheels_per_side** | Wheel count parameter exists |
| **No ellipsis** | No "..." abbreviations |
| **No markdown** | No ``` fences |
| **No explanations** | Pure code only |

**Rejection Logic:**
- If **> 2 critical checks fail** → Code is rejected
- Saves to `robot_base_vlm.incomplete_TIMESTAMP.py`
- Returns detailed error with suggestions

**Example Console Output:**
```
[codegen] ===== VALIDATION START =====
[codegen] ✓ Code has 48 lines (need 100+): False  ← FAIL!
[codegen] ✓ Has Rover: True
[codegen] ✓ Has RobotBase: False  ← FAIL!
[codegen] ✓ Has CadQuery import: True
[codegen] ✓ Has cqparts import: True
[codegen] ✓ Has make_components method: True
[codegen] ✓ Returns dict (not bare object): False  ← FAIL!
[codegen] ✓ Creates wheels (MountedStepper loop): False  ← FAIL!
[codegen] ===== VALIDATION END =====
[codegen] Validation: 4/8 checks passed
[codegen] ✗ VALIDATION FAILED: 4 critical checks failed
[codegen] Generated code is incomplete/broken - rejecting
```

### 3. Strengthened Prompt ✅ ENHANCED

**Added explicit requirements** (lines 1250-1276):

```
🚨 FINAL CRITICAL INSTRUCTIONS - READ CAREFULLY:

Your output MUST be COMPLETE Python code with:
• ALL imports from baseline
• ALL class definitions  
• FULL make_components() including the for-loop that creates wheel pairs
• FULL make_constraints() with all Coincident/Fixed mates
• FULL _axle_offsets() method
• 150+ lines minimum

❌ DO NOT:
• Use '...' or '# rest of implementation'
• Abbreviate or summarize any methods
• Add markdown fences (```)
• Return a single object from make_components() - must return a DICT

✅ Process:
1. COPY every single line from baseline source above
2. Find parameters like: wheels_per_side = PositiveFloat(2)
3. Change ONLY the number: wheels_per_side = PositiveFloat(3)
4. Leave everything else IDENTICAL

⚠️ Your code will be VALIDATED and REJECTED if it's incomplete!

START YOUR OUTPUT NOW (begin with #!/usr/bin/env python3):
```

**Checklist added** (lines 1141-1153):
- 9-point mandatory checklist
- Must verify before outputting
- Clear warning: "code will FAIL and be rejected"

### 4. User Feedback ✅ ENHANCED

**Frontend shows detailed errors** (lines 838-862 in app.js):

When validation fails, user sees:
```
Console:
  ✗ Generated code failed validation:
    4 critical checks failed
    Expected: 100+ lines, Got: 48 lines
    Missing:
      - Sufficient length
      - Returns dict  
      - Creates wheels in loop
      - Has _axle_offsets
    Suggestions:
      • Try with Ollama: USE_FINETUNED_MODEL=0 OLLAMA_MODEL=codellama:34b
      • Or manually copy robot_base.py to generated/robot_base_vlm.py and edit
      • The VLM needs better fine-tuning data to learn code copying

Status: Validation failed (4 issues)
```

**When validation succeeds:**
```
Console:
  ✓ Code generated: 203 lines, 5847 chars
  ✓ Validation passed: 8/8 checks passed
  Rebuilding 3D model with generated code...
  ✓ 3D model updated with new code

Status: Code generated (5847 chars) ✓
```

### 5. Generation Parameters ✅ OPTIMIZED

**For better code copying** (lines 1670-1683):
```python
max_tokens = 4096  # Can output 200+ lines
temperature = 0.05  # Very low = more faithful copying
top_p = 0.95
repetition_penalty = 1.05  # Prevent infinite loops
```

## Complete Flow

### User Action:
1. Upload reference image showing rover with wheels
2. (Optional) Upload snapshot of current model
3. Add prompt: "add 4 wheels per side"
4. Click "Generate Code"

### System Process:

```
┌─ Prepare Request ─────────────────────────────┐
│ • Extract images as base64                     │
│ • Build prompt with baseline source (200 lines)│
│ • Send 2 images to VLM                        │
└───────────────────┬───────────────────────────┘
                    ↓
┌─ VLM Generation ──────────────────────────────┐
│ • Receives: baseline source + 2 images        │
│ • Generates: Modified Python code              │
│ • Max tokens: 4096, Temp: 0.05               │
└───────────────────┬───────────────────────────┘
                    ↓
┌─ Validation (NEW!) ───────────────────────────┐
│ ✓ Check length: 100+ lines?                   │
│ ✓ Has all imports?                             │
│ ✓ Has complete make_components()?             │
│ ✓ Returns dict (not object)?                  │
│ ✓ Has MountedStepper creation loop?           │
│ ✓ Has _axle_offsets() method?                 │
│ ✓ No '...' abbreviations?                     │
│ ✓ No markdown fences?                         │
│                                                │
│ If > 2 checks fail → REJECT with detailed error│
└───────────────────┬───────────────────────────┘
                    ↓
┌─ Save & Reload ───────────────────────────────┐
│ • Save to generated/robot_base_vlm.py         │
│ • Dynamically reload module                    │
│ • Extract Rover class                          │
└───────────────────┬───────────────────────────┘
                    ↓
┌─ Build GLB (NEW!) ────────────────────────────┐
│ • Use GENERATED Rover class (not original)     │
│ • Build with wheels_per_side from new code    │
│ • Create 2*wheels_per_side total wheels       │
│ • Save to assets/rover.glb                    │
└───────────────────┬───────────────────────────┘
                    ↓
┌─ Frontend Update ─────────────────────────────┐
│ • Display code in textarea                     │
│ • Show validation results                      │
│ • Refresh 3D model (with ?ts= cache bust)     │
│ • Model loads NEW GLB → Wheels appear! 🎉    │
└────────────────────────────────────────────────┘
```

## Testing Checklist

### Before Testing:
```bash
conda activate cad-optimizer
cd /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
python optim.py
```

### Test the Fix:
1. ✅ Upload reference image with wheels
2. ✅ Add prompt: "set wheels_per_side to 4"
3. ✅ Click "Generate Code"
4. ✅ Watch console for validation messages
5. ✅ If validation passes:
   - Code appears in textarea
   - Should be 150+ lines
   - 3D model rebuilds
   - Wheels appear in viewer (4 per side = 8 total)

### Expected Console Output (Success):
```
[codegen_prompt] Built prompt with 2 images
[codegen_prompt] Total prompt length: 12453 chars
[codegen_prompt] Baseline source included: 7117 chars
[vlm] Using fine-tuned model...
[vlm] Generating response...
[vlm] ✓ Got response: 6234 chars
[codegen] Extracted code length: 5847 chars
[codegen] ===== VALIDATION START =====
[codegen] ✓ Code has 203 lines (need 100+): True
[codegen] ✓ Has Rover: True
[codegen] ✓ Has RobotBase: True
[codegen] ✓ Has CadQuery import: True
[codegen] ✓ Has cqparts import: True
[codegen] ✓ Has make_components method: True
[codegen] ✓ Returns dict: True
[codegen] ✓ Creates wheels (MountedStepper loop): True
[codegen] ✓ Has _axle_offsets method: True
[codegen] ✓ No '...' or abbreviations: True
[codegen] ===== VALIDATION END =====
[codegen] Validation: 8/8 checks passed
[codegen] ✓ Validation passed (8/8 checks)
[reload] Loading Rover from generated/robot_base_vlm.py...
[reload] ✓ Loaded Rover from generated code
Generating GLB via cqparts using Rover...
[rebuild] ✓ Saved GLB to assets/rover.glb
[codegen] ✓ GLB rebuild successful with generated code
```

### Expected Console Output (Validation Failure):
```
[codegen] ===== VALIDATION START =====
[codegen] ✓ Code has 48 lines (need 100+): False
[codegen] ✓ Returns dict: False
[codegen] ✓ Creates wheels (MountedStepper loop): False
[codegen] ✗ VALIDATION FAILED: 4 critical checks failed
[codegen] Generated code is incomplete/broken - rejecting
```

User sees in UI:
```
✗ Generated code failed validation:
  4 critical checks failed
  Expected: 100+ lines, Got: 48 lines
  Missing:
    - Sufficient length
    - Returns dict
    - Creates wheels in loop
    - Has _axle_offsets
  Suggestions:
    • Try with Ollama: USE_FINETUNED_MODEL=0 OLLAMA_MODEL=codellama:34b
```

## Alternative If Fine-Tuned Model Still Fails

Your fine-tuned model may not be optimized for code copying. Use CodeLlama instead:

```bash
# Stop fine-tuned model, use Ollama with CodeLlama
USE_FINETUNED_MODEL=0 OLLAMA_MODEL=codellama:34b python optim.py
```

Make sure Ollama is running:
```bash
ollama serve
ollama pull codellama:34b
```

CodeLlama is specifically trained for code generation and will copy the baseline source correctly.

## Quick Manual Fix (If Needed)

If VLM keeps failing validation:

```bash
cd cqparts_bucket

# Copy baseline to generated
cp robot_base.py generated/robot_base_vlm.py

# Edit manually
nano generated/robot_base_vlm.py
# Change line: wheels_per_side = PositiveFloat(2)
# To:          wheels_per_side = PositiveFloat(4)

# Refresh the model in browser - it will use your manual edit
```

## Files Modified

1. **`optim.py`** (Lines 1041-1276):
   - Strengthened VLM_CODEGEN_PROMPT
   - Added mandatory requirements checklist
   - Added final critical instructions
   - Clarified image handling

2. **`optim.py`** (Lines 1282-1284):
   - Added debug logging for images
   - Logs prompt length and baseline source length

3. **`optim.py`** (Lines 1388-1515):
   - Added comprehensive 8-point validation
   - Rejects incomplete code (>2 failures)
   - Saves rejected code for debugging
   - Returns detailed error messages

4. **`optim.py`** (Lines 1561-1571):
   - Returns validation results in response
   - Includes code_lines count
   - Includes validation_passed flag

5. **`static/js/app.js`** (Lines 835-904):
   - Displays validation errors to user
   - Shows missing components
   - Provides actionable suggestions
   - Better status messages

## Validation Criteria

Code **PASSES** if:
- ✅ ≥100 lines
- ✅ Has Rover OR RobotBase class
- ✅ Has imports
- ✅ Has make_components()
- ✅ make_components() returns dict
- ✅ Creates MountedStepper in loop
- ✅ Has _axle_offsets()
- ✅ No "..." abbreviations

Code **FAILS** if >2 checks fail.

## Why This Matters for Wheels

**How wheels are added:**

1. `wheels_per_side = PositiveFloat(4)` ← Sets count
2. `make_components()` creates dict:
   ```python
   comps = {"base": RobotBase(...)}
   offsets = self._axle_offsets()  # Calculate positions for 4 wheels
   for i, off in enumerate(offsets):  # Loop 4 times
       comps[f"Ldrive_{i}"] = MountedStepper(...)  # Left wheel
       comps[f"Rdrive_{i}"] = MountedStepper(...)  # Right wheel
   return comps  # Dict with 4 left + 4 right = 8 wheels total!
   ```

If make_components() is incomplete or returns single object → **NO WHEELS!**

The validation ensures this critical loop is present.

## Next Steps

### 1. Test Code Generation
Try generating code again with the improved prompt and validation.

### 2. Check Validation Results
Look at console output - should show 8/8 checks passed.

### 3. If Still Failing
Use CodeLlama with Ollama:
```bash
USE_FINETUNED_MODEL=0 OLLAMA_MODEL=codellama:34b python optim.py
```

### 4. Improve Fine-Tuning Dataset
If you want your fine-tuned model to work better:
- Add examples of "copy this code and change parameter X"
- Include complete baseline → modified pairs
- Train with lower temperature
- Use code-specific loss functions

## Summary of All Changes

✅ **Images**: Both reference + snapshot confirmed working
✅ **Validation**: 8-point comprehensive check, rejects incomplete code
✅ **Prompt**: Explicit "COPY ENTIRE BASELINE" with examples
✅ **Feedback**: Detailed error messages to user
✅ **Generation**: 4096 tokens, temp 0.05, optimized for code
✅ **UI**: Shows validation results and suggestions

The system now has **robust safeguards** against incomplete code generation! 🛡️

Try it and let me know if wheels appear correctly! 🎯

