# VLM Code Generation for Robot Base

## Overview

The `/codegen` endpoint uses a Vision Language Model (VLM) to automatically modify `robot_base.py` based on:
1. **Reference Image** - Your target design (photo, sketch, or render)
2. **Current CAD Snapshot** - Orthogonal views of the current model
3. **User Intent** - Your qualitative feedback and instructions

## How It Works

### 1. The Prompt Flow

```
┌─────────────────────┐
│ Reference Image     │ ──┐
└─────────────────────┘   │
                          │
┌─────────────────────┐   │    ┌───────────────┐
│ CAD Snapshot        │ ──┼───▶│  VLM Model    │
└─────────────────────┘   │    └───────────────┘
                          │           │
┌─────────────────────┐   │           │
│ robot_base.py       │ ──┤           ▼
│ (current source)    │   │    ┌───────────────┐
└─────────────────────┘   │    │ Modified      │
                          │    │ robot_base.py │
┌─────────────────────┐   │    └───────────────┘
│ User Feedback Text  │ ──┘
└─────────────────────┘
```

### 2. The Source Code Variable

The current `robot_base.py` source is loaded via the `_baseline_cqparts_source()` function:

```python
def _baseline_cqparts_source(max_chars: int = 20000) -> str:
    """
    Returns a compact string containing the actual Rover/RobotBase/wheel/stepper
    source from your project, trimmed to fit in the prompt.
    """
    chunks = []
    
    # Extract source from imported modules
    import robot_base as _rb
    mod_src = inspect.getsource(_rb)
    chunks.append(mod_src)
    
    # Extract individual class sources
    from robot_base import Rover, RobotBase
    from wheel import BuiltWheel, SpokeWheel, SimpleWheel
    from pan_tilt import PanTilt
    
    # ... collect all sources ...
    
    merged = "\n\n# ----\n\n".join(chunks)
    merged = _strip_docstrings_and_comments(merged)
    
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n# ... [truncated] ..."
    
    return merged
```

### 3. Making an API Call

#### Using curl:

```bash
curl -X POST http://localhost:5160/codegen \
  -F "reference=@my_target_design.jpg" \
  -F "snapshot=@current_cad_orthogonal.png" \
  -F "prompt=Make the base wider and add 4 wheels per side instead of 2"
```

#### Using Python requests:

```python
import requests

with open('reference.jpg', 'rb') as ref, \
     open('snapshot.png', 'rb') as snap:
    
    response = requests.post(
        'http://localhost:5160/codegen',
        files={
            'reference': ref,
            'snapshot': snap,
        },
        data={
            'prompt': '''
            The reference shows a longer, narrower chassis. 
            Current model is too square.
            Target proportions: length should be 1.8x width.
            Increase wheels_per_side to 3.
            '''
        }
    )

result = response.json()
if result['ok']:
    print(f"Generated code saved to: {result['module_path']}")
    print(f"Backup saved to: {result['backup_path']}")
else:
    print(f"Error: {result['error']}")
```

### 4. The Prompt Structure

The VLM receives a carefully structured prompt:

```
=== YOUR TASK ===
You will receive:
1. REFERENCE IMAGE(s) - showing the desired robot rover design
2. CURRENT CAD SNAPSHOT - orthogonal views of the current CAD model
3. BASELINE PYTHON SOURCE - the current robot_base.py implementation
4. USER INTENT - human qualitative assessment and modification requests

Your job: Modify the baseline Python source to create a parametric rover 
that matches the reference image.

=== CRITICAL OUTPUT RULES ===
⚠️ Output ONLY Python code. NO explanations, NO markdown fences...
⚠️ The LAST line must be exactly: # END_OF_MODULE

=== CODE REQUIREMENTS ===
1. Produce a complete, valid Python 3.10 module
2. Keep the same imports and class structure from the baseline
3. Modify parameters like:
   - Base dimensions (length, width, chamfer, thickness)
   - Wheel count (wheels_per_side)
   - Wheel spacing (axle_spacing_mm, wheelbase_span_mm)
   ...

================================================================================
<<<BASELINE_PYTHON_SOURCE>>>
# File: robot_base.py
# This is the current implementation you should modify

[YOUR ACTUAL robot_base.py SOURCE CODE HERE]

<<<END_BASELINE_PYTHON_SOURCE>>>
================================================================================

<<<CURRENT_CAD_STATE>>>
{
  "current_params": {...},
  "context": {...},
  ...
}
<<<END_CURRENT_CAD_STATE>>>
================================================================================

<<<USER_INTENT_AND_FEEDBACK>>>
[YOUR PROMPT TEXT HERE]
<<<END_USER_INTENT>>>
================================================================================

=== IMAGES PROVIDED ===
- Image 0: REFERENCE (target design)
- Image 1: CURRENT CAD SNAPSHOT (orthogonal views)

=== NOW OUTPUT YOUR MODIFIED robot_base.py ===
Remember: Python code ONLY, no markdown, no explanations.
End with: # END_OF_MODULE
```

## Example Workflow

### 1. Capture Reference and Snapshot

```bash
# Take a photo of your target design
# Save as: reference_design.jpg

# Export current CAD orthogonal views
# Save as: current_snapshot.png
```

### 2. Write Your Intent

```text
The reference image shows a more compact rover with:
- Shorter wheelbase (current: 280mm → target: ~200mm)
- Wider stance (current span: 170mm → target: ~220mm)
- 3 wheels per side instead of 2
- Smaller wheel diameter (looks like 80mm vs current 90mm)
```

### 3. Call the API

```python
result = call_codegen_api(
    reference='reference_design.jpg',
    snapshot='current_snapshot.png',
    prompt=your_intent_text
)
```

### 4. Review Generated Code

The VLM will generate modified Python code saved to:
- `generated/robot_base_vlm.py` (latest)
- `generated/robot_base_vlm_{timestamp}.py` (timestamped backup)

### 5. Integrate Changes

Review the generated code and either:
- **Option A**: Copy relevant changes to your main `robot_base.py`
- **Option B**: Use the generated file directly (rename/replace)
- **Option C**: Cherry-pick specific parameter changes

## Key Variables the VLM Can Modify

### RobotBase Parameters
```python
class RobotBase(Lasercut):
    length     = PositiveFloat(250)      # Base length
    width      = PositiveFloat(240)      # Base width
    thickness  = PositiveFloat(6)        # Plate thickness
    chamfer    = PositiveFloat(30)       # Corner chamfer
```

### Rover Parameters
```python
class Rover(cqparts.Assembly):
    length     = PositiveFloat(280)          # Overall length
    width      = PositiveFloat(170)          # Overall width
    chamfer    = PositiveFloat(55)           # Corner chamfer
    thickness  = PositiveFloat(6)            # Base thickness
    
    # Multi-axle params
    wheels_per_side  = PositiveFloat(2)      # Wheels per side
    axle_spacing_mm  = PositiveFloat(70)     # Spacing between axles
    wheelbase_span_mm= PositiveFloat(0)      # Total wheelbase span
```

### Wheel Parameters
```python
class ThisWheel(SpokeWheel):
    diameter  = PositiveFloat(90)       # Wheel diameter
    thickness = PositiveFloat(15)       # Wheel width
    outset    = PositiveFloat(10)       # Rim outset
```

## Tips for Best Results

### 1. Clear Reference Images
- Use well-lit, clear photos
- Multiple angles help (front, side, top)
- Include scale reference if possible

### 2. Specific User Intent
✅ Good:
```
"Make the chassis 40mm longer. Current length=280mm, target=320mm.
Keep width the same. Add one more wheel per side (2→3)."
```

❌ Vague:
```
"Make it bigger and add more wheels"
```

### 3. Provide Context
```
"Reference shows an off-road configuration with:
- Ground clearance increased (wheel z-offset should be +10mm)
- Wider wheelbase for stability on uneven terrain
- Reinforced chassis (thickness 6mm → 8mm)"
```

### 4. Iterative Refinement
Don't expect perfect results in one shot:
1. First pass: Major structural changes
2. Review and test
3. Second pass: Fine-tune proportions
4. Third pass: Detail adjustments

## Troubleshooting

### Issue: "VLM output wasn't valid Python"
- The VLM generated invalid syntax
- Check `generated/robot_base_vlm.reject_TIMESTAMP.txt` for raw output
- Try simplifying your prompt
- Provide clearer reference images

### Issue: "Generated code missing Rover or RobotBase class"
- The VLM didn't output the expected structure
- This is a safety check - the code won't be saved
- Try being more explicit: "Keep the existing Rover class structure"

### Issue: Generated code doesn't match reference
- The VLM made conservative or incorrect assumptions
- Provide more specific numerical targets in your prompt
- Include multiple reference images from different angles
- Add explicit constraints: "DO NOT change the electronics mount position"

## Advanced: Customizing the Prompt

You can modify `VLM_CODEGEN_PROMPT` in `optim.py` to:
- Add domain-specific constraints
- Include additional context about your robot's purpose
- Specify coding style preferences
- Add safety checks or validation rules

Example addition:
```python
VLM_CODEGEN_PROMPT = """
...existing prompt...

ADDITIONAL CONSTRAINTS:
- Wheel diameter must stay between 60mm and 120mm
- Always maintain length > width for the base
- Preserve all existing import statements
- Do not modify the electronics or sensor mounts
"""
```

## Output Structure

Successful response:
```json
{
  "ok": true,
  "module_path": "generated/robot_base_vlm.py",
  "backup_path": "generated/robot_base_vlm_1704123456.py",
  "code_length": 8542,
  "glb_updated": false,
  "message": "Generated robot_base.py - review and integrate manually if needed"
}
```

Error response:
```json
{
  "ok": false,
  "error": "VLM output wasn't valid Python: unexpected EOF",
  "reject_path": "generated/robot_base_vlm.reject_1704123456.txt",
  "raw_length": 4231
}
```

---

## Summary

The VLM code generation system:
1. ✅ Reads your current `robot_base.py` source via `_baseline_cqparts_source()`
2. ✅ Accepts reference images and user qualitative feedback
3. ✅ Outputs **Python code only** (no explanations)
4. ✅ Saves generated code with timestamped backups
5. ✅ Validates syntax before saving
6. ✅ Provides detailed error messages for debugging

This allows you to rapidly iterate on CAD designs by showing the VLM what you want and letting it modify the parametric Python code accordingly!

