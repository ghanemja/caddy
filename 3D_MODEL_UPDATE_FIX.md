# 3D Model Update Fix

## Problem

After clicking "Generate Code", the system showed "3D model updated with new code" but the 3D rendering didn't actually change.

## Root Cause

The system had two issues:

1. **Code not being used**: Generated code was saved to `generated/robot_base_vlm.py` but the GLB builder was still using the original `robot_base.py` import
2. **No module reload**: Python imports are static - the `Rover` class loaded at startup never changed

## Solution

### 1. Dynamic Module Loading

Added `_reload_rover_from_generated()` function (lines 3072-3124):
- Checks if `generated/robot_base_vlm.py` exists
- Dynamically loads it using `importlib.util.spec_from_file_location()`
- Extracts the `Rover` or `RobotBase` class
- Falls back to original if loading fails

### 2. Parameterized GLB Builder

Modified `build_rover_scene_glb_cqparts()` to accept optional `RoverClass` parameter (line 2748):
```python
def build_rover_scene_glb_cqparts(RoverClass=None) -> bytes:
    if RoverClass is None:
        RoverClass = Rover  # Use original by default
    
    print(f"Generating GLB via cqparts using {RoverClass.__name__}...")
    rv = RoverClass(...)  # Build with the specified class
```

### 3. Rebuild with Generated Code

Updated `_rebuild_and_save_glb()` to use generated code (lines 3127-3140):
```python
def _rebuild_and_save_glb(use_generated=False):
    # Reload from generated file if requested
    RoverClass = _reload_rover_from_generated() if use_generated else Rover
    
    # Build GLB with the new class
    glb = build_rover_scene_glb_cqparts(RoverClass=RoverClass)
    
    # Save to file
    with open(ROVER_GLB_PATH, "wb") as f:
        f.write(glb)
```

### 4. Codegen Endpoint Update

Changed `/codegen` endpoint to use generated code (line 1374):
```python
_rebuild_and_save_glb(use_generated=True)  # Now uses generated code!
```

### 5. UI Improvements

**HTML Template** (`templates/partials/_gen.html`):
- Reformatted to match other sections (header + section-body structure)
- Added status pill showing generation state
- Added code display textarea (readonly, monospace)
- Added copy button
- Shows file path where code is saved
- Consistent spacing and styling

**JavaScript** (`static/js/app.js`):
- Displays generated code in textarea
- Updates status during generation (orange → green/red)
- Adds loading indicators
- Logs progress to console
- Handles copy to clipboard
- Already had `refreshModel()` call - now it works!

## How It Works Now

### User Flow:
1. Upload reference image
2. (Optional) Add prompt
3. Click "Generate Code"

### System Flow:
```
Click "Generate Code"
  ↓
Send to /codegen endpoint
  ↓
VLM generates new code
  ↓
Save to generated/robot_base_vlm.py  
  ↓
Reload module from generated file  ← NEW!
  ↓
Build GLB using new Rover class    ← NEW!
  ↓
Save GLB to assets/rover.glb
  ↓
Return code + success to frontend
  ↓
Display code in textarea
  ↓
refreshModel() with ?ts=timestamp
  ↓
Frontend loads NEW GLB
  ↓
3D rendering updates! ✅
```

## Console Output (Backend)

```
[codegen] Building prompt with user_text: make it bigger
[codegen] Final prompt length: 15234 chars
[codegen] Calling VLM with 2 image(s)...
[vlm] Using fine-tuned model...
[vlm] Generating response...
[vlm] ✓ Got response from fine-tuned model: 3456 chars
[codegen] Got 3456 chars from VLM
[codegen] Extracted code length: 2134 chars
[codegen] Saved to generated/robot_base_vlm.py
[codegen] Attempting to rebuild GLB with new generated code...
[reload] Loading Rover from generated/robot_base_vlm.py...
[reload] ✓ Loaded Rover from generated code
Generating GLB via cqparts using Rover...
[cqparts] Building Rover...
[rebuild] ✓ Saved GLB to assets/rover.glb
[codegen] ✓ GLB rebuild successful with generated code
```

## Console Output (Frontend)

```
Generating code from VLM...
✓ Code generated successfully
Rebuilding 3D model...
Loading model…
Model loaded.
✓ 3D model updated with new code
```

## UI Changes

### Before:
```
[Codegen]                [Collapse]
  Reference image: [____]
  Optional snapshot: [____]
  [Generate Code]
```

### After:
```
[Code Generation]         [Collapse]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Code generated successfully] ✓

  Reference image: [____]
  Optional snapshot: [____]
  
  [Generate Code]
  
  Generated Code:          [Copy]
  ┌────────────────────────────────┐
  │ import cadquery as cq          │
  │ class Rover(cqparts.Assembly): │
  │     # ... generated code ...   │
  └────────────────────────────────┘
  Code saved to: generated/robot_base_vlm.py
```

## Testing

1. Start server: `python optim.py`
2. Upload reference image in "Code Generation" section
3. (Optional) Add prompt like "make the wheels bigger"
4. Click "Generate Code"
5. Wait for generation (~2-3 min first time, ~5s after)
6. **Observe**:
   - Status changes to "Generating code..."
   - Generated code appears in textarea
   - 3D model rebuilds and updates
   - Console shows progress messages
   - You can copy the code with one click

## Fallback Behavior

If generated code has errors or can't be loaded:
- System logs detailed error
- Falls back to original `robot_base.py`
- GLB still rebuilds (with original code)
- User sees error message
- Generated code is still saved for manual fixing

## Files Modified

1. `optim.py`:
   - Added `_reload_rover_from_generated()` function
   - Modified `build_rover_scene_glb_cqparts()` to accept RoverClass
   - Updated `_rebuild_and_save_glb()` to use generated code
   - Changed codegen endpoint to pass `use_generated=True`
   - Added `"code"` to JSON response

2. `templates/partials/_gen.html`:
   - Reformatted to match other sections
   - Added status pill
   - Added code display textarea
   - Added copy button

3. `static/js/app.js`:
   - Enhanced button handler
   - Added code display logic
   - Added status updates
   - Added copy functionality
   - Improved error handling

## Benefits

✅ **Actually works**: 3D model now updates with generated code
✅ **Visual feedback**: See the code that was generated
✅ **Easy copy**: One-click copy to clipboard
✅ **Consistent UI**: Matches other sections
✅ **Better errors**: Detailed fallback and error handling
✅ **Fast iteration**: Generate → See → Copy → Iterate

Your code generation workflow is now complete! 🎉

