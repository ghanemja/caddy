# VLM Code Generation - Complete Summary

## What Was Implemented

You now have a complete VLM-powered code generation system that can:

1. ✅ **Read your current `robot_base.py` source code** and pass it to the VLM
2. ✅ **Accept reference images** showing your target design
3. ✅ **Accept CAD snapshots** for comparison (orthogonal views)
4. ✅ **Accept user intent/feedback** as natural language
5. ✅ **Generate modified Python code** that matches your target design
6. ✅ **Output ONLY Python code** (no explanations, no markdown)
7. ✅ **Save with backups** (timestamped versions)
8. ✅ **Validate syntax** before saving

---

## How the Python Source is Passed to VLM

### The Key Function: `_baseline_cqparts_source()`

Located at **line 2042** in `optim.py`, this function:

```python
def _baseline_cqparts_source(max_chars: int = 20000) -> str:
    """
    Returns a compact string containing the actual Rover/RobotBase/wheel/stepper
    source from your project, trimmed to fit in the prompt.
    """
    chunks = []
    
    # 1. Get the entire robot_base module
    import robot_base as _rb
    mod_src = inspect.getsource(_rb)
    chunks.append(mod_src)
    
    # 2. Get individual classes
    from robot_base import Rover, RobotBase
    from wheel import BuiltWheel, SpokeWheel, SimpleWheel
    from pan_tilt import PanTilt
    
    # ... extract source for each ...
    
    # 3. Merge and clean
    merged = "\n\n# ----\n\n".join(chunks)
    merged = _strip_docstrings_and_comments(merged)  # Remove bloat
    
    # 4. Truncate if needed
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n# ... [truncated] ..."
    
    return merged
```

**This string becomes a variable** that gets injected into the VLM prompt!

### The Prompt Builder: `_build_codegen_prompt()`

Located at **line 955** in `optim.py`:

```python
def _build_codegen_prompt(ref_url, snapshot_url, user_text):
    # Get the source code as a string
    baseline_src = _baseline_cqparts_source()  # ← HERE!
    
    # Build the complete prompt
    parts = [
        VLM_CODEGEN_PROMPT,  # Instructions
        "\n<<<BASELINE_PYTHON_SOURCE>>>\n",
        baseline_src,  # ← Your robot_base.py as a string variable!
        "\n<<<END_BASELINE_PYTHON_SOURCE>>>\n",
        # ... current state, user intent, etc. ...
    ]
    
    return "".join(parts), images
```

---

## How to Use It

### Method 1: Python Script (Recommended)

```bash
# Basic usage
python codegen_helper.py reference.jpg --prompt "Make it wider"

# With snapshot comparison
python codegen_helper.py ref.jpg \
    --snapshot current_cad.png \
    --prompt "Match the proportions in the reference"

# Full example with detailed intent
python codegen_helper.py target_design.jpg \
    --snapshot current.png \
    --prompt "
    The reference shows:
    - Longer chassis (280mm → 320mm)
    - 3 wheels per side instead of 2
    - Wider wheelbase (170mm → 220mm)
    - Smaller wheel diameter (90mm → 75mm)
    " \
    --output ../robot_base_v2.py
```

### Method 2: Direct API Call

```python
import requests

with open('reference.jpg', 'rb') as ref:
    response = requests.post(
        'http://localhost:5160/codegen',
        files={'reference': ref},
        data={'prompt': 'Make the base 40mm longer'}
    )

result = response.json()
if result['ok']:
    print(f"Code saved to: {result['module_path']}")
```

### Method 3: curl

```bash
curl -X POST http://localhost:5160/codegen \
  -F "reference=@target.jpg" \
  -F "snapshot=@current.png" \
  -F "prompt=Add one more wheel per side"
```

---

## The Complete Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  1. YOU provide:                                             │
│     • Reference image (target design)                        │
│     • Current CAD snapshot (optional)                        │
│     • User intent text                                       │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  2. _baseline_cqparts_source() extracts:                     │
│     • Your current robot_base.py source                      │
│     • Wheel class definitions                                │
│     • Other component classes                                │
│     └──► Returns as STRING VARIABLE                          │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  3. _build_codegen_prompt() assembles:                       │
│     • VLM instructions                                       │
│     • <<<BASELINE_PYTHON_SOURCE>>>                          │
│     •     [robot_base.py as text]                           │
│     • <<<END_BASELINE_PYTHON_SOURCE>>>                      │
│     • Current CAD state (JSON)                               │
│     • User intent text                                       │
│     • Images (reference + snapshot)                          │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  4. VLM Model receives everything and:                       │
│     • Analyzes reference vs snapshot                         │
│     • Reads baseline Python source                           │
│     • Understands user intent                                │
│     • Generates MODIFIED robot_base.py                       │
│     └──► Outputs ONLY Python code                            │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  5. extract_python_module() validates:                       │
│     • Strips markdown fences                                 │
│     • Removes explanatory text                               │
│     • Validates Python syntax                                │
│     └──► Returns clean Python code                           │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  6. /codegen endpoint saves:                                 │
│     • generated/robot_base_vlm.py (latest)                   │
│     • generated/robot_base_vlm_TIMESTAMP.py (backup)         │
│     • Returns paths and statistics                           │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Code Locations in optim.py

| What | Where | Purpose |
|------|-------|---------|
| `VLM_CODEGEN_PROMPT` | Line 913 | VLM instructions (Python-only output) |
| `_baseline_cqparts_source()` | Line 2042 | **Extracts robot_base.py as string** |
| `_build_codegen_prompt()` | Line 955 | **Assembles prompt with source code** |
| `/codegen` endpoint | Line 1014 | API handler |
| `extract_python_module()` | Line 1193 | Validates and cleans VLM output |
| `call_vlm()` | Line 1112 | Sends request to Ollama/LLaVA |

---

## Example Prompt the VLM Sees

```
You are a CAD code generator that writes complete Python modules for CadQuery/cqparts.

=== YOUR TASK ===
You will receive:
1. REFERENCE IMAGE(s) - showing the desired robot rover design
2. CURRENT CAD SNAPSHOT - orthogonal views of the current CAD model
3. BASELINE PYTHON SOURCE - the current robot_base.py implementation
4. USER INTENT - human qualitative assessment and modification requests

Your job: Modify the baseline Python source to create a parametric rover 
that matches the reference image.

=== CRITICAL OUTPUT RULES ===
⚠️ Output ONLY Python code. NO explanations, NO markdown fences, NO backticks...
⚠️ The LAST line must be exactly: # END_OF_MODULE

================================================================================
<<<BASELINE_PYTHON_SOURCE>>>
# File: robot_base.py
# This is the current implementation you should modify

import cadquery as cq
import cqparts
from cqparts.params import PositiveFloat
...

class RobotBase(Lasercut):
    length     = PositiveFloat(250)
    width      = PositiveFloat(240)
    thickness  = PositiveFloat(6)
    chamfer    = PositiveFloat(30)
    ...

class Rover(cqparts.Assembly):
    length     = PositiveFloat(280)
    width      = PositiveFloat(170)
    wheels_per_side  = PositiveFloat(2)
    axle_spacing_mm  = PositiveFloat(70)
    ...

<<<END_BASELINE_PYTHON_SOURCE>>>
================================================================================

<<<CURRENT_CAD_STATE>>>
{
  "current_params": {
    "wheel_diameter": 90.0,
    "wheel_width": 15.0,
    "wheels_per_side": 2.0,
    ...
  },
  ...
}
<<<END_CURRENT_CAD_STATE>>>
================================================================================

<<<USER_INTENT_AND_FEEDBACK>>>
Make the base 40mm longer. Current length=280mm, target=320mm.
Keep width the same. Add one more wheel per side (2→3).
<<<END_USER_INTENT>>>
================================================================================

=== IMAGES PROVIDED ===
- Image 0: REFERENCE (target design)
- Image 1: CURRENT CAD SNAPSHOT (orthogonal views)

=== NOW OUTPUT YOUR MODIFIED robot_base.py ===
Remember: Python code ONLY, no markdown, no explanations.
End with: # END_OF_MODULE
```

---

## Expected VLM Output

The VLM should output something like:

```python
import cadquery as cq
import cqparts
from cqparts.params import PositiveFloat
from cqparts.display import render_props
from cqparts.constraint import Fixed, Coincident, Mate
from cqparts.utils.geometry import CoordSystem
from cqparts.search import register

from partref import PartRef
from manufacture import Lasercut
from motor_mount import MountedStepper
from cqparts_motors.stepper import Stepper
from wheel import SpokeWheel
from electronics import type1 as Electronics
from pan_tilt import PanTilt


class RobotBase(Lasercut):
    length     = PositiveFloat(320)      # CHANGED: 250 → 320
    width      = PositiveFloat(240)
    thickness  = PositiveFloat(6)
    chamfer    = PositiveFloat(30)
    _render    = render_props(template="wood")

    def make(self):
        base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
        base = base.edges("|Z and >X").chamfer(self.chamfer)
        return base
    
    # ... mate methods unchanged ...


class Rover(cqparts.Assembly):
    length     = PositiveFloat(320)          # CHANGED: 280 → 320
    width      = PositiveFloat(170)
    wheels_per_side  = PositiveFloat(3)      # CHANGED: 2 → 3
    axle_spacing_mm  = PositiveFloat(70)
    
    # ... rest unchanged ...

# END_OF_MODULE
```

**No explanations. No markdown. Just pure Python code.**

---

## Files Created

```
cqparts_bucket/
├── optim.py                          # [Modified] Main server with VLM integration
├── robot_base.py                     # [Unchanged] Your original source
├── codegen_helper.py                 # [New] Python CLI helper
├── VLM_CODEGEN_USAGE.md             # [New] Detailed usage guide
├── CODEGEN_SUMMARY.md               # [New] This summary
└── generated/                        # [New] Output directory
    ├── robot_base_vlm.py            # Latest generated code
    ├── robot_base_vlm_1704123456.py # Timestamped backup
    └── robot_base_vlm.reject_*.txt  # Failed attempts (debugging)
```

---

## Quick Start

### 1. Ensure server is running
```bash
cd /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
python optim.py
```

### 2. Prepare your images
- **Reference image**: Photo/sketch of target design
- **Snapshot**: Orthogonal views of current CAD model

### 3. Run code generation
```bash
# Simple version
python codegen_helper.py my_reference.jpg --prompt "Make it longer"

# Full version with comparison
python codegen_helper.py reference.jpg \
    --snapshot current_cad.png \
    --prompt "Target: 320mm long, 3 wheels/side, 75mm diameter"
```

### 4. Review generated code
```bash
# Check the output
cat generated/robot_base_vlm.py

# Compare with original
diff robot_base.py generated/robot_base_vlm.py
```

### 5. Integrate changes
```bash
# Option A: Copy manually after review
# Review first, then:
cp generated/robot_base_vlm.py robot_base.py

# Option B: Cherry-pick specific changes
# Edit robot_base.py manually with desired values
```

---

## Configuration

### VLM Endpoint
Edit in `optim.py`:
```python
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava-llama3:latest")
```

Or set environment variables:
```bash
export OLLAMA_URL="http://your-gpu-server:11434"
export OLLAMA_MODEL="llava:13b"
```

### Source Code Length Limit
```python
# In _baseline_cqparts_source()
baseline_src = _baseline_cqparts_source(max_chars=20000)  # Adjust as needed
```

---

## Tips for Best Results

### ✅ DO:
- Use clear, well-lit reference images
- Provide specific numerical targets ("280mm → 320mm")
- Include both reference and snapshot for comparison
- Be explicit about what should NOT change
- Start with major changes, then iterate for details

### ❌ DON'T:
- Use blurry or low-quality images
- Give vague instructions ("make it better")
- Expect perfect results on first try
- Modify too many things at once
- Forget to backup your original code

---

## Troubleshooting

### "Could not connect to server"
```bash
# Check if server is running
curl http://localhost:5160/mode

# Start server if needed
python optim.py
```

### "VLM output wasn't valid Python"
- Check `generated/robot_base_vlm.reject_*.txt`
- The VLM generated invalid syntax or added explanations
- Try simplifying your prompt
- Try better quality reference images

### "Generated code missing Rover class"
- The VLM didn't output expected structure
- Add to prompt: "Preserve all class definitions and structure"

### Generated code doesn't match reference
- VLM made conservative assumptions
- Be MORE specific in your prompt with exact numbers
- Provide multiple reference angles
- Iterate: run again with corrections

---

## Advanced Customization

### Modify the VLM Prompt
Edit `VLM_CODEGEN_PROMPT` in `optim.py` (line 913):

```python
VLM_CODEGEN_PROMPT = """
... existing instructions ...

ADDITIONAL CONSTRAINTS:
- Wheel diameter must be between 60mm and 120mm
- Length must always be greater than width
- Preserve all import statements exactly
- Do not modify mate_* methods
"""
```

### Add Validation Rules
Edit the `/codegen` endpoint:

```python
# After line 1077
if "class Rover" not in code_txt:
    return jsonify({"ok": False, "error": "Missing Rover class"}), 400

# Add custom checks
if "wheels_per_side" not in code_txt:
    return jsonify({"ok": False, "error": "Missing wheels_per_side param"}), 400
```

---

## Summary

You now have a complete system where:

1. **Your robot_base.py is extracted as a string variable** via `_baseline_cqparts_source()`
2. **This string is inserted into the VLM prompt** via `_build_codegen_prompt()`
3. **The VLM sees**: reference image + snapshot + Python source + your intent
4. **The VLM outputs**: Modified Python code ONLY (no explanations)
5. **The system validates and saves**: `generated/robot_base_vlm.py` with backups

All you need to do is:
```bash
python codegen_helper.py reference.jpg --prompt "Your intent here"
```

Then review and integrate the generated code! 🚀

