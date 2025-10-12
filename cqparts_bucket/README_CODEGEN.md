# VLM Code Generation for Parametric CAD

## Quick Start

```bash
# 1. Start the server
python optim.py

# 2. Generate code from reference image
python codegen_helper.py reference.jpg --prompt "Make the base 40mm longer"

# 3. Review generated code
cat generated/robot_base_vlm.py

# 4. Integrate changes
# Review the changes and copy what you need to robot_base.py
```

## What This Does

🎯 **Takes**: Reference image + User feedback + Current robot_base.py source  
🤖 **Generates**: Modified robot_base.py that matches your target design  
💾 **Saves**: `generated/robot_base_vlm.py` (with timestamped backups)

## How It Works

1. **Your Python file is extracted as a string** via `inspect.getsource(robot_base)`
2. **This string is inserted into the VLM prompt** along with your images and intent
3. **The VLM reads and modifies** the Python code to match your reference
4. **Output is pure Python code** (no explanations, ready to use)

## Key Files

- `optim.py` - Main server with `/codegen` endpoint
- `codegen_helper.py` - Easy-to-use Python CLI
- `VLM_CODEGEN_USAGE.md` - Detailed usage guide
- `CODEGEN_SUMMARY.md` - Complete implementation summary
- `FLOW_DIAGRAM.txt` - Visual flow diagram

## Key Functions in optim.py

| Function | Line | What It Does |
|----------|------|--------------|
| `_baseline_cqparts_source()` | 2042 | **Extracts robot_base.py as string** |
| `_build_codegen_prompt()` | 955 | **Assembles prompt with source code** |
| `/codegen` endpoint | 1014 | API handler |

## Examples

### Basic
```bash
python codegen_helper.py reference.jpg --prompt "Increase length to 320mm"
```

### With Snapshot Comparison
```bash
python codegen_helper.py ref.jpg --snapshot current.png \
  --prompt "Make chassis match reference proportions"
```

### With Detailed Intent
```bash
python codegen_helper.py target.jpg --prompt "
  Target changes:
  - Length: 280mm → 320mm
  - Wheels per side: 2 → 3
  - Wheel diameter: 90mm → 75mm
  - Wider wheelbase for stability
"
```

## Configuration

Set VLM endpoint:
```bash
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="llava-llama3:latest"
```

## Output

Generated files:
```
generated/
├── robot_base_vlm.py              # Latest version
├── robot_base_vlm_1704123456.py   # Timestamped backup
└── robot_base_vlm.reject_*.txt    # Failed attempts (for debugging)
```

## Tips

✅ **DO**:
- Use clear, well-lit reference images
- Provide specific numerical targets
- Include both reference and snapshot for comparison
- Review generated code before using

❌ **DON'T**:
- Use blurry images
- Give vague instructions ("make it better")
- Expect perfection on first try
- Skip reviewing the generated code

## Troubleshooting

**Server not responding?**
```bash
curl http://localhost:5160/mode  # Check if running
python optim.py  # Start if needed
```

**Invalid Python output?**
- Check `generated/robot_base_vlm.reject_*.txt`
- Simplify your prompt
- Use better quality images

**Generated code doesn't match?**
- Be more specific with numbers
- Provide multiple reference angles
- Iterate: run again with corrections

## Need Help?

1. Read `VLM_CODEGEN_USAGE.md` for detailed examples
2. Read `CODEGEN_SUMMARY.md` for implementation details
3. Check `FLOW_DIAGRAM.txt` for visual flow
4. Look at server logs for debugging info

## Example Workflow

```bash
# 1. Capture your target design
# (photo, sketch, or render)

# 2. Generate code
python codegen_helper.py target_design.jpg \
  --snapshot current_cad.png \
  --prompt "
    Reference shows longer, narrower chassis.
    Target: 320x180mm (current: 280x170mm)
    Add 1 more wheel per side.
  "

# 3. Check output
cat generated/robot_base_vlm.py

# 4. Compare changes
diff robot_base.py generated/robot_base_vlm.py

# 5. Integrate
# Review and copy relevant changes to robot_base.py
# OR rename generated file to replace original
```

---

**Questions?** Check the detailed docs:
- `VLM_CODEGEN_USAGE.md` - Usage guide
- `CODEGEN_SUMMARY.md` - Implementation details
- `FLOW_DIAGRAM.txt` - Visual reference

