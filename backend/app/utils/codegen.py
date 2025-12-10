"""
Code generation utilities
Functions for normalizing and extracting Python code from VLM output
"""
import re
import ast
from typing import Optional


def normalize_generated_code(code: str) -> str:
    """
    Normalize/fix common errors in VLM-generated code.
    
    Common issues:
    - Markdown fences (```python ... ```)
    - Extra explanations before/after code
    - Missing imports
    - Syntax errors
    """
    if not code:
        return ""
    
    # Remove markdown code fences
    code = re.sub(r'^```(?:python|py)?\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```(?:python|py)?', '', code)
    code = re.sub(r'```$', '', code)
    
    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        r'^Here (?:is|are) (?:the|a) (?:modified|updated|generated) code:?\s*\n',
        r'^Here (?:is|are) (?:the|a) (?:code|implementation):?\s*\n',
        r'^```python\s*\n',
        r'^```\s*\n',
    ]
    for prefix in prefixes_to_remove:
        code = re.sub(prefix, '', code, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove trailing explanations
    suffixes_to_remove = [
        r'\n```\s*$',
        r'\n(?:Note|Note:|This code|The code).*$',
    ]
    for suffix in suffixes_to_remove:
        code = re.sub(suffix, '', code, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Fix common indentation issues
    lines = code.split('\n')
    if lines:
        # Find first non-empty line to determine base indentation
        first_line_idx = next((i for i, line in enumerate(lines) if line.strip()), 0)
        if first_line_idx < len(lines):
            first_line = lines[first_line_idx]
            base_indent = len(first_line) - len(first_line.lstrip())
            # Remove excessive indentation
            if base_indent > 0:
                lines = [line[base_indent:] if len(line) > base_indent else line 
                        for line in lines]
                code = '\n'.join(lines)
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    return code


def extract_python_module(text: str) -> str:
    """
    Extract the largest contiguous region of Python code from text.
    Prefers fenced blocks, else heuristics from first code-ish line.
    """
    if not text:
        return ""
    
    # Try to find fenced code blocks first
    fence_pattern = r'```(?:python|py)?\s*\n(.*?)\n```'
    matches = re.findall(fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        # Return the longest match
        return max(matches, key=len).strip()
    
    # No fenced blocks, try to find code by heuristics
    lines = text.split('\n')
    code_start = None
    
    # Find start: look for shebang, import, from, class, def, @decorator, if __name__
    code_indicators = [
        r'^#!/usr/bin/env python',
        r'^#!.*python',
        r'^import\s+',
        r'^from\s+\w+\s+import',
        r'^class\s+\w+',
        r'^def\s+\w+',
        r'^@\w+',
        r'^if\s+__name__',
    ]
    
    for i, line in enumerate(lines):
        if any(re.match(pattern, line.strip()) for pattern in code_indicators):
            code_start = i
            break
    
    if code_start is None:
        return ""
    
    # Find end: look for blank line followed by non-code, or end of text
    code_end = len(lines)
    for i in range(code_start + 1, len(lines)):
        line = lines[i].strip()
        # Stop if we hit a markdown header, explanation text, or multiple blank lines
        if line.startswith('#') and any(word in line.lower() for word in ['note', 'example', 'warning']):
            if i > code_start + 10:  # Only stop if we have substantial code
                code_end = i
                break
        elif line == '' and i < len(lines) - 1:
            # Check if next line looks like explanation
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if next_line and not any(re.match(pattern, next_line) for pattern in code_indicators):
                if not next_line.startswith(' ') and not next_line.startswith('\t'):
                    code_end = i
                    break
    
    extracted = '\n'.join(lines[code_start:code_end])
    
    # Validate it's valid Python
    try:
        ast.parse(extracted)
        return extracted.strip()
    except SyntaxError:
        # Try to fix common issues and retry
        fixed = normalize_generated_code(extracted)
        try:
            ast.parse(fixed)
            return fixed
        except SyntaxError:
            # Return what we have anyway - let the caller handle validation
            return extracted.strip()


def normalize_generated_code_full(code: str) -> str:
    """
    Full normalization with advanced fixes for VLM-generated code.
    This is a more comprehensive version that handles many edge cases.
    """
    if not code:
        return ""
    
    original_code = code
    fixes_applied = []
    
    # Remove markdown fences
    code = normalize_generated_code(code)
    
    # Advanced fixes for common VLM errors
    # Fix hyphens in attribute names
    code = re.sub(r'self\.([a-z]+)-([a-z]+)', r'self.\1_\2', code)
    
    # Fix missing self. prefix on common attributes
    common_attrs = ['length', 'width', 'height', 'diameter', 'thickness']
    for attr in common_attrs:
        pattern = rf'(?<!self\.)(?<!\.){attr}\s*='
        code = re.sub(pattern, f'self.{attr} =', code)
    
    # Fix undefined 'offsets' variable
    if 'offsets' in code and 'self.offsets' not in code:
        code = re.sub(r'\boffsets\b', 'self.offsets', code)
    
    return code


def normalize_generated_code_advanced(code: str) -> str:
    """
    Advanced normalization with CAD-specific fixes.
    This extends the basic normalize_generated_code from codegen.py.
    """
    # Start with basic normalization
    code = normalize_generated_code(code)
    
    print("[normalize] Applying CAD-specific fixes...")
    fixes_applied = []
    
    # CAD-specific fixes (keep these here as they're domain-specific)
    required_imports = [
        "import cadquery as cq",
        "import cqparts",
        "from cqparts.params import PositiveFloat",
        "from cqparts.display import render_props",
        "from cqparts.constraint import Fixed, Coincident, Mate",
        "from cqparts.utils.geometry import CoordSystem",
        "from cqparts.search import register",
        "from partref import PartRef",
        "from manufacture import Lasercut",
        "from motor_mount import MountedStepper",
        "from cqparts_motors.stepper import Stepper",
        "from wheel import SpokeWheel",
        "from electronics import type1 as Electronics",
        "from pan_tilt import PanTilt",
    ]
    
    # Check if file is missing shebang and imports
    if not code.strip().startswith("#!/usr/bin/env python3"):
        missing_imports = [imp for imp in required_imports if imp not in code]
        if missing_imports:
            header = "#!/usr/bin/env python3\n\n" + "\n".join(required_imports) + "\n\n"
            code = header + code
            fixes_applied.append(f"Added missing imports ({len(missing_imports)} imports restored)")
            print(f"[normalize] ✗ VLM truncated file - restored {len(missing_imports)} missing imports")
    
    # Fix RobotBase class if missing
    if "class RobotBase" not in code and "class ThisWheel" in code:
        robot_base_class = '''class RobotBase(Lasercut):
    length = PositiveFloat(280)
    width = PositiveFloat(170)
    chamfer = PositiveFloat(55)
    thickness = PositiveFloat(6)

    def make(self):
        base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
        base = base.edges("|Z and >X").chamfer(self.chamfer)
        return base

    def mate_back(self, offset=5):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, 0, self.thickness),
                xDir=(1, 0, 0),
                normal=(0, 0, 1),
            ),
        )

    def mate_front(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(self.length / 2 - offset, 0, self.thickness),
                xDir=(1, 0, 0),
                normal=(0, 0, 1),
            ),
        )

    def mate_RL(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, self.width / 2, 0),
                xDir=(1, 0, 0),
                normal=(0, 0, -1),
            ),
        )

    def mate_RR(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, -self.width / 2, 0),
                xDir=(-1, 0, 0),
                normal=(0, 0, -1),
            ),
        )

'''
        code = code.replace("class ThisWheel", robot_base_class + "class ThisWheel")
        fixes_applied.append("Added missing RobotBase class")
        print(f"[normalize] ✗ VLM skipped RobotBase - restored it")
    
    # Additional CAD-specific fixes
    hyphen_fixes = {
        r'\.wheelbase_span-mm': '.wheelbase_span_mm',
        r'\.axle_spacing-mm': '.axle_spacing_mm',
        r'\.wheel_z_offset-mm': '.wheel_z_offset_mm',
        r'\.wheel-diameter': '.wheel_diameter',
        r'\.wheel-width': '.wheel_width',
    }
    
    for pattern, replacement in hyphen_fixes.items():
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            fixes_applied.append(f"Fixed hyphenated attribute: {pattern} → {replacement}")
    
    # Fix undefined 'offsets' variable
    if re.search(r'for\s+i,\s+off\s+in\s+enumerate\(offsets\)', code):
        code = re.sub(
            r'for\s+i,\s+off\s+in\s+enumerate\(offsets\)',
            'for i, off in enumerate(self._axle_offsets())',
            code
        )
        fixes_applied.append("Fixed undefined 'offsets' → 'self._axle_offsets()'")
    
    # Fix _axle_offsets to allow 0 wheels
    if 'n = max(1, int(round(float(self.wheels_per_side))))' in code:
        code = code.replace(
            'n = max(1, int(round(float(self.wheels_per_side))))',
            'n = max(0, int(round(float(self.wheels_per_side))))'
        )
        fixes_applied.append("Fixed _axle_offsets to allow 0 wheels")
    
    # Remove trailing incomplete lines
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('cq.display.') or \
           (line.strip().startswith('register(') and 'model=' not in line):
            continue
        cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)
    
    # Detect and truncate VLM hallucinations
    lines = code.split('\n')
    class_names_seen = {}
    truncate_at = None
    
    for i, line in enumerate(lines):
        class_match = re.match(r'^class\s+(\w+)', line)
        if class_match:
            class_name = class_match.group(1)
            if class_name in class_names_seen:
                truncate_at = class_names_seen[class_name]
                print(f"[normalize] ✗ Detected VLM hallucination: class '{class_name}' repeated")
                fixes_applied.append(f"Truncated hallucination: repeated class '{class_name}'")
                break
            else:
                class_names_seen[class_name] = i
    
    if truncate_at is not None:
        code = '\n'.join(lines[:truncate_at])
        if code and not code.endswith('\n'):
            code += '\n'
        code += '\n# === End of generated code ===\n'
    
    if fixes_applied:
        print(f"[normalize] Applied {len(fixes_applied)} CAD-specific fixes")
    
    return code

