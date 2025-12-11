"""
Code inspection utilities
Functions for introspecting classes and extracting source code.
"""
import os
import re
import inspect
from typing import Dict, Any, Optional
from pathlib import Path


def strip_docstrings_and_comments(src: str) -> str:
    """Very light scrubbing to keep the prompt small."""
    # (keeps string literals inside code; removes triple-quoted docstrings and # lines)
    src = re.sub(r'(?s)^\s*("""|\'\'\').*?\1\s*', "", src)  # file header docstring
    src = re.sub(r'(?s)([^f])("""|\'\'\').*?\2', r"\1", src)  # other docstrings (rough)
    src = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    return src


def try_get_source(obj) -> Optional[str]:
    """Try to get source code for an object."""
    try:
        return inspect.getsource(obj)
    except Exception:
        return None


def introspect_params_from_cls(cls) -> Dict[str, Any]:
    """Extract parameter information from a class."""
    d: Dict[str, Any] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            mod = getattr(getattr(val, "__class__", object), "__module__", "")
            if "cqparts.params" in mod:
                d[name] = str(val)
        except Exception:
            pass
    for name in dir(cls):
        if name in d or name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            if isinstance(val, (int, float)):
                d[name] = val
        except Exception:
            pass
    return d


def baseline_cqparts_source(base_dir: str, max_chars: int = 20000) -> str:
    """
    Returns a compact string containing the *actual* Rover / RobotBase / wheel / stepper
    source from your project, trimmed to fit in the prompt.
    """
    import sys
    import importlib
    
    chunks = []

    # Try direct file read first (most reliable)
    # Check in app/models/cad first, then root
    robot_base_path = os.path.join(base_dir, "app", "models", "cad", "robot_base.py")
    if not os.path.exists(robot_base_path):
        robot_base_path = os.path.join(base_dir, "robot_base.py")
    if os.path.exists(robot_base_path):
        print(f"[baseline_source] Reading robot_base.py directly from {robot_base_path}")
        try:
            with open(robot_base_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content:
                    chunks.append(f"# === robot_base.py ===\n{content}")
                    print(f"[baseline_source] ✓ Read {len(content)} chars from robot_base.py")
        except Exception as e:
            print(f"[baseline_source] ✗ Failed to read robot_base.py: {e}")

    # Try inspect.getsource as backup
    if not chunks:
        print("[baseline_source] Trying inspect.getsource method...")
        try:
            import robot_base as _rb
            mod_src = inspect.getsource(_rb)
            chunks.append(mod_src)
            print(f"[baseline_source] ✓ Got {len(mod_src)} chars via inspect.getsource")
        except Exception as e:
            print(f"[baseline_source] ✗ inspect.getsource failed: {e}")

    # Try class-by-class extraction
    if not chunks:
        print("[baseline_source] Trying class-by-class extraction...")
        try:
            from robot_base import Rover, RobotBase
            for obj in (Rover, RobotBase):
                s = try_get_source(obj)
                if s:
                    chunks.append(s)
                    print(f"[baseline_source] ✓ Got {obj.__name__}")
        except Exception as e:
            print(f"[baseline_source] ✗ Class extraction failed: {e}")

    # Also try to get wheel classes
    try:
        from wheel import BuiltWheel, SpokeWheel, SimpleWheel
        for obj in (BuiltWheel, SpokeWheel, SimpleWheel):
            s = try_get_source(obj)
            if s:
                chunks.append(f"# === {obj.__name__} ===\n{s}")
    except Exception as e:
        print(f"[baseline_source] Note: Could not get wheel classes: {e}")

    # Fallback: try to open files by searching sys.modules
    if not chunks:
        print("[baseline_source] Trying sys.modules fallback...")
        for modname in ("robot_base", "wheel", "pan_tilt"):
            try:
                m = sys.modules.get(modname) or importlib.import_module(modname)
                path = inspect.getsourcefile(m) or inspect.getfile(m)
                if path and os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        chunks.append(f"# === {modname} ===\n{content}")
                        print(f"[baseline_source] ✓ Read {modname} from {path}")
            except Exception as e:
                print(f"[baseline_source] Could not read {modname}: {e}")

    # Merge and clean
    if not chunks:
        error_msg = "# ERROR: Could not extract robot_base.py source code\n"
        error_msg += f"# Tried path: {robot_base_path}\n"
        error_msg += "# Please ensure robot_base.py exists in the same directory as optim.py\n"
        print(f"[baseline_source] ✗ FAILED - no source code extracted!")
        return error_msg
    
    merged = "\n\n# ----\n\n".join(chunks)
    print(f"[baseline_source] Total merged: {len(merged)} chars before cleaning")
    
    # Light cleaning (optional - may want to keep comments for VLM)
    # merged = strip_docstrings_and_comments(merged)
    
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n# ... [truncated for prompt] ..."
        print(f"[baseline_source] Truncated to {max_chars} chars")
    
    print(f"[baseline_source] ✓ Final output: {len(merged)} chars")
    return merged

