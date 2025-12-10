"""
Parameter normalization utilities
Functions for normalizing and parsing CAD parameter changes
"""
import re
from typing import Dict, Any, Optional, List
from app.utils.helpers import num as _num, clean_num as _clean_num


def intent_to_changes(text: str) -> list[dict]:
    """
    Converts natural-language text like 'add 6 wheels' or
    'add six wheels on the base and align midpoints' into
    structured change objects for /apply.
    """
    if not text:
        return []
    s = text.strip().lower()

    # number words
    num_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    # detect numeric count
    m = re.search(r"\badd\s+(\d+)\s+wheels?\b", s)
    count = None
    if m:
        count = int(m.group(1))
    else:
        m2 = re.search(r"\badd\s+([a-z]+)\s+wheels?\b", s)
        if m2:
            count = num_words.get(m2.group(1))

    changes = []
    if count:
        changes.append(
            {
                "target_component": "wheel",
                "action": "add",
                "parameters": {"count": count, "wheels_per_side": max(1, count // 2)},
            }
        )

    # midpoint → zero z-offset
    if "midpoint" in s and ("join" in s or "touch" in s) and "base" in s:
        changes.append(
            {
                "target_component": "wheel",
                "action": "modify",
                "parameters": {"wheel_z_offset_mm": 0},
            }
        )

    return changes


def normalize_params(target: str, action: str, params: dict) -> dict:
    """Normalize parameters for a given target and action."""
    p = {}
    params = params or {}
    tgt = (target or "").lower()

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "on", "y")

    # Aliases (imported from run.py)
    from run import TARGET_ALIASES, ACTION_ALIASES
    tgt = TARGET_ALIASES.get(tgt, tgt)
    action = ACTION_ALIASES.get(action, action)

    # Normalize based on target and action
    if tgt in ("wheel", "wheels"):
        if "diameter" in params:
            p["wheel_diameter"] = _num(params["diameter"])
        if "width" in params:
            p["wheel_width"] = _num(params["width"])
        if "count" in params or "wheels_per_side" in params:
            p["wheels_per_side"] = int(_num(params.get("wheels_per_side") or params.get("count"), 2))
        if "z_offset" in params or "z" in params:
            p["wheel_z_offset_mm"] = _num(params.get("z_offset") or params.get("z"))
    elif tgt in ("pan_tilt", "pan-tilt", "pantilt", "sensors"):
        for axis in ("x", "y", "z"):
            key = f"pan_tilt_offset_{axis}"
            if axis in params or key in params:
                p[key] = _num(params.get(key) or params.get(axis))
    elif tgt in ("rover", "base", "chassis", "roverbase"):
        if "yaw" in params or "rover_yaw_deg" in params:
            p["rover_yaw_deg"] = _num(params.get("rover_yaw_deg") or params.get("yaw"))
        if "mirror_lr" in params:
            p["mirror_lr"] = _as_bool(params["mirror_lr"])

    # Copy any remaining params that match known keys
    known_keys = {
        "wheel_diameter", "wheel_width", "wheels_per_side", "wheel_z_offset_mm",
        "pan_tilt_offset_x", "pan_tilt_offset_y", "pan_tilt_offset_z",
        "rover_yaw_deg", "axle_spacing_mm", "wheelbase_span_mm", "mirror_lr", "hide_wheels"
    }
    for k, v in params.items():
        if k in known_keys:
            p[k] = _clean_num(v) if k != "mirror_lr" and k != "hide_wheels" else _as_bool(v)

    return p


def normalize_change(ch: dict) -> Optional[dict]:
    """Normalize a single change object."""
    if not isinstance(ch, dict):
        return None
    
    action = (ch.get("action") or "").strip().lower()
    target = (ch.get("target_component") or ch.get("target") or "").strip().lower()
    params = ch.get("parameters") or ch.get("params") or {}
    
    if not action or not target:
        return None
    
    # Normalize parameters
    norm_params = normalize_params(target, action, params)
    
    return {
        "action": action,
        "target_component": target,
        "parameters": norm_params,
    }


def coerce_changes(payload: Any) -> List[dict]:
    """Coerce various input formats to a list of change dicts."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [c for c in payload if isinstance(c, dict)]
    return []


def mk_change(action: str, target: str, params: dict) -> dict:
    """Create a normalized change object."""
    ch = {"action": action, "target_component": target, "parameters": params}
    norm = normalize_change(ch)
    return norm or ch

