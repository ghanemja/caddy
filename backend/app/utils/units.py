"""
Unit conversion utilities
Functions for handling units in parameter values
"""
import re
from typing import Optional
from app.utils.helpers import strip_units_to_float as _strip_units_to_float


def mm_from_value_token(tok: str) -> str:
    """
    Convert a value token like "2.5m" or "2500mm" to millimeters as a string.
    Returns the numeric value in mm as a string, or the original token if parsing fails.
    """
    if not tok or not isinstance(tok, str):
        return tok
    tok = tok.strip()
    # Try to extract number and unit
    m = re.match(r"(-?\d+(?:\.\d+)?)\s*([a-z]+)?", tok, re.IGNORECASE)
    if not m:
        return tok
    num_str, unit = m.groups()
    try:
        val = float(num_str)
    except Exception:
        return tok
    unit = (unit or "").lower()
    # Convert to mm
    if unit in ("m", "meter", "meters", "metre", "metres"):
        val *= 1000.0
    elif unit in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        val *= 10.0
    elif unit in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        pass  # already mm
    elif unit in ("in", "inch", "inches"):
        val *= 25.4
    elif unit in ("ft", "foot", "feet"):
        val *= 304.8
    else:
        # Unknown unit, assume mm
        pass
    return f"{val:.3f}".rstrip("0").rstrip(".")


def repair_units_in_json_text(block: str) -> str:
    """
    Repair units in JSON text by converting values like "2.5m" to "2500".
    This helps parse JSON that has unit strings in numeric fields.
    """
    if not block:
        return block
    
    # Pattern: find numeric values with units in JSON value positions
    # Match: "key": "2.5m" or "key": 2.5m (without quotes)
    def replace_units(match):
        full_match = match.group(0)
        value_part = match.group(1)  # The value part
        # Try to convert
        try:
            converted = mm_from_value_token(value_part)
            # Replace the value in the original match
            return full_match.replace(value_part, converted)
        except Exception:
            return full_match
    
    # Pattern to match JSON values with units
    # Matches: "key": "2.5m" or "key": 2.5m
    pattern = r'(":\s*")(-?\d+(?:\.\d+)?\s*[a-z]+)(")'
    block = re.sub(pattern, lambda m: f'{m.group(1)}{mm_from_value_token(m.group(2))}{m.group(3)}', block, flags=re.IGNORECASE)
    
    # Also handle unquoted numeric values with units (less common but possible)
    pattern2 = r'(":\s*)(-?\d+(?:\.\d+)?\s*[a-z]+)(\s*[,}])'
    block = re.sub(pattern2, lambda m: f'{m.group(1)}"{mm_from_value_token(m.group(2))}"{m.group(3)}', block, flags=re.IGNORECASE)
    
    return block

