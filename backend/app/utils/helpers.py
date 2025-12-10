"""
Utility helper functions
"""
import re
from typing import Any, Optional


def truthy(x) -> bool:
    """Convert value to boolean."""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "on")


def num(v, default=None) -> Optional[float]:
    """Extract numeric value from various types."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    dv = getattr(v, "default", None)
    if isinstance(dv, (int, float)):
        return float(dv)
    val = getattr(v, "value", None)
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(v)
    except Exception:
        s = str(v)
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return default


def strip_units_to_float(val) -> Optional[float]:
    """Strip units from string and return float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Remove common unit suffixes
    for unit in ["mm", "cm", "m", "in", "deg", "°", "rad"]:
        if s.lower().endswith(unit):
            s = s[:-len(unit)].strip()
    try:
        return float(s)
    except ValueError:
        return None


def clean_num(v) -> Optional[float]:
    """Clean and convert value to number."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

