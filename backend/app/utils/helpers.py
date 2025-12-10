"""
Utility helper functions
"""
import re
import io
from typing import Any, Optional
import trimesh
from cadquery import exporters


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
            try:
                return float(m.group(1))
            except Exception:
                pass
        return default


def strip_units_to_float(val) -> Optional[float]:
    """Strip units from string and return float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # Remove all non-numeric characters except eE+-.
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def clean_num(v) -> Optional[float]:
    """Clean and convert value to number."""
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def percent_to_abs(token, base) -> Optional[float]:
    """Convert percentage token to absolute value."""
    if token is None or base is None:
        return None
    try:
        s = str(token).strip()
        if s.endswith("%"):
            return float(base) * (float(s[:-1]) / 100.0)
        f = float(s)
        return float(base) * f if 0.0 < f <= 2.0 else f
    except Exception:
        return None


def cq_to_trimesh(obj, tol=0.6):
    """Convert CadQuery object to trimesh."""
    try:
        stl_txt = exporters.toString(obj, "STL", tolerance=tol).encode("utf-8")
        m = trimesh.load(io.BytesIO(stl_txt), file_type="stl")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        return m
    except Exception as e:
        print("[mesh] STL export failed:", e)
        return None

