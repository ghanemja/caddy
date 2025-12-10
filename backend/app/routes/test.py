"""
Test routes blueprint
For testing CAD operations.
"""
from flask import Blueprint, request, jsonify

bp = Blueprint("test", __name__)


@bp.post("/rotate")
def test_rotate():
    """Test rotation operation."""
    # Lazy import to avoid circular dependency
    from run import _mk_change, _apply_changes_list
    
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or "rover").lower()
    angle = data.get("angle_deg")
    orient = data.get("orientation_deg")
    params = {}
    if angle is not None:
        params["rover_yaw_deg"] = float(angle)
    if orient is not None:
        if isinstance(orient, (list, tuple)) and len(orient) >= 3:
            params["rx"] = float(orient[0])
            params["ry"] = float(orient[1])
            params["rz"] = float(orient[2])
    ch = _mk_change("rotate", target, params)
    code, payload = _apply_changes_list([ch], None)
    return jsonify(payload), code


@bp.post("/translate")
def test_translate():
    """Test translation operation."""
    # Lazy import to avoid circular dependency
    from run import _mk_change, _apply_changes_list
    
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or "wheel").lower()
    pos = data.get("position_mm")
    params = {}
    if pos is not None and isinstance(pos, (list, tuple)) and len(pos) >= 3:
        params["x"] = float(pos[0])
        params["y"] = float(pos[1])
        params["z"] = float(pos[2])
    ch = _mk_change("translate", target, params)
    code, payload = _apply_changes_list([ch], None)
    return jsonify(payload), code


@bp.post("/modify")
def test_modify():
    """Test modify operation."""
    # Lazy import to avoid circular dependency
    from run import _mk_change, _apply_changes_list
    
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or "wheel").lower()
    params = data.get("parameters") or {}
    ch = _mk_change("modify", target, params)
    code, payload = _apply_changes_list([ch], None)
    return jsonify(payload), code

