"""
General API routes blueprint
"""
from flask import Blueprint, request, jsonify
from app.services.state_service import get_state, reset_state, apply_changes
from app.services.cad_service import get_glb_path
import os
import sys

bp = Blueprint("api", __name__)

# Import from legacy optim.py for now
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


@bp.get("/state")
def state():
    """Get current application state."""
    from optim import _ensure_initial_history
    _ensure_initial_history()
    which = (request.args.get("which") or "all").lower()
    return jsonify(get_state(which))


@bp.post("/state/reset")
def state_reset():
    """Reset state to initial values."""
    result = reset_state()
    return jsonify(result)


@bp.post("/apply")
def apply():
    """Apply changes to the CAD model."""
    from optim import _parse_apply_request
    changes, excerpt = _parse_apply_request()
    if not changes:
        return jsonify({"ok": False, "error": "no changes provided"}), 400
    
    status_code, response = apply_changes(changes, excerpt)
    return jsonify(response), status_code


@bp.get("/mode")
def mode():
    """Get current mode."""
    from optim import USE_CQPARTS
    return jsonify({"mode": "cqparts" if USE_CQPARTS else "legacy"})


@bp.post("/label")
def label():
    """Set label for a component."""
    from optim import STATE
    data = request.get_json() or {}
    key = data.get("key")
    label = data.get("label")
    if key and label:
        if "labels" not in STATE:
            STATE["labels"] = {}
        STATE["labels"][key] = label
    return jsonify({"ok": True})


@bp.get("/labels")
def labels():
    """Get all labels."""
    from optim import STATE
    return jsonify(STATE.get("labels", {}))


@bp.get("/model/glb")
def model_glb():
    """Get current GLB model file."""
    from flask import send_file
    glb_path = get_glb_path()
    if not os.path.exists(glb_path):
        return jsonify({"ok": False, "error": "GLB not found"}), 404
    return send_file(glb_path, mimetype="model/gltf-binary")


@bp.post("/undo")
def undo():
    """Undo last change."""
    from optim import HISTORY, H_PTR, _restore, _rebuild_and_save_glb
    if H_PTR > 0:
        H_PTR -= 1
        _restore(HISTORY[H_PTR])
        _rebuild_and_save_glb()
        return jsonify({"ok": True, "ptr": H_PTR})
    return jsonify({"ok": False, "error": "nothing to undo"}), 400


@bp.post("/redo")
def redo():
    """Redo last undone change."""
    from optim import HISTORY, H_PTR, _restore, _rebuild_and_save_glb
    if H_PTR < len(HISTORY) - 1:
        H_PTR += 1
        _restore(HISTORY[H_PTR])
        _rebuild_and_save_glb()
        return jsonify({"ok": True, "ptr": H_PTR})
    return jsonify({"ok": False, "error": "nothing to redo"}), 400


@bp.get("/params")
def params():
    """Get current parameters."""
    from optim import CURRENT_PARAMS
    return jsonify(CURRENT_PARAMS)

