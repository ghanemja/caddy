"""
General API routes blueprint
"""
from flask import Blueprint, request, jsonify
from app.services.state_service import get_state, reset_state, apply_changes
from app.services.cad_service import get_glb_path
import os
import sys

bp = Blueprint("api", __name__)

# Import from run.py (legacy optim.py renamed)
# All functions will be gradually moved to service modules


@bp.get("/state")
def state():
    """Get current application state."""
    from run import _ensure_initial_history, HISTORY, H_PTR, INIT_SNAPSHOT, _snapshot, CONTEXT, COMPONENT_REGISTRY, PENDING_ADDS
    _ensure_initial_history()
    which = (request.args.get("which") or "all").lower()
    payload = {
        "initial": HISTORY[0] if HISTORY else INIT_SNAPSHOT or _snapshot(),
        "current": _snapshot(),
        "context": CONTEXT,
        "known_classes": sorted(list(COMPONENT_REGISTRY.keys())),
        "history": HISTORY[: H_PTR + 1],
        "pending_adds": list(PENDING_ADDS),
    }
    if which in payload:
        return jsonify({"ok": True, which: payload[which]})
    return jsonify({"ok": True, "state": payload})


@bp.post("/state/reset")
def state_reset():
    """Reset state to initial values."""
    from run import HISTORY, H_PTR, CURRENT_PARAMS, PENDING_ADDS, STATE, _rebuild_and_save_glb, _snapshot
    try:
        global HISTORY, H_PTR, CURRENT_PARAMS, PENDING_ADDS, STATE
        for k in list(CURRENT_PARAMS.keys()):
            CURRENT_PARAMS[k] = None
        PENDING_ADDS.clear()
        STATE["selected_parts"] = []
        HISTORY = [{k: None for k in CURRENT_PARAMS.keys()}]
        H_PTR = 0
        try:
            _rebuild_and_save_glb()
        except Exception as e:
            import logging
            logging.warning("reset: rebuild failed: %s", e)
        return (
            jsonify(
                {
                    "ok": True,
                    "current": _snapshot(),
                    "history_len": len(HISTORY),
                    "pending_adds_len": len(PENDING_ADDS),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/apply")
def apply():
    """Apply changes to the CAD model."""
    from run import _parse_apply_request
    from app.services.state_service import apply_changes_list as _apply_changes_list
    try:
        changes, excerpt = _parse_apply_request()
        if not changes:
            return jsonify({"ok": False, "error": "No change objects supplied"}), 400
        code, payload = _apply_changes_list(changes, excerpt)
        return jsonify(payload), code
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.get("/mode")
def mode():
    """Get current mode."""
    from run import USE_CQPARTS
    from app.config import Config
    import os
    model_glb = Config.ASSETS_DIR / "model.glb"
    mode = (
        "model.glb"
        if os.path.exists(model_glb)
        else ("cqparts" if USE_CQPARTS else "no_model")
    )
    return jsonify({"mode": mode})


@bp.post("/label")
def label():
    """Set label for a component."""
    from run import STATE
    data = request.get_json(force=True, silent=True) or {}
    part = (data.get("part_name") or "").strip()
    if part:
        STATE["selected_parts"].append(part)
        return jsonify(
            {"ok": True, "part": part, "count": len(STATE["selected_parts"])}
        )
    return jsonify({"ok": False, "error": "no part_name"})


@bp.get("/labels")
def labels():
    """Get all labels."""
    from run import STATE
    return jsonify({"ok": True, "selected_parts": STATE["selected_parts"]})


@bp.get("/model/glb")
@bp.get("/model.glb")  # Legacy route
def model_glb():
    """Get current GLB model file."""
    from flask import send_file, Response
    from app.config import Config
    from app.services.cad_service import get_glb_path
    import io
    
    # Check if model GLB exists
    glb_path = get_glb_path()
    if os.path.exists(glb_path):
        try:
            with open(glb_path, "rb") as f:
                cached_glb = f.read()
            if len(cached_glb) > 1000:
                print(f"[model.glb] ✓ Serving GLB ({len(cached_glb)} bytes)", flush=True)
                return send_file(io.BytesIO(cached_glb), mimetype="model/gltf-binary")
        except Exception as e:
            print(f"[model.glb] Failed to read GLB: {e}", flush=True)
    
    # No model loaded yet
    print("[model.glb] ⚠ No model GLB found - returning 404", flush=True)
    return Response(
        "Model not yet loaded. Please upload a mesh or CadQuery model first.",
        status=404,
        mimetype="text/plain"
    )


@bp.post("/undo")
def undo():
    """Undo last change."""
    from run import HISTORY, H_PTR, _restore, _rebuild_and_save_glb
    global H_PTR
    if H_PTR <= 0:
        return jsonify({"ok": False, "error": "Nothing to undo"}), 400
    H_PTR -= 1
    _restore(HISTORY[H_PTR])
    try:
        _rebuild_and_save_glb()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/redo")
def redo():
    """Redo last undone change."""
    from run import HISTORY, H_PTR, _restore, _rebuild_and_save_glb
    global H_PTR
    if H_PTR >= len(HISTORY) - 1:
        return jsonify({"ok": False, "error": "Nothing to redo"}), 400
    H_PTR += 1
    _restore(HISTORY[H_PTR])
    try:
        _rebuild_and_save_glb()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.get("/params")
def params():
    """Get current parameters."""
    from run import CURRENT_PARAMS, _snapshot, USE_CQPARTS, _introspect_params_from_cls, CONTEXT
    info = {"current": _snapshot(), "introspected": {}}
    # Legacy rover component introspection removed
    info["context"] = {"terrain_mode": CONTEXT["terrain_mode"]}
    return jsonify({"ok": True, "params": info})

