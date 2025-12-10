"""
Model/GLB routes blueprint
"""
from flask import Blueprint, send_file, jsonify
from app.services.cad_service import get_glb_path, build_rover_glb
import os

bp = Blueprint("model", __name__)


@bp.get("/glb")
def get_glb():
    """Get current GLB model file."""
    glb_path = get_glb_path()
    if not os.path.exists(glb_path):
        return jsonify({"ok": False, "error": "GLB not found"}), 404
    return send_file(glb_path, mimetype="model/gltf-binary")


@bp.post("/rebuild")
def rebuild():
    """Rebuild GLB from current state."""
    try:
        result = build_rover_glb()
        if result:
            return jsonify({"ok": True, "path": result})
        return jsonify({"ok": False, "error": "Build failed"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

