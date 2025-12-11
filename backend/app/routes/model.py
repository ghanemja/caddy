"""
Model/GLB routes blueprint
"""
from flask import Blueprint, send_file, jsonify
from app.config import Config
import os

bp = Blueprint("model", __name__)


@bp.get("/glb")
def get_glb():
    """Get current GLB model file."""
    # GLB files will be generated from uploaded models
    # For now, check if a model GLB exists in assets
    glb_path = Config.ASSETS_DIR / "model.glb"
    if not os.path.exists(glb_path):
        return jsonify({"ok": False, "error": "No model loaded. Please upload a CadQuery model or Python file first."}), 404
    return send_file(glb_path, mimetype="model/gltf-binary")


@bp.post("/rebuild")
def rebuild():
    """Rebuild GLB from current state."""
    # TODO: Implement rebuild from uploaded model
    return jsonify({"ok": False, "error": "Rebuild not yet implemented for uploaded models"}), 501

