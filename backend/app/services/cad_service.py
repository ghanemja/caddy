"""
CAD Building Service
Handles CAD model building and GLB generation.
"""
import os
import sys
import subprocess
import threading
import math
import trimesh
import cadquery as cq
from typing import Dict, Any, Optional

# Import from legacy optim.py for now (will be migrated)
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


def get_glb_path() -> str:
    """Get the path to the current GLB file."""
    from app.config import Config
    return str(Config.ASSETS_DIR / "model.glb")


def build_rover_scene_glb_cqparts_hybrid(use_generated: bool = False) -> bytes:
    """Legacy function - rover system removed."""
    raise NotImplementedError("Rover build system removed. Use uploaded CadQuery models instead.")


def reload_rover_from_generated():
    """Legacy function - rover system removed. Returns None."""
    print("[reload] ⚠ Legacy rover system removed - function is no-op")
    return None


def rebuild_and_save_glb(use_generated: bool = False):
    """
    Legacy function - rover system removed.
    TODO: Replace with generic model rebuild from uploaded CadQuery/Python files.
    """
    print("[rebuild] ⚠ Legacy rover rebuild removed - function is no-op")
    # TODO: Implement rebuild from uploaded user model


def build_rover_scene_glb_cqparts(RoverClass=None) -> bytes:
    """Legacy function - rover system removed."""
    raise NotImplementedError("Rover build system removed. Use uploaded CadQuery models instead.")


def build_rover_scene_glb(_: Optional[Dict[str, Any]] = None) -> bytes:
    """Legacy function - rover system removed."""
    raise NotImplementedError("Rover build system removed. Use uploaded CadQuery models instead.")
