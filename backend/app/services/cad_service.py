"""
CAD Building Service
Handles CAD model building and GLB generation.
"""
import os
import sys
from typing import Dict, Any, Optional

# Import from legacy optim.py for now (will be migrated)
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


def build_rover_glb() -> Optional[str]:
    """
    Build the rover GLB model.
    
    Returns:
        Path to generated GLB file, or None on error
    """
    from optim import _rebuild_and_save_glb, ROVER_GLB_PATH
    try:
        _rebuild_and_save_glb()
        return ROVER_GLB_PATH if os.path.exists(ROVER_GLB_PATH) else None
    except Exception as e:
        print(f"[cad_service] Build failed: {e}")
        return None


def get_glb_path() -> str:
    """Get the path to the current GLB file."""
    from optim import ROVER_GLB_PATH
    return ROVER_GLB_PATH

