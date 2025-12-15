"""
Application State Management
Runtime state variables for the application.
"""

from typing import Dict, Any, Optional, List, Union

# Runtime state - these change during application execution
STATE: Dict[str, Any] = {"selected_parts": []}

CURRENT_PARAMS: Dict[str, Optional[float]] = {
    # wheel geometry
    "wheel_diameter": None,
    "wheel_width": None,
    # pan/tilt offsets
    "pan_tilt_offset_x": None,
    "pan_tilt_offset_y": None,
    "pan_tilt_offset_z": None,
    # layout
    "wheels_per_side": None,
    "axle_spacing_mm": None,
    "wheelbase_span_mm": None,
    # rover pose + wheel vertical offset + visibility
    "rover_yaw_deg": None,
    "wheel_z_offset_mm": None,
    "hide_wheels": None,
    # new: optional left/right mirror (swap sides)
    "mirror_lr": None,
}

CONTEXT: Dict[str, Any] = {"terrain_mode": "flat"}  # or "uneven"
HISTORY: List[Dict[str, Optional[float]]] = []
H_PTR: int = -1

# Cache for IngestResult (keyed by mesh_path)
_INGEST_RESULT_CACHE: Dict[str, Any] = {}
INIT_SNAPSHOT: Optional[Dict[str, Optional[float]]] = None

# queued ops for true geometry adds
PENDING_ADDS: List[Dict[str, Any]] = []
HIDDEN_PREFIXES: List[str] = []
