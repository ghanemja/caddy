#!/usr/bin/env python3
"""
CAD Optimizer Server
Main entry point for the Flask application.
"""
import io, os, sys, json, re, base64, threading, mimetypes, ast, math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

# ----------------- Setup Python path FIRST -----------------
# Set up paths BEFORE importing any packages that might conflict
BACKEND_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BACKEND_DIR.parent

# Add paths, but ensure current directory doesn't interfere with package imports
if '' in sys.path:
    sys.path.remove('')
if str(os.getcwd()) in sys.path:
    sys.path.remove(str(os.getcwd()))

# Add our paths
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now import packages (after path is set up correctly)
import requests
import numpy as np
import trimesh
from trimesh.transformations import euler_matrix
from flask import (
    Flask,
    Response,
    request,
    jsonify,
    send_file,
    send_from_directory,
    abort,
    render_template,
)

# ----------------- Initialize FreeCAD and CadQuery -----------------
# Use service modules for initialization
from app.services.freecad_service import FreeCAD, init_cadquery_with_freecad
import cadquery as cq
from cadquery import exporters, Workplane

# Initialize CadQuery with FreeCAD
cq, exporters, Workplane = init_cadquery_with_freecad(FreeCAD)

# ----------------- Repo components -----------------
# CAD components are now optional - no hardcoded rover dependency

# Add app/models/cad to path so imports work (if the directory exists)
_cad_models_path = os.path.join(BACKEND_DIR, "app", "models", "cad")
if os.path.exists(_cad_models_path) and _cad_models_path not in sys.path:
    sys.path.insert(0, _cad_models_path)

# Try to import CAD components, but don't fail if they're not available
# This allows the server to run without rover-specific dependencies
Rover = None
_Electronics = None
_PanTilt = None
_ThisWheel = None
_Stepper = None
SensorFork = None

try:
    from app.models.cad import Electronics as _Electronics, PanTilt as _PanTilt
    from app.models.cad import BuiltWheel as _ThisWheel, SensorFork
    print("[imports] ✓ Loaded some CAD components from app.models.cad")
    # Try to import Rover separately (may fail due to cqparts_motors dependency)
    try:
        from app.models.cad import Rover
        print("[imports] ✓ Loaded Rover component")
    except ImportError as rover_err:
        print(f"[imports] ⚠ Rover not available (optional): {rover_err}")
        Rover = None
except ImportError as e:
    # Fallback: try direct imports without Rover
    try:
        from app.models.cad.electronics import type1 as _Electronics
        from app.models.cad.pan_tilt import PanTilt as _PanTilt
        from app.models.cad.wheel import BuiltWheel as _ThisWheel
        from app.models.cad.sensor_fork import SensorFork
        print("[imports] ✓ Loaded CAD components (direct, without Rover)")
        # Try Rover separately
        try:
            from app.models.cad.robot_base import Rover
            print("[imports] ✓ Loaded Rover component")
        except ImportError:
            print("[imports] ⚠ Rover not available (optional)")
            Rover = None
    except ImportError as e2:
        print(f"[imports] ⚠ Some CAD components not available (optional)")
        print(f"[imports] ⚠ Error: {e2}")
        # Continue without CAD components - server can still run for other features
except ImportError:
    # Fallback: try direct import from current directory
    try:
        from robot_base import Rover
        from electronics import type1 as _Electronics
        from pan_tilt import PanTilt as _PanTilt
        from wheel import BuiltWheel as _ThisWheel
        from cqparts_motors.stepper import Stepper as _Stepper
        from sensor_fork import SensorFork
    except ImportError:
        # Last resort: try from app.models.cad directly
        from app.models.cad.robot_base import Rover
        from app.models.cad.electronics import type1 as _Electronics
        from app.models.cad.pan_tilt import PanTilt as _PanTilt
        from app.models.cad.wheel import BuiltWheel as _ThisWheel
        from cqparts_motors.stepper import Stepper as _Stepper
        from app.models.cad.sensor_fork import SensorFork

# MIME fix for ESM
mimetypes.add_type("application/javascript", ".js")

# ----------------- VLM config -----------------
from app.services.vlm_service import (
    USE_FINETUNED_MODEL,
    FINETUNED_MODEL_PATH,
    OLLAMA_URL,
    OLLAMA_MODEL,
    LLAVA_URL,
    load_finetuned_model,
    get_finetuned_model as _get_finetuned_model,
    get_finetuned_processor as _get_finetuned_processor,
    call_vlm,  # Now imported from service
    stitch_images_side_by_side as _stitch_images_side_by_side,  # Renamed for compatibility
)

# Keep backward compatibility
_finetuned_model = None  # Will be lazy-loaded
_finetuned_processor = None  # Will be lazy-loaded


# ----------------- Aliases -----------------
TARGET_ALIASES = {
    "motor_controllerboard": "motor_controller_board",
    "motorcontrollerboard": "motor_controller_board",
    "motor controller board": "motor_controller_board",
    "motorcontroller": "motor_controller",
    "motor": "motor_controller",
    "sensorsbase": "sensor_fork",
    "sensor": "sensor_fork",
    "sensors": "sensor_fork",
    "wheels": "wheel",
    "roverbase": "rover",
    "chassis": "rover",
    "base": "rover",
}
ACTION_ALIASES = {
    "move": "translate",
    "position": "translate",
    "pos": "translate",
    "orientation": "rotate",
    "orient": "rotate",
    "size": "modify",
    "count": "add",
    "wheels_per_side": "modify",
    "scale": "resize",
    "shrink": "resize",
}

# ----------------- App, paths, state -----------------
PORT = int(os.environ.get("PORT", "5160"))
# ----------------- App configuration -----------------
# Use config for paths
from app.config import Config

BASE_DIR = BACKEND_DIR  # backend/ directory
ASSETS_DIR = Config.ASSETS_DIR  # frontend/assets/
os.makedirs(ASSETS_DIR, exist_ok=True)
ROVER_GLB_PATH = str(ASSETS_DIR / "rover.glb")  # Keep as string for compatibility
USE_CQPARTS = os.environ.get("USE_CQPARTS", "1") == "1"

# Initialize Flask app with correct template and static folders
app = Flask(
    __name__,
    template_folder=str(Config.TEMPLATES_FOLDER),
    static_folder=str(Config.STATIC_FOLDER),
    static_url_path="/static"
)

# Register blueprints for organized routes
from app.routes import main, api, vlm, mesh, model, test
app.register_blueprint(main.bp)  # Routes: /, /debug, /demo/*, /static/*
app.register_blueprint(api.bp)  # Routes: /state, /apply, /params, /model.glb, etc.
app.register_blueprint(vlm.bp)  # Routes: /codegen, /vlm, /recommend
app.register_blueprint(mesh.bp, url_prefix="/api/mesh")  # Routes: /ingest_mesh_segment, etc.
app.register_blueprint(model.bp, url_prefix="/api/model")
app.register_blueprint(test.bp, url_prefix="/test")  # Routes: /test/rotate, /test/translate, /test/modify

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

# non-numeric context
CONTEXT: Dict[str, Any] = {"terrain_mode": "flat"}  # or "uneven"
HISTORY: List[Dict[str, Optional[float]]] = []
H_PTR: int = -1

# Cache for IngestResult (keyed by mesh_path)
_INGEST_RESULT_CACHE: Dict[str, Any] = {}
INIT_SNAPSHOT: Optional[Dict[str, Optional[float]]] = None

# ----------------- cqparts shim -----------------
try:
    from cqparts.utils.geometry import CoordSystem
except Exception:

    class CoordSystem:
        def __sub__(self, other):
            return self

        def __rsub__(self, other):
            return other


cq.Workplane.world_coords = property(lambda self: CoordSystem())
cq.Workplane.local_coords = property(lambda self: CoordSystem())


# ----------------- Component registry -----------------
from app.core.component_registry import ComponentSpec, register_component, get_component_spec, COMPONENT_REGISTRY


# queued ops for true geometry adds
PENDING_ADDS: List[dict] = []
INSTALLED_ADDS: List[dict] = []
HIDDEN_PREFIXES: List[str] = []


# ----------------- Utility helpers -----------------
from app.utils.helpers import (
    truthy as _truthy,
    num as _num,
    strip_units_to_float as _strip_units_to_float,
    clean_num as _clean_num,
    percent_to_abs as _percent_to_abs,
    cq_to_trimesh as _cq_to_trimesh,
)
from app.utils.param_normalization import (
    normalize_params as _normalize_params,
    normalize_change as _normalize_change,
    coerce_changes as _coerce_changes,
    intent_to_changes as _intent_to_changes,
    mk_change as _mk_change,
)
from app.utils.json_parsing import (
    extract_json_loose as _extract_json_loose,
    split_multi_json_and_summaries as _split_multi_json_and_summaries,
    find_all_balanced_json_blocks as _find_all_balanced_json_blocks,
)
from app.utils.units import (
    mm_from_value_token as _mm_from_value_token,
    repair_units_in_json_text as _repair_units_in_json_text,
)


# State management functions moved to app.services.state_service
from app.services.state_service import (
    snapshot_global as _snapshot,
    ensure_initial_history_global as _ensure_initial_history,
    push_history_global as _push_history,
    restore_global as _restore
)


# Functions moved to app.utils.param_normalization, app.utils.json_parsing, app.utils.units
# Imported above


# ----------------- Model adapter -----------------
class ModelAdapter:
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def add(self, kind: str, **params):
        PENDING_ADDS.append({"kind": kind.lower(), "params": params})


ADAPTER = ModelAdapter(Rover)


# ----------------- Component registry entries -----------------
def proxy_wheels(scene):
    """Detailed wheel proxy with rim/spokes/tread, named nodes."""
    try:
        n_side = int((CURRENT_PARAMS.get("wheels_per_side") or 0))
        if n_side <= 0 or _truthy(CURRENT_PARAMS.get("hide_wheels")):
            return
        dia = float(CURRENT_PARAMS.get("wheel_diameter") or 130.0)
        wid = float(CURRENT_PARAMS.get("wheel_width") or 40.0)
        span = float(CURRENT_PARAMS.get("wheelbase_span_mm") or 320.0)
        axle = float(CURRENT_PARAMS.get("axle_spacing_mm") or 180.0)
        zoff = float(CURRENT_PARAMS.get("wheel_z_offset_mm") or 0.0)
    except Exception:
        return

    R = dia * 0.5
    rim_thick = max(2.0, min(0.06 * dia, 6.0))
    hub_rad = max(6.0, 0.12 * R)
    spoke_w = max(2.0, 0.06 * R)
    n_spokes = 8
    n_treads = max(12, int(0.6 * R))

    rim = cq.Workplane("XY").circle(R).circle(R - rim_thick).extrude(wid)
    hub = cq.Workplane("XY").circle(hub_rad).extrude(wid)
    spoke_len = (R - rim_thick) - hub_rad
    spoke = (
        cq.Workplane("XY")
        .rect(spoke_len, spoke_w, centered=(True, True))
        .extrude(wid)
        .translate(((hub_rad + (spoke_len / 2.0)), 0, 0))
    )
    spokes = cq.Workplane("XY")
    for i in range(n_spokes):
        spokes = spokes.union(
            spoke.rotate((0, 0, 0), (0, 0, 1), i * (360.0 / n_spokes))
        )
    wheel_solid = rim.union(hub).union(spokes)

    # grooves (best-effort)
    try:
        groove_w = max(1.0, wid * 0.08)
        groove_d = max(0.8, rim_thick * 0.5)
        groove_rad = R - groove_d * 0.5
        grooves = cq.Workplane("XY")
        ang_step = 360.0 / n_treads
        for i in range(n_treads):
            g = (
                cq.Workplane("XY")
                .circle(groove_w * 0.5)
                .extrude(rim_thick * 1.2)
                .translate(
                    (
                        groove_rad * math.cos(math.radians(i * ang_step)),
                        groove_rad * math.sin(math.radians(i * ang_step)),
                        wid * 0.5,
                    )
                )
                .rotate((0, 0, 0), (1, 0, 0), 90)
            )
            grooves = grooves.union(g)
        wheel_solid = wheel_solid.cut(grooves)
    except Exception:
        pass

    wheel_solid = wheel_solid.rotate((0, 0, 0), (0, 1, 0), 90)

    left_x = -span * 0.5
    right_x = +span * 0.5
    z0 = R + zoff

    if n_side == 1:
        ys = [0.0]
    else:
        step = axle / (n_side - 1) if n_side > 1 else 0.0
        ys = [-axle * 0.5 + i * step for i in range(n_side)]

    def add_one(x, y, z, label):
        tm = _cq_to_trimesh(wheel_solid.translate((x, y, z)), tol=0.45)
        if tm and not getattr(tm, "is_empty", False):
            scene.add_geometry(
                tm, node_name=f"wheel/{label}", geom_name=f"wheel_geom/{label}"
            )

    for i, y in enumerate(ys, start=1):
        add_one(left_x, y, z0, f"L{i}")
        add_one(right_x, y, z0, f"R{i}")


register_component(
    "wheel",
    ComponentSpec(
        cls=_ThisWheel,
        add_fn=None,
        param_map={"wheel_diameter": "diameter", "wheel_width": "width"},
        proxy_fn=proxy_wheels,
    ),
)

register_component(
    "pan_tilt",
    ComponentSpec(
        cls=_PanTilt,
        add_fn=None,
        param_map={
            "pan_tilt_offset_x": "pan_tilt_offset_x",
            "pan_tilt_offset_y": "pan_tilt_offset_y",
            "pan_tilt_offset_z": "pan_tilt_offset_z",
        },
    ),
)

register_component(
    "sensor_fork",
    ComponentSpec(
        cls=SensorFork,
        add_fn=lambda adapter, **p: ADAPTER.add("sensor_fork", **p),
        param_map={
            "width_mm": "width",
            "depth_mm": "depth",
            "height_mm": "height",
            "wall_mm": "wall",
            "hole_diam_mm": "hole_diam",
        },
    ),
)

register_component(
    "rover",
    ComponentSpec(
        cls=Rover,
        add_fn=None,
        param_map={"rover_yaw_deg": "rover_yaw_deg"},
    ),
)


def apply_params_to_rover(rv, params: Dict[str, Any] | None):
    if params:
        for k, v in params.items():
            if k in CURRENT_PARAMS:
                CURRENT_PARAMS[k] = _clean_num(v)
    if CURRENT_PARAMS["wheel_diameter"] is not None:
        setattr(_ThisWheel, "diameter", float(CURRENT_PARAMS["wheel_diameter"]))
    if CURRENT_PARAMS["wheel_width"] is not None:
        setattr(_ThisWheel, "width", float(CURRENT_PARAMS["wheel_width"]))

    if CURRENT_PARAMS["wheels_per_side"] is not None:
        wps = int(CURRENT_PARAMS["wheels_per_side"])
        try:
            setattr(rv, "wheels_per_side", wps)
        except Exception:
            pass
        setattr(Rover, "wheels_per_side", wps)
        try:
            setattr(_ThisWheel, "count", max(2, 2 * wps))
        except Exception:
            pass

    for k in ("axle_spacing_mm", "wheelbase_span_mm"):
        if CURRENT_PARAMS[k] is not None:
            val = float(CURRENT_PARAMS[k])
            try:
                setattr(rv, k, val)
            except Exception:
                pass
            setattr(Rover, k, val)

    for axis in ("x", "y", "z"):
        key = f"pan_tilt_offset_{axis}"
        if CURRENT_PARAMS[key] is not None:
            val = float(CURRENT_PARAMS[key])
            try:
                setattr(rv, key, val)
            except Exception:
                pass
            try:
                setattr(_PanTilt, key, val)
            except Exception:
                pass


def _apply_rotation_to_wheels(scene):
    rot = CURRENT_PARAMS.get("wheel_rotation")
    if not rot or not isinstance(rot, (list, tuple)):
        return
    try:
        rx, ry, rz = [float(x) for x in rot]
    except Exception:
        rx = ry = rz = 0.0
    if abs(rx) < 1e-6 and abs(ry) < 1e-6 and abs(rz) < 1e-6:
        return
    R = trimesh.transformations.euler_matrix(
        math.radians(rx), math.radians(ry), math.radians(rz)
    )
    for name, geom in list(scene.geometry.items()):
        if name.lower().startswith("wheel/"):
            g = geom.copy()
            g.apply_transform(R)
            scene.geometry[name] = g


# ----------------- VLM prompt -----------------
# Prompts are now loaded from app/services/vlm/prompts/ directory
from app.services.vlm.prompts_loader import get_system_prompt, get_codegen_prompt

# Backward compatibility: lazy-load prompts when accessed
VLM_SYSTEM_PROMPT = get_system_prompt()
VLM_CODEGEN_PROMPT = get_codegen_prompt()

# Old inline prompt removed - now loaded from file:
# See: app/services/vlm/prompts/system_prompt.txt

# Old inline codegen prompt removed - now loaded from file:
# See: app/services/vlm/prompts/codegen_prompt.txt

def _build_codegen_prompt(
    ref_url: Optional[str], snapshot_url: Optional[str], user_text: str = ""
) -> tuple[str, List[str]]:
    """
    Build the complete prompt for VLM code generation.
    
    Args:
        ref_url: Reference image (target design)
        snapshot_url: Current CAD screenshot (orthogonal views)
        user_text: Human qualitative feedback/intent
    
    Returns:
        (prompt_text, list_of_image_urls)
    """
    cad_state = _cad_state_json()
    baseline_src = _baseline_cqparts_source()

    parts = [
        get_codegen_prompt(),
        "\n\n",
        "=" * 80,
        "\n<<<BASELINE_PYTHON_SOURCE>>>\n",
        "# File: robot_base.py\n",
        "# This is the current implementation - copy exact lines from here\n",
        "# Look for parameter lines like: wheels_per_side = PositiveFloat(N)\n\n",
        baseline_src if baseline_src else "# (baseline source unavailable)",
        "\n<<<END_BASELINE_PYTHON_SOURCE>>>\n",
        "=" * 80,
        "\n\n<<<CURRENT_CAD_STATE>>>\n",
        json.dumps(cad_state, indent=2),
        "\n<<<END_CURRENT_CAD_STATE>>>\n",
        "=" * 80,
    ]
    
    if user_text:
        parts += [
            "\n\n<<<USER_INTENT_AND_FEEDBACK>>>\n",
            user_text,
            "\n<<<END_USER_INTENT>>>\n",
            "=" * 80,
        ]
    
    parts += [
        "\n\n=== IMAGES PROVIDED ===",
        "\n- Image 0: REFERENCE (target design showing desired rover)",
    ]
    
    if snapshot_url:
        parts.append("\n- Image 1: CURRENT CAD SNAPSHOT (orthogonal views of current model)")
        parts.append("\n\nCompare these TWO images to understand what needs to change.")
    else:
        parts.append("\n\n(No current snapshot - generate from reference image only)")
    
    parts += [
        "\n\n" + "=" * 80,
        "\n=== NOW OUTPUT THE COMPLETE MODIFIED robot_base.py ===",
        "\n" + "=" * 80,
        "\n",
        "\n🚨 CRITICAL INSTRUCTIONS:",
        "\n",
        "\n1. READ the user's instruction carefully - translate it to parameter changes!",
        "\n   • 'remove all wheels' → wheels_per_side = PositiveFloat(0)",
        "\n   • '3 wheels per side' → wheels_per_side = PositiveFloat(3)",
        "\n   • 'more space between wheels' → axle_spacing_mm = PositiveFloat(90) [increase from 70]",
        "\n   • 'bigger wheels' → diameter = PositiveFloat(100) [increase from 90]",
        "\n",
        "\n2. COPY the ENTIRE baseline source above (all 180+ lines)",
        "\n",
        "\n3. Modify ONLY the specific parameter VALUE that matches the user request",
        "\n   • Find the line with that parameter",
        "\n   • Change ONLY the number inside PositiveFloat(...)",
        "\n   • Keep everything else identical",
        "\n",
        "\n4. DO NOT just copy the baseline unchanged - YOU MUST MAKE THE CHANGE!",
        "\n   • If user says 'remove all wheels', wheels_per_side MUST be 0, not 4",
        "\n   • If user says 'increase spacing', axle_spacing_mm MUST be larger, not the same",
        "\n",
        "\n5. Keep ALL method implementations identical (make_components, make_constraints, etc.)",
        "\n",
        "\n⚠️ OUTPUT REQUIREMENTS:",
        "\n• NO markdown fences (```python or ```) - output raw Python only",
        "\n• NO explanations like 'Here is the modified code'",
        "\n• Start with: #!/usr/bin/env python3",
        "\n• Copy every import, every class, every method from baseline",
        "\n• Your output should be 150-250 lines (same length as baseline)",
        "\n• DO NOT use '...' or abbreviate any methods",
        "\n",
        "\n✅ Example 1 - User says 'remove all wheels':",
        "\n• Translate: 'remove all wheels' means wheels_per_side = 0",
        "\n• Find line: wheels_per_side = PositiveFloat(4)  # default 4 per side",
        "\n• Change to: wheels_per_side = PositiveFloat(0)  # no wheels",
        "\n• Copy everything else EXACTLY",
        "\n• Result: 180 lines with ONE number changed from 4 to 0",
        "\n",
        "\n✅ Example 2 - User says 'increase spacing between wheels':",
        "\n• Translate: 'increase spacing' means axle_spacing_mm should be larger",
        "\n• Find line: axle_spacing_mm = PositiveFloat(70)",
        "\n• Change to: axle_spacing_mm = PositiveFloat(90)  # increased by ~30%",
        "\n• Copy everything else EXACTLY",
        "\n",
        "\n✅ Example 3 - User says 'make diameter 15mm smaller':",
        "\n• Step 1: Find baseline diameter in ThisWheel class: diameter = PositiveFloat(90)",
        "\n• Step 2: Calculate: 90 - 15 = 75",
        "\n• Step 3: Change to: diameter = PositiveFloat(75)  # 15mm smaller than 90mm",
        "\n• Copy everything else EXACTLY",
        "\n",
        "\n❌ WRONG - Do NOT do:",
        '\n• Output ```python at start',
        "\n• Abbreviate methods with '# ... rest of code'",
        "\n• Return single object from make_components (must return dict)",
        "\n• Change parameters that user didn't request",
        "\n• Modify imports or method logic",
        "\n",
        "\n⚠️ Your output will be compiled and validated. It must be syntactically perfect.",
        "\n",
        "\nSTART YOUR PYTHON CODE NOW (begin with #!/usr/bin/env python3, no fences):",
        "\n",
    ]
    
    images = [u for u in [ref_url, snapshot_url] if u]
    
    # Debug logging
    print(f"[codegen_prompt] Built prompt with {len(images)} images")
    print(f"[codegen_prompt] Total prompt length: {len(''.join(parts))} chars")
    print(f"[codegen_prompt] Baseline source included: {len(baseline_src)} chars")
    
    return "".join(parts), images


# /codegen route moved to app.routes.vlm blueprint
# Legacy function removed - route is now in vlm.py blueprint


# ----------------- Helpers: uploads & VLM -----------------
# _data_url_from_upload moved to app.utils.helpers
from app.utils.helpers import data_url_from_upload as _data_url_from_upload


# _stitch_images_side_by_side and call_vlm are now imported from app.services.vlm_service


# Code generation utilities are now in app.utils.codegen
from app.utils.codegen import (
    normalize_generated_code,
    extract_python_module,
    normalize_generated_code_advanced as _normalize_generated_code_advanced
)

# Removed _normalize_generated_code_advanced - now imported from app.utils.codegen


# ----------------- CAD state JSON for grounding -----------------
def _cad_state_json():
    return {
        "current_params": _snapshot(),
        "context": CONTEXT,
        "known_classes": sorted(list(COMPONENT_REGISTRY.keys())),
        "selected_parts": list(STATE.get("selected_parts", [])),
        "history": HISTORY[: H_PTR + 1],
        "pending_adds": list(PENDING_ADDS),
    }


def _apply_changes_list(changes: List[dict], excerpt: str | None = None):
    if not changes:
        return 400, {"ok": False, "error": "No change objects supplied"}

    _ensure_initial_history()
    _push_history()

    rv = Rover(
        stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel
    )
    highlight_key = None

    for raw in changes:
        ch = _normalize_change(raw) or raw
        action = (ch.get("action") or "").strip().lower()
        target = (ch.get("target_component") or "").strip().lower()
        params = ch.get("parameters") or {}
        if not action or not target:
            print(f"[apply] skipping malformed change: {ch}")
            continue

        comp = get_component_spec(target) or (
            get_component_spec(target.split()[0]) if target.split() else None
        )

        # wheel add count → wheels_per_side
        if action == "add" and (
            target.startswith("wheel") or comp is get_component_spec("wheel")
        ):
            cnt = params.get("count")
            if params.get("wheels_per_side") is None and cnt is not None:
                try:
                    params["wheels_per_side"] = max(1, (int(cnt) + 1) // 2)
                except Exception:
                    pass

        # param map to class attrs
        if comp and action in (
            "modify",
            "resize",
            "replace",
            "translate",
            "rotate",
            "add",
        ):
            for jkey, attr in (comp.param_map or {}).items():
                if jkey in params and params[jkey] is not None:
                    try:
                        setattr(comp.cls, attr, float(params[jkey]))
                    except Exception:
                        pass

        # apply non-geom context first (wheel attach mode, LR mirror)
        wa = params.get("wheel_attach")
        if isinstance(wa, str) and wa:
            CONTEXT["wheel_attach"] = (
                "center" if "center" in wa or "mid" in wa else "bottom"
            )
        if "mirror_lr" in params and params["mirror_lr"] is not None:
            CURRENT_PARAMS["mirror_lr"] = bool(params["mirror_lr"])

        # true add
        if action == "add" and comp and callable(comp.add_fn):
            comp.add_fn(adapter=ADAPTER, **params)

        # apply model-level params
        apply_params_to_rover(rv, params)

        # rover rotate convenience
        if (action == "rotate" or "rover_yaw_deg" in params) and target in (
            "rover",
            "base",
            "chassis",
        ):
            yaw = (
                params.get("rover_yaw_deg")
                or params.get("rz")
                or params.get("angle")
                or params.get("angle_deg")
            )
            if yaw is not None:
                try:
                    CURRENT_PARAMS["rover_yaw_deg"] = float(yaw)
                except Exception:
                    pass

        # delete/hide unified semantics
        if action == "delete":
            t = target
            if t in ("wheel", "wheels"):
                CURRENT_PARAMS["hide_wheels"] = True
                HIDDEN_PREFIXES.append("wheel/")
            elif t in ("pan_tilt", "pan-tilt", "pantilt"):
                HIDDEN_PREFIXES.extend(["pan_tilt", "pan-tilt"])
            elif t in ("sensor_fork", "sensor", "sensors"):
                HIDDEN_PREFIXES.append("sensor_fork")

        if not highlight_key:
            highlight_key = target

    _push_history()
    try:
        _rebuild_and_save_glb()
        return 200, {
            "ok": True,
            "highlight_key": highlight_key or "wheel",
            "excerpt": excerpt,
        }
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}



# Function moved to app.utils.request_parsing
from app.utils.request_parsing import parse_apply_request as _parse_apply_request


# Route moved to app.routes.api


def apply_mesh_params():
    """
    Apply parameter changes to deform a mesh using the full ParametricMeshDeformer.
    
    Expects JSON: {
        "parameters": { "param_name": value, ... },
        "mesh_path": "...",  # Optional, will use cached if not provided
        "enabled_parts": { part_id: bool, ... }  # Optional, for part add/remove
    }
    Returns: { "ok": True, "glb_path": "...", "message": "..." }
    """
    try:
        data = request.get_json()
        if not data or "parameters" not in data:
            return jsonify({"ok": False, "error": "Missing 'parameters' in request"}), 400
        
        parameters = data["parameters"]
        if not isinstance(parameters, dict):
            return jsonify({"ok": False, "error": "'parameters' must be a dictionary"}), 400
        
        # Get mesh path from request or use cached one
        mesh_path = data.get("mesh_path")
        if not mesh_path:
            # Try to find the most recent cached result
            if _INGEST_RESULT_CACHE:
                mesh_path = list(_INGEST_RESULT_CACHE.keys())[-1]
            else:
                return jsonify({
                    "ok": False,
                    "error": "No mesh path provided and no cached ingestion result found. Please run mesh ingestion first."
                }), 400
        
        if not os.path.exists(mesh_path):
            return jsonify({
                "ok": False,
                "error": f"Mesh file not found: {mesh_path}"
            }), 400
        
        # Import mesh deformation modules
        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from meshml.mesh_deform import ParametricMeshDeformer, MeshData
        from meshml.semantics.ingest_mesh import build_deformer_from_ingest_result, IngestResult
        import trimesh
        import numpy as np
        
        # Get cached IngestResult
        ingest_result = _INGEST_RESULT_CACHE.get(mesh_path)
        if not ingest_result:
            return jsonify({
                "ok": False,
                "error": f"No cached ingestion result for mesh: {mesh_path}. Please run mesh ingestion first."
            }), 400
        
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Get part labels from cached result
        vertex_labels = None
        if ingest_result.part_table:
            # Get vertex labels from PartTable
            vertex_labels = ingest_result.part_table.vertex_part_labels
            if len(vertex_labels) != len(vertices):
                # Pad or truncate to match
                if len(vertex_labels) < len(vertices):
                    vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
                else:
                    vertex_labels = vertex_labels[:len(vertices)]
        
        if vertex_labels is None:
            # Fallback: try to get from extra dict
            if "vertex_labels" in ingest_result.extra:
                vertex_labels = np.array(ingest_result.extra["vertex_labels"])
            else:
                # Last resort: create dummy labels (all vertices in part 0)
                print(f"[apply_mesh_params] Warning: No vertex labels found, using dummy labels", flush=True)
                vertex_labels = np.zeros(len(vertices), dtype=np.int32)
        
        # Build part_label_names from PartTable or use defaults
        part_label_names = {}
        if ingest_result.part_table:
            for part_id, part_info in ingest_result.part_table.parts.items():
                part_label_names[part_id] = part_info.name or f"part_{part_id}"
        else:
            # Fallback: use generic names
            unique_labels = np.unique(vertex_labels)
            for label_id in unique_labels:
                part_label_names[int(label_id)] = f"part_{int(label_id)}"
        
        # Build deformer from IngestResult
        deformer = build_deformer_from_ingest_result(
            ingest_result=ingest_result,
            vertices=vertices,
            faces=faces,
            part_labels=vertex_labels,
            part_label_names=part_label_names,
        )
        
        # Handle part enable/disable (for add/remove operations)
        enabled_parts = data.get("enabled_parts")
        if enabled_parts:
            # Convert to proper format
            enabled_parts_dict = {
                int(k): bool(v) for k, v in enabled_parts.items()
            }
        else:
            enabled_parts_dict = None
        
        # Apply deformations
        deformed_mesh_data = deformer.deform(
            new_parameters=parameters,
            enabled_parts=enabled_parts_dict,
        )
        
        # Create trimesh object from deformed mesh
        deformed_mesh = trimesh.Trimesh(
            vertices=deformed_mesh_data.vertices,
            faces=deformed_mesh_data.faces,
        )
        
        # Save deformed mesh
        output_dir = os.path.join(os.path.dirname(mesh_path), "deformed")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_deformed.glb")
        deformed_mesh.export(output_path)
        
        print(f"[apply_mesh_params] ✓ Applied {len(parameters)} parameter changes, saved to {output_path}", flush=True)
        
        return jsonify({
            "ok": True,
            "glb_path": output_path,
            "message": f"Applied {len(parameters)} parameter changes"
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[apply_mesh_params] Error: {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }), 500


# ----------------- Introspection -----------------


# Route get_params moved to blueprint
# Route undo_change moved to blueprint
# Route redo_change moved to blueprint
def patch_cqparts_brittleness():
    # keep imports consistent with this project layout
    from electronics import OtherBatt, OtherController, MotorController, type1
    from robot_base import Rover

    roots = (Rover, type1, OtherController, MotorController, OtherBatt)

    for cls in roots:
        try:
            if not hasattr(cls, "world_coords"):
                cls.world_coords = CoordSystem()
        except Exception:
            pass

    def _no_constraints(self):
        return []

    for cls in (OtherController, MotorController, OtherBatt, type1, Rover):
        try:
            cls.make_constraints = _no_constraints
        except Exception:
            pass

    def _ob_box(self, x=60, y=30, z=15):
        self.local_obj = cq.Workplane("XY").box(x, y, z)
        return {}

    def _oc_box(self, x=65, y=55, z=12):
        self.local_obj = cq.Workplane("XY").box(x, y, z)
        return {}

    try:
        OtherBatt.make_components = _ob_box
    except Exception:
        pass
    try:
        OtherController.make_components = _oc_box
    except Exception:
        pass


def _emit_missing_proxies(scene):
    for key, spec in COMPONENT_REGISTRY.items():
        if hasattr(spec, "proxy_fn") and callable(spec.proxy_fn):
            print(f"[proxy] emitting for {key} …")
            spec.proxy_fn(scene)


def _apply_runtime_params_to_instance(rv):
    if CURRENT_PARAMS.get("wheel_diameter") is not None:
        try:
            setattr(_ThisWheel, "diameter", float(CURRENT_PARAMS["wheel_diameter"]))
        except:
            pass
    if CURRENT_PARAMS.get("wheel_width") is not None:
        try:
            setattr(_ThisWheel, "width", float(CURRENT_PARAMS["wheel_width"]))
        except:
            pass
    if CURRENT_PARAMS.get("wheels_per_side") is not None:
        try:
            setattr(rv, "wheels_per_side", int(CURRENT_PARAMS["wheels_per_side"]))
        except:
            pass
    if CURRENT_PARAMS.get("axle_spacing_mm") is not None:
        try:
            setattr(rv, "axle_spacing_mm", float(CURRENT_PARAMS["axle_spacing_mm"]))
        except:
            pass
    if CURRENT_PARAMS.get("wheelbase_span_mm") is not None:
        try:
            setattr(rv, "wheelbase_span_mm", float(CURRENT_PARAMS["wheelbase_span_mm"]))
        except:
            pass


# Inspection utilities moved to app.utils.inspection
from app.utils.inspection import (
    strip_docstrings_and_comments as _strip_docstrings_and_comments,
    try_get_source as _try_get_source,
    introspect_params_from_cls as _introspect_params_from_cls
)
import inspect, textwrap, importlib, sys, os, re


# _baseline_cqparts_source moved to app.utils.inspection
from app.utils.inspection import baseline_cqparts_source
def _baseline_cqparts_source(max_chars: int = 20000) -> str:
    """Wrapper that passes BASE_DIR to inspection.baseline_cqparts_source."""
    return baseline_cqparts_source(BASE_DIR, max_chars)


def build_rover_scene_glb_cqparts_hybrid(use_generated=False) -> bytes:
    """
    Build GLB using hybrid environment approach:
    - Call build_glb.py in freecad environment (CadQuery 1.x, no .wrapped issues)
    - Return GLB bytes to main server (cad-optimizer env)
    """
    import subprocess
    
    script_path = os.path.join(BASE_DIR, "build_glb.py")
    cmd = f"source ~/.bashrc && conda activate freecad && python {script_path}"
    
    if use_generated:
        cmd += " --generated"
    
    print(f"[hybrid] Building GLB in freecad environment...", flush=True)
    
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            timeout=120
        )
        
        if result.returncode == 0 and len(result.stdout) > 1000:
            print(f"[hybrid] ✓ GLB built: {len(result.stdout)} bytes", flush=True)
            return result.stdout
        else:
            print(f"[hybrid] ✗ Build failed (code {result.returncode}):", flush=True)
            print(f"[hybrid] stderr: {result.stderr.decode()[:1000]}", flush=True)
            raise RuntimeError(f"Hybrid build failed: {result.stderr.decode()[:500]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("GLB build timed out")

def build_rover_scene_glb_cqparts(RoverClass=None) -> bytes:
    """Build GLB from cqparts, optionally using a custom Rover class."""
    if RoverClass is None:
        RoverClass = Rover
    
    print(f"Generating GLB via cqparts using {RoverClass.__name__}...")
    rv = RoverClass(
        stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel
    )
    for name, cls in (
        ("stepper", _Stepper),
        ("electronics", _Electronics),
        ("sensors", _PanTilt),
        ("wheel", _ThisWheel),
    ):
        if not getattr(rv, name, None):
            setattr(rv, name, cls)
    _apply_runtime_params_to_instance(rv)

    saved_pending_attr = hasattr(RoverClass, "_pending_adds")
    saved_pending_val = getattr(RoverClass, "_pending_adds", None)
    setattr(RoverClass, "_pending_adds", [])

    build_err = [None]

    def _run_build():
        try:
            patch_cqparts_brittleness()
            print("[cqparts] Building Rover...")
            rv.build()
        except Exception as e:
            build_err[0] = e

    t = threading.Thread(target=_run_build, daemon=True)
    t.start()
    t.join(40.0)
    if t.is_alive():
        print("[error] build timed out after 40 seconds", flush=True)
        raise RuntimeError("Rover build timed out after 40 seconds")
    if build_err[0] is not None:
        print(f"[error] build failed: {build_err[0]}", flush=True)
        import traceback
        if hasattr(build_err[0], '__traceback__'):
            traceback.print_exception(type(build_err[0]), build_err[0], build_err[0].__traceback__)
        raise RuntimeError(f"Rover build failed: {build_err[0]}") from build_err[0]
    if saved_pending_attr:
        setattr(RoverClass, "_pending_adds", saved_pending_val)
    else:
        try:
            delattr(RoverClass, "_pending_adds")
        except Exception:
            pass

    scene = trimesh.Scene()
    added_rover = False

    def _add_geom(m, node_name: str):
        nonlocal added_rover
        if m and not getattr(m, "is_empty", False):
            try:
                scene.add_geometry(m, node_name=node_name)
                if node_name == "Rover" or node_name.startswith("Rover/"):
                    added_rover = True
            except Exception as e:
                print(f"[scene] add {node_name} failed:", e)

    def _get_shape(component):
        for attr in (
            "world_obj",
            "toCompound",
            "obj",
            "to_cadquery",
            "shape",
            "local_obj",
            "make",
        ):
            if hasattr(component, attr):
                try:
                    v = getattr(component, attr)
                    shp = v() if callable(v) else v
                    if shp is not None:
                        return shp
                except Exception as e:
                    print(
                        f"[get_shape] {component.__class__.__name__}.{attr} failed:", e
                    )
        return None

    def _iter_components(root):
        comps = getattr(root, "components", None)
        if isinstance(comps, dict):
            return comps.items()
        if comps:
            try:
                return list(comps)
            except Exception:
                pass
        return []

    def _walk(node, prefix="Rover"):
        shp = _get_shape(node)
        if shp is not None:
            tm = _cq_to_trimesh(shp, tol=0.6)
            _add_geom(tm, prefix)
        for child_name, child in _iter_components(node):
            _walk(child, f"{prefix}/{child_name}")

    whole = None
    for attr in ("world_obj", "toCompound", "obj", "to_cadquery"):
        if hasattr(rv, attr):
            try:
                v = getattr(rv, attr)
                whole = v() if callable(v) else v
                if whole is not None:
                    break
            except Exception as e:
                print(f"[asm] rv.{attr} failed:", e)
    if whole is not None:
        mesh = _cq_to_trimesh(whole, tol=0.6)
        if mesh and not getattr(mesh, "is_empty", False):
            _add_geom(mesh, "Rover")
        else:
            _walk(rv, "Rover")
    else:
        _walk(rv, "Rover")

    def _emit_parametric_wheels(scene):
        if _truthy(CURRENT_PARAMS.get("hide_wheels")):
            print("[wheels] hidden via hide_wheels")
            return

        n_side = int(_num(CURRENT_PARAMS.get("wheels_per_side"), 1))
        n_side = max(1, min(n_side, 6))
        axle = _num(CURRENT_PARAMS.get("axle_spacing_mm"), 160.0)
        span = _num(CURRENT_PARAMS.get("wheelbase_span_mm"), 280.0)
        diam = _num(
            CURRENT_PARAMS.get("wheel_diameter"),
            _num(getattr(_ThisWheel, "diameter", None), 120.0),
        )
        width = _num(
            CURRENT_PARAMS.get("wheel_width"),
            _num(getattr(_ThisWheel, "width", None), 35.0),
        )

        try:
            setattr(_ThisWheel, "diameter", diam)
            setattr(_ThisWheel, "width", width)
        except Exception:
            pass

        # get a base shape for the wheel
        part = None
        try:
            part = _ThisWheel()
        except Exception:
            part = _ThisWheel
        shp = None
        for attr in (
            "world_obj",
            "toCompound",
            "obj",
            "to_cadquery",
            "shape",
            "local_obj",
            "make",
        ):
            if hasattr(part, attr):
                try:
                    v = getattr(part, attr)
                    shp = v() if callable(v) else v
                    if shp is not None:
                        break
                except Exception:
                    pass
        if shp is None:
            shp = cq.Workplane("XY").circle(diam * 0.5).extrude(width)
        base_tm = _cq_to_trimesh(shp, tol=0.6)
        if not base_tm or getattr(base_tm, "is_empty", False):
            return

        # positions along axle (x) and along span (y)
        if n_side == 1:
            xs = [0.0]
        elif n_side == 2:
            xs = [-axle * 0.5, axle * 0.5]
        else:
            half = (n_side - 1) / 2.0
            xs = [axle * (i - half) for i in range(n_side)]

        # NEW: mount mode — center vs bottom
        mount = (CONTEXT.get("wheel_attach") or "center").lower()
        center_mount = ("center" in mount) or ("mid" in mount)
        z = float(CURRENT_PARAMS.get("wheel_z_offset_mm") or 0.0)
        if not center_mount:
            z = (diam * 0.5) + z  # bottom-at-ground legacy

        # NEW: optional left/right mirror (swap sides)
        side_sign = -1.0 if _truthy(CURRENT_PARAMS.get("mirror_lr")) else 1.0
        y_off = span * 0.5

        def add_copy(name, x, y, zc):
            m = base_tm.copy()
            m.apply_translation([x, y, zc])
            try:
                scene.add_geometry(m, node_name=name, geom_name=f"{name}")
            except Exception as e:
                print(f"[glb wheels] add {name} failed:", e)

        # Left side uses +y; Right uses -y (then optionally mirrored)
        for i, x in enumerate(xs):
            add_copy(f"wheel/L{i+1}", x, +y_off * side_sign, z)
        for i, x in enumerate(xs):
            add_copy(f"wheel/R{i+1}", x, -y_off * side_sign, z)

        _apply_rotation_to_wheels(scene)

    _emit_parametric_wheels(scene)
    _emit_missing_proxies(scene)

    # consume queued adds
    def _safe_float(x, default):
        return _num(x, default)

    def _add_sensor_fork(params: Dict[str, Any]):
        width = max(5.0, min(_safe_float(params.get("width_mm"), 40.0), 500.0))
        depth = max(5.0, min(_safe_float(params.get("depth_mm"), 25.0), 500.0))
        height = max(5.0, min(_safe_float(params.get("height_mm"), 30.0), 500.0))
        wall = max(1.0, min(_safe_float(params.get("wall_mm"), 3.0), 25.0))
        hole_d = max(1.0, min(_safe_float(params.get("hole_diam_mm"), 3.2), 25.0))
        try:
            part = SensorFork()
        except Exception as e_ctor:
            print("[sensor_fork] no-arg construction failed:", e_ctor)
            return
        for k, v in (
            ("width", width),
            ("depth", depth),
            ("height", height),
            ("wall", wall),
            ("hole_diam", hole_d),
        ):
            try:
                setattr(part, k, v)
            except Exception as ee:
                print(f"[sensor_fork] setattr {k}={v} failed:", ee)
        try:
            shp = part.make() if hasattr(part, "make") else getattr(part, "shape", None)
        except Exception as e_make:
            print("[sensor_fork] make() failed, fallback box:", e_make)
            shp = cq.Workplane("XY").box(
                width, depth, height, centered=(True, True, False)
            )
        if shp is None:
            return
        pos = params.get("position_mm")
        if not (
            isinstance(pos, (list, tuple))
            and len(pos) == 3
            and all(isinstance(t, (int, float)) for t in pos)
        ):
            pos = [220.0, 0.0, 160.0]
        ori = params.get("orientation_deg")
        if not (
            isinstance(ori, (list, tuple))
            and len(ori) == 3
            and all(isinstance(t, (int, float)) for t in ori)
        ):
            ori = [0.0, 0.0, 0.0]
        if CONTEXT.get("terrain_mode") == "uneven":
            ori = [ori[0] or 6.0, ori[1] or 0.0, ori[2] or 0.0]
            pos = [pos[0], pos[1], pos[2] + 10.0]
        w = shp.rotate((0, 0, 0), (1, 0, 0), float(ori[0]))
        w = w.rotate((0, 0, 0), (0, 1, 0), float(ori[1]))
        w = w.rotate((0, 0, 0), (0, 0, 1), float(ori[2]))
        w = w.translate(tuple(float(x) for x in pos))
        tm = _cq_to_trimesh(w, tol=0.5)
        if tm:
            try:
                scene.add_geometry(tm, node_name="sensor_fork")
            except Exception as e:
                print("[sensor_fork] add geom failed:", e)

    for op in list(PENDING_ADDS):
        kind = (op.get("kind") or "").lower()
        params = op.get("params") or {}
        if kind == "sensor_fork":
            _add_sensor_fork(params)
        elif kind == "wheel":
            pass
        else:
            print(f"[adds] unknown kind '{kind}', skipping")
    PENDING_ADDS.clear()

    # yaw rotate rover + delete hidden prefixes
    try:
        yaw = float(CURRENT_PARAMS.get("rover_yaw_deg") or 0.0)
    except Exception:
        yaw = 0.0
    if abs(yaw) > 1e-6:
        R = trimesh.transformations.euler_matrix(0.0, 0.0, math.radians(yaw))
        for name, geom in list(scene.geometry.items()):
            g = geom.copy()
            g.apply_transform(R)
            scene.geometry[name] = g

    if HIDDEN_PREFIXES:
        to_delete = []
        for name in list(scene.geometry.keys()):
            low = name.lower()
            if any(
                low.startswith(pref) or f"/{pref}/" in low for pref in HIDDEN_PREFIXES
            ):
                to_delete.append(name)
        for name in to_delete:
            try:
                del scene.geometry[name]
            except Exception:
                pass

    if not scene.geometry:
        if os.path.exists(ROVER_GLB_PATH):
            with open(ROVER_GLB_PATH, "rb") as f:
                return f.read()
        raise RuntimeError("No geometry exported")

    return scene.export(file_type="glb")


def build_rover_scene_glb(_: Optional[Dict[str, Any]] = None) -> bytes:
    if USE_CQPARTS:
        return build_rover_scene_glb_cqparts()
    if os.path.exists(ROVER_GLB_PATH):
        with open(ROVER_GLB_PATH, "rb") as f:
            return f.read()
    raise FileNotFoundError("cqparts disabled, and assets/rover.glb not found")


def _reload_rover_from_generated():
    """
    Dynamically reload the Rover class from generated/robot_base_vlm.py if it exists.
    Returns the Rover class (either new or original).
    """
    gen_path = os.path.join(BASE_DIR, "generated", "robot_base_vlm.py")
    
    if not os.path.exists(gen_path):
        print("[reload] No generated code found, using original Rover")
        return Rover
    
    try:
        print(f"[reload] Loading Rover from {gen_path}...")
        import importlib.util
        
        
        # Load the module from the generated file
        spec = importlib.util.spec_from_file_location("robot_base_vlm", gen_path)
        if spec is None or spec.loader is None:
            print("[reload] ✗ Failed to create module spec")
            return Rover
            
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules temporarily so imports work
        sys.modules["robot_base_vlm"] = module
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Get the Rover class
        if hasattr(module, 'Rover'):
            new_rover = module.Rover
            print(f"[reload] ✓ Loaded Rover from generated code")
            return new_rover
        elif hasattr(module, 'RobotBase'):
            new_rover = module.RobotBase
            print(f"[reload] ✓ Loaded RobotBase from generated code")
            return new_rover
        else:
            print("[reload] ✗ Generated code has no Rover or RobotBase class")
            return Rover
            
    except Exception as e:
        print(f"[reload] ✗ Failed to load generated code: {e}")
        import traceback
        traceback.print_exc()
        return Rover


def _rebuild_and_save_glb(use_generated=False):
    """
    Rebuild the GLB file.
    If use_generated=True, uses generated/robot_base_vlm.py instead of robot_base.py
    """
    # Reload Rover from generated code if requested
    RoverClass = _reload_rover_from_generated() if use_generated else Rover
    
    # Build the GLB with the selected Rover class
    glb = build_rover_scene_glb_cqparts(RoverClass=RoverClass)
    
    with open(ROVER_GLB_PATH, "wb") as f:
        f.write(glb)
    print(f"[rebuild] ✓ Saved GLB to {ROVER_GLB_PATH}")

if __name__ == "__main__":
    import sys
    
    # Check for test flag - run test BEFORE other imports that might fail
    if "--test-vlm" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "test-vlm"):
        from app.utils.test_vlm import test_vlm_model
        include_mesh = "--no-mesh" not in sys.argv
        try:
            test_vlm_model(include_mesh_analysis=include_mesh)
            sys.exit(0)
        except Exception as e:
            print(f"\n[test] ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # VLM setup - prefer Ollama if available, otherwise fine-tuned model
    # Check if Ollama is available and has the requested model
    ollama_available = False
    ollama_model_exists = False
    if OLLAMA_URL:
        try:
            import requests
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                ollama_available = True
                # Check if the requested model exists
                models_data = r.json()
                model_names = [m.get("name", "") for m in models_data.get("models", [])]
                ollama_model_exists = OLLAMA_MODEL in model_names
                if ollama_model_exists:
                    print(f"[startup] ✓ Ollama is available (URL: {OLLAMA_URL}, Model: {OLLAMA_MODEL})")
                else:
                    print(f"[startup] ⚠ Ollama is available but model '{OLLAMA_MODEL}' not found")
                    print(f"[startup] Available models: {', '.join(model_names)}")
                    print(f"[startup] Will use fine-tuned model instead")
            else:
                print(f"[startup] ⚠ Ollama URL responded with {r.status_code}")
        except Exception as e:
            print(f"[startup] ⚠ Ollama not available: {e}")
            print("[startup] Make sure Ollama is running: ollama serve")
    
    # Only use Ollama if it's available AND has the requested model
    # Otherwise, fall back to fine-tuned model
    if not (ollama_available and ollama_model_exists) and USE_FINETUNED_MODEL:
        print(f"[startup] Preloading fine-tuned VLM model...")
        print(f"[startup] Model path: {FINETUNED_MODEL_PATH}")
        print(f"[startup] Model path exists: {os.path.exists(FINETUNED_MODEL_PATH)}")
        print("[startup] Loading model (base model will be loaded from cache if available)...")
        try:
            load_finetuned_model()
            if _finetuned_model is not None:
                print("[startup] ✓ VLM model preloaded and ready")
            else:
                print("[startup] ⚠ VLM model not loaded (will load on first use)")
        except Exception as e:
            print(f"[startup] ⚠ Could not preload VLM model: {e}")
            print("[startup] Model will be loaded on first use")
    elif ollama_available and ollama_model_exists:
        print(f"[startup] ✓ Using Ollama VLM (Model: {OLLAMA_MODEL})")
        if USE_FINETUNED_MODEL and os.path.exists(FINETUNED_MODEL_PATH):
            print(f"[startup] Fine-tuned model available at {FINETUNED_MODEL_PATH} but using Ollama")
    else:
        if ollama_available and not ollama_model_exists:
            print(f"[startup] ⚠ Ollama available but model '{OLLAMA_MODEL}' not found")
            print(f"[startup] To use Ollama: ollama pull {OLLAMA_MODEL}")
            if USE_FINETUNED_MODEL and os.path.exists(FINETUNED_MODEL_PATH):
                print(f"[startup] Will use fine-tuned model instead (lazy load on first use)")
        else:
            print("[startup] ⚠ No VLM configured - VLM features will not work")
    
    # Disabled warm build - don't build rover on startup
    # Wait for user to upload mesh or parametric model instead
    # threading.Thread(target=_warm_build, daemon=True).start()
    print("[startup] Skipping warm build - waiting for user to upload mesh or parametric model", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False)
