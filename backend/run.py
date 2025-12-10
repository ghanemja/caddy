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
from app.routes import main, api, vlm, mesh, model
app.register_blueprint(main.bp)  # Routes: /, /debug
app.register_blueprint(api.bp)  # Routes: /state, /apply, /params, etc.
app.register_blueprint(vlm.bp)  # Routes: /codegen, /vlm, /recommend
app.register_blueprint(mesh.bp, url_prefix="/api/mesh")
app.register_blueprint(model.bp, url_prefix="/api/model")

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


def _snapshot():
    snap = {}
    for k, v in CURRENT_PARAMS.items():
        if v is None:
            snap[k] = None
        elif isinstance(v, (int, float)):
            snap[k] = float(v)
        elif isinstance(v, (list, tuple)):
            snap[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
        else:
            try:
                snap[k] = float(v)
            except Exception:
                snap[k] = str(v)
    return snap


def _ensure_initial_history():
    global INIT_SNAPSHOT, HISTORY, H_PTR
    if H_PTR == -1:
        INIT_SNAPSHOT = _snapshot()
        HISTORY = [INIT_SNAPSHOT.copy()]
        H_PTR = 0


def _push_history():
    _ensure_initial_history()
    global H_PTR, HISTORY
    if H_PTR < len(HISTORY) - 1:
        HISTORY = HISTORY[: H_PTR + 1]
    HISTORY.append(_snapshot())
    H_PTR = len(HISTORY) - 1


def _restore(snapshot: Dict[str, Optional[float]]):
    for k in CURRENT_PARAMS.keys():
        CURRENT_PARAMS[k] = snapshot.get(k, CURRENT_PARAMS[k])


def _normalize_params(target: str, action: str, params: dict) -> dict:
    p = {}
    params = params or {}
    tgt = (target or "").lower()

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "on", "y")

    for k, v in params.items():
        kk = str(k).strip().lower().replace(" ", "_")

        # common numeric aliases
        if kk in ("x", "dx", "x_mm"):
            p["dx_mm"] = _strip_units_to_float(v)
        elif kk in ("y", "dy", "y_mm"):
            p["dy_mm"] = _strip_units_to_float(v)
        elif kk in ("z", "dz", "z_mm"):
            p["dz_mm"] = _strip_units_to_float(v)
        elif kk in ("diameter", "diameter_mm"):
            p["wheel_diameter"] = _strip_units_to_float(v)
        elif kk in (
            "width_mm",
            "height_mm",
            "depth_mm",
            "wall_mm",
            "hole_diam_mm",
            "axle_spacing_mm",
            "wheelbase_span_mm",
            "wheels_per_side",
            "wheel_diameter",
            "wheel_width",
            "wheel_z_offset_mm",
        ):
            p[kk] = _strip_units_to_float(v)
        elif kk in ("position_mm", "orientation_deg"):
            if isinstance(v, (list, tuple)) and len(v) in (2, 3):
                p[kk] = [_strip_units_to_float(x) for x in v]
            else:
                p[kk] = v
        elif kk in ("mirror_lr", "flip_lr", "swap_sides"):
            p["mirror_lr"] = _as_bool(v)
        elif kk in ("wheel_mount", "wheel_attach", "mount"):
            # store as a plain string; we'll fold into CONTEXT later
            vs = str(v).strip().lower()
            if "center" in vs or "mid" in vs:
                p["wheel_attach"] = "center"
            elif "bottom" in vs or "ground" in vs:
                p["wheel_attach"] = "bottom"
        else:
            num = _strip_units_to_float(v)
            p[kk] = num if num is not None else v

    # wheel percent/factor helpers
    if tgt == "wheel":
        base_diam = CURRENT_PARAMS.get("wheel_diameter") or getattr(
            _ThisWheel, "diameter", None
        )
        base_width = CURRENT_PARAMS.get("wheel_width") or getattr(
            _ThisWheel, "width", None
        )
        for key in ("scale", "factor", "percent", "diameter_percent", "wheel_diameter"):
            if key in params and base_diam is not None:
                val = (
                    _percent_to_abs(params.get(key), base_diam)
                    if key != "wheel_diameter"
                    else _strip_units_to_float(params[key])
                )
                if val is not None:
                    p["wheel_diameter"] = float(val)
                    break
        for key in ("width_percent", "width_scale", "width_factor"):
            if key in params and base_width is not None:
                neww = _percent_to_abs(params.get(key), base_width)
                if neww is not None:
                    p["wheel_width"] = neww
                    break

    # yaw on rover/base: accept several aliases (e.g. base_yaw_deg)
    if tgt == "rover" and (
        action == "rotate"
        or any(
            k in params
            for k in (
                "rover_yaw_deg",
                "base_yaw_deg",
                "yaw",
                "rz",
                "angle",
                "angle_deg",
            )
        )
    ):
        yaw = (
            params.get("rover_yaw_deg")
            or params.get("base_yaw_deg")
            or params.get("yaw")
            or params.get("rz")
            or params.get("angle")
            or params.get("angle_deg")
        )
        if yaw is not None:
            p["rover_yaw_deg"] = _strip_units_to_float(yaw)

    # pass-through of a few keys that aren't purely numeric
    for k in ("rover_yaw_deg", "wheel_z_offset_mm", "hide_wheels"):
        if k in params and params[k] is not None:
            p[k] = params[k]

    return p


def _normalize_change(ch: dict) -> Optional[dict]:
    if not isinstance(ch, dict):
        return None
    action = ACTION_ALIASES.get(
        (ch.get("action") or "").strip().lower(),
        (ch.get("action") or "").strip().lower(),
    )
    target = TARGET_ALIASES.get(
        (ch.get("target_component") or "").strip().lower(),
        (ch.get("target_component") or "").strip().lower(),
    )
    params = _normalize_params(target, action, ch.get("parameters") or {})
    if action == "modify" and "wheels_per_side" in (ch.get("action", "").lower()):
        params.setdefault(
            "wheels_per_side",
            _strip_units_to_float(ch.get("parameters", {}).get("wheels_per_side")),
        )
    if not target or not action:
        return None
    out = dict(ch)
    out["action"] = action
    out["target_component"] = target
    out["parameters"] = params
    return out


def _coerce_changes(payload: Any) -> List[dict]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "response" in payload and isinstance(payload["response"], dict):
            payload = payload["response"].get("json") or payload
        if "actions" in payload or "changes" in payload:
            payload = payload.get("actions") or payload.get("changes")
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


# JSON/unit tolerant parsing
def _mm_from_value_token(tok: str) -> str:
    s = tok.strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*m", s, flags=re.IGNORECASE)
    if m:
        mm = float(m.group(1)) * 1000.0
        return str(int(mm)) if abs(mm - round(mm)) < 1e-9 else str(mm)
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*mm", s, flags=re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)", s)
    if m:
        v = float(m.group(1))
        return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)
    return tok


def _repair_units_in_json_text(block: str) -> str:
    def repl(match):
        prefix = match.group(1)
        token = match.group(2).strip().strip('"').strip("'")
        return f"{prefix}{_mm_from_value_token(token)}"

    pattern = r'(:\s*)(-?\d+(?:\.\d+)?\s*(?:m|mm)|"-?\d+(?:\.\d+)?\s*(?:m|mm)"|\'-?\d+(?:\.\d+)?\s*(?:m|mm)\')(?=\s*[,\}\]])'
    return re.sub(pattern, repl, block, flags=re.IGNORECASE)


def _find_all_balanced_json_blocks(text: str):
    spans = []
    if not text:
        return spans
    in_string = False
    escape = False
    stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            if len(stack) == 1:
                start_idx = i
        elif ch in "}]":
            if stack:
                opening = stack[-1]
                if (opening == "{" and ch == "}") or (opening == "[" and ch == "]"):
                    stack.pop()
                    if not stack and start_idx is not None:
                        spans.append((start_idx, i))
                        start_idx = None
                else:
                    stack.clear()
                    start_idx = None
    return spans


def _split_multi_json_and_summaries(raw_text: str):
    """
    Accepts model output that may contain multiple JSON objects/arrays (possibly fenced),
    followed by one or more SUMMARY: lines.
    Returns (list_of_change_dicts or None, list_of_summaries, list_of_raw_json_blocks_kept).
    """
    if not raw_text:
        return None, [], []

    # Tolerate code fences & leading chatter
    text = raw_text
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="ignore")
    text = text.strip()
    # Strip Markdown fences and tokens if present
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    # Remove any leading non-{[ … so we start at the first JSON block if model spoke first
    text = re.sub(r"^[^\[\{]*", "", text, count=1).strip()

    # Collect SUMMARY lines (keep all of them)
    summaries = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.upper().startswith("SUMMARY:"):
            summaries.append(s[len("SUMMARY:") :].strip())

    # Only parse blocks BEFORE the first SUMMARY
    first_summary_pos = text.upper().find("SUMMARY:")
    parse_zone = text if first_summary_pos == -1 else text[:first_summary_pos]

    # Find ALL balanced JSON blocks (objects or arrays)
    blocks = []
    for s, e in _find_all_balanced_json_blocks(parse_zone):
        blocks.append(parse_zone[s : e + 1])

    parsed_changes = []
    kept_blocks = []

    for block in blocks:
        # Normalize units inside JSON-like text (e.g. "2.5m" -> "2500")
        fixed_units = _repair_units_in_json_text(block)

        # Be tolerant to single quotes / Python literals and None/True/False
        candidates = [
            fixed_units,
            fixed_units.replace("'", '"'),
            fixed_units.replace(" None", " null")
            .replace(": None", ": null")
            .replace(" True", " true")
            .replace(": True", ": true")
            .replace(" False", " false")
            .replace(": False", ": false"),
        ]

        obj = None
        for cand in candidates:
            try:
                obj = json.loads(cand)
                break
            except Exception:
                try:
                    obj = ast.literal_eval(cand)
                    break
                except Exception:
                    obj = None

        if obj is None:
            # Log and keep going—don't fail the whole parse
            print("[recommend][parser] could not parse block:\n", block)
            continue

        kept_blocks.append(block)

        # Flatten to a list of dicts
        if isinstance(obj, list):
            parsed_changes += [it for it in obj if isinstance(it, dict)]
        elif isinstance(obj, dict):
            parsed_changes.append(obj)

    return (parsed_changes or None), summaries, kept_blocks


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


# _clean_num is now imported from app.utils.helpers


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

VLM_SYSTEM_PROMPT = """You are a vision model proposing JSON edits for a parametric rover (CadQuery/cqparts).

INPUTS
- (0) REFERENCE image — intended appearance.
- (1) SNAPSHOT image — current CAD rendering.
- (2) CAD STATE JSON — current parameters, known classes, selection, history, pending adds.

OBJECTIVE
Compare REFERENCE vs SNAPSHOT and propose the smallest set of conservative changes that move CAD toward REFERENCE.

ABSOLUTE OUTPUT CONTRACT — DO NOT VIOLATE
1) Output ONLY a JSON ARRAY (no wrapper object, no keys like "changes", no markdown).
2) Then output a single blank line.
3) Then output exactly one line starting with: SUMMARY: <brief one sentence>.
4) Use only double quotes. No comments. No trailing commas. No units in numbers.
5) If no changes are justified, output [] then the SUMMARY line.
6) Do NOT include any additional text, headings, rationale paragraphs, or code fences.

ALLOWED OBJECT SHAPE (whitelisted keys only)
Each array element MUST be an object with these keys ONLY:
{
  "target_component": "<one of: rover | wheel | pan_tilt | sensor_fork>",
  "action": "<one of: modify | add | delete | rotate | translate | resize | replace>",
  "parameters": { /* FLAT key→value, whitelisted below */ },
  "rationale": "<one brief sentence>",
  "title": "<optional short label, <= 8 words>",
  "confidence": <number 0..1>
}

PARAMETER WHITELIST + RANGES (drop everything else)
- rover:
  - rover_yaw_deg: number in [-180, 180]
- wheel:
  - wheel_diameter: number in [20, 400]          # mm
  - wheel_width:    number in [10, 120]          # mm
  - wheels_per_side:number in [1, 6] integer
  - axle_spacing_mm:number in [80, 600]
  - wheelbase_span_mm:number in [120, 800]
  - wheel_z_offset_mm:number in [-80, 80]
- pan_tilt:
  - pan_tilt_offset_x: number in [-200, 200]
  - pan_tilt_offset_y: number in [-200, 200]
  - pan_tilt_offset_z: number in [-200, 200]
- sensor_fork (for add/modify only):
  - width_mm:      number in [5, 500]
  - height_mm:     number in [5, 500]
  - depth_mm:      number in [5, 500]
  - wall_mm:       number in [1, 25]
  - hole_diam_mm:  number in [1, 25]
  - position_mm:   array of 3 numbers [x,y,z] (omit if unknown)
  - orientation_deg: array of 3 numbers [rx,ry,rz] (omit if unknown)

DISALLOWED / NEVER OUTPUT
- Any key not listed above (e.g., "wheels_", "wheel_positions", "layout", "scale", "factor", "percent", "units").
- Null values, empty arrays, or placeholder vectors like [0,0,0].
- Duplicate or contradictory changes.
- More than 5 change objects total.

DELETE SEMANTICS
- { target_component: "sensor_fork", action: "delete" } → remove most recently added sensor_fork
- { target_component: "wheel", action: "delete" }       → hide wheels
- { target_component: "pan_tilt", action: "delete" }    → hide pan-tilt

GUIDELINES
- Prefer small deltas when uncertain.
- Use wheels_per_side (typical 1–3) when wheel count must change.
- Keep numbers unitless (mm/deg), already normalized.
- If you’re not confident, reduce change magnitude and lower confidence.
- If reference and snapshot already match for a component, omit it.

FEW-SHOT EXAMPLES (for format only; DO NOT COPY VALUES VERBATIM)

GOOD:
[
  {
    "target_component": "wheel",
    "action": "modify",
    "parameters": { "wheels_per_side": 2, "wheel_diameter": 130, "wheel_width": 40 },
    "rationale": "Wheel size/count appear slightly larger than snapshot.",
    "title": "Wheels: 2/side @130×40",
    "confidence": 0.72
  }
]

GOOD (no changes):
[]

BAD (do NOT do these):
- {"changes":[ ... ]}                # wrapper object
- [{"target":"roverbase", ...}]      # unknown target name
- extra keys like "wheels_", "layout", "units"
- numbers with units like "120mm" or "15 deg"
- [ ... ] plus any prose or markdown

SELF-CHECK BEFORE YOU PRINT
- Are ALL objects limited to the whitelisted keys?
- Are ALL parameter names whitelisted for that component?
- Are ALL numbers within the specified ranges and unitless?
- Is the top-level a pure JSON ARRAY (no wrapper object)?
- Is there a blank line and then exactly one SUMMARY: line?
If any answer is NO → output [] and: SUMMARY: No change.

Remember: emit ONLY the array, blank line, and SUMMARY line."""

VLM_CODEGEN_PROMPT = """You are a CAD code generator that modifies parametric Python code for CadQuery/cqparts framework.

=== YOUR TASK ===
You will receive:
1. REFERENCE IMAGE(s) - showing the desired robot rover design
2. CURRENT CAD SNAPSHOT - orthogonal views of the current CAD model
3. BASELINE PYTHON SOURCE - the current robot_base.py implementation
4. USER INTENT/INSTRUCTION - specific modification request from user

Your job: Copy the baseline source and modify ONLY the parameter values that the user requested OR that are needed to match the reference image.

=== CRITICAL OUTPUT RULES ===
⚠️ Output ONLY valid Python code - NO explanations, NO markdown fences (```), NO extra text
⚠️ Start immediately with #!/usr/bin/env python3
⚠️ COPY the ENTIRE baseline source (all imports, classes, methods)
⚠️ ONLY modify parameter VALUES (numbers in PositiveFloat(...)) where needed
⚠️ If user asks for specific change (e.g., "set wheels to 4"), make ONLY that change
⚠️ If user just says "match the image", identify what differs and change those parameters
⚠️ DO NOT modify method implementations (make_components, make_constraints, etc.)
⚠️ DO NOT add/remove classes or methods
⚠️ Keep all comments, imports, and structure identical

=== CQPARTS FRAMEWORK UNDERSTANDING ===

**What cqparts IS:**
- A parametric CAD assembly framework built on CadQuery
- Uses `cqparts.Assembly` for assemblies, `cqparts.Part` for parts
- `make_components()` creates child components and returns a dict
- `make_constraints()` defines spatial relationships using Mates and returns a list
- `PartRef()` is a reference to a component class (DO NOT modify these!)

**CRITICAL - DO NOT REWRITE:**
1. The `make_components()` method logic - just copy it as-is
2. The `make_constraints()` method logic - just copy it as-is  
3. Any `MountedStepper` instantiations - copy exactly
4. Any `Mate()` or `CoordSystem()` calls - copy exactly
5. The `_axle_offsets()` helper method - copy exactly
6. Import statements - copy exactly

**WHAT TO CHANGE:**
Only parameter VALUES in class definitions:
- `wheels_per_side = PositiveFloat(2)` → Change the 2
- `length = PositiveFloat(280)` → Change the 280
- `wheel_diameter = PositiveFloat(120)` → Change the 120

**⚠️ TRANSLATING USER REQUESTS TO PARAMETER CHANGES:**

READ THE USER INSTRUCTION CAREFULLY and map it to specific parameter changes:

"remove all wheels" / "no wheels" → wheels_per_side = PositiveFloat(0)
"3 wheels per side" / "6 wheels total" → wheels_per_side = PositiveFloat(3)
"increase spacing" / "more space between wheels" → axle_spacing_mm = PositiveFloat(LARGER NUMBER, e.g., 70→90)
"wheels closer" / "less space" → axle_spacing_mm = PositiveFloat(SMALLER NUMBER, e.g., 70→50)
"bigger wheels" / "100mm diameter" → diameter = PositiveFloat(100) in ThisWheel class
"thicker wheels" / "wider wheels" → thickness = PositiveFloat(20) in ThisWheel class
"longer base" → length = PositiveFloat(LARGER, e.g., 280→350)
"wider base" → width = PositiveFloat(LARGER, e.g., 170→220)

**ARITHMETIC CHANGES (requires calculation):**
⚠️ When user says "X mm smaller/larger", you MUST do the math:

"diameter 15mm smaller" → Look at baseline diameter (90), calculate 90-15=75, set diameter = PositiveFloat(75)
"diameter 20mm larger" → Look at baseline diameter (90), calculate 90+20=110, set diameter = PositiveFloat(110)
"spacing 10mm more" → Look at baseline axle_spacing_mm (70), calculate 70+10=80, set axle_spacing_mm = PositiveFloat(80)
"spacing 20mm less" → Look at baseline axle_spacing_mm (70), calculate 70-20=50, set axle_spacing_mm = PositiveFloat(50)

IMPORTANT: Use the BASELINE value (from robot_base.py above) for calculations, NOT the current CAD state!

**Example - CORRECT search-replace output:**
```json
[
  {
    "search": "    wheels_per_side = PositiveFloat(6)  # default 6 per side (12 total)",
    "replace": "    wheels_per_side = PositiveFloat(4)  # default 4 per side (8 total)",
    "reason": "Reduce wheels from 6 to 4 per side"
  }
]
```

**Example - WRONG:**
```json
[
  {
    "search": "wheels_per_side",  // ❌ Not specific enough
    "replace": "wheels_per_side = PositiveFloat(4)"  // ❌ Must copy EXACT line from baseline
  },
  {
    "search": "wheels_per_side = PositiveFloat(2)",  // ❌ Wrong value! Baseline has 6, not 2
    "replace": "wheels_per_side = PositiveFloat(4)"
  }
]
```

=== SEARCH STRING RULES - CRITICAL ===
⚠️ Copy the EXACT line from the BASELINE SOURCE above (including comments!)
⚠️ Do NOT use values from examples - use the ACTUAL values you see in baseline
⚠️ Include exact whitespace (spaces/tabs) at start of line
⚠️ Include any trailing comments if present
⚠️ Copy the entire line character-for-character from baseline source

**How to get it right:**
1. FIND the parameter in baseline source (e.g., search for "wheels_per_side")
2. COPY that exact line (with spaces, comments, everything)
3. PASTE into "search"
4. MODIFY only the number in "replace" (keep structure/comments identical)

=== STEP-BY-STEP PROCESS ===
1. LOOK at the REFERENCE image - what does the rover look like?
2. LOOK at the SNAPSHOT image (if provided) - what does it look like now?
3. IDENTIFY the differences (more wheels? bigger? longer?)
4. FIND the parameters in baseline source that control those aspects
5. OUTPUT JSON array with search-replace pairs for ONLY those parameters
6. Maximum 10 changes - be selective and focused

=== COMMON PARAMETERS TO MODIFY ===

**For wheel count changes:**
- `wheels_per_side` in Rover class (e.g., 2 → 3 for 6 wheels total)

**For chassis size:**
- `length` and `width` in Rover or RobotBase classes
- Keep length > width typically

**For wheel size:**
- `diameter` and `thickness` in ThisWheel class

**For wheel spacing:**
- `axle_spacing_mm` (space between adjacent wheels)
- `wheelbase_span_mm` (total span, or leave at 0 for auto)

=== VALIDATION CHECKLIST ===
Before outputting your JSON array, verify:
- [ ] Each "search" string has exact indentation from baseline
- [ ] Each "search" string is unique (won't match multiple lines)
- [ ] Each "replace" string has same indentation as search
- [ ] Only parameter VALUES changed (not variable names or structure)
- [ ] Valid JSON syntax (double quotes, no trailing commas)
- [ ] Maximum 10 changes (be selective!)
- [ ] Each change has a brief "reason"

=== YOUR INPUTS FOLLOW ===
"""

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
        VLM_CODEGEN_PROMPT,
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
def _data_url_from_upload(file_storage) -> Optional[str]:
    if not file_storage:
        return None
    raw = file_storage.read()
    mime = file_storage.mimetype or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


# _stitch_images_side_by_side and call_vlm are now imported from app.services.vlm_service


# Code generation utilities are now in app.utils.codegen
from app.utils.codegen import normalize_generated_code, extract_python_module

def _normalize_generated_code_advanced(code: str) -> str:
    """
    Advanced normalization with CAD-specific fixes.
    This extends the basic normalize_generated_code from codegen.py.
    """
    # Start with basic normalization
    code = normalize_generated_code(code)
    
    print("[normalize] Applying CAD-specific fixes...")
    fixes_applied = []
    
    # CAD-specific fixes (keep these here as they're domain-specific)
    required_imports = [
        "import cadquery as cq",
        "import cqparts",
        "from cqparts.params import PositiveFloat",
        "from cqparts.display import render_props",
        "from cqparts.constraint import Fixed, Coincident, Mate",
        "from cqparts.utils.geometry import CoordSystem",
        "from cqparts.search import register",
        "from partref import PartRef",
        "from manufacture import Lasercut",
        "from motor_mount import MountedStepper",
        "from cqparts_motors.stepper import Stepper",
        "from wheel import SpokeWheel",
        "from electronics import type1 as Electronics",
        "from pan_tilt import PanTilt",
    ]
    
    # Check if file is missing shebang and imports
    if not code.strip().startswith("#!/usr/bin/env python3"):
        missing_imports = [imp for imp in required_imports if imp not in code]
        if missing_imports:
            header = "#!/usr/bin/env python3\n\n" + "\n".join(required_imports) + "\n\n"
            code = header + code
            fixes_applied.append(f"Added missing imports ({len(missing_imports)} imports restored)")
            print(f"[normalize] ✗ VLM truncated file - restored {len(missing_imports)} missing imports")
    
    # Fix RobotBase class if missing
    if "class RobotBase" not in code and "class ThisWheel" in code:
        robot_base_class = '''class RobotBase(Lasercut):
    length = PositiveFloat(280)
    width = PositiveFloat(170)
    chamfer = PositiveFloat(55)
    thickness = PositiveFloat(6)

    def make(self):
        base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
        base = base.edges("|Z and >X").chamfer(self.chamfer)
        return base

    def mate_back(self, offset=5):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, 0, self.thickness),
                xDir=(1, 0, 0),
                normal=(0, 0, 1),
            ),
        )

    def mate_front(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(self.length / 2 - offset, 0, self.thickness),
                xDir=(1, 0, 0),
                normal=(0, 0, 1),
            ),
        )

    def mate_RL(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, self.width / 2, 0),
                xDir=(1, 0, 0),
                normal=(0, 0, -1),
            ),
        )

    def mate_RR(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, -self.width / 2, 0),
                xDir=(-1, 0, 0),
                normal=(0, 0, -1),
            ),
        )

'''
        code = code.replace("class ThisWheel", robot_base_class + "class ThisWheel")
        fixes_applied.append("Added missing RobotBase class")
        print(f"[normalize] ✗ VLM skipped RobotBase - restored it")
    
    # Additional CAD-specific fixes
    hyphen_fixes = {
        r'\.wheelbase_span-mm': '.wheelbase_span_mm',
        r'\.axle_spacing-mm': '.axle_spacing_mm',
        r'\.wheel_z_offset-mm': '.wheel_z_offset_mm',
        r'\.wheel-diameter': '.wheel_diameter',
        r'\.wheel-width': '.wheel_width',
    }
    
    for pattern, replacement in hyphen_fixes.items():
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            fixes_applied.append(f"Fixed hyphenated attribute: {pattern} → {replacement}")
    
    # Fix undefined 'offsets' variable
    if re.search(r'for\s+i,\s+off\s+in\s+enumerate\(offsets\)', code):
        code = re.sub(
            r'for\s+i,\s+off\s+in\s+enumerate\(offsets\)',
            'for i, off in enumerate(self._axle_offsets())',
            code
        )
        fixes_applied.append("Fixed undefined 'offsets' → 'self._axle_offsets()'")
    
    # Fix _axle_offsets to allow 0 wheels
    if 'n = max(1, int(round(float(self.wheels_per_side))))' in code:
        code = code.replace(
            'n = max(1, int(round(float(self.wheels_per_side))))',
            'n = max(0, int(round(float(self.wheels_per_side))))'
        )
        fixes_applied.append("Fixed _axle_offsets to allow 0 wheels")
    
    # Remove trailing incomplete lines
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('cq.display.') or \
           (line.strip().startswith('register(') and 'model=' not in line):
            continue
        cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)
    
    # Detect and truncate VLM hallucinations
    lines = code.split('\n')
    class_names_seen = {}
    truncate_at = None
    
    for i, line in enumerate(lines):
        class_match = re.match(r'^class\s+(\w+)', line)
        if class_match:
            class_name = class_match.group(1)
            if class_name in class_names_seen:
                truncate_at = class_names_seen[class_name]
                print(f"[normalize] ✗ Detected VLM hallucination: class '{class_name}' repeated")
                fixes_applied.append(f"Truncated hallucination: repeated class '{class_name}'")
                break
            else:
                class_names_seen[class_name] = i
    
    if truncate_at is not None:
        code = '\n'.join(lines[:truncate_at])
        if code and not code.endswith('\n'):
            code += '\n'
        code += '\n# === End of generated code ===\n'
    
    if fixes_applied:
        print(f"[normalize] Applied {len(fixes_applied)} CAD-specific fixes")
    
    return code


# extract_python_module is now imported from app.utils.codegen


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


# ----------------- Routes -----------------
# Main routes (/, /debug) moved to app.routes.main blueprint


@app.get("/state")
def state():
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


@app.post("/state/reset")
def state_reset():
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
            app.logger.warning("reset: rebuild failed: %s", e)
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


def _intent_to_changes(text: str) -> list[dict]:
    """
    Converts natural-language text like 'add 6 wheels' or
    'add six wheels on the base and align midpoints' into
    structured change objects for /apply.
    """
    if not text:
        return []
    s = text.strip().lower()

    # number words
    num_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    # detect numeric count
    m = re.search(r"\badd\s+(\d+)\s+wheels?\b", s)
    count = None
    if m:
        count = int(m.group(1))
    else:
        m2 = re.search(r"\badd\s+([a-z]+)\s+wheels?\b", s)
        if m2:
            count = num_words.get(m2.group(1))

    changes = []
    if count:
        changes.append(
            {
                "target_component": "wheel",
                "action": "add",
                "parameters": {"count": count, "wheels_per_side": max(1, count // 2)},
            }
        )

    # midpoint → zero z-offset
    if "midpoint" in s and ("join" in s or "touch" in s) and "base" in s:
        changes.append(
            {
                "target_component": "wheel",
                "action": "modify",
                "parameters": {"wheel_z_offset_mm": 0},
            }
        )

    return changes


def _parse_apply_request():
    """
    Robustly extracts a list[dict] of change objects from the request.
    Accepts:
      - JSON body with {changes: [...]} or {actions: [...]}
      - Raw text containing JSON
      - Plain natural language (fallback to _intent_to_changes)
    """
    excerpt = None
    data = request.get_json(silent=True)

    # --- structured JSON ---
    if isinstance(data, dict):
        excerpt = data.get("excerpt") or data.get("summary")
        payload = data.get("actions") or data.get("changes")

        # nested chat format {response:{json:[…]}}
        if not payload and "response" in data and isinstance(data["response"], dict):
            payload = (
                data["response"].get("json")
                or data["response"].get("actions")
                or data["response"].get("changes")
            )
        items = _coerce_changes(payload)
        changes = [c for c in (_normalize_change(x) for x in items) if c]
        if not changes:
            # fallback from text
            text_src = (
                data.get("prompt") or data.get("text") or data.get("message") or ""
            )
            changes = _intent_to_changes(text_src)
        return changes, excerpt

    # --- list form ---
    if isinstance(data, list):
        return [c for c in (_normalize_change(x) for x in data) if c], None

    # --- form-data ---
    if request.form:
        raw_text = (
            request.form.get("json")
            or request.form.get("changes")
            or request.form.get("actions")
            or request.form.get("prompt")
            or request.form.get("text")
            or ""
        )
        excerpt = request.form.get("excerpt") or request.form.get("summary")
        parsed_list, _, _ = _split_multi_json_and_summaries(raw_text)
        changes = [c for c in (_normalize_change(x) for x in (parsed_list or [])) if c]
        if not changes:
            changes = _intent_to_changes(raw_text)
        return changes, excerpt

    # --- raw text body ---
    raw_body = request.get_data(as_text=True) or ""
    parsed_list, _, _ = _split_multi_json_and_summaries(raw_body)
    changes = [c for c in (_normalize_change(x) for x in (parsed_list or [])) if c]
    if not changes:
        changes = _intent_to_changes(raw_body)
    return changes, None


@app.post("/apply")
def apply_change():
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


@app.get("/mode")
def mode():
    mode = (
        "GLB: assets/rover.glb"
        if os.path.exists(ROVER_GLB_PATH)
        else ("cqparts" if USE_CQPARTS else "fallback")
    )
    return jsonify({"mode": mode})


@app.post("/label")
def label():
    data = request.get_json(force=True, silent=True) or {}
    part = (data.get("part_name") or "").strip()
    if part:
        STATE["selected_parts"].append(part)
        return jsonify(
            {"ok": True, "part": part, "count": len(STATE["selected_parts"])}
        )
    return jsonify({"ok": False, "error": "no part_name"})


@app.get("/labels")
def labels():
    return jsonify({"ok": True, "selected_parts": STATE["selected_parts"]})


# --------- parse helpers (single, robust copy) ----------
def _extract_json_loose(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])\s*$", text.strip())
        if m:
            block = m.group(1)
            try:
                return json.loads(block)
            except Exception:
                try:
                    return ast.literal_eval(block)
                except Exception:
                    return None
    return None


def _split_multi_json_and_summaries(raw_text: str):
    """
    Accepts model output that may contain multiple JSON objects/arrays (possibly fenced),
    followed by one or more SUMMARY: lines.
    Returns (list_of_change_dicts or None, list_of_summaries, list_of_raw_json_blocks_kept).
    """
    if not raw_text:
        return None, [], []

    # Tolerate code fences & leading chatter
    text = raw_text
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="ignore")
    text = text.strip()
    # Strip Markdown fences and tokens if present
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    # Remove any leading non-{[ … so we start at the first JSON block if model spoke first
    text = re.sub(r"^[^\[\{]*", "", text, count=1).strip()

    # Collect SUMMARY lines (keep all of them)
    summaries = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.upper().startswith("SUMMARY:"):
            summaries.append(s[len("SUMMARY:") :].strip())

    # Only parse blocks BEFORE the first SUMMARY
    first_summary_pos = text.upper().find("SUMMARY:")
    parse_zone = text if first_summary_pos == -1 else text[:first_summary_pos]

    # Find ALL balanced JSON blocks (objects or arrays)
    blocks = []
    for s, e in _find_all_balanced_json_blocks(parse_zone):
        blocks.append(parse_zone[s : e + 1])

    parsed_changes = []
    kept_blocks = []

    for block in blocks:
        # Normalize units inside JSON-like text (e.g. "2.5m" -> "2500")
        fixed_units = _repair_units_in_json_text(block)

        # Be tolerant to single quotes / Python literals and None/True/False
        candidates = [
            fixed_units,
            fixed_units.replace("'", '"'),
            fixed_units.replace(" None", " null")
            .replace(": None", ": null")
            .replace(" True", " true")
            .replace(": True", ": true")
            .replace(" False", " false")
            .replace(": False", ": false"),
        ]

        obj = None
        for cand in candidates:
            try:
                obj = json.loads(cand)
                break
            except Exception:
                try:
                    obj = ast.literal_eval(cand)
                    break
                except Exception:
                    obj = None

        if obj is None:
            # Log and keep going—don't fail the whole parse
            print("[recommend][parser] could not parse block:\n", block)
            continue

        kept_blocks.append(block)

        # Flatten to a list of dicts
        if isinstance(obj, list):
            parsed_changes += [it for it in obj if isinstance(it, dict)]
        elif isinstance(obj, dict):
            parsed_changes.append(obj)

    return (parsed_changes or None), summaries, kept_blocks


@app.post("/recommend")
def recommend():
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        classes = json.loads(request.form.get("classes") or "[]")
        if not isinstance(classes, list):
            classes = []

        # Images
        ref_url = _data_url_from_upload(request.files.get("reference"))
        if not ref_url:
            return jsonify({"ok": False, "error": "no reference image"}), 400
        snapshot_url = _data_url_from_upload(request.files.get("snapshot"))

        # Grounding JSON for the VLM
        cad_state = _cad_state_json()
        grounding_lines = [
            "Goal: Compare the REFERENCE image (photo/render) to the CURRENT CAD and propose precise, conservative changes that align CAD to the image.",
            "",
            "You are given:",
            "1) REFERENCE image (index 0).",
        ]
        if snapshot_url:
            grounding_lines.append("2) CURRENT CAD SNAPSHOT image (index 1).")
        grounding_lines += [
            "3) CURRENT CAD STATE JSON (below):",
            json.dumps(cad_state, indent=2),
            "",
            "Known classes (from client):",
            *[f"- {c}" for c in classes],
            "",
        ]
        if prompt_text:
            grounding_lines += ["User prompt:", prompt_text]

        images = [ref_url, snapshot_url] if snapshot_url else [ref_url]
        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding_lines)

        provider_out = call_vlm(final_prompt, images)

        # --- sanitize model output before JSON parsing ---
        raw = provider_out.get("raw", "")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        raw = raw.strip()
        raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"^[^\[\{]*", "", raw, count=1).strip()

        # DEBUG log
        print("\n" + "=" * 80)
        print("[recommend] Raw (sanitized) VLM output:\n")
        print(raw)
        print("=" * 80 + "\n")

        # Parse possibly-multiple JSON blocks + summaries
        parsed_list, summaries, json_blocks = _split_multi_json_and_summaries(raw)

        # Optional helper: translate parsed changes into a small CadQuery script
        def _translate_json_to_cadquery(changes: list[dict]) -> str:
            header = """#!/usr/bin/env python3
import cadquery as cq
from robot_base import Rover
from electronics import type1 as _Electronics
from pan_tilt import PanTilt as _PanTilt
from wheel import BuiltWheel as _ThisWheel
from sensor_fork import SensorFork
from cqparts_motors.stepper import Stepper as _Stepper

# Auto-generated from VLM recommendation
"""
            lines = [header, ""]
            rover_params = {}
            components = []
            for ch in changes or []:
                tgt = ch.get("target_component", "")
                params = ch.get("parameters", {}) or {}
                if tgt in ("rover", "base", "chassis"):
                    rover_params.update(params)
                elif tgt == "wheel":
                    d = params.get("wheel_diameter")
                    w = params.get("wheel_width")
                    n = params.get("wheels_per_side")
                    lines.append(
                        f"# Wheel modification: diameter={d}, width={w}, count={n}"
                    )
                elif tgt == "sensor_fork":
                    pos = params.get("position_mm", [220, 0, 160])
                    w, h, d = (
                        params.get("width_mm", 40),
                        params.get("height_mm", 30),
                        params.get("depth_mm", 25),
                    )
                    components.append(
                        f"SensorFork(width={w}, height={h}, depth={d}).local_obj.translate(({pos[0]}, {pos[1]}, {pos[2]}))"
                    )
                elif tgt == "pan_tilt":
                    offs = [
                        params.get(f"pan_tilt_offset_{a}", 0) for a in ("x", "y", "z")
                    ]
                    lines.append(f"# Pan-tilt offsets: {offs}")
            rover_args = ", ".join(
                [
                    "stepper=_Stepper",
                    "electronics=_Electronics",
                    "sensors=_PanTilt",
                    "wheel=_ThisWheel",
                ]
            )
            lines.append(f"rover = Rover({rover_args})")
            for k, v in rover_params.items():
                lines.append(f"setattr(rover, '{k}', {v})")
            if components:
                lines.append("# Add-on components")
                for c in components:
                    lines.append(c)
            lines.append("\nfrom cadquery import exporters")
            lines.append("exporters.export(rover.local_obj, 'rover_vlm.glb')")
            return "\n".join(lines)

        translated_script = (
            _translate_json_to_cadquery(parsed_list) if parsed_list else None
        )

        return jsonify(
            {
                "ok": True,
                "response": {
                    "raw": raw,
                    "json": parsed_list,
                    "summaries": summaries,
                    "json_blocks": json_blocks,
                    "python_script": translated_script,
                },
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# /vlm route moved to app.routes.vlm blueprint
# Legacy function removed - route is now in vlm.py blueprint


@app.post("/ingest_mesh_segment")
def ingest_mesh_segment():
    """
    Step 1: Run segmentation only (fast, ~1-5 seconds).
    Returns part information for user labeling.
    
    Accepts:
    - mesh: mesh file (OBJ/STL/PLY) - required
    
    Returns:
    - segmentation: part segmentation results
    - part_table: PartTable JSON for labeling UI
    - mesh_path: path to uploaded mesh (for step 2)
    """
    try:
        import tempfile
        import os
        from pathlib import Path
        
        # Get uploaded mesh file
        mesh_file = request.files.get("mesh")
        if not mesh_file:
            return jsonify({"ok": False, "error": "mesh file required"}), 400
        
        # Save uploaded file to temp directory
        temp_dir = tempfile.mkdtemp(prefix="mesh_ingest_")
        mesh_filename = mesh_file.filename or "mesh.obj"
        mesh_path = os.path.join(temp_dir, mesh_filename)
        mesh_file.save(mesh_path)
        
        print(f"[ingest_mesh_segment] Processing mesh: {mesh_path}", flush=True)
        
        # Import segmentation only (no VLM)
        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from meshml.segmentation import create_segmentation_backend
        from meshml.pointnet_seg.labels import get_category_from_flat_label
        from meshml.pointnet_seg.geometry import compute_part_statistics, compute_part_bounding_boxes
        from meshml.parts.parts import build_part_table_from_segmentation, part_table_to_labeling_json
        import numpy as np
        import trimesh
        
        # Create segmentation backend
        print(f"[ingest_mesh_segment] Initializing segmentation backend...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
        print(f"[ingest_mesh_segment] Using segmentation backend: {backend_kind}")
        
        try:
            model = create_segmentation_backend(kind=backend_kind, device=device)
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": f"Failed to initialize segmentation backend '{backend_kind}': {str(e)}"
            }), 500
        
        # Run segmentation only (fast, ~1-5 seconds)
        print(f"[ingest_mesh_segment] Running part segmentation...")
        seg_result = model.segment(mesh_path, num_points=2048)
        points = seg_result.points
        labels = seg_result.labels
        unique_labels = np.unique(labels)
        
        print(f"[ingest_mesh_segment] ✓ Part segmentation complete!")
        print(f"[ingest_mesh_segment]   Segmented into {seg_result.num_parts} parts")
        print(f"[ingest_mesh_segment]   Point cloud: {seg_result.num_points} points")
        
        # Build part statistics for visualization
        part_stats = compute_part_statistics(points, labels)
        part_bboxes = compute_part_bounding_boxes(points, labels)
        
        # Build part label names
        part_label_names = {}
        for label_id in unique_labels:
            label_id_int = int(label_id)
            result = get_category_from_flat_label(label_id_int)
            if result:
                cat, part_name = result
                part_label_names[label_id_int] = part_name
            else:
                part_label_names[label_id_int] = f"part_{label_id_int}"
        
        # Create segmentation summary
        segmentation_summary = {
            "num_parts": len(unique_labels),
            "num_points": len(points),
            "parts": []
        }
        
        for label_id in unique_labels:
            label_id_int = int(label_id)
            part_name = part_label_names.get(label_id_int, f"part_{label_id_int}")
            count = np.sum(labels == label_id_int)
            bbox = part_bboxes.get(label_id_int, {})
            
            # Convert bbox to JSON-serializable format
            bbox_data = None
            if label_id_int in part_bboxes and bbox:
                def to_list(v):
                    """Convert NumPy arrays/scalars to native Python types."""
                    if v is None:
                        return [0.0, 0.0, 0.0]
                    if hasattr(v, 'tolist'):
                        # NumPy array - convert to list and ensure all values are native Python floats
                        return [float(x) for x in v.tolist()]
                    if isinstance(v, (list, tuple)):
                        # Already a list/tuple - ensure all values are native Python floats
                        return [float(x) for x in v]
                    # Single value - convert to float
                    return [float(v)]
                
                bbox_data = {
                    "min": to_list(bbox.get("min")),
                    "max": to_list(bbox.get("max")),
                    "center": to_list(bbox.get("center")),
                    "extent": to_list(bbox.get("extent")),
                }
            
            segmentation_summary["parts"].append({
                "id": int(label_id_int),
                "name": part_name,
                "point_count": int(count),
                "percentage": float(count / len(points) * 100),
                "bbox": bbox_data,
            })
        
        # Print part details
        print(f"[ingest_mesh_segment] Part breakdown:")
        for part in segmentation_summary["parts"]:
            print(f"[ingest_mesh_segment]   • Part {part['id']} ({part['name']}): {part['point_count']} points ({part['percentage']:.1f}%)")
        
        # Build PartTable for user labeling
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
        # Map point labels to vertex labels (approximate)
        vertex_labels = seg_result.labels
        if len(vertex_labels) != len(vertices):
            if len(vertex_labels) < len(vertices):
                vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
            else:
                vertex_labels = vertex_labels[:len(vertices)]
        
        # Build PartTable
        part_table = build_part_table_from_segmentation(
            vertices=vertices,
            part_labels=vertex_labels,
            ground_plane_z=None,
        )
        part_table_json = part_table_to_labeling_json(part_table)
        
        print(f"[ingest_mesh_segment] ✓ PartTable created with {len(part_table.parts)} parts for user labeling")
        
        # Save colored point cloud visualization
        try:
            colors = np.zeros((len(points), 3))
            for i, label_id in enumerate(labels):
                np.random.seed(int(label_id))
                color = np.random.rand(3)
                colors[i] = color
            pc = trimesh.PointCloud(vertices=points, colors=colors)
            viz_path = os.path.join(temp_dir, "segmentation_colored.ply")
            pc.export(viz_path)
            segmentation_summary["visualization_path"] = viz_path
        except Exception as e:
            print(f"[ingest_mesh_segment] Warning: Could not save visualization: {e}")
            segmentation_summary["visualization_path"] = None
        
        # Include vertex labels for coloring (convert to list for JSON)
        vertex_labels_list = vertex_labels.tolist() if hasattr(vertex_labels, 'tolist') else list(vertex_labels)
        
        # Return segmentation results only (user will label parts, then call /ingest_mesh_label)
        response_data = {
            "ok": True,
            "segmentation": segmentation_summary,
            "part_table": part_table_json,
            "vertex_labels": vertex_labels_list,  # Include vertex labels for coloring
            "mesh_path": mesh_path,  # Store for step 2 (VLM processing)
            "temp_dir": temp_dir,  # Keep temp dir for step 2
        }
        
        # Cleanup temp directory (keep renders for now, they might be useful)
        # shutil.rmtree(temp_dir)  # Commented out - might want to keep renders
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Mesh segmentation error: {str(e)}"
        print(f"[ingest_mesh_segment] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.post("/convert_mesh_to_glb")
def convert_mesh_to_glb():
    """
    Convert uploaded mesh file (STL/PLY/OBJ) to GLB format for display in viewer.
    """
    try:
        if 'mesh' not in request.files:
            return jsonify({"ok": False, "error": "No mesh file provided"}), 400
        
        mesh_file = request.files['mesh']
        if not mesh_file.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400
        
        import tempfile
        import os
        import trimesh
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(mesh_file.filename)[1]) as tmp:
            mesh_file.save(tmp.name)
            mesh_path = tmp.name
        
        try:
            # Load mesh
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Export as GLB
            import io
            glb_buffer = io.BytesIO()
            mesh.export(file_obj=glb_buffer, file_type='glb')
            glb_buffer.seek(0)
            
            from flask import send_file
            return send_file(
                glb_buffer,
                mimetype='model/gltf-binary',
                as_attachment=False,
                download_name='mesh.glb'
            )
        finally:
            # Clean up temp file
            if os.path.exists(mesh_path):
                os.unlink(mesh_path)
                
    except Exception as e:
        import traceback
        error_msg = f"Mesh conversion error: {str(e)}"
        print(f"[convert_mesh_to_glb] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.post("/ingest_mesh_label")
def ingest_mesh_label():
    """
    Step 2: Run VLM with user-provided part labels.
    
    Accepts:
    - mesh_path: path to mesh (from step 1)
    - temp_dir: temp directory (from step 1)
    - part_labels: JSON with user-assigned part names (from labeling UI)
    
    Returns:
    - category: object category
    - final_parameters: list of semantic parameters
    - raw_parameters: list of raw geometric parameters
    """
    try:
        import json
        from pathlib import Path

        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "JSON body required"}), 400
        
        mesh_path = data.get("mesh_path")
        temp_dir = data.get("temp_dir")
        part_labels_json = data.get("part_labels")  # User-provided labels
        
        if not mesh_path or not os.path.exists(mesh_path):
            return jsonify({"ok": False, "error": "mesh_path required and must exist"}), 400
        
        print(f"[ingest_mesh_label] Processing mesh with user labels: {mesh_path}", flush=True)
        
        # Import pipeline
        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from meshml.segmentation import create_segmentation_backend
        from meshml.semantics.vlm_client_finetuned import FinetunedVLMClient
        from meshml.semantics.vlm_client_ollama import OllamaVLMClient
        from meshml.semantics.vlm_client import DummyVLMClient
        from meshml.semantics.ingest_mesh import ingest_mesh_to_semantic_params
        from meshml.parts.parts import build_part_table_from_segmentation, apply_labels_from_json
        import trimesh
        import numpy as np
        
        # Initialize segmentation backend (reuse from step 1)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
        model = create_segmentation_backend(kind=backend_kind, device=device)
        
        # Re-run segmentation to get PartTable (or we could cache it from step 1)
        seg_result = model.segment(mesh_path, num_points=2048)
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
        vertex_labels = seg_result.labels
        if len(vertex_labels) != len(vertices):
            if len(vertex_labels) < len(vertices):
                vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
            else:
                vertex_labels = vertex_labels[:len(vertices)]
        
        # Build PartTable
        part_table = build_part_table_from_segmentation(
            vertices=vertices,
            part_labels=vertex_labels,
            ground_plane_z=None,
        )
        
        # Apply user-provided labels
        if part_labels_json:
            part_table = apply_labels_from_json(part_table, part_labels_json)
            print(f"[ingest_mesh_label] Applied user labels to {len(part_labels_json.get('parts', []))} parts", flush=True)
        
        # Initialize VLM (same logic as before)
        vlm = None
        if device == "cpu":
            try:
                vlm = OllamaVLMClient()
                print("[ingest_mesh_label] Using Ollama VLM (fast on CPU)")
            except Exception as e:
                print(f"[ingest_mesh_label] Warning: Could not use Ollama: {e}")
            try:
                vlm = FinetunedVLMClient()
                print("[ingest_mesh_label] Using fine-tuned VLM (slower on CPU)")
            except Exception as e2:
                print(f"[ingest_mesh_label] Warning: Could not use fine-tuned VLM: {e2}")
                vlm = DummyVLMClient()
                print("[ingest_mesh_label] Using dummy VLM (for testing)")
        else:
            try:
                vlm = FinetunedVLMClient()
                print("[ingest_mesh_label] Using fine-tuned VLM (pretrained model on GPU)")
            except Exception as e:
                print(f"[ingest_mesh_label] Warning: Could not use fine-tuned VLM: {e}")
                try:
                    vlm = OllamaVLMClient()
                    print("[ingest_mesh_label] Using Ollama VLM (fallback)")
                except Exception as e2:
                    print(f"[ingest_mesh_label] Warning: Could not use Ollama: {e2}")
                    vlm = DummyVLMClient()
                    print("[ingest_mesh_label] Using dummy VLM (for testing)")
        
        # Run ingestion pipeline with user-labeled PartTable
        render_dir = os.path.join(temp_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)
        
        print(f"[ingest_mesh_label] Running VLM pipeline with user labels...", flush=True)
        result = ingest_mesh_to_semantic_params(
            mesh_path=mesh_path,
            vlm=vlm,
            model=model,
            render_output_dir=render_dir,
            num_points=2048,
        )
        
        # Update result with user-labeled PartTable
        result.part_table = part_table
        # Also store vertex labels in extra for easier access
        result.extra["vertex_labels"] = vertex_labels.tolist() if hasattr(vertex_labels, 'tolist') else list(vertex_labels)
        
        # Cache IngestResult for later use in /apply_mesh_params
        _INGEST_RESULT_CACHE[mesh_path] = result
        print(f"[ingest_mesh_label] ✓ Cached IngestResult for mesh: {mesh_path}", flush=True)
        
        # Convert to JSON
        def param_to_dict(p):
            d = {
                "id": p.id,
                "semantic_name": p.semantic_name,
                "value": p.value,
                "units": p.units,
                "description": p.description,
                "confidence": p.confidence,
                "raw_sources": p.raw_sources,
            }
            d["name"] = p.semantic_name
            return d
        
        from meshml.parts.parts import part_table_to_labeling_json
        part_table_json = part_table_to_labeling_json(part_table)
        
        response_data = {
            "ok": True,
            "category": result.category,
            "raw_parameters": [
                {
                    "id": p.id,
                    "value": p.value,
                    "units": p.units,
                    "description": p.description,
                    "part_labels": p.part_labels or [],
                }
                for p in result.raw_parameters[:20]
            ],
            "proposed_parameters": [param_to_dict(p) for p in result.proposed_parameters],
            "final_parameters": [param_to_dict(p) for p in result.proposed_parameters],
            "metadata": {
                "num_points": result.extra.get("num_points", 0),
                "num_parts": result.extra.get("num_parts", 0),
            },
            "part_table": part_table_json,
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Mesh labeling/VLM error: {str(e)}"
        print(f"[ingest_mesh_label] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# Backward compatibility: keep old endpoint but make it call segment + label
@app.post("/ingest_mesh")
def ingest_mesh():
    """
    Legacy endpoint: Runs segmentation then VLM (auto-labels parts).
    For new workflow, use /ingest_mesh_segment then /ingest_mesh_label.
    """
    # This will be kept for backward compatibility but can call the new endpoints internally
    # For now, just redirect to the old behavior
    pass


# Cache for IngestResult (keyed by mesh_path)
_INGEST_RESULT_CACHE: Dict[str, Any] = {}


@app.post("/apply_mesh_params")
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
def _introspect_params_from_cls(cls) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            mod = getattr(getattr(val, "__class__", object), "__module__", "")
            if "cqparts.params" in mod:
                d[name] = str(val)
        except Exception:
            pass
    for name in dir(cls):
        if name in d or name.startswith("_"):
            continue
        try:
            val = getattr(cls, name)
            if isinstance(val, (int, float)):
                d[name] = val
        except Exception:
            pass
    return d


@app.get("/params")
def get_params():
    info = {"current": _snapshot(), "introspected": {}}
    if USE_CQPARTS:
        try:
            info["introspected"]["wheel"] = _introspect_params_from_cls(_ThisWheel)
            info["introspected"]["pan_tilt"] = _introspect_params_from_cls(_PanTilt)
        except Exception:
            pass
    info["context"] = {"terrain_mode": CONTEXT["terrain_mode"]}
    return jsonify({"ok": True, "params": info})


# ----------------- Undo/Redo -----------------
@app.post("/undo")
def undo_change():
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


@app.post("/redo")
def redo_change():
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


# ----------------- Build & GLB export -----------------
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


import inspect, textwrap, importlib, sys, os, re


def _strip_docstrings_and_comments(src: str) -> str:
    # very light scrubbing to keep the prompt small
    # (keeps string literals inside code; removes triple-quoted docstrings and # lines)
    src = re.sub(r'(?s)^\s*("""|\'\'\').*?\1\s*', "", src)  # file header docstring
    src = re.sub(r'(?s)([^f])("""|\'\'\').*?\2', r"\1", src)  # other docstrings (rough)
    src = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    return src


def _try_get_source(obj):
    try:
        return inspect.getsource(obj)
    except Exception:
        return None


def _baseline_cqparts_source(max_chars: int = 20000) -> str:
    """
    Returns a compact string containing the *actual* Rover / RobotBase / wheel / stepper
    source from your project, trimmed to fit in the prompt.
    """
    chunks = []

    # Try direct file read first (most reliable)
    # Check in app/models/cad first, then root
    robot_base_path = os.path.join(BASE_DIR, "app", "models", "cad", "robot_base.py")
    if not os.path.exists(robot_base_path):
        robot_base_path = os.path.join(BASE_DIR, "robot_base.py")
    if os.path.exists(robot_base_path):
        print(f"[baseline_source] Reading robot_base.py directly from {robot_base_path}")
        try:
            with open(robot_base_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content:
                    chunks.append(f"# === robot_base.py ===\n{content}")
                    print(f"[baseline_source] ✓ Read {len(content)} chars from robot_base.py")
        except Exception as e:
            print(f"[baseline_source] ✗ Failed to read robot_base.py: {e}")

    # Try inspect.getsource as backup
    if not chunks:
        print("[baseline_source] Trying inspect.getsource method...")
        try:
            import robot_base as _rb
            mod_src = inspect.getsource(_rb)
            chunks.append(mod_src)
            print(f"[baseline_source] ✓ Got {len(mod_src)} chars via inspect.getsource")
        except Exception as e:
            print(f"[baseline_source] ✗ inspect.getsource failed: {e}")

    # Try class-by-class extraction
    if not chunks:
        print("[baseline_source] Trying class-by-class extraction...")
        try:
            from robot_base import Rover, RobotBase
            for obj in (Rover, RobotBase):
                s = _try_get_source(obj)
                if s:
                    chunks.append(s)
                    print(f"[baseline_source] ✓ Got {obj.__name__}")
        except Exception as e:
            print(f"[baseline_source] ✗ Class extraction failed: {e}")

    # Also try to get wheel classes
    try:
        from wheel import BuiltWheel, SpokeWheel, SimpleWheel
        for obj in (BuiltWheel, SpokeWheel, SimpleWheel):
            s = _try_get_source(obj)
            if s:
                chunks.append(f"# === {obj.__name__} ===\n{s}")
    except Exception as e:
        print(f"[baseline_source] Note: Could not get wheel classes: {e}")

    # Fallback: try to open files by searching sys.modules
    if not chunks:
        print("[baseline_source] Trying sys.modules fallback...")
        for modname in ("robot_base", "wheel", "pan_tilt"):
            try:
                m = sys.modules.get(modname) or importlib.import_module(modname)
                path = inspect.getsourcefile(m) or inspect.getfile(m)
                if path and os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        chunks.append(f"# === {modname} ===\n{content}")
                        print(f"[baseline_source] ✓ Read {modname} from {path}")
            except Exception as e:
                print(f"[baseline_source] Could not read {modname}: {e}")

    # Merge and clean
    if not chunks:
        error_msg = "# ERROR: Could not extract robot_base.py source code\n"
        error_msg += f"# Tried path: {robot_base_path}\n"
        error_msg += "# Please ensure robot_base.py exists in the same directory as optim.py\n"
        print(f"[baseline_source] ✗ FAILED - no source code extracted!")
        return error_msg
    
    merged = "\n\n# ----\n\n".join(chunks)
    print(f"[baseline_source] Total merged: {len(merged)} chars before cleaning")
    
    # Light cleaning (optional - may want to keep comments for VLM)
    # merged = _strip_docstrings_and_comments(merged)
    
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n# ... [truncated for prompt] ..."
        print(f"[baseline_source] Truncated to {max_chars} chars")
    
    print(f"[baseline_source] ✓ Final output: {len(merged)} chars")
    return merged


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


# ----------------- GLB route & static -----------------
@app.get("/model.glb")
def model_glb():
    """
    Serve the GLB model file.
    Only builds the rover if explicitly requested (not on automatic page load).
    """
    # First, check if cached GLB exists - if it does, serve it without building
    # This prevents auto-building on page load
    if os.path.exists(ROVER_GLB_PATH):
        try:
            with open(ROVER_GLB_PATH, "rb") as f:
                cached_glb = f.read()
            if len(cached_glb) > 1000:  # Ensure it's a valid GLB
                print(f"[model.glb] ✓ Serving cached GLB ({len(cached_glb)} bytes) - NO BUILD", flush=True)
                return send_file(io.BytesIO(cached_glb), mimetype="model/gltf-binary")
        except Exception as e:
            print(f"[model.glb] Failed to read cached GLB: {e}", flush=True)
    
    # No cached file exists - check if this is an explicit request
    # Look for explicit request indicators
    explicit_request = (
        request.args.get('force_rebuild') == '1' or  # Explicit rebuild flag
        request.args.get('use_generated') == '1' or  # Explicit generated code request
        (request.referrer and '/viewer' in request.referrer)  # Coming from viewer page
    )
    
    # If no cached file and not an explicit request, don't auto-build
    # This prevents the rover from being built automatically on page load
    if not explicit_request:
        print("[model.glb] ⚠ No cached GLB and not an explicit request - returning 404 to prevent auto-build", flush=True)
        print(f"[model.glb]   Referrer: {request.referrer}, Args: {dict(request.args)}, Purpose: {request.headers.get('Purpose', 'N/A')}", flush=True)
        return Response(
            "Model not yet loaded. Please upload a mesh or parametric model first, or use the 'Load Model' button.",
            status=404,
            mimetype="text/plain"
        )
    
    # This is an explicit request - proceed with building
    print("[model.glb] ⚠ Explicit request detected - building rover model (this should not happen on page load)", flush=True)
    
    try:
        # Check if we should use generated code
        # Can be forced via ?use_generated=1 or auto-detect
        force_generated = request.args.get('use_generated') == '1'
        gen_path = os.path.join(BASE_DIR, "generated", "robot_base_vlm.py")
        use_generated = force_generated or os.path.exists(gen_path)
        
        # Try hybrid approach first (if build_glb.py exists)
        script_path = os.path.join(BASE_DIR, "build_glb.py")
        use_hybrid = os.path.exists(script_path)
        
        if use_hybrid:
            if use_generated and os.path.exists(gen_path):
                print("[model.glb] Using hybrid approach with freecad environment (generated)", flush=True)
                try:
                    # Use hybrid builder (runs in freecad env)
                    glb = build_rover_scene_glb_cqparts_hybrid(use_generated=True)
                    
                    # Save to disk so file size/timestamp updates
                    with open(ROVER_GLB_PATH, "wb") as f:
                        f.write(glb)
                    print(f"[model.glb] ✓ Saved generated GLB to {ROVER_GLB_PATH} ({len(glb)} bytes)", flush=True)
                    
                    return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
                except Exception as gen_error:
                    print(f"[model.glb] ✗ Hybrid approach failed: {gen_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    print("[model.glb] Falling back to direct build method", flush=True)
                    # Fall through to direct build
            
            print("[model.glb] Using hybrid approach with freecad environment (original)", flush=True)
            try:
                # Use hybrid builder for original code too
                glb = build_rover_scene_glb_cqparts_hybrid(use_generated=False)
                
                # Save to disk so file size/timestamp updates
                with open(ROVER_GLB_PATH, "wb") as f:
                    f.write(glb)
                print(f"[model.glb] ✓ Saved GLB to {ROVER_GLB_PATH} ({len(glb)} bytes)", flush=True)
                
                return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
            except Exception as hybrid_error:
                print(f"[model.glb] ✗ Hybrid approach failed: {hybrid_error}", flush=True)
                import traceback
                traceback.print_exc()
                print("[model.glb] Falling back to direct build method", flush=True)
                # Fall through to direct build
        
        # Fallback to direct build method (no hybrid/subprocess)
        print("[model.glb] Using direct build method", flush=True)
        if use_generated and os.path.exists(gen_path):
            print("[model.glb] Loading generated robot_base_vlm.py", flush=True)
            RoverClass = _reload_rover_from_generated()
        else:
            RoverClass = Rover
        
        glb = build_rover_scene_glb_cqparts(RoverClass=RoverClass)
        
        # Save to disk so file size/timestamp updates
        with open(ROVER_GLB_PATH, "wb") as f:
            f.write(glb)
        print(f"[model.glb] ✓ Saved GLB to {ROVER_GLB_PATH} ({len(glb)} bytes)", flush=True)
        
        return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
        
    except Exception as e:
        import traceback
        error_msg = f"model.glb build failed:\n{traceback.format_exc()}"
        print(f"[model.glb] FATAL ERROR: {error_msg}", flush=True)
        traceback.print_exc()
        
        # Fallback: try to serve cached GLB if it exists
        if os.path.exists(ROVER_GLB_PATH):
            try:
                print(f"[model.glb] Attempting to serve cached GLB from {ROVER_GLB_PATH}", flush=True)
                with open(ROVER_GLB_PATH, "rb") as f:
                    cached_glb = f.read()
                if len(cached_glb) > 1000:  # Ensure it's a valid GLB
                    print(f"[model.glb] Serving cached GLB ({len(cached_glb)} bytes)", flush=True)
                    return send_file(io.BytesIO(cached_glb), mimetype="model/gltf-binary")
            except Exception as cache_error:
                print(f"[model.glb] Failed to serve cached GLB: {cache_error}", flush=True)
        
        return Response(
            error_msg,
            status=500,
            mimetype="text/plain",
        )


@app.route("/demo/rover.png")
@app.route("/demo/mars_rover.jpg")
def demo_mars_rover():
    """Serve the demo Mars rover image."""
    # Try rover.png first (user's file), then fall back to mars_rover.jpg
    demo_paths = [
        os.path.join(ASSETS_DIR, "demo", "rover.png"),
        os.path.join(ASSETS_DIR, "demo", "mars_rover.jpg"),
    ]
    
    for demo_path in demo_paths:
        if os.path.exists(demo_path):
            mimetype = "image/png" if demo_path.endswith(".png") else "image/jpeg"
            return send_file(demo_path, mimetype=mimetype)
    
    # Return a placeholder message if demo image doesn't exist
    return Response(
        "Demo image not found. Please save the Mars rover image to assets/demo/rover.png or assets/demo/mars_rover.jpg",
        status=404,
        mimetype="text/plain"
    )

@app.route("/demo/curiosity_rover.stl")
def demo_curiosity_rover():
    """Serve the demo Curiosity Rover STL file."""
    # Try multiple possible locations
    possible_paths = [
        os.path.join(ASSETS_DIR, "demo", "curiosity_rover.stl"),
        os.path.join(ASSETS_DIR, "demo", "rover.stl"),
        os.path.join(ASSETS_DIR, "demo", "body-small.STL"),
        "/Users/janelleg/Downloads/Curiosity Rover 3D Printed Model/Simplified Curiosity Model (Small)/STL Files/body-small.STL",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return send_file(path, mimetype="model/stl")
    
    # If no file found, return a helpful message
    return Response(
        f"Demo STL file not found. Please place a demo STL file at: {os.path.join(ASSETS_DIR, 'demo', 'curiosity_rover.stl')}",
        status=404,
        mimetype="text/plain"
    )

@app.route("/static/<path:filename>")
def custom_static(filename):
    root = app.static_folder
    full = os.path.join(root, filename)
    if not os.path.exists(full):
        abort(404)
    if filename.endswith(".js"):
        return send_from_directory(root, filename, mimetype="application/javascript")
    return send_from_directory(root, filename)


# ----------------- Test routes -----------------
def _mk_change(action: str, target: str, params: dict) -> dict:
    ch = {"action": action, "target_component": target, "parameters": params}
    norm = _normalize_change(ch)
    return norm or ch


@app.post("/test/rotate")
def test_rotate():
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or "rover").lower()
    angle = data.get("angle_deg")
    orient = data.get("orientation_deg")
    params = {}
    if target in ("rover", "roverbase", "base", "chassis"):
        if angle is not None:
            params["rover_yaw_deg"] = angle
    if isinstance(orient, (list, tuple)) and len(orient) == 3:
        params["orientation_deg"] = orient
    change = _mk_change("rotate", target, params)
    code, payload = _apply_changes_list([change])
    return jsonify(payload), code


@app.post("/test/translate")
def test_translate():
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or "pan_tilt").lower()
    params = {k: data.get(k) for k in ("dx_mm", "dy_mm", "dz_mm") if k in data}
    change = _mk_change("translate", target, params)
    code, payload = _apply_changes_list([change])
    return jsonify(payload), code


@app.post("/test/modify")
def test_modify():
    data = request.get_json(silent=True) or {}
    target = (data.get("target") or "wheel").lower()
    params = {k: v for k, v in data.items() if k != "target"}
    change = _mk_change("modify", target, params)
    code, payload = _apply_changes_list([change])
    return jsonify(payload), code


# ----------------- Warm build & main -----------------
def _warm_build():
    try:
        print("[warm] starting initial build…")
        glb = build_rover_scene_glb({})
        with open(ROVER_GLB_PATH, "wb") as f:
            f.write(glb)
        print("[warm] initial GLB ready.")
    except Exception as e:
        print("[warm] initial build failed:", e)


def test_vlm_model(include_mesh_analysis=True):
    """
    Test function to load and test the VLM model without running the full UI.
    This can be run directly: python -c "from optim import test_vlm_model; test_vlm_model()"
    Or: python optim.py --test-vlm
    
    Args:
        include_mesh_analysis: If True, also test the mesh ingestion pipeline
    """
    import sys
    import os
    import tempfile
    
    print("=" * 80)
    print("Testing VLM Model Loading and Inference")
    print("=" * 80)
    
    # Check model path
    print(f"\n[test] Model path: {FINETUNED_MODEL_PATH}")
    print(f"[test] Path exists: {os.path.exists(FINETUNED_MODEL_PATH)}")
    
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"[test] ✗ ERROR: Model path does not exist!")
        print(f"[test] Expected: {FINETUNED_MODEL_PATH}")
        return False
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []
    for f in required_files:
        file_path = os.path.join(FINETUNED_MODEL_PATH, f)
        if not os.path.exists(file_path):
            missing_files.append(f)
    else:
            print(f"[test] ✓ Found {f}")
    
    if missing_files:
        print(f"[test] ✗ ERROR: Missing required files: {missing_files}")
        return False
    
    # Load model
    print(f"\n[test] Loading model...")
    print(f"[test] Note: This may take a few minutes if the base model needs to be downloaded")
    try:
        load_finetuned_model()
        if _finetuned_model is None or _finetuned_processor is None:
            print("[test] ✗ ERROR: Model failed to load")
            return False
        print("[test] ✓ Model loaded successfully")
    except KeyboardInterrupt:
        print("\n[test] Model loading interrupted by user")
        return False
    except Exception as e:
        print(f"[test] ✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference with a simple prompt
    print(f"\n[test] Testing inference with a simple prompt...")
    try:
        test_prompt = "What is 2+2? Answer with just the number."
        result = call_vlm(
            final_prompt=test_prompt,
            image_data_urls=None,
            expect_json=False
        )
        
        if result and "raw" in result:
            response = result["raw"]
            print(f"[test] ✓ Inference successful!")
            print(f"[test] Provider: {result.get('provider', 'unknown')}")
            print(f"[test] Response length: {len(response)} chars")
            print(f"[test] Response preview: {response[:200]}...")
        else:
            print(f"[test] ✗ ERROR: Invalid response format: {result}")
            return False
    except KeyboardInterrupt:
        print("\n[test] Inference interrupted by user")
        return False
    except Exception as e:
        print(f"[test] ✗ ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test mesh analysis if requested
    if include_mesh_analysis:
        print(f"\n" + "=" * 80)
        print("Testing Mesh Analysis Pipeline")
        print("=" * 80)
        
        try:
            # Check PointNet++ model
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "pointnet2", "pointnet2_part_seg_msg.pth"
            )
            checkpoint_path = os.path.abspath(checkpoint_path)
            print(f"\n[test] PointNet++ model path: {checkpoint_path}")
            print(f"[test] Path exists: {os.path.exists(checkpoint_path)}")
            
            if not os.path.exists(checkpoint_path):
                print(f"[test] ⚠ WARNING: PointNet++ model not found, skipping mesh analysis test")
                print(f"[test] Expected: {checkpoint_path}")
                return True  # VLM test passed, just mesh analysis missing
            
            # Create a simple test mesh
            print(f"\n[test] Creating test mesh...")
            try:
                import trimesh
                # Create a simple box mesh for testing
                test_mesh = trimesh.creation.box(extents=[1.0, 2.0, 0.5])
                temp_dir = tempfile.mkdtemp(prefix="test_mesh_")
                test_mesh_path = os.path.join(temp_dir, "test_box.obj")
                test_mesh.export(test_mesh_path)
                print(f"[test] ✓ Created test mesh at {test_mesh_path}")
            except Exception as e:
                print(f"[test] ✗ ERROR: Failed to create test mesh: {e}")
                import traceback
                traceback.print_exc()
                return True  # VLM test passed
            
            # Import mesh ingestion components
            print(f"\n[test] Loading mesh ingestion pipeline...")
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from meshml.pointnet_seg.model import load_pretrained_model
            from meshml.semantics.vlm_client_finetuned import FinetunedVLMClient
            from meshml.semantics.ingest_mesh import ingest_mesh_to_semantic_params
            
            # Load PointNet++ model
            print(f"[test] Loading PointNet++ model...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[test] Using device: {device}")
            
            pointnet_model = load_pretrained_model(
                checkpoint_path=checkpoint_path,
                num_classes=50,
                use_normals=True,
                device=device,
            )
            print(f"[test] ✓ PointNet++ model loaded")
            
            # Initialize VLM client (use the already loaded model)
            print(f"[test] Initializing VLM client...")
            vlm_client = FinetunedVLMClient()
            print(f"[test] ✓ VLM client initialized")
            
            # Run mesh ingestion
            print(f"\n[test] Running mesh ingestion pipeline...")
            print(f"[test] Note: This may take a few minutes (VLM calls + segmentation)")
            render_dir = os.path.join(temp_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            
            try:
                result = ingest_mesh_to_semantic_params(
                    mesh_path=test_mesh_path,
                    vlm=vlm_client,
                    model=pointnet_model,
                    render_output_dir=render_dir,
                    num_points=2048,
                )
                
                print(f"[test] ✓ Mesh ingestion completed!")
                print(f"[test] Category: {result.category}")
                print(f"[test] Raw parameters: {len(result.raw_parameters)}")
                print(f"[test] Proposed parameters: {len(result.proposed_parameters)}")
                print(f"[test] Number of parts: {result.extra.get('num_parts', 0)}")
                
                # Show some example parameters
                if result.proposed_parameters:
                    print(f"\n[test] Example proposed parameters:")
                    for i, param in enumerate(result.proposed_parameters[:5]):
                        print(f"  {i+1}. {param.semantic_name}: {param.value} {param.units} (confidence: {param.confidence:.2f})")
                
                # Cleanup
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
                print(f"\n[test] ✓ Mesh analysis test passed!")
                return True
            except KeyboardInterrupt:
                print("\n[test] Mesh analysis interrupted by user")
                # Cleanup
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                return True  # VLM test passed, mesh analysis interrupted
            
        except KeyboardInterrupt:
            print("\n[test] Mesh analysis setup interrupted by user")
            return True  # VLM test passed
        except Exception as e:
            print(f"[test] ✗ ERROR: Mesh analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return True  # VLM test passed, mesh analysis failed
    
    return True


if __name__ == "__main__":
    import sys
    
    # Check for test flag - run test BEFORE other imports that might fail
    if "--test-vlm" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "test-vlm"):
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
