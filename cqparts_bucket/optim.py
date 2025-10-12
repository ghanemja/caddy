#!/usr/bin/env python3
# server.py — Rover CAD viewer/editor (GLB pipeline with real “add” support)
# Import and run build()
import io, os, sys, json, re, base64, threading, mimetypes, ast, math
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import requests
import numpy as np
import cadquery as cq
import trimesh
from trimesh.transformations import euler_matrix
from cadquery import exporters
from cadquery import Workplane
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

# ----------------- FreeCAD bootstrap -----------------
import importlib.util


def load_freecad():
    path = "/home/ec2-user/Documents/cad-optimizer/squashfs-root/usr/lib/FreeCAD.so"
    spec = importlib.util.spec_from_file_location("FreeCAD", path)
    FreeCAD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FreeCAD)
    sys.modules["FreeCAD"] = FreeCAD
    return FreeCAD


FreeCAD = load_freecad()

# ----------------- Repo components -----------------
from robot_base import Rover
from electronics import type1 as _Electronics
from pan_tilt import PanTilt as _PanTilt
from wheel import BuiltWheel as _ThisWheel
from cqparts_motors.stepper import Stepper as _Stepper
from sensor_fork import SensorFork

# MIME fix for ESM
mimetypes.add_type("application/javascript", ".js")

# ----------------- VLM config -----------------
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava-llama3:latest")
LLAVA_URL = os.environ.get("LLAVA_URL")  # optional

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
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
ROVER_GLB_PATH = os.path.join(ASSETS_DIR, "rover.glb")
USE_CQPARTS = os.environ.get("USE_CQPARTS", "1") == "1"

app = Flask(__name__, static_folder="static")
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
class ComponentSpec:
    def __init__(self, cls, add_fn=None, param_map=None, proxy_fn=None):
        self.cls = cls
        self.add_fn = add_fn
        self.param_map = param_map or {}
        self.proxy_fn = proxy_fn


COMPONENT_REGISTRY: Dict[str, ComponentSpec] = {}


def register_component(key: str, spec: ComponentSpec):
    COMPONENT_REGISTRY[key.lower()] = spec


def get_component_spec(key: str) -> Optional[ComponentSpec]:
    return COMPONENT_REGISTRY.get(key.lower())


# queued ops for true geometry adds
PENDING_ADDS: List[dict] = []
INSTALLED_ADDS: List[dict] = []
HIDDEN_PREFIXES: List[str] = []


def _truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "on")


# ----------------- Utility helpers -----------------
def _num(v, default=None):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    dv = getattr(v, "default", None)
    if isinstance(dv, (int, float)):
        return float(dv)
    val = getattr(v, "value", None)
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(v)
    except Exception:
        s = str(v)
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return default


def _strip_units_to_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def _percent_to_abs(token, base):
    if token is None or base is None:
        return None
    try:
        s = str(token).strip()
        if s.endswith("%"):
            return float(base) * (float(s[:-1]) / 100.0)
        f = float(s)
        return float(base) * f if 0.0 < f <= 2.0 else f
    except Exception:
        return None


def _cq_to_trimesh(obj, tol=0.6):
    try:
        stl_txt = exporters.toString(obj, "STL", tolerance=tol).encode("utf-8")
        m = trimesh.load(io.BytesIO(stl_txt), file_type="stl")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        return m
    except Exception as e:
        print("[mesh] STL export failed:", e)
        return None


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


# ----------------- Param application -----------------
def _clean_num(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


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
4. USER INTENT - human qualitative assessment and modification requests

Your job: Make MINIMAL parameter changes to match the reference image. DO NOT rewrite logic or framework code.

=== CRITICAL OUTPUT RULES ===
⚠️ Output ONLY Python code. NO explanations, NO markdown fences, NO backticks, NO comments about your changes.
⚠️ Start immediately with import statements.
⚠️ Output a complete, valid, syntactically correct Python module.
⚠️ COPY the baseline source and make MINIMAL changes - only modify parameter VALUES.

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

**WHAT YOU SHOULD MODIFY:**
Only change PositiveFloat/Int parameter VALUES in class definitions:

```python
class Rover(cqparts.Assembly):
    length = PositiveFloat(280)        # ← ONLY modify these numbers
    width = PositiveFloat(170)         # ← Change these values
    wheels_per_side = PositiveFloat(2) # ← e.g., 2 → 3 for 6 wheels
    axle_spacing_mm = PositiveFloat(70)# ← Adjust spacing
```

**Example - CORRECT minimal change:**
```python
# BEFORE (baseline):
wheels_per_side = PositiveFloat(2)  # 2 per side = 4 total

# AFTER (your output):
wheels_per_side = PositiveFloat(3)  # 3 per side = 6 total
```

**Example - WRONG (DO NOT DO THIS):**
```python
# DO NOT rewrite make_components() like this:
def make_components(self):
    for i in range(3):  # ← WRONG! Don't rewrite the logic!
        left_wheel = self.wheel(...)
```

=== STEP-BY-STEP PROCESS ===
1. READ the baseline source code carefully
2. IDENTIFY which parameters control what you see in the image
3. CALCULATE new parameter values based on visual comparison
4. COPY the baseline source entirely
5. CHANGE only the specific parameter values (the numbers)
6. DO NOT modify any method implementations
7. VERIFY syntax is correct (balanced parentheses, proper indentation)

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
Before outputting, verify:
- [ ] All parentheses are balanced
- [ ] All method signatures match the baseline exactly  
- [ ] All imports are copied exactly
- [ ] No undefined attributes (e.g., self.diameter when it should be in ThisWheel class)
- [ ] make_components() returns a dict
- [ ] make_constraints() returns a list
- [ ] No typos in parameter names (e.g., "axle_spacings" vs "axle_spacing_mm")

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
        "# This is the current implementation you should modify\n\n",
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
        "\n- Image 0: REFERENCE (target design)",
    ]
    
    if snapshot_url:
        parts.append("\n- Image 1: CURRENT CAD SNAPSHOT (orthogonal views)")
    
    parts += [
        "\n\n=== NOW OUTPUT YOUR MODIFIED robot_base.py ===",
        "\nRemember: Python code ONLY, no markdown, no explanations.",
        "\nStart with imports, output complete classes.\n",
    ]
    
    images = [u for u in [ref_url, snapshot_url] if u]
    return "".join(parts), images


@app.post("/codegen")
def codegen():
    """
    VLM-powered code generation endpoint.
    
    Accepts:
    - reference: image file (required) - target design
    - snapshot: image file (optional) - current CAD orthogonal views
    - prompt: text (optional) - user qualitative feedback/intent
    
    Returns:
    - Generated robot_base.py code
    - Saved to generated/robot_base_vlm.py
    - Also creates a backup in generated/robot_base_vlm_TIMESTAMP.py
    """
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        ref_url = _data_url_from_upload(request.files.get("reference"))
        snap_url = _data_url_from_upload(request.files.get("snapshot"))

        if not ref_url:
            return jsonify({"ok": False, "error": "reference image required"}), 400

        print(f"[codegen] Building prompt with user_text: {prompt_text[:100] if prompt_text else '(none)'}")
        
        # Quick pre-check: verify source extraction works
        # Use smaller size for faster processing and to fit in context window
        # Vision models need extra space for image tokens
        max_source_chars = 10000 if "vision" in OLLAMA_MODEL.lower() else 15000
        baseline_test = _baseline_cqparts_source(max_chars=max_source_chars)
        print(f"[codegen] Using max_source_chars={max_source_chars} for model {OLLAMA_MODEL}")
        if len(baseline_test) < 1000:
            print(f"[codegen] ✗ ERROR: Source extraction failed! Only {len(baseline_test)} chars")
            return jsonify({
                "ok": False,
                "error": "Failed to extract robot_base.py source code",
                "source_length": len(baseline_test),
                "source_preview": baseline_test[:500],
                "help": "Check that robot_base.py exists in the same directory as optim.py"
            }), 500
        
        final_prompt, images = _build_codegen_prompt(ref_url, snap_url, prompt_text)
        
        print(f"[codegen] Final prompt length: {len(final_prompt)} chars")
        print(f"[codegen] Calling VLM with {len(images)} image(s)...")
        # Ask the VLM for Python code (NOT JSON)
        out = call_vlm(final_prompt, images, expect_json=False)
        raw_txt = out.get("raw", "")
        
        print(f"[codegen] Got {len(raw_txt)} chars from VLM")
        print(f"[codegen] Raw VLM output (first 500 chars):")
        print(raw_txt[:500])
        print(f"[codegen] Raw VLM output (last 200 chars):")
        print(raw_txt[-200:])

        # Strip the sentinel if present
        end_ix = raw_txt.find("# END_OF_MODULE")
        if end_ix != -1:
            print(f"[codegen] Found END_OF_MODULE marker at position {end_ix}")
            raw_txt = raw_txt[:end_ix]
        else:
            print(f"[codegen] No END_OF_MODULE marker found")

        gen_dir = os.path.join(BASE_DIR, "generated")
        os.makedirs(gen_dir, exist_ok=True)

        # Extract valid Python code
        try:
            code_txt = extract_python_module(raw_txt.strip())
        except Exception as e:
            # Save the raw response for debugging
            import time
            reject_path = os.path.join(gen_dir, f"robot_base_vlm.reject_{int(time.time())}.txt")
            with open(reject_path, "w", encoding="utf-8") as rf:
                rf.write(raw_txt)
            print(f"[codegen] Failed to parse Python. Saved to {reject_path}")
            print(f"[codegen] Error: {e}")
            print(f"[codegen] VLM output preview (first 1000 chars):")
            print(raw_txt[:1000])
            print(f"[codegen] VLM output preview (last 500 chars):")
            print(raw_txt[-500:])
            
            # Try to extract even with errors and save it
            attempted_code = None
            try:
                # Try to find code blocks even if they don't parse
                fence_match = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", raw_txt, flags=re.I)
                if fence_match:
                    attempted_code = fence_match[0].strip()
                    # Save it anyway for manual fixing
                    save_path = os.path.join(gen_dir, "robot_base_vlm_WITH_ERRORS.py")
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(f"# WARNING: This code has syntax errors!\n")
                        f.write(f"# Error: {e}\n")
                        f.write(f"# Fix the errors manually and use this file\n\n")
                        f.write(attempted_code)
                    print(f"[codegen] ⚠️ Saved code WITH ERRORS to: {save_path}")
                    print(f"[codegen] You can manually fix the syntax error and use it")
            except:
                pass
            
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"VLM output wasn't valid Python: {e}",
                        "reject_path": reject_path,
                        "raw_length": len(raw_txt),
                        "raw_preview_first": raw_txt[:500],
                        "raw_preview_last": raw_txt[-200:],
                        "code_with_errors": attempted_code[:500] if attempted_code else None,
                        "help": f"Check {reject_path} for full VLM output. Code saved to robot_base_vlm_WITH_ERRORS.py - you can manually fix the syntax error."
                    }
                ),
                400,
            )

        # Log what we got
        print(f"[codegen] Extracted code length: {len(code_txt)} chars")
        print(f"[codegen] First 300 chars: {code_txt[:300]}")
        
        # Save with timestamp backup (save BEFORE validation so we can debug)
        import time
        timestamp = int(time.time())
        backup_path = os.path.join(gen_dir, f"robot_base_vlm_{timestamp}.py")
        mod_path = os.path.join(gen_dir, "robot_base_vlm.py")
        
        # Save even if validation fails (for debugging)
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        
        print(f"[codegen] Saved to {mod_path} (backup: {backup_path})")
        
        # Safety check: ensure it contains essential class definitions
        # Be lenient - check case-insensitive and look for variations
        code_lower = code_txt.lower()
        has_rover = "class rover" in code_lower
        has_robotbase = "class robotbase" in code_lower
        has_imports = "import" in code_lower and ("cadquery" in code_lower or "cq" in code_lower)
        has_defs = "def " in code_lower
        
        print(f"[codegen] Validation checks:")
        print(f"  - Has 'class Rover': {has_rover}")
        print(f"  - Has 'class RobotBase': {has_robotbase}")
        print(f"  - Has imports: {has_imports}")
        print(f"  - Has function defs: {has_defs}")
        
        if not (has_rover or has_robotbase):
            print(f"[codegen] ⚠️ Warning: Generated code missing expected classes!")
            print(f"[codegen] Code preview (first 500 chars):")
            print(code_txt[:500])
            
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Generated code missing Rover or RobotBase class definition",
                        "saved_anyway": True,
                        "module_path": mod_path,
                        "backup_path": backup_path,
                        "code_length": len(code_txt),
                        "code_preview": code_txt[:500],
                        "help": "Check the saved file to see what the VLM generated. May need to adjust prompt."
                    }
                ),
                400,
            )

        # Try to rebuild the GLB with the new code
        # (This is optional - the generated module might need manual integration)
        try:
            print("[codegen] Attempting to rebuild GLB with new code...")
            _rebuild_and_save_glb()
            print("[codegen] GLB rebuild successful")
            glb_updated = True
        except Exception as e:
            print(f"[codegen] GLB rebuild failed (expected): {e}")
            glb_updated = False

        return jsonify({
            "ok": True,
            "module_path": mod_path,
            "backup_path": backup_path,
            "code_length": len(code_txt),
            "glb_updated": glb_updated,
            "message": "Generated robot_base.py - review and integrate manually if needed"
        })
        
    except requests.exceptions.Timeout as e:
        print(f"[codegen] ⏱️ TIMEOUT: VLM took too long (>5 minutes)")
        return jsonify({
            "ok": False,
            "error": "VLM request timed out after 5 minutes",
            "suggestions": [
                "Try a faster/smaller VLM model (e.g., codellama:7b instead of 13b)",
                "Reduce image resolution",
                "Simplify your prompt",
                "Check if Ollama is running and has GPU access",
                "Try without snapshot image (only reference)",
            ],
            "help": "The VLM is taking too long to process. This usually means the model is too slow or the context is too large."
        }), 500
    except Exception as e:
        import traceback
        print(f"[codegen] ERROR: {e}")
        traceback.print_exc()
        
        # Better error message for connection issues
        error_msg = str(e)
        suggestions = []
        
        if "Connection" in error_msg or "connect" in error_msg.lower():
            suggestions = [
                "Check if Ollama is running: curl http://localhost:11434/api/tags",
                "Start Ollama if needed: ollama serve",
                "Verify OLLAMA_URL environment variable is correct",
            ]
        elif "timeout" in error_msg.lower():
            suggestions = [
                "VLM is slow - try a smaller/faster model",
                "Reduce context size",
                "Check Ollama server logs for errors",
            ]
        
        return (
            jsonify({
                "ok": False,
                "error": error_msg,
                "suggestions": suggestions,
                "trace": traceback.format_exc()
            }),
            500,
        )


# ----------------- Helpers: uploads & VLM -----------------
def _data_url_from_upload(file_storage) -> Optional[str]:
    if not file_storage:
        return None
    raw = file_storage.read()
    mime = file_storage.mimetype or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _stitch_images_side_by_side(img1_data_url: str, img2_data_url: str) -> str:
    """
    Combine two images side-by-side for models that only support one image.
    Returns a data URL of the combined image.
    """
    try:
        from PIL import Image
        
        # Decode base64 images
        def decode_data_url(url):
            if url.startswith("data:"):
                header, data = url.split(",", 1)
            else:
                data = url
            return base64.b64decode(data)
        
        img1_bytes = decode_data_url(img1_data_url)
        img2_bytes = decode_data_url(img2_data_url)
        
        # Open images
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))
        
        # Resize to same height if needed (preserve aspect ratio)
        target_height = 1024  # Max height for combined image
        
        # Calculate new dimensions
        h1, h2 = img1.height, img2.height
        w1, w2 = img1.width, img2.width
        
        if h1 > target_height or h2 > target_height:
            scale1 = target_height / h1
            scale2 = target_height / h2
            img1 = img1.resize((int(w1 * scale1), target_height), Image.Resampling.LANCZOS)
            img2 = img2.resize((int(w2 * scale2), target_height), Image.Resampling.LANCZOS)
        elif h1 != h2:
            # Make same height
            target_h = min(h1, h2)
            if h1 > target_h:
                scale = target_h / h1
                img1 = img1.resize((int(w1 * scale), target_h), Image.Resampling.LANCZOS)
            if h2 > target_h:
                scale = target_h / h2
                img2 = img2.resize((int(w2 * scale), target_h), Image.Resampling.LANCZOS)
        
        # Create combined image
        combined_width = img1.width + img2.width
        combined_height = max(img1.height, img2.height)
        combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        
        # Paste images side by side
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))
        
        # Add labels
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined)
            # Use default font
            font_size = 30
            try:
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add text labels
            draw.text((10, 10), "REFERENCE (Target)", fill=(255, 0, 0), font=font)
            draw.text((img1.width + 10, 10), "CURRENT CAD", fill=(0, 0, 255), font=font)
        except Exception as e:
            print(f"[stitch] Could not add labels: {e}")
        
        # Convert to base64
        output = io.BytesIO()
        combined.save(output, format='JPEG', quality=85)
        output.seek(0)
        combined_b64 = base64.b64encode(output.read()).decode('ascii')
        
        print(f"[stitch] ✓ Combined {img1.width}x{img1.height} + {img2.width}x{img2.height} → {combined.width}x{combined.height}")
        
        return f"data:image/jpeg;base64,{combined_b64}"
        
    except ImportError:
        print("[stitch] ✗ PIL not available, cannot stitch images")
        print("[stitch] Install with: pip install Pillow")
        return img1_data_url  # Return first image as fallback
    except Exception as e:
        print(f"[stitch] ✗ Failed to stitch images: {e}")
        return img1_data_url  # Return first image as fallback


def call_vlm(
    final_prompt: str,
    image_data_urls: Optional[List[str] | str],
    *,
    expect_json: bool = True,
) -> Dict[str, Any]:
    def _normalize(imgs):
        if not imgs:
            return None
        if isinstance(imgs, str):
            imgs = [imgs]
        out = []
        for u in imgs:
            if not u:
                continue
            out.append(u.split(",", 1)[1] if u.startswith("data:") else u)
        return out or None

    images_payload = _normalize(image_data_urls)
    err = None
    
    # Handle models that only support 1 image (like llama3.2-vision)
    # If we have 2 images, stitch them side-by-side
    if images_payload and len(images_payload) > 1:
        model_name = OLLAMA_MODEL.lower()
        single_image_models = ["llama3.2-vision", "llama3.2", "llava:7b"]
        
        if any(m in model_name for m in single_image_models):
            print(f"[vlm] Model {OLLAMA_MODEL} supports only 1 image")
            print(f"[vlm] Stitching {len(images_payload)} images together...")
            
            # Reconstruct data URLs for stitching
            img1_url = f"data:image/jpeg;base64,{images_payload[0]}"
            img2_url = f"data:image/jpeg;base64,{images_payload[1]}"
            
            combined_url = _stitch_images_side_by_side(img1_url, img2_url)
            
            # Extract just the base64 part
            if combined_url.startswith("data:"):
                images_payload = [combined_url.split(",", 1)[1]]
            else:
                images_payload = [combined_url]
            
            print(f"[vlm] ✓ Now sending 1 combined image")

    if OLLAMA_URL:
        try:
            # set different stop sequences depending on mode
            # For code generation: NO STOP SEQUENCES - let VLM finish naturally
            # The VLM often wants to explain first, then output code in fences
            # Stop sequences would cut it off prematurely
            if not expect_json:
                # Code generation mode - no stop sequences, let it finish
                stops = []  # Let VLM generate complete response
            else:
                # JSON mode - stop on summary or fences
                stops = ["```", "SUMMARY:"]

            payload = {
                "model": OLLAMA_MODEL,
                "system": (
                    VLM_CODEGEN_PROMPT if not expect_json else VLM_SYSTEM_PROMPT
                ),
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9,
                    # Reduce context for vision models - they often have smaller windows
                    "num_ctx": 4096 if "vision" in OLLAMA_MODEL.lower() else 8192,
                    "stop": stops,
                },
            }
            if expect_json:
                payload["format"] = "json"
            if images_payload:
                payload["images"] = images_payload

            # Code generation can be slow - use longer timeout
            timeout_seconds = 300 if not expect_json else 120  # 5 min for code, 2 min for JSON
            
            print(f"[vlm] Sending to Ollama (timeout: {timeout_seconds}s, context: {payload['options']['num_ctx']})")
            print(f"[vlm] Model: {OLLAMA_MODEL}")
            print(f"[vlm] Images: {len(images_payload) if images_payload else 0}")
            print(f"[vlm] System prompt length: {len(payload.get('system', ''))}")
            print(f"[vlm] User prompt length: {len(payload.get('prompt', ''))}")
            
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout_seconds)
            
            # Check for errors before raising
            if r.status_code != 200:
                error_detail = r.text
                print(f"[vlm] ✗ Ollama returned {r.status_code}")
                print(f"[vlm] Error response: {error_detail}")
                
                # Try to parse error message
                try:
                    error_json = r.json()
                    error_msg = error_json.get("error", error_detail)
                except:
                    error_msg = error_detail
                
                raise RuntimeError(f"Ollama error ({r.status_code}): {error_msg}")
            
            r.raise_for_status()
            response = r.json().get("response", "")
            print(f"[vlm] ✓ Got response: {len(response)} chars")
            return {"provider": "ollama", "raw": response}
        except requests.exceptions.RequestException as e:
            err = f"Ollama error: {e}"
            print(f"[vlm] ✗ Request failed: {err}")
        except Exception as e:
            err = f"Ollama error: {e}"
            print(f"[vlm] ✗ Unexpected error: {err}")

    if LLAVA_URL:
        try:
            payload = {
                # LLAVA_URL may not support system; bake it into the prompt if needed
                "prompt": (
                    (VLM_CODEGEN_PROMPT + "\n\n" + final_prompt)
                    if not expect_json
                    else (VLM_SYSTEM_PROMPT + "\n\n" + final_prompt)
                )
            }
            imgs = images_payload or []
            if imgs:
                payload["image"] = imgs[0]
            r = requests.post(LLAVA_URL, json=payload, timeout=120)
            r.raise_for_status()
            try:
                js = r.json()
                if isinstance(js, dict) and "response" in js:
                    return {"provider": "llava_url", "raw": js["response"]}
                return {"provider": "llava_url", "raw": json.dumps(js)}
            except Exception:
                return {"provider": "llava_url", "raw": r.text}
        except Exception as e:
            err = (err or "") + f" ; LLAVA_URL error: {e}"

    raise RuntimeError(err or "No VLM endpoint configured")


def extract_python_module(text: str) -> str:
    """
    Take possibly chatty model output and return the largest contiguous region of
    lines that compiles as Python. Prefers fenced blocks, else heuristics from first
    code-ish line (shebang/import/from/class/def/@decorator/if __name__ ...).
    Raises ValueError if nothing plausible is found.
    """
    if not text:
        raise ValueError("empty code output")

    t = text.replace("\r\n", "\n")

    # 1) Prefer fenced blocks (```python ...``` or ``` ... ```)
    fence_match = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", t, flags=re.I)
    candidates = []
    candidates += fence_match

    # 2) If no fences: slice from the first code-like line
    if not candidates:
        lines = t.splitlines()
        start = None
        codey = re.compile(
            r'^\s*(#\!|from\s+\w+|import\s+\w+|class\s+\w+|def\s+\w+|@|if\s+__name__\s*==\s*[\'"]__main__[\'"])'
        )
        for i, ln in enumerate(lines):
            if codey.match(ln):
                start = i
                break
        if start is not None:
            candidates.append("\n".join(lines[start:]))

    # 3) Also try the whole thing (maybe it’s already clean)
    candidates.append(t)

    # 4) Light scrub on each candidate, then choose the first that parses
    def _scrub(s: str) -> str:
        s = re.sub(r"^```(?:python)?\s*", "", s.strip(), flags=re.I)
        s = re.sub(r"\s*```$", "", s, flags=re.I).strip()
        # Drop common trailing chatter that sometimes sneaks in
        s = re.split(r"\n(?:Explanation|Notes|Rationale|SUMMARY:|Here is|Here's)\b", s, maxsplit=1)[0]
        # Remove leading explanations like "Here's the modified code:"
        if not s.startswith(("import", "from", "class", "def", "@", "#!")):
            lines = s.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith(("import", "from", "class", "def", "@", "#!")):
                    s = "\n".join(lines[i:])
                    break
        return s.strip()

    seen = set()
    errors = []
    
    for idx, raw in enumerate(candidates):
        if not raw:
            continue
        s = _scrub(raw)
        if s in seen or len(s) < 50:  # Skip very short snippets
            continue
        seen.add(s)
        try:
            ast.parse(s)
            print(f"[extract] ✓ Found valid Python (candidate {idx}, {len(s)} chars)")
            return s
        except SyntaxError as e:
            errors.append(f"Candidate {idx}: SyntaxError at line {e.lineno}: {e.msg}")
            
            # Try some common fixes for VLM-generated code
            if "unmatched" in str(e.msg).lower() or "parenthes" in str(e.msg).lower():
                print(f"[extract] Attempting to fix unmatched parentheses...")
                # Try to balance parentheses
                fixed = s
                # Count and try to fix
                open_count = fixed.count('(')
                close_count = fixed.count(')')
                if open_count != close_count:
                    print(f"[extract] Unbalanced: {open_count} '(' vs {close_count} ')'")
                    # This is tricky, but let's try removing trailing incomplete lines
                    lines = fixed.splitlines()
                    for trim_count in range(1, min(5, len(lines))):
                        trimmed = "\n".join(lines[:-trim_count])
                        try:
                            ast.parse(trimmed)
                            print(f"[extract] ✓ Fixed by removing last {trim_count} lines")
                            return trimmed
                        except:
                            pass
            
            # Try trimming leading non-code lines progressively
            lines = s.splitlines()
            codey2 = re.compile(r"^\s*(from|import|class|def|@|if\s+__name__)")
            for i in range(len(lines)):
                if codey2.match(lines[i]):
                    chunk = "\n".join(lines[i:])
                    try:
                        ast.parse(chunk)
                        print(f"[extract] ✓ Found valid Python after trimming {i} lines")
                        return chunk
                    except Exception:
                        pass
        except Exception as e:
            errors.append(f"Candidate {idx}: {type(e).__name__}: {e}")
    
    # If we got here, nothing worked
    error_msg = "no valid Python block found in model output"
    if errors:
        error_msg += f"\n  Tried {len(candidates)} candidates, errors:\n  " + "\n  ".join(errors[:3])
    raise ValueError(error_msg)


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
@app.get("/")
def index():
    return render_template("viewer.html")


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


@app.post("/vlm")
def vlm():
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        selected = (request.form.get("selected_class") or "").strip() or None
        classes = json.loads(request.form.get("classes") or "[]")
        if not isinstance(classes, list):
            classes = []
        data_url = _data_url_from_upload(request.files.get("image"))
        grounding = ["Known component classes:", *[f"- {c}" for c in classes]]
        if selected:
            grounding.append(f"\nUser highlighted class: {selected}")
        grounding.append("\nUser prompt:\n" + prompt_text)
        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding)
        resp = call_vlm(final_prompt, data_url)
        raw = resp.get("raw", "")
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}\s*$", raw.strip())
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        return jsonify(
            {
                "ok": True,
                "provider": resp.get("provider"),
                "response": {"raw": raw, "json": parsed},
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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


def build_rover_scene_glb_cqparts() -> bytes:
    print("Generating GLB via cqparts…")
    rv = Rover(
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

    saved_pending_attr = hasattr(Rover, "_pending_adds")
    saved_pending_val = getattr(Rover, "_pending_adds", None)
    setattr(Rover, "_pending_adds", [])

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
    if t.is_alive() or build_err[0] is not None:
        print("[warn] build timed out or errored:", build_err[0])
    if saved_pending_attr:
        setattr(Rover, "_pending_adds", saved_pending_val)
    else:
        try:
            delattr(Rover, "_pending_adds")
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


def _rebuild_and_save_glb():
    setattr(Rover, "_pending_adds", list(PENDING_ADDS))
    glb = build_rover_scene_glb_cqparts()
    with open(ROVER_GLB_PATH, "wb") as f:
        f.write(glb)


# ----------------- GLB route & static -----------------
@app.get("/model.glb")
def model_glb():
    try:
        glb = build_rover_scene_glb({})
        return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
    except Exception as e:
        import traceback

        return Response(
            "model.glb build failed:\n" + traceback.format_exc(),
            status=500,
            mimetype="text/plain",
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


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    threading.Thread(target=_warm_build, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
