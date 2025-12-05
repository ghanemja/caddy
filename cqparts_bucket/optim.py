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
    """
    Load FreeCAD module. Tries multiple locations:
    1. Direct import (if available in Python path)
    2. Conda environment lib directory
    3. Extracted AppImage location
    """
    # Try 1: Direct import (conda-installed FreeCAD may work this way)
    try:
        import FreeCAD
        print("[freecad] ✓ Loaded FreeCAD from system/conda")
        return FreeCAD
    except ImportError:
        pass
    
    # Try 2: Load from conda environment's lib directory
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_freecad_path = os.path.join(conda_prefix, "lib", "FreeCAD.so")
        if os.path.exists(conda_freecad_path):
            try:
                spec = importlib.util.spec_from_file_location("FreeCAD", conda_freecad_path)
                FreeCAD = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(FreeCAD)
                sys.modules["FreeCAD"] = FreeCAD
                print(f"[freecad] ✓ Loaded FreeCAD from conda: {conda_freecad_path}")
                return FreeCAD
            except Exception as e:
                print(f"[freecad] ⚠ Failed to load from conda: {e}")
    
    # Try 3: Load from AppImage extraction directory
    appimage_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "squashfs-root", "usr", "lib", "FreeCAD.so"
    )
    if os.path.exists(appimage_path):
        try:
            spec = importlib.util.spec_from_file_location("FreeCAD", appimage_path)
            FreeCAD = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(FreeCAD)
            sys.modules["FreeCAD"] = FreeCAD
            print(f"[freecad] ✓ Loaded FreeCAD from AppImage: {appimage_path}")
            return FreeCAD
        except Exception as e:
            print(f"[freecad] ⚠ Failed to load from AppImage: {e}")
    
    # If all methods fail, raise error with helpful message
    raise ImportError(
        "Could not load FreeCAD module. Tried:\n"
        f"  1. Direct import from Python path\n"
        f"  2. Conda environment: {conda_freecad_path if conda_prefix else 'N/A'}\n"
        f"  3. AppImage: {appimage_path}\n"
        "Install FreeCAD with: conda install -c conda-forge freecad"
    )


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
USE_FINETUNED_MODEL = os.environ.get("USE_FINETUNED_MODEL", "1") == "1"  # Use fine-tuned by default
# Default to runs/onevision_lora_small/checkpoint-4 relative to project root (parent of cqparts_bucket)
# BASE_DIR will be defined later, so we compute it here
_vlm_base_dir = os.path.dirname(__file__)
_default_model_path = os.path.join(os.path.dirname(_vlm_base_dir), "runs", "onevision_lora_small", "checkpoint-4")
FINETUNED_MODEL_PATH = os.environ.get(
    "FINETUNED_MODEL_PATH", 
    _default_model_path
)
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava:latest")  # Default to llava:latest (user has this)
LLAVA_URL = os.environ.get("LLAVA_URL")  # optional

# Global variable to hold the fine-tuned model and processor
_finetuned_model = None
_finetuned_processor = None


def load_finetuned_model():
    """Load the fine-tuned VLM model with LoRA adapter."""
    global _finetuned_model, _finetuned_processor
    
    if not USE_FINETUNED_MODEL:
        print("[vlm] Fine-tuned model disabled, will use Ollama/LLAVA")
        return
    
    if _finetuned_model is not None:
        print("[vlm] Fine-tuned model already loaded")
        return
    
    try:
        print(f"[vlm] Loading fine-tuned model from {FINETUNED_MODEL_PATH}...")
        
        # Import necessary libraries
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        from peft import PeftModel
        import torch
        
        # Load base model
        base_model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        print(f"[vlm] Loading base model: {base_model_name}")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[vlm] Using device: {device}")
        
        # Load processor
        print(f"[vlm] Loading processor...")
        _finetuned_processor = AutoProcessor.from_pretrained(base_model_name)
        print(f"[vlm] ✓ Processor loaded")
        
        # Load base model with optimizations for faster loading
        # Note: If model is cached, this will be fast. First-time download may take a few minutes.
        print(f"[vlm] Loading base model from cache (or downloading if first time)...")
        base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            # Speed up loading by using safetensors
            use_safetensors=True,
        )
        print(f"[vlm] ✓ Base model loaded to {device}")
        
        # Load LoRA adapter
        print(f"[vlm] Loading LoRA adapter from {FINETUNED_MODEL_PATH}")
        _finetuned_model = PeftModel.from_pretrained(
            base_model,
            FINETUNED_MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # Set to evaluation mode
        _finetuned_model.eval()
        
        print(f"[vlm] ✓ Fine-tuned model loaded successfully on {device}")
        
    except ImportError as e:
        print(f"[vlm] ✗ Failed to import required libraries: {e}")
        print("[vlm] Install with: pip install transformers peft torch pillow accelerate")
        _finetuned_model = None
        _finetuned_processor = None
    except Exception as e:
        print(f"[vlm] ✗ Failed to load fine-tuned model: {e}")
        import traceback
        traceback.print_exc()
        _finetuned_model = None
        _finetuned_processor = None


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
        # Ask the VLM for full Python code (NOT JSON)
        out = call_vlm(final_prompt, images, expect_json=False)
        raw_txt = out.get("raw", "")
        
        print(f"[codegen] Got {len(raw_txt)} chars from VLM")
        print(f"[codegen] Raw VLM output (first 500 chars):")
        print(raw_txt[:500])

        gen_dir = os.path.join(BASE_DIR, "generated")
        os.makedirs(gen_dir, exist_ok=True)

        # Extract valid Python code from VLM output
        try:
            code_txt = extract_python_module(raw_txt.strip())
            print(f"[codegen] ✓ Extracted {len(code_txt)} chars of Python code")
        except Exception as e:
            # Save the raw response for debugging
            import time
            reject_path = os.path.join(gen_dir, f"robot_base_vlm.reject_{int(time.time())}.txt")
            with open(reject_path, "w", encoding="utf-8") as rf:
                rf.write(raw_txt)
            print(f"[codegen] ✗ Failed to extract Python code. Saved to {reject_path}")
            print(f"[codegen] Error: {e}")
            
            return jsonify({
                "ok": False,
                "error": f"Could not extract valid Python from VLM output: {e}",
                "reject_path": reject_path,
                "raw_preview": raw_txt[:1000],
                "help": f"Check {reject_path} for full VLM output."
            }), 400

        # Apply automatic fixes/normalization
        code_txt = normalize_generated_code(code_txt)

        # Validate code compiles (after normalization)
        try:
            compile(code_txt, "robot_base_vlm.py", "exec")
            print(f"[codegen] ✓ Normalized code compiles successfully")
        except SyntaxError as e:
            import time
            reject_path = os.path.join(gen_dir, f"robot_base_vlm.syntax_error_{int(time.time())}.py")
            with open(reject_path, "w", encoding="utf-8") as rf:
                rf.write(f"# SYNTAX ERROR\n# {e}\n\n")
                rf.write(code_txt)
            
            return jsonify({
                "ok": False,
                "error": f"Generated code has syntax error: {e}",
                "reject_path": reject_path
            }), 400
        
        # Save the generated code
        import time
        timestamp = int(time.time())
        backup_path = os.path.join(gen_dir, f"robot_base_vlm_{timestamp}.py")
        mod_path = os.path.join(gen_dir, "robot_base_vlm.py")
        
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        
        print(f"[codegen] ✓ Saved to {mod_path}")
        print(f"[codegen] ✓ Backup: {backup_path}")
        print(f"[codegen] Code: {len(code_txt)} chars, {len(code_txt.splitlines())} lines")

        # Try to rebuild the GLB with the new code
        # (This is optional - the generated module might need manual integration)
        try:
            print("[codegen] Attempting to rebuild GLB with new generated code...")
            _rebuild_and_save_glb(use_generated=True)  # Use the generated code!
            print("[codegen] ✓ GLB rebuild successful with generated code")
            glb_updated = True
        except Exception as e:
            print(f"[codegen] GLB rebuild failed: {e}")
            import traceback
            traceback.print_exc()
            glb_updated = False

        return jsonify({
            "ok": True,
            "code": code_txt,  # Include the generated code for display
            "module_path": mod_path,
            "backup_path": backup_path,
            "code_length": len(code_txt),
            "code_lines": len(code_txt.splitlines()),
            "glb_updated": glb_updated,
            "message": "Generated complete robot_base.py with modifications"
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
    
    # Try Ollama first (fastest, no download needed) if model exists
    # Check if Ollama model exists before trying it
    ollama_model_available = False
    if OLLAMA_URL:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                models_data = r.json()
                model_names = [m.get("name", "") for m in models_data.get("models", [])]
                ollama_model_available = OLLAMA_MODEL in model_names
        except:
            pass
    
    # Try fine-tuned model first if it's loaded
    if _finetuned_model is not None and _finetuned_processor is not None:
        try:
            print("[vlm] Using fine-tuned model...")
            import torch
            from PIL import Image
            import io
            import base64
            
            # Prepare images
            images = []
            if images_payload:
                for img_b64 in images_payload:
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    images.append(img)
            
            # Prepare the conversation format
            system_prompt = VLM_CODEGEN_PROMPT if not expect_json else VLM_SYSTEM_PROMPT
            
            # Format as conversation (LLaVA OneVision format)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt + "\n\n" + final_prompt}
                    ]
                }
            ]
            
            # Add images to the user message
            if images:
                for img in images:
                    conversation[0]["content"].insert(0, {"type": "image"})
            
            # Apply chat template and process
            prompt_text = _finetuned_processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = _finetuned_processor(
                images=images if images else None,
                text=prompt_text,
                return_tensors="pt"
            )
            
            # Move to same device as model
            device = next(_finetuned_model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            print(f"[vlm] Generating response...")
            
            # Generate
            # For code generation, use more tokens and very low temperature for faithful copying
            max_tokens = 6144 if not expect_json else 1024  # Code needs lots of tokens for 200+ lines
            temp = 0.01 if not expect_json else 0.1  # Very low temp for precise copying
            
            with torch.no_grad():
                output_ids = _finetuned_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=0.98,
                    do_sample=True if temp > 0 else False,
                    repetition_penalty=1.1,  # Prevent getting stuck in loops
                )
                
            print(f"[vlm] Generated with max_tokens={max_tokens}, temp={temp}")
            
            # Decode response
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            response = _finetuned_processor.decode(generated_ids, skip_special_tokens=True)
            
            print(f"[vlm] ✓ Got response from fine-tuned model: {len(response)} chars")
            return {"provider": "finetuned", "raw": response}
            
        except Exception as e:
                err = f"Fine-tuned model error: {e}"
                print(f"[vlm] ✗ Fine-tuned model failed: {err}")
                import traceback
                traceback.print_exc()
                # Fall back to Ollama
    
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
        # If Ollama fails, continue to try other options
    
    # Try fine-tuned model if Ollama not available, model doesn't exist, or failed
    # Load fine-tuned model if not already loaded and Ollama model doesn't exist
    if USE_FINETUNED_MODEL:
        if _finetuned_model is None and not ollama_model_available:
            print("[vlm] Loading fine-tuned model (Ollama model not available)...")
            load_finetuned_model()
        
        if _finetuned_model is not None and _finetuned_processor is not None:
            try:
                print("[vlm] Using fine-tuned model...")
                import torch
                from PIL import Image
                import io
                import base64
                
                # Prepare images
                images = []
                if images_payload:
                    for img_b64 in images_payload:
                        img_bytes = base64.b64decode(img_b64)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        images.append(img)
                
                # Prepare the conversation format
                system_prompt = VLM_CODEGEN_PROMPT if not expect_json else VLM_SYSTEM_PROMPT
                
                # Format as conversation (LLaVA OneVision format)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt + "\n\n" + final_prompt}
                        ]
                    }
                ]
                
                # Add images to the user message
                if images:
                    for img in images:
                        conversation[0]["content"].insert(0, {"type": "image"})
                
                # Apply chat template and process
                prompt_text = _finetuned_processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                # Process inputs
                inputs = _finetuned_processor(
                    images=images if images else None,
                    text=prompt_text,
                    return_tensors="pt"
                )
                
                # Move to same device as model
                device = next(_finetuned_model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                print(f"[vlm] Generating response...")
                
                # Generate
                max_tokens = 6144 if not expect_json else 1024
                temp = 0.01 if not expect_json else 0.1
                
                with torch.no_grad():
                    output_ids = _finetuned_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_p=0.98,
                        do_sample=True if temp > 0 else False,
                        repetition_penalty=1.1,
                    )
                
                # Decode response
                generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
                response = _finetuned_processor.decode(generated_ids, skip_special_tokens=True)
                
                print(f"[vlm] ✓ Got response from fine-tuned model: {len(response)} chars")
                return {"provider": "finetuned", "raw": response}
                
            except Exception as e:
                err = f"Fine-tuned model error: {e}"
                print(f"[vlm] ✗ Fine-tuned model failed: {err}")
                import traceback
                traceback.print_exc()
    
    # Fallback: Try LLAVA_URL if configured
    if LLAVA_URL:
        try:
            payload = {
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


def normalize_generated_code(code: str) -> str:
    """
    Normalize/fix common errors in VLM-generated code.
    
    Common issues:
    - Hyphens instead of underscores in attribute names
    - Missing self. prefix
    - Undefined 'offsets' variable
    - Missing class parameters
    - Missing imports (VLM truncated the file)
    """
    print("[normalize] Applying automatic fixes to generated code...")
    
    original_code = code
    fixes_applied = []
    
    # Fix 0: Check if imports are missing and add them
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
        # VLM truncated the file! Add back the header
        missing_imports = []
        for imp in required_imports:
            if imp not in code:
                missing_imports.append(imp)
        
        if missing_imports:
            header = "#!/usr/bin/env python3\n\n" + "\n".join(required_imports) + "\n\n"
            code = header + code
            fixes_applied.append(f"Added missing imports ({len(missing_imports)} imports restored)")
            print(f"[normalize] ✗ VLM truncated file - restored {len(missing_imports)} missing imports")
    
    # Fix 0b: Check if RobotBase class is missing (VLM sometimes skips it)
    if "class RobotBase" not in code:
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
        # Insert RobotBase before the first ThisWheel or ThisStepper class
        if "class ThisWheel" in code:
            code = code.replace("class ThisWheel", robot_base_class + "class ThisWheel")
            fixes_applied.append("Added missing RobotBase class")
            print(f"[normalize] ✗ VLM skipped RobotBase - restored it")
    
    # Fix 1: Replace hyphenated attribute names with underscores
    # self.wheelbase_span-mm → self.wheelbase_span_mm
    # self.axle_spacing-mm → self.axle_spacing_mm
    hyphen_fixes = {
        r'\.wheelbase_span-mm': '.wheelbase_span_mm',
        r'\.axle_spacing-mm': '.axle_spacing_mm',
        r'\.wheel_z_offset-mm': '.wheel_z_offset_mm',
        r'\.wheel-diameter': '.wheel_diameter',
        r'\.wheel-width': '.wheel_width',
    }
    
    for pattern, replacement in hyphen_fixes.items():
        import re
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            fixes_applied.append(f"Fixed hyphenated attribute: {pattern} → {replacement}")
    
    # Fix 2: Undefined 'offsets' variable in make_constraints
    # for i, off in enumerate(offsets): → for i, off in enumerate(self._axle_offsets()):
    if re.search(r'for\s+i,\s+off\s+in\s+enumerate\(offsets\)', code):
        code = re.sub(
            r'for\s+i,\s+off\s+in\s+enumerate\(offsets\)',
            'for i, off in enumerate(self._axle_offsets())',
            code
        )
        fixes_applied.append("Fixed undefined 'offsets' → 'self._axle_offsets()'")
    
    # Fix 3: Missing class parameters - add them if Rover class exists but missing params
    rover_match = re.search(r'class Rover\(cqparts\.Assembly\):\s*\n((?:\s+\w+.*\n)*)', code)
    if rover_match:
        rover_body = rover_match.group(1)
        # Check if essential parameters are missing
        has_length = 'length = ' in rover_body
        has_width = 'width = ' in rover_body
        has_chamfer = 'chamfer = ' in rover_body
        has_thickness = 'thickness = ' in rover_body
        
        if not has_length or not has_width:
            # Need to add missing parameters
            # Find where to insert (after class declaration, before other params)
            insert_params = []
            if not has_length:
                insert_params.append('    length = PositiveFloat(280)')
            if not has_width:
                insert_params.append('    width = PositiveFloat(170)')
            if not has_chamfer:
                insert_params.append('    chamfer = PositiveFloat(55)')
            if not has_thickness:
                insert_params.append('    thickness = PositiveFloat(6)')
            
            if insert_params:
                # Insert after "class Rover..." line
                code = re.sub(
                    r'(class Rover\(cqparts\.Assembly\):\s*\n)',
                    r'\1' + '\n'.join(insert_params) + '\n',
                    code
                )
                fixes_applied.append(f"Added missing Rover parameters: {', '.join([p.split('=')[0].strip() for p in insert_params])}")
    
    # Fix 4: Wrong base length calculation
    # length=self.wheels_per_side * self.axle_spacing_mm → length=self.length
    if 'length=self.wheels_per_side' in code:
        code = code.replace(
            'length=self.wheels_per_side * self.axle_spacing_mm',
            'length=self.length'
        )
        fixes_applied.append("Fixed RobotBase length calculation")
    
    # Fix 5: Return statement in _axle_offsets with syntax error
    # return [self.axle_spacing-mm] → return [self.chamfer]
    code = re.sub(
        r'return \[\s*self\.axle_spacing-mm\s*\]',
        'return [self.length / 2 - self.chamfer]',
        code
    )
    
    # Fix 6: max_off calculation error
    # max_off = self.axle_spacing-mm → max_off = self.length - self.chamfer
    code = re.sub(
        r'max_off = self\.axle_spacing-mm',
        'max_off = self.length - self.chamfer',
        code
    )
    
    # Fix 7: Electronics/sensors should not have target=base in make_components
    # self.electronics(target=base) → self.electronics()
    # self.sensors(target=base) → self.sensors(target=base) [keep for sensors!]
    if 'self.electronics(target=base)' in code:
        code = code.replace('self.electronics(target=base)', 'self.electronics()')
        fixes_applied.append("Removed target=base from electronics() call")
    
    # Fix 8: max_off should be length - chamfer, not axle_spacing_mm
    if 'max_off = self.axle_spacing_mm' in code:
        code = code.replace(
            'max_off = self.axle_spacing_mm',
            'max_off = self.length - self.chamfer'
        )
        fixes_applied.append("Fixed max_off calculation in _axle_offsets")
    
    # Fix 8b: Fix _axle_offsets to allow 0 wheels
    # The baseline has: n = max(1, int(...)) which forces minimum 1 wheel
    # Change to: n = max(0, int(...)) to allow 0 wheels
    if 'n = max(1, int(round(float(self.wheels_per_side))))' in code:
        code = code.replace(
            'n = max(1, int(round(float(self.wheels_per_side))))',
            'n = max(0, int(round(float(self.wheels_per_side))))'
        )
        fixes_applied.append("Fixed _axle_offsets to allow 0 wheels (changed max(1,...) to max(0,...))")
    
    # Fix 8c: Add check for 0 wheels at the start of _axle_offsets
    if 'def _axle_offsets(self):' in code and 'if n == 0:' not in code:
        # Insert "if n == 0: return []" after n = max(0, ...)
        code = code.replace(
            'n = max(0, int(round(float(self.wheels_per_side))))',
            'n = max(0, int(round(float(self.wheels_per_side))))\n        if n == 0:\n            return []'
        )
        fixes_applied.append("Added early return for 0 wheels in _axle_offsets")
    
    # Fix 9: Remove any trailing incomplete register() calls or display calls
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip incomplete or problematic lines at the end
        if line.strip().startswith('cq.display.') or \
           line.strip().startswith('register(') and 'model=' not in line:
            continue
        cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)
    
    # Fix 10: Detect and truncate VLM hallucinations (repetitive class definitions)
    # If we see the same class name defined multiple times with incrementing values,
    # truncate at the first repetition
    lines = code.split('\n')
    class_names_seen = {}
    truncate_at = None
    
    for i, line in enumerate(lines):
        # Check for class definitions
        class_match = re.match(r'^class\s+(\w+)', line)
        if class_match:
            class_name = class_match.group(1)
            if class_name in class_names_seen:
                # We've seen this class before - this is likely a repetition/hallucination
                # Truncate here
                truncate_at = class_names_seen[class_name]
                print(f"[normalize] ✗ Detected VLM hallucination: class '{class_name}' repeated at line {i}")
                print(f"[normalize] Truncating at line {truncate_at} (first occurrence)")
                fixes_applied.append(f"Truncated hallucination: repeated class '{class_name}'")
                break
            else:
                class_names_seen[class_name] = i
    
    if truncate_at is not None:
        code = '\n'.join(lines[:truncate_at])
        # Ensure we end cleanly - add a comment if the last line is incomplete
        if code and not code.endswith('\n'):
            code += '\n'
        code += '\n# === End of generated code ===\n'
    
    # Report fixes
    if fixes_applied:
        print(f"[normalize] Applied {len(fixes_applied)} automatic fixes:")
        for fix in fixes_applied:
            print(f"[normalize]   - {fix}")
    else:
        print("[normalize] No fixes needed - code looks clean")
    
    return code


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


@app.get("/debug")
def debug_viewer():
    """Simple viewer for inspecting GLB output."""
    return render_template("simple_viewer.html")


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
        import traceback
        error_msg = f"VLM endpoint error: {str(e)}"
        print(f"[vlm endpoint] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.post("/ingest_mesh")
def ingest_mesh():
    """
    Mesh ingestion endpoint: PointNet++ segmentation + VLM semantics.
    
    Accepts:
    - mesh: mesh file (OBJ/STL/PLY) - required
    
    Returns:
    - category: object category
    - final_parameters: list of semantic parameters
    - raw_parameters: list of raw geometric parameters
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
        
        print(f"[ingest_mesh] Processing mesh: {mesh_path}", flush=True)
        print(f"[ingest_mesh] Note: This may take 2-3 minutes on first run (VLM model download)", flush=True)
        
        # Import the ingestion pipeline
        # Add parent directory to path so we can import vlm_cad
        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from vlm_cad.pointnet_seg.model import load_pretrained_model
        from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient
        from vlm_cad.semantics.ingest_mesh import ingest_mesh_to_semantic_params
        
        # Load PointNet++ model
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "pointnet2", "pointnet2_part_seg_msg.pth"
        )
        
        if not os.path.exists(checkpoint_path):
            return jsonify({
                "ok": False,
                "error": f"PointNet++ model not found at {checkpoint_path}"
            }), 500
        
        print(f"[ingest_mesh] Loading PointNet++ model from {checkpoint_path}")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_pretrained_model(
            checkpoint_path=checkpoint_path,
            num_classes=50,
            use_normals=True,
            device=device,
        )
        
        # Initialize VLM client
        # Prefer Ollama if available, otherwise use fine-tuned model
        vlm = None
        
        # Check if Ollama is available
        ollama_available = False
        if OLLAMA_URL:
            try:
                import requests
                r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
                ollama_available = (r.status_code == 200)
            except:
                pass
        
        if ollama_available:
            try:
                from vlm_cad.semantics.vlm_client_ollama import OllamaVLMClient
                vlm = OllamaVLMClient()
                print("[ingest_mesh] Using Ollama VLM")
            except Exception as e:
                print(f"[ingest_mesh] Warning: Could not use Ollama: {e}")
        
        if vlm is None:
            try:
                vlm = FinetunedVLMClient()
                print("[ingest_mesh] Using fine-tuned VLM")
            except Exception as e2:
                print(f"[ingest_mesh] Warning: Could not use fine-tuned VLM: {e2}")
                from vlm_cad.semantics.vlm_client import DummyVLMClient
                vlm = DummyVLMClient()
                print("[ingest_mesh] Using dummy VLM (for testing)")
        
        # Run ingestion pipeline
        render_dir = os.path.join(temp_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)
        
        print(f"[ingest_mesh] Running ingestion pipeline...")
        result = ingest_mesh_to_semantic_params(
            mesh_path=mesh_path,
            vlm=vlm,
            model=model,
            render_output_dir=render_dir,
            num_points=2048,
        )
        
        # Convert result to JSON-serializable format
        def param_to_dict(p):
            """Convert FinalParameter to dict with backward compatibility."""
            d = {
                "id": p.id,
                "semantic_name": p.semantic_name,
                "value": p.value,
                "units": p.units,
                "description": p.description,
                "confidence": p.confidence,
                "raw_sources": p.raw_sources,
            }
            # Backward compatibility: also include "name"
            d["name"] = p.semantic_name
            return d
        
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
                for p in result.raw_parameters[:20]  # Limit to first 20 for response size
            ],
            "proposed_parameters": [param_to_dict(p) for p in result.proposed_parameters],
            "final_parameters": [param_to_dict(p) for p in result.proposed_parameters],  # Backward compat
            "metadata": {
                "num_points": result.extra.get("num_points", 0),
                "num_parts": result.extra.get("num_parts", 0),
            }
        }
        
        # Cleanup temp directory (keep renders for now, they might be useful)
        # shutil.rmtree(temp_dir)  # Commented out - might want to keep renders
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Mesh ingestion error: {str(e)}"
        print(f"[ingest_mesh] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
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
    """Serve the default Curiosity Rover STL file."""
    default_path = "/Users/janelleg/Downloads/Curiosity Rover 3D Printed Model/Simplified Curiosity Model (Small)/STL Files/body-small.STL"
    
    if os.path.exists(default_path):
        return send_file(default_path, mimetype="model/stl")
    
    return Response(
        f"Default STL file not found at {default_path}",
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
            
            from vlm_cad.pointnet_seg.model import load_pretrained_model
            from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient
            from vlm_cad.semantics.ingest_mesh import ingest_mesh_to_semantic_params
            
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
    
    threading.Thread(target=_warm_build, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
