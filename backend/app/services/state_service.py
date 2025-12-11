"""
State Management Service
Handles application state, history, and parameters.
"""
from typing import Dict, Any, Optional, List

# Note: Global state variables are defined in run.py
# This module provides functions that operate on those globals
# Functions are imported into run.py to maintain backward compatibility


def snapshot(current_params: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Create a snapshot of current parameters."""
    snap = {}
    for k, v in current_params.items():
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


def ensure_initial_history(current_params, history, h_ptr, init_snapshot, snapshot_fn):
    """Ensure history is initialized with initial snapshot."""
    if h_ptr == -1:
        init_snapshot = snapshot_fn(current_params)
        history[:] = [init_snapshot.copy()]
        h_ptr = 0
    return init_snapshot, history, h_ptr


def push_history(current_params, history, h_ptr, snapshot_fn):
    """Push current state to history."""
    if h_ptr == -1:
        init_snapshot = snapshot_fn(current_params)
        history[:] = [init_snapshot.copy()]
        h_ptr = 0
    if h_ptr < len(history) - 1:
        history[:] = history[: h_ptr + 1]
    history.append(snapshot_fn(current_params))
    h_ptr = len(history) - 1
    return h_ptr


def restore(snapshot: Dict[str, Optional[float]], current_params: Dict[str, Optional[float]]):
    """Restore state from a snapshot."""
    for k in current_params.keys():
        current_params[k] = snapshot.get(k, current_params[k])


# Wrapper functions that access globals from run.py for backward compatibility
def snapshot_global() -> Dict[str, Optional[float]]:
    """Create a snapshot using global CURRENT_PARAMS."""
    from run import CURRENT_PARAMS
    return snapshot(CURRENT_PARAMS)


def ensure_initial_history_global():
    """Ensure history is initialized with initial snapshot using globals."""
    from run import CURRENT_PARAMS, HISTORY, H_PTR, INIT_SNAPSHOT
    global INIT_SNAPSHOT
    if H_PTR == -1:
        INIT_SNAPSHOT = snapshot(CURRENT_PARAMS)
        HISTORY[:] = [INIT_SNAPSHOT.copy()]
        return 0
    return H_PTR


def push_history_global():
    """Push current state to history using globals."""
    from run import CURRENT_PARAMS, HISTORY, H_PTR, INIT_SNAPSHOT
    global H_PTR
    if H_PTR == -1:
        INIT_SNAPSHOT = snapshot(CURRENT_PARAMS)
        HISTORY[:] = [INIT_SNAPSHOT.copy()]
        H_PTR = 0
    if H_PTR < len(HISTORY) - 1:
        HISTORY[:] = HISTORY[: H_PTR + 1]
    HISTORY.append(snapshot(CURRENT_PARAMS))
    H_PTR = len(HISTORY) - 1


def restore_global(snapshot_data: Dict[str, Optional[float]]):
    """Restore state from a snapshot using globals."""
    from run import CURRENT_PARAMS
    restore(snapshot_data, CURRENT_PARAMS)


def get_state(which: str = "all") -> Dict[str, Any]:
    """Get current application state."""
    # Import from run.py to access global state
    from run import (
        CURRENT_PARAMS, CONTEXT, HISTORY, H_PTR, INIT_SNAPSHOT,
        PENDING_ADDS, STATE
    )
    from app.core.component_registry import COMPONENT_REGISTRY
    
    ensure_initial_history_global()
    
    payload = {
        "initial": HISTORY[0] if HISTORY else INIT_SNAPSHOT or snapshot_global(),
        "current": snapshot_global(),
        "context": CONTEXT,
        "known_classes": sorted(list(COMPONENT_REGISTRY.keys())),
        "history": HISTORY[: H_PTR + 1],
        "pending_adds": list(PENDING_ADDS),
    }
    
    if which in payload:
        return {"ok": True, which: payload[which]}
    return {"ok": True, "state": payload}


def reset_state() -> Dict[str, Any]:
    """Reset state to initial values."""
    # Import from run.py to access global state
    from run import (
        CURRENT_PARAMS, CONTEXT, HISTORY, H_PTR, INIT_SNAPSHOT,
        PENDING_ADDS, STATE
    )
    
    for k in list(CURRENT_PARAMS.keys()):
        CURRENT_PARAMS[k] = None
    PENDING_ADDS.clear()
    STATE["selected_parts"] = []
    HISTORY[:] = [{k: None for k in CURRENT_PARAMS.keys()}]
    H_PTR = -1
    
    # Try to rebuild GLB
    try:
        from run import _rebuild_and_save_glb
        _rebuild_and_save_glb()
    except Exception as e:
        import logging
        logging.warning("reset: rebuild failed: %s", e)
    
    return {
        "ok": True,
        "current": snapshot_global(),
        "history_len": len(HISTORY),
        "pending_adds_len": len(PENDING_ADDS),
    }


def cad_state_json() -> Dict[str, Any]:
    """Get current CAD state as JSON."""
    from run import (
        CURRENT_PARAMS, CONTEXT, HISTORY, H_PTR, STATE, 
        PENDING_ADDS, COMPONENT_REGISTRY
    )
    return {
        "current_params": snapshot(CURRENT_PARAMS),
        "context": CONTEXT,
        "known_classes": sorted(list(COMPONENT_REGISTRY.keys())),
        "selected_parts": list(STATE.get("selected_parts", [])),
        "history": HISTORY[: H_PTR + 1],
        "pending_adds": list(PENDING_ADDS),
    }


def apply_changes(changes: List[dict], excerpt: Optional[str] = None) -> tuple[int, Dict[str, Any]]:
    """
    Apply a list of changes to the CAD model.
    
    This is a wrapper that calls apply_changes_list (the full implementation).
    
    Returns:
        Tuple of (status_code, response_dict)
    """
    return apply_changes_list(changes, excerpt)


def apply_changes_list(changes: List[dict], excerpt: Optional[str] = None) -> tuple[int, Dict[str, Any]]:
    """
    Apply a list of changes to the CAD model.
    This is the full implementation that accesses globals from run.py.
    
    Returns:
        Tuple of (status_code, response_dict)
    """
    from run import (
        CURRENT_PARAMS, CONTEXT, HISTORY, H_PTR, PENDING_ADDS, STATE,
        HIDDEN_PREFIXES, Rover, _Stepper, _Electronics, _PanTilt, _ThisWheel,
        _rebuild_and_save_glb,
        _normalize_change
    )
    from app.core.component_registry import get_component_spec
    from app.utils.param_normalization import normalize_change
    
    if not changes:
        return 400, {"ok": False, "error": "No change objects supplied"}

    ensure_initial_history_global()
    push_history_global()

    rv = Rover(
        stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel
    )
    highlight_key = None

    for raw in changes:
        ch = normalize_change(raw) or raw
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
            from app.core.component_registry import ModelAdapter
            adapter = ModelAdapter(Rover)
            comp.add_fn(adapter=adapter, **params)

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

    push_history_global()
    try:
        _rebuild_and_save_glb()
        return 200, {
            "ok": True,
            "highlight_key": highlight_key or "wheel",
            "excerpt": excerpt,
        }
    except Exception as e:
        return 500, {"ok": False, "error": str(e)}
