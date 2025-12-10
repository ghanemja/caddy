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
        PENDING_ADDS, STATE, _snapshot, _ensure_initial_history
    )
    from app.core.component_registry import COMPONENT_REGISTRY
    
    _ensure_initial_history()
    
    payload = {
        "initial": HISTORY[0] if HISTORY else INIT_SNAPSHOT or _snapshot(),
        "current": _snapshot(),
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
        PENDING_ADDS, STATE, _snapshot
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
        "current": _snapshot(),
        "history_len": len(HISTORY),
        "pending_adds_len": len(PENDING_ADDS),
    }


def apply_changes(changes: List[dict], excerpt: Optional[str] = None) -> tuple[int, Dict[str, Any]]:
    """
    Apply a list of changes to the CAD model.
    
    This function wraps the _apply_changes_list function from run.py.
    Eventually this logic will be fully moved to state_service.
    
    Returns:
        Tuple of (status_code, response_dict)
    """
    from run import _apply_changes_list
    return _apply_changes_list(changes, excerpt)
