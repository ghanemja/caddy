"""
State Management Service
Handles application state, history, and parameters.
"""
import os
import sys
from typing import Dict, Any, Optional, List

# Import from legacy optim.py for now (will be migrated)
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


def get_state(which: str = "all") -> Dict[str, Any]:
    """Get current application state."""
    from optim import STATE, CURRENT_PARAMS, CONTEXT, HISTORY, H_PTR
    if which == "all":
        return {
            "state": STATE,
            "params": CURRENT_PARAMS,
            "context": CONTEXT,
            "history": HISTORY,
            "history_ptr": H_PTR,
        }
    elif which == "params":
        return CURRENT_PARAMS
    elif which == "context":
        return CONTEXT
    elif which == "history":
        return {"history": HISTORY, "ptr": H_PTR}
    return {"state": STATE}


def reset_state():
    """Reset state to initial values."""
    from optim import (
        CURRENT_PARAMS, CONTEXT, HISTORY, H_PTR, INIT_SNAPSHOT,
        _restore, _ensure_initial_history
    )
    if INIT_SNAPSHOT:
        _restore(INIT_SNAPSHOT)
    else:
        _ensure_initial_history()
    return {"ok": True}


def apply_changes(changes: List[dict], excerpt: Optional[str] = None) -> tuple[int, Dict[str, Any]]:
    """
    Apply a list of changes to the CAD model.
    
    Returns:
        Tuple of (status_code, response_dict)
    """
    from optim import _apply_changes_list
    return _apply_changes_list(changes, excerpt)

