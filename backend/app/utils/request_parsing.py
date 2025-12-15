"""
Request parsing utilities
Functions for parsing Flask request data into change objects
"""

from flask import request
from typing import Tuple, Optional, List, Dict, Any
from app.utils.param_normalization import (
    normalize_change,
    coerce_changes,
    intent_to_changes,
)
from app.utils.json_parsing import split_multi_json_and_summaries


def parse_apply_request() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Robustly extracts a list[dict] of change objects from the request.
    Accepts:
      - JSON body with {changes: [...]} or {actions: [...]}
      - Raw text containing JSON
      - Plain natural language (fallback to intent_to_changes)

    Returns:
      Tuple of (changes_list, excerpt_string)
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
        items = coerce_changes(payload)
        changes = [c for c in (normalize_change(x) for x in items) if c]
        if not changes:
            # fallback from text
            text_src = (
                data.get("prompt") or data.get("text") or data.get("message") or ""
            )
            changes = intent_to_changes(text_src)
        return changes, excerpt

    # --- list form ---
    if isinstance(data, list):
        return [c for c in (normalize_change(x) for x in data) if c], None

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
        parsed_list, _, _ = split_multi_json_and_summaries(raw_text)
        changes = [c for c in (normalize_change(x) for x in (parsed_list or [])) if c]
        if not changes:
            changes = intent_to_changes(raw_text)
        return changes, excerpt

    # --- raw text body ---
    raw_body = request.get_data(as_text=True) or ""
    parsed_list, _, _ = split_multi_json_and_summaries(raw_body)
    changes = [c for c in (normalize_change(x) for x in (parsed_list or [])) if c]
    if not changes:
        changes = intent_to_changes(raw_body)
    return changes, None
