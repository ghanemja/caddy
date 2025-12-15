"""
JSON parsing utilities
Functions for parsing JSON from VLM output and other text sources
"""

import re
import json
import ast
from typing import Optional, List, Tuple, Dict, Any


def extract_json_loose(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text, trying multiple parsing strategies."""
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


def find_all_balanced_json_blocks(text: str) -> List[Tuple[int, int]]:
    """
    Find all balanced JSON object/array blocks in text.
    Returns list of (start_index, end_index) tuples.
    """
    blocks = []
    i = 0
    while i < len(text):
        if text[i] in ("{", "["):
            start = i
            depth = 0
            in_string = False
            escape = False
            opener = text[i]
            closer = "}" if opener == "{" else "]"
            i += 1
            while i < len(text):
                c = text[i]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if c == opener:
                        depth += 1
                    elif c == closer:
                        depth -= 1
                        if depth == 0:
                            blocks.append((start, i))
                            break
                i += 1
        i += 1
    return blocks


def split_multi_json_and_summaries(
    raw_text: str,
) -> Tuple[Optional[List[Dict[str, Any]]], List[str], List[str]]:
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
    for s, e in find_all_balanced_json_blocks(parse_zone):
        blocks.append(parse_zone[s : e + 1])

    parsed_changes = []
    kept_blocks = []

    for block in blocks:
        # Normalize units inside JSON-like text (e.g. "2.5m" -> "2500")
        from app.utils.units import repair_units_in_json_text

        fixed_units = repair_units_in_json_text(block)

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
