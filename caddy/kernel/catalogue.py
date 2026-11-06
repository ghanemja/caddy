"""Catalogues: one CSV per family (named in the family's spec), plus ISO 7-1 thread data for pipe fittings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
FAMILIES = DATA / "families"


def iso7() -> dict[str, dict]:
    with open(DATA / "iso7_1.csv") as f:
        return {r["size"]: {k: (v if k == "size" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)}


TEXT_COLUMNS = {"inlet", "outlet", "code", "pose", "size", "variant"}   # sizes like "1" (one inch) stay names, not numbers


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def spec(family: str = "bspt_nipple") -> dict:
    return json.loads((FAMILIES / f"{family}.checks.json").read_text())


def rows(path: Path | None = None, family: str = "bspt_nipple") -> list[dict]:
    path = path or DATA / spec(family).get("catalogue", "bspt_nipple.csv")
    with open(path) as f:
        return [{k: (v if k in TEXT_COLUMNS else _num(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def values_for(row: dict) -> dict[str, float]:
    """Row variables a graph can use: every numeric catalogue column; for pipe fittings also ISO 7-1 thread data for
    each end, prefixed in_ (inlet, the larger end) and out_ (outlet)."""
    v = {k: x for k, x in row.items() if isinstance(x, float)}
    if "inlet" in row and "outlet" in row:
        t = iso7()
        for prefix, size in (("in_", row["inlet"]), ("out_", row["outlet"])):
            iso = t[size]
            v.update({prefix + "major_d": iso["major_d_gauge_mm"], prefix + "gauge_len": iso["gauge_len_mm"],
                      prefix + "pitch": iso["pitch_mm"], prefix + "tpi": iso["tpi"],
                      prefix + "useful_thread_min": iso["useful_thread_min_mm"]})
    return v


def size_name(row: dict, fmt: str | None = None) -> str:
    if fmt:
        return fmt.format(**{k: (f"{v:g}" if isinstance(v, float) else v) for k, v in row.items()})
    return f'{row["inlet"]} x {row["outlet"]}'
