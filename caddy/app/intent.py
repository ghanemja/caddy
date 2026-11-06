"""Design intent for the board: the feature tree and the constraints, both read off the kernel's results.

- Feature tree: parts → features (the smlibquery calls that make geometry), each with its parameter expressions,
  where every number comes from (catalogue column, ISO 7-1, a design parameter) and the script line. Boolean and
  move bookkeeping is kept but folded.
- Constraints: driving dimensions (catalogue/parameter → features), catalogue checks, rules and fits, each with a
  live status across every configuration. Engineers add constraints in words: Claude proposes one from a fixed
  library of kernel measurements, caddy previews it on every configuration, the engineer approves it.
"""
from __future__ import annotations

import copy
import json
from collections import OrderedDict

from caddy.kernel import graph as G
from caddy.kernel.measure import MEASURES, run_checks

BOOKKEEPING = {"union", "difference", "intersect", "move"}
FIT = {"gap", "overlap"}
ISO_PREFIX = ("in_", "out_")


def _sources(params: dict, rows: list[dict], design: set[str]) -> list[dict]:
    cols = {k for k, v in (rows[0] if rows else {}).items() if isinstance(v, float)}
    out, seen = [], set()
    for e in params.values():
        for n in sorted(G.names_in(e)):
            if n in seen:
                continue
            seen.add(n)
            kind = "iso" if n.startswith(ISO_PREFIX) else "col" if n in cols else "design" if n in design else "other"
            out.append(dict(name=n, kind=kind))
    return out


def feature_tree(s, blocks: list[dict]) -> list[dict]:
    """[{part, ok, features: [...], bookkeeping: [...]}] in build order."""
    if not s.graph:
        return []
    design = {p["name"] for p in s.graph.get("parameters", [])}
    parts: OrderedDict[str, dict] = OrderedDict()
    for b in blocks:
        p = parts.setdefault(b["part"], dict(part=b["part"], features=[], bookkeeping=[], failing=set()))
        f = dict(id=b["id"], op=b["op"], role=b["role"], params=b.get("params", {}), line=b.get("line"),
                 inputs=b.get("inputs", []), failing=b["failing"], optional=b.get("optional", False),
                 sources=_sources(b.get("params", {}), s.rows, design))
        (p["bookkeeping"] if b["op"] in BOOKKEEPING else p["features"]).append(f)
        p["failing"] |= set(b["failing"])
    for r in s.reports:                      # a part also fails if a check measured on it fails
        for c in r.checks:
            if not c["ok"] and c["part"] in parts:
                parts[c["part"]]["failing"].add(r.size)
    return [dict(p, failing=sorted(p["failing"])) for p in parts.values()]


def _status(s, check_id: str) -> dict:
    applies, ok, failing, measured = 0, 0, [], []
    for r in s.reports:
        c = next((c for c in r.checks if c["id"] == check_id), None)
        if c is None:
            continue
        applies += 1
        ok += c["ok"]
        if not c["ok"]:
            failing.append(r.size)
        measured.append(dict(size=r.size, measured=c["measured"], expected=c["expected"], ok=c["ok"]))
    return dict(applies=applies, ok=ok, failing=failing, measured=measured)


def _features_for(s, chk: dict) -> list[str]:
    """Blocks whose faces a check measures (by face name) or whose body it relates (fits)."""
    res = next((r.result for r in s.reports if r.result is not None), None)
    if res is None:
        return []
    if chk.get("measure") in FIT:
        bodies = {chk.get("a"), chk.get("b")}
        return sorted({f.block for f in res.faces if f.body in bodies or f.part in bodies})
    t = chk.get("target")
    return sorted({f.block for f in res.by_alias(t)}) if t else []


def constraints(s) -> list[dict]:
    if not s.graph:
        return []
    out = []
    used: OrderedDict[str, list[str]] = OrderedDict()
    for b in s.graph["blocks"]:
        for e in b.get("params", {}).values():
            for n in sorted(G.names_in(e)):
                used.setdefault(n, [])
                if b["id"] not in used[n]:
                    used[n].append(b["id"])
    design = {p["name"]: p for p in s.graph.get("parameters", [])}
    by_column = {c.get("column"): c for c in s.spec["checks"] if c.get("column")}
    for c in s.spec["checks"]:              # an ISO check whose expected value is exactly a name drives it too
        if not c.get("column") and isinstance(c.get("expect"), str) and c["expect"] in used and c.get("measure") not in FIT:
            by_column.setdefault(c["expect"], c)
    row0 = s.rows[0] if s.rows else {}
    for n, feats in used.items():
        kind = "iso" if n.startswith(ISO_PREFIX) else "design" if n in design else "col"
        if kind == "col" and n not in row0:
            continue
        c = by_column.get(n)
        st = _status(s, c["id"]) if c else None
        out.append(dict(kind="driving", id=f"drive:{n}", name=n, source=kind, features=feats,
                        value=design[n].get("default") if n in design else row0.get(n),
                        status=st, check=c["id"] if c else None))
    for c in s.spec["checks"]:
        kind = "catalogue" if c.get("column") else "fit" if c.get("measure") in FIT else "rule"
        if any(o["kind"] == "driving" and o["check"] == c["id"] for o in out):
            continue                         # shown with its driving dimension
        out.append(dict(kind=kind, id=c["id"], text=c.get("rule") or describe(c), by=c.get("by", "spec"),
                        status=_status(s, c["id"]), features=_features_for(s, c), check=c["id"], spec=c))
    return out


def describe(c: dict) -> str:
    rhs = c.get("column") or c.get("expect")
    op = {"==": "=", "<=": "≤", ">=": "≥"}.get(c.get("op", "=="), "=")
    m, t = c["measure"], c.get("target")
    what = {"overall_length": "overall length", "across_flats": f"across flats of {t}",
            "axial_length": f"length of {t}", "diameter": f"Ø of {t}", "max_diameter": f"largest Ø of {t}",
            "diameter_at": f"Ø of {t} at z = {c.get('z')}", "gap": f"gap {c.get('a')} ↔ {c.get('b')}",
            "overlap": f"overlap {c.get('a')} ↔ {c.get('b')}"}.get(m, m)
    src = f"{c['source']}: " if c.get("source") and c.get("source") not in ("rule", "catalogue") else ""
    return f"{src}{what} {op} {rhs}"


# --- constraints from words ------------------------------------------------------------------------------------

MEASURE_DOCS = {
    "overall_length": "overall length of the part along z (no target)",
    "across_flats": "distance between opposite flat faces of a named face group (target), e.g. a hex",
    "axial_length": "length along z covered by a named face group (target)",
    "diameter": "diameter of a cylindrical named face (target)",
    "diameter_at": "diameter of a conical/cylindrical named face (target) at height z (field z: expression)",
    "max_diameter": "largest diameter of a named face group (target)",
    "gap": "smallest distance between two bodies (a, b); 0 when touching or overlapping",
    "overlap": "volume shared by two bodies (a, b); 0 when they only touch or are apart",
}

PROPOSE_TOOL = {
    "name": "propose_constraint",
    "description": "Propose ONE kernel-checkable constraint that captures the engineer's sentence.",
    "input_schema": {"type": "object", "additionalProperties": False, "properties": {
        "id": {"type": "string", "description": "short snake_case id"},
        "rule": {"type": "string", "description": "the constraint in plain words, as the engineer would say it"},
        "measure": {"type": "string", "enum": sorted(MEASURE_DOCS)},
        "target": {"type": "string", "description": "a face name from the list (for face measures)"},
        "a": {"type": "string", "description": "body name (fits)"},
        "b": {"type": "string", "description": "body name (fits)"},
        "z": {"type": "string", "description": "height expression (diameter_at only)"},
        "op": {"type": "string", "enum": ["==", "<=", ">="]},
        "expect": {"type": "string", "description": "expression over catalogue names and numbers, e.g. 'hex_af - 1'"},
        "tol": {"type": "number"},
        "when": {"type": "string", "description": "optional expression; the constraint applies where it is non-zero"},
        "cannot": {"type": "string", "description": "if the sentence can't be checked with these measures, say why"}},
        "required": ["id", "rule", "measure", "op", "expect"]},
}


def propose(s, text: str, client, model: str = "claude-opus-5-5") -> dict:
    res = next((r.result for r in s.reports if r.result is not None), None)
    if res is None:
        raise G.GraphError("build the model first")
    names = sorted({f.alias.split("[")[0] for f in res.faces})
    bodies = sorted(res.bodies) if len(res.bodies) > 1 else []
    row = {k: v for k, v in res.row_env.items()}
    prompt = (f"An engineer wants this constraint on a parametric CAD family: \"{text}\"\n\n"
              f"Turn it into ONE constraint the geometry kernel can measure on every configuration, by calling "
              f"propose_constraint. Use only these measurements:\n"
              + "\n".join(f"- {k}: {v}" for k, v in MEASURE_DOCS.items())
              + f"\n\nFace names: {', '.join(names)}"
              + (f"\nBodies: {', '.join(bodies)}" if bodies else "")
              + "\nCatalogue names usable in expressions (first configuration's values): "
              + ", ".join(f"{k}={v:g}" for k, v in row.items())
              + "\nDon't invent face names or measurements. If it truly can't be measured, fill 'cannot'.")
    resp = client.messages.create(model=model, max_tokens=4000, tools=[PROPOSE_TOOL],
                                  messages=[{"role": "user", "content": prompt}], output_config={"effort": "low"})
    use = next((b for b in resp.content if b.type == "tool_use"), None)
    if use is None:
        txt = " ".join(b.text for b in resp.content if b.type == "text")
        raise G.GraphError(f"no constraint proposed: {txt[:300]}")
    chk = {k: v for k, v in dict(use.input).items() if v not in (None, "")}
    if chk.get("cannot"):
        return dict(check=chk, preview=None, cannot=chk["cannot"])
    chk["id"] = "you_" + "".join(ch for ch in chk["id"].lower() if ch.isalnum() or ch == "_")[:40]
    chk["by"] = "you"
    if chk.get("target") and chk["target"] not in names:
        raise G.GraphError(f"proposal used an unknown face name {chk['target']!r}")
    return dict(check=chk, preview=preview(s, chk), usage=resp.usage.model_dump(exclude_none=True))


def preview(s, chk: dict) -> dict:
    """Run a proposed constraint on every configuration before it's approved."""
    out, ok, applies = [], 0, 0
    for r in s.reports:
        if r.result is None:
            continue
        got = run_checks(r.result, r.row, [chk])
        if not got:
            continue
        c = got[0]
        applies += 1
        ok += c["ok"]
        out.append(dict(size=r.size, ok=c["ok"], measured=c["measured"], expected=c["expected"], error=c["error"]))
    return dict(applies=applies, ok=ok, rows=out)


def approve(s, chk: dict) -> None:
    if chk.get("measure") not in MEASURES:
        raise G.GraphError(f"unknown measurement {chk.get('measure')!r}")
    if any(c["id"] == chk["id"] for c in s.spec["checks"]):
        raise G.GraphError(f"a constraint named {chk['id']} already exists")
    s.spec = copy.deepcopy(s.spec)
    s.spec["checks"].append(dict(chk, by="you"))
    if s.graph is not None:
        s._run(s.graph)                      # cached blocks: only the checks re-run


def remove(s, check_id: str) -> None:
    s.spec = copy.deepcopy(s.spec)
    before = len(s.spec["checks"])
    s.spec["checks"] = [c for c in s.spec["checks"] if not (c["id"] == check_id and c.get("by") == "you")]
    if len(s.spec["checks"]) == before:
        raise G.GraphError("only constraints you added can be removed here")
    if s.graph is not None:
        s._run(s.graph)


def dumps(x) -> str:
    return json.dumps(x, default=str)
