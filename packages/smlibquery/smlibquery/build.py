"""A minimal IR builder: evaluate expressions for one set of values and run the ops on SMLib.

(caddy has a fuller evaluator with caching, face provenance and measurement; this one keeps smlibquery usable on
its own.)"""
from __future__ import annotations

import usd_brep as u

from . import expr, kernel
from .errors import SMLQError


def build(graph: dict, values: dict[str, float] | None = None) -> dict[str, u.Brep]:
    """Return {body name: Brep}. A graph without "bodies" yields {"result": <last unconsumed solid>}."""
    env = dict(values or {})
    for p in graph.get("parameters", []):
        if "default" in p and p["name"] not in env:
            env[p["name"]] = p["default"]
    for p in graph.get("parameters", []):
        if "expr" in p and p["name"] not in env:
            env[p["name"]] = expr.evaluate(p["expr"], env)
    live: dict[str, u.Brep] = {}
    for b in graph["blocks"]:
        q = {k: expr.evaluate(e, env) for k, e in b.get("params", {}).items()}
        op = b["op"]
        try:
            if op in kernel.PRIMITIVE_OPS:
                brep = kernel.make_primitive(op, q, b.get("stock", ""))
            elif op in kernel.MOVE_OPS:
                brep = kernel.apply_move(live.pop(b["inputs"][0]), q)
            elif op in kernel.BOOLEANS:
                a, c = (live.pop(i) for i in b["inputs"])
                brep = kernel.BOOLEANS[op](a, c)
            else:
                raise SMLQError(f"unknown op {op!r}")
        except RuntimeError as e:
            raise SMLQError(f"block {b['id']} ({op}): kernel error: {e}") from None
        live[b["id"]] = brep
    if graph.get("bodies"):
        return {name: live[bid] for name, bid in graph["bodies"].items()}
    if len(live) != 1:
        raise SMLQError(f"graph leaves {len(live)} solids; name them in 'bodies'")
    return {"result": next(iter(live.values()))}


def export_usd(solid_or_bodies, path: str) -> None:
    """Exact B-rep to USD (NVIDIA's BrepArray schema)."""
    if isinstance(solid_or_bodies, dict):
        u.usd.export_breps([b.copy() for b in solid_or_bodies.values()], path)
    else:
        u.usd.export_brep(solid_or_bodies.copy(), path)
