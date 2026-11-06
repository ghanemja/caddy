"""The CAD accessibility tree: what the model reads instead of renders.

Compact, deterministic text (golden-master friendly): per part, its blocks and named faces with exact sizes from the
kernel, then the checks that fail with the face and block to look at. Aliases (`hex.flat[3]`, `bore.wall`) are a
semantic layer on top of the face index; `why(alias)` and `measure` answer follow-up questions.
"""
from __future__ import annotations

import math
from collections import OrderedDict

from .graph import Result
from .measure import _radius_at


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _group(alias: str) -> str:
    return alias.split("[")[0]


def _range(idx: list[int]) -> str:
    return f"[0..{len(idx) - 1}]" if len(idx) > 1 else ""


def describe_group(res: Result, name: str, labels: list) -> str:
    bfs = res.face_objs
    kinds = {l.kind for l in labels}
    kind = "/".join(sorted(kinds))
    zs = [bfs[l.index].bounding_box(tight=True) for l in labels]
    z0, z1 = min(b[0][2] for b in zs), max(b[1][2] for b in zs)
    bits = [f"{name}{_range(labels)}", kind if len(labels) == 1 else f"{len(labels)}x {kind}"]
    if kinds == {"cone"} and len(labels) == 1:
        f = bfs[labels[0].index]
        bits.append(f"Ø{_fmt(2 * _radius_at(f, z0))}→Ø{_fmt(2 * _radius_at(f, z1))}")
    elif kinds == {"cylinder"} and len(labels) == 1:
        p = bfs[labels[0].index].interior_point()
        bits.append(f"Ø{_fmt(2 * math.hypot(p[0], p[1]))}")
    if abs(z1 - z0) < 1e-6:
        bits.append(f"z={_fmt(z0)}")
    else:
        bits.append(f"z {_fmt(z0)}–{_fmt(z1)}")
    small = [l for l in labels if l.area < 1.0]
    if small and len(small) != len(labels):
        bits.append(f"{len(small)} sliver(s) <1mm²")
    return " ".join(bits)


def digest(res: Result, checks: list[dict] | None = None, smoke: list[dict] | None = None, size: str = "") -> str:
    g = res.graph
    lines = [f"family {g['family']}" + (f" · size {size}" if size else "")
             + f" · closed {'✓' if all(b.is_manifold_solid() for b in res.bodies.values()) else '✗'}"
             + f" · volume {sum(b.volume() for b in res.bodies.values()):.1f} mm³ · {len(res.faces)} faces"
             + (f" · bodies {', '.join(res.bodies)}" if len(res.bodies) > 1 else "")]
    parts: OrderedDict[str, OrderedDict[str, list]] = OrderedDict()
    for f in sorted(res.faces, key=lambda f: [b["id"] for b in g["blocks"]].index(f.block)):
        parts.setdefault(f.part, OrderedDict()).setdefault(_group(f.alias), []).append(f)
    for part, groups in parts.items():
        blocks = [b["id"] for b in g["blocks"] if b["part"] == part]
        params = [p["name"] for p in g["parameters"] if p["part"] == part]
        lines.append(f"part {part}  blocks: {', '.join(blocks)}  params: {', '.join(params)}")
        for name, labels in groups.items():
            lines.append("  " + describe_group(res, name, labels))
    if checks is not None:
        ok = [c for c in checks if c["ok"]]
        lines.append(f"checks {len(ok)}/{len(checks)} ✓  "
                     + " · ".join(f"{c['id']} {c['measured']}" for c in ok))
        for c in checks:
            if not c["ok"]:
                what = c["rule"] or f"{c['source']} {c['column'] or c['id']}"
                lines.append(f"  ✗ {c['id']}: measured {c['measured']} {c['op']} expected {c['expected']}"
                             f" (Δ {c['delta']}) on {', '.join(c['faces']) or '?'} · part {c['part']} · {what}")
    if smoke is not None:
        bad = [s for s in smoke if not s["ok"]]
        lines.append(f"smoke {len(smoke) - len(bad)}/{len(smoke)} ✓" + "".join(f"\n  ✗ {s['id']}" for s in bad))
    return "\n".join(lines)


def why(res: Result, alias: str) -> str:
    """Which block, op, part and parameters made this face."""
    labels = res.by_alias(alias) or res.by_alias(_group(alias))
    if not labels:
        return f"no face named {alias!r}"
    f = labels[0]
    b = next(b for b in res.graph["blocks"] if b["id"] == f.block)
    params = ", ".join(f"{k}={v}" for k, v in b.get("params", {}).items())
    return f"{alias}: {f.kind} made by block {b['id']} ({b['op']}, part {b['part']}, role {b['role']}); params {params}"
