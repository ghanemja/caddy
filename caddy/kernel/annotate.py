"""Where a check was measured, as a dimension the page can draw: two points on the exact solid (model coordinates).

Each measurement in `measure.py` returns a number; this returns the line that number spans, taken from the same
faces with the same kernel queries, so the drawing and the verdict can't disagree about what was measured.
"""
from __future__ import annotations

import math

import numpy as np
import usd_brep as u

from .graph import evaluate_expr
from .measure import _faces, _plane_normal, _radius_at


def _pt(p) -> list[float]:
    return [round(float(x), 4) for x in p]


def _extent(res):
    (x0, y0, z0), (x1, y1, z1) = res.brep.bounding_box(tight=True)
    return max(abs(x0), abs(x1), abs(y0), abs(y1)), z0, z1


def across_flats(res, chk):
    labels, faces = _faces(res, chk["target"])
    data = []
    for l, f in zip(labels, faces):
        if l.area <= 1.0:
            continue
        n = _plane_normal(f)
        n[2] = 0.0
        n /= np.linalg.norm(n)
        c = np.array(l.centroid, dtype=float)
        data.append((l, n, float(np.dot(n, c)), c))
    best = None
    for i, (l, n, d, c) in enumerate(data):
        for l2, n2, d2, c2 in data[i + 1:]:
            if np.dot(n, n2) < -0.999 and (best is None or abs(d) + abs(d2) < best[0]):
                z = c[2]
                best = (abs(d) + abs(d2), [*(n * d)[:2], z], [*(n2 * d2)[:2], z], [l.alias, l2.alias])
    if best is None:
        return None
    return dict(a=_pt(best[1]), b=_pt(best[2]), faces=best[3])


def _across(r: float, z: float, ang: float = 0.0):
    c, s = math.cos(ang), math.sin(ang)
    return dict(a=_pt([r * c, r * s, z]), b=_pt([-r * c, -r * s, z]))


def diameter(res, chk):
    labels, faces = _faces(res, chk["target"])
    p = faces[0].interior_point()
    return dict(_across(math.hypot(p[0], p[1]), p[2], math.atan2(p[1], p[0])), faces=[labels[0].alias])


def diameter_at(res, chk):
    labels, faces = _faces(res, chk["target"])
    z = evaluate_expr(chk["z"], res.row_env)
    for l, f in zip(labels, faces):
        (_, _, z0), (_, _, z1) = f.bounding_box(tight=True)
        if z0 - 1e-6 <= z <= z1 + 1e-6:
            return dict(_across(_radius_at(f, z), z), faces=[l.alias])
    return None


def max_diameter(res, chk):
    labels, faces = _faces(res, chk["target"])
    best = None
    for l, f in zip(labels, faces):
        (x0, y0, z0), (x1, y1, z1) = f.bounding_box(tight=True)
        span = max(x1 - x0, y1 - y0)
        if best is None or span > best[0]:
            best = (span, (z0 + z1) / 2, l.alias)
    return dict(_across(best[0] / 2, best[1]), faces=[best[2]]) if best else None


def axial_length(res, chk):
    labels, faces = _faces(res, chk["target"])
    boxes = [f.bounding_box(tight=True) for f in faces]
    z0, z1 = min(b[0][2] for b in boxes), max(b[1][2] for b in boxes)
    r = max(max(abs(b[0][0]), abs(b[1][0]), abs(b[0][1]), abs(b[1][1])) for b in boxes)
    off = _extent(res)[0] * 1.25 + 2       # drawn beside the part, like a drawing's dimension line
    return dict(a=_pt([0, -off, z0]), b=_pt([0, -off, z1]), ext=[_pt([0, -r, z0]), _pt([0, -r, z1])],
                faces=[l.alias for l in labels])


def overall_length(res, chk):
    r, z0, z1 = _extent(res)
    off = r * 1.6 + 4
    return dict(a=_pt([0, off, z0]), b=_pt([0, off, z1]), ext=[_pt([0, r, z0]), _pt([0, r, z1])], faces=[])


def gap(res, chk):
    """Closest pair between two bodies, by alternating closest-point queries on the exact surfaces."""
    a, b = res.bodies[chk["a"]], res.bodies[chk["b"]]
    (lo, hi) = b.bounding_box(tight=True)
    q = tuple((np.array(lo) + np.array(hi)) / 2)
    p = q
    for _ in range(12):
        p, _ = u.brep_closest_point(a, tuple(q))
        q2, _ = u.brep_closest_point(b, tuple(p))
        if np.linalg.norm(np.array(q2) - np.array(q)) < 1e-6:
            q = q2
            break
        q = q2
    return dict(a=_pt(p), b=_pt(q), faces=[chk["a"], chk["b"]])


def overlap(res, chk):
    return dict(faces=[chk["a"], chk["b"]])


DIMENSIONS = dict(across_flats=across_flats, diameter=diameter, diameter_at=diameter_at, max_diameter=max_diameter,
                  axial_length=axial_length, overall_length=overall_length, gap=gap, overlap=overlap)


def dimensions(res, checks: list[dict]) -> dict[str, dict]:
    """{check id: {a, b, ext?, faces}} for every check whose measurement can be drawn on this result."""
    out = {}
    for chk in checks:
        if "when" in chk and not evaluate_expr(chk["when"], res.row_env):
            continue
        fn = DIMENSIONS.get(chk.get("measure"))
        try:
            d = fn(res, chk) if fn else None
        except Exception:      # a check that can't be measured has nothing to draw; the verdict says why
            d = None
        if d:
            out[chk["id"]] = d
    return out
