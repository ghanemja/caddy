"""The spec oracle: every catalogue cell and rule measured on named faces of the solid. No AI here.

Measurements assume the family's axis is z (true for every family in scope).
"""
from __future__ import annotations

import math

import numpy as np
import usd_brep as u

from .graph import Result, evaluate_expr


def _faces(res: Result, target: str):
    labels = res.by_alias(target)
    return labels, [res.face_objs[f.index] for f in labels]


def _plane_normal(face) -> np.ndarray:
    pts = np.array([v.point() for v in face.vertices()], dtype=float)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    return vt[-1] / np.linalg.norm(vt[-1])


def _radius_at(face, z: float) -> float:
    """Radius of a surface of revolution about z at height z (linear between two samples: exact for a cone)."""
    def sample(zz):
        p, _ = face.closest_point((0.0, 0.0, zz))
        return p[2], math.hypot(p[0], p[1])
    (za, ra), (zb, rb) = sample(z - 1.0), sample(z + 1.0)
    if abs(zb - za) < 1e-12:
        return ra
    return ra + (rb - ra) * (z - za) / (zb - za)


def overall_length(res, chk):
    (_, _, z0), (_, _, z1) = res.brep.bounding_box(tight=True)
    return z1 - z0, []


def across_flats(res, chk):
    labels, faces = _faces(res, chk["target"])
    big = [(l, f) for l, f in zip(labels, faces) if l.area > 1.0]   # ignore slivers
    data = []
    for l, f in big:
        n = _plane_normal(f)
        n[2] = 0.0
        n /= np.linalg.norm(n)
        data.append((l, n, abs(float(np.dot(n, np.array(l.centroid))))))
    spans = []
    for i, (l, n, d) in enumerate(data):
        for l2, n2, d2 in data[i + 1:]:
            if np.dot(n, n2) < -0.999:
                spans.append(d + d2)
    if not spans:
        return float("nan"), [l.alias for l, _ in big]
    return min(spans), [l.alias for l, _ in big]


def axial_length(res, chk):
    labels, faces = _faces(res, chk["target"])
    zs = [f.bounding_box(tight=True) for f in faces]
    return max(b[1][2] for b in zs) - min(b[0][2] for b in zs), [l.alias for l in labels]


def diameter(res, chk):
    labels, faces = _faces(res, chk["target"])
    p = faces[0].interior_point()
    return 2 * math.hypot(p[0], p[1]), [labels[0].alias]


def diameter_at(res, chk):
    labels, faces = _faces(res, chk["target"])
    z = evaluate_expr(chk["z"], res.row_env)
    for l, f in zip(labels, faces):   # the face whose axial range contains z
        (_, _, z0), (_, _, z1) = f.bounding_box(tight=True)
        if z0 - 1e-6 <= z <= z1 + 1e-6:
            return 2 * _radius_at(f, z), [l.alias]
    return float("nan"), [l.alias for l in labels]


def max_diameter(res, chk):
    labels, faces = _faces(res, chk["target"])
    spans = []
    for f in faces:
        (x0, y0, _), (x1, y1, _) = f.bounding_box(tight=True)
        spans.append(max(x1 - x0, y1 - y0))
    return max(spans), [l.alias for l in labels]


# --- assemblies: fit between two bodies ---------------------------------------------------------------------

def _pair(res, chk):
    try:
        return res.bodies[chk["a"]], res.bodies[chk["b"]]
    except KeyError as e:
        raise KeyError(f"no body {e.args[0]!r}; bodies: {sorted(res.bodies)}") from None


def gap(res, chk):
    """Smallest distance between two bodies (0 when they touch or overlap)."""
    a, b = _pair(res, chk)
    d = float(u.brep_distance(a, b))
    if d < 1e-6 and _skinned(res, chk["a"], chk["b"]):
        # second opinion: on skinned (lofted B-spline) faces the kernel's distance can read a clear gap as touching
        try:
            x = u.boolean_intersection(a.copy(), b.copy())
            shared = float(x.volume()) if x.face_count() else 0.0
        except RuntimeError:
            shared = 0.0
        if shared <= 1e-6:                       # no shared material: measure the gap on the surfaces themselves
            d = min(_sampled_gap(a, b), _sampled_gap(b, a))
            d = 0.0 if d < 0.5 else d
    return d, [chk["a"], chk["b"]]


def _skinned(res, *bodies) -> bool:
    ops = {b["id"]: b["op"] for b in res.graph.get("blocks", [])}
    return any(f.body in bodies and ops.get(f.block) == "skin" for f in res.faces)


def _sampled_gap(a, b) -> float:
    """Smallest distance from the tessellation vertices of `a` to the exact (trimmed) surface of `b`."""
    lo, hi = a.bounding_box(tight=True)
    chord = max(0.05, max(h - l for l, h in zip(lo, hi)) * 2e-4)     # fine enough for a clearance, not a render
    pts = u.tessellate(a, chord_height_tolerance=chord, surface_angle_tolerance_deg=8.0,
                       curve_angle_tolerance_deg=8.0).to_numpy_arrays()["points"]
    return min(float(u.brep_closest_point(b, tuple(float(x) for x in p))[1]) for p in pts)


def overlap(res, chk):
    """Volume shared by two bodies, from a boolean intersection: the second opinion on 'touching', since the
    relationship classifier can read a shallow overlap as touching (NVIDIA queries guide)."""
    a, b = _pair(res, chk)
    rel = u.brep_relationship(a, b)["relationship"]
    if rel == "separate":
        return 0.0, [chk["a"], chk["b"]]
    try:
        x = u.boolean_intersection(a.copy(), b.copy())
        return (float(x.volume()) if x.face_count() else 0.0), [chk["a"], chk["b"]]
    except RuntimeError:
        return 0.0, [chk["a"], chk["b"]]


_AX = {"x": 0, "y": 1, "z": 2}


def _box(res, chk):
    brep = res.bodies.get(chk["a"]) if chk.get("a") else res.brep
    if brep is None:
        raise ValueError(f"no body {chk.get('a')!r}")
    return brep.bounding_box(tight=True), _AX[chk.get("axis", "z")]


def extent(res, chk):
    """Size of a body (default: the first) along an axis, from its tight bounding box."""
    (lo, hi), i = _box(res, chk)
    return hi[i] - lo[i], []


def min_coord(res, chk):
    """Lowest coordinate of a body along an axis (e.g. ground clearance: the body's lowest y)."""
    (lo, _), i = _box(res, chk)
    return lo[i], []


def max_coord(res, chk):
    (_, hi), i = _box(res, chk)
    return hi[i], []


def surface_share(res, chk):
    """Share of a body's surface area made by the given ops (e.g. ["skin"]). The envelope checks pass for a stack of
    boxes too, so a family that wants a skinned shell or NACA wings measures how much of the body they make."""
    ops = {b["id"]: b["op"] for b in res.graph.get("blocks", [])}
    want = set(chk["ops"])
    faces = [f for f in res.faces if not chk.get("a") or f.body == chk["a"]]
    total = sum(f.area for f in faces)
    if not total:
        raise ValueError(f"no faces on body {chk.get('a')!r}")
    return sum(f.area for f in faces if ops.get(f.block) in want) / total, []


MEASURES = dict(extent=extent, min_coord=min_coord, max_coord=max_coord, overall_length=overall_length, across_flats=across_flats, axial_length=axial_length,
                diameter=diameter, diameter_at=diameter_at, max_diameter=max_diameter, gap=gap, overlap=overlap,
                surface_share=surface_share)


def run_checks(res: Result, row: dict | None = None, checks: list[dict] | None = None) -> list[dict]:
    """Each check: expected value from the catalogue row (column) or an expression (expect), measured on the solid.
    Expressions in checks see the row variables, never the model's graph parameters."""
    out = []
    env = res.row_env
    for chk in (checks if checks is not None else res.graph.get("checks", [])):
        if "when" in chk and not evaluate_expr(chk["when"], env):
            continue          # e.g. a fit check that applies to the closed pose only
        if "column" in chk:
            if row is None or chk["column"] not in row:
                continue
            expected, source = float(row[chk["column"]]), "catalogue"
        else:
            expected, source = evaluate_expr(chk["expect"], env), chk.get("source", "rule")
        try:
            measured, faces = MEASURES[chk["measure"]](res, chk)
        except Exception as e:  # a measurement that can't run is a failure, with the reason
            measured, faces, err = float("nan"), [], f"{type(e).__name__}: {e}"
        else:
            err = None
        tol = chk.get("tol", 0.01)
        op = chk.get("op", "==")
        if math.isnan(measured):
            ok = False
        elif op == "<=":
            ok = measured <= expected + tol
        elif op == ">=":
            ok = measured >= expected - tol
        else:
            ok = abs(measured - expected) <= tol
        parts = (sorted({f.part for f in res.faces if f.alias in faces})
                 or ([chk["a"]] if "a" in chk else [chk.get("target", "?").split(".")[0]]))
        out.append(dict(id=chk["id"], source=source, column=chk.get("column"), rule=chk.get("rule"), op=op,
                        expected=round(expected, 4), measured=round(measured, 4) if not math.isnan(measured) else None,
                        delta=None if math.isnan(measured) else round(measured - expected, 4), tol=tol, ok=ok,
                        faces=faces, part=parts[0], error=err))
    return out


def required_aliases(res: Result, names: list[str]) -> list[dict]:
    """The spec needs these named faces; a graph that doesn't name them can't be measured."""
    have = {f.alias.split("[")[0] for f in res.faces}
    return [dict(id=f"named_face:{n}", ok=n in have) for n in names]


def smoke(res: Result) -> list[dict]:
    """Smoke tests and runtime invariants (VERIFY.md), per body."""
    out = []
    bodies = res.bodies or {"part": res.brep}
    for name, b in bodies.items():
        sfx = "" if len(bodies) == 1 else f":{name}"
        exact = u.compute_volume(b)
        lo, hi = b.bounding_box(tight=True)
        chord = max(0.01, max(h_ - l_ for l_, h_ in zip(lo, hi)) * 2e-4)   # 0.01 mm on a fitting, ~1 mm on a car
        pb = u.tessellate(b, chord_height_tolerance=chord, surface_angle_tolerance_deg=8, curve_angle_tolerance_deg=8)
        mesh = u.poly_brep_volume_via_api(pb)
        vgap = abs(exact - mesh) / exact if exact else 1.0
        out += [
            dict(id="closed_solid" + sfx, ok=bool(b.is_manifold_solid()), part=name),
            dict(id="volume_positive" + sfx, ok=exact > 0, measured=round(exact, 2), part=name),
            dict(id="mesh_vs_exact_volume" + sfx, ok=vgap < 0.005, measured=round(vgap, 6), tol=0.005, part=name),
        ]
    out.append(dict(id="every_face_labelled", ok=all(f.block and f.alias for f in res.faces), measured=len(res.faces)))
    for s in res.steps:
        out.append(dict(id=f"feature_exists:{s['block']}", ok=s["feature_exists"] or s.get("optional", False),
                        part=s["part"], warn=not s["feature_exists"]))
    return out
