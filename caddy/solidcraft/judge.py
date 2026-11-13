"""Image → CAD, the judging half: SolidCraft builds the parametric model, SMLib decides whether it stands.

SolidCraft exports STEP; smlibpy reads it into SMLib solids; they become ordinary caddy bodies (stock blocks), so the
smoke invariants (closed solid, positive volume, mesh vs exact volume), the verdict, "done" refused while a check
fails, the viewer and the drawing sheets all work unchanged. On top of that:

- export_intact: the volume SMLib computes for the solids it read back against the volume SolidCraft reports. A gap
  means the export lost or damaged geometry (an open body, a missing solid, wrong units).
- symmetric: points on the SMLib surface, mirrored across the declared plane, must land back on the surface
  (exact closest points). A hole, boss or leg on one side only shows up as a deviation in mm.

Facts only SolidCraft can know (which features fail to rebuild, which parameters move the geometry, how the outline
compares with the picture) are measured by caddy through SolidCraft's own commands, never claimed by the model.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import usd_brep as u

from caddy.kernel.measure import MEASURES

AXIS = {"YZ": 0, "XZ": 1, "XY": 2}
FACTS: dict[str, dict] = {}      # session id -> what caddy measured through SolidCraft for the current design


def import_step(step: Path, stock_dir: Path, tag: str) -> list[str]:
    """STEP → one SMLib .smb per solid in `stock_dir`; returns the stock names (smlibquery loads them by name)."""
    import smlibpy
    parts = smlibpy.import_step(str(step))
    parts = list(parts) if isinstance(parts, (list, tuple)) else [parts]
    names = []
    for i, p in enumerate(parts):
        brep = getattr(p, "_brep", p)
        name = f"sc_{tag}_{i}"
        brep.write_to_file(str(stock_dir / f"{name}.smb"), ascii=True)
        names.append(name)
    return names


def ident(name: str, taken: set[str]) -> str:
    """A SolidCraft body name ("Leg (2)") as a caddy identifier ("leg_2"), unique among `taken`."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "body"
    if not base[0].isalpha():
        base = "b_" + base
    out, k = base, 2
    while out in taken:
        out, k = f"{base}_{k}", k + 1
    taken.add(out)
    return out


def bbox_of(bodies: dict) -> tuple[np.ndarray, np.ndarray]:
    boxes = [b.bounding_box(tight=True) for b in bodies.values()]
    return (np.array([min(b[0][i] for b in boxes) for i in range(3)]),
            np.array([max(b[1][i] for b in boxes) for i in range(3)]))


def _surface_points(brep, chord: float, n: int, rng) -> np.ndarray:
    """Points on the solid's surface, spread by area (random points on the tessellation's triangles)."""
    pb = u.tessellate(brep, chord_height_tolerance=chord, surface_angle_tolerance_deg=8, curve_angle_tolerance_deg=8)
    a = pb.to_numpy_arrays()
    P, flat = a["points"].astype(float), a["faces"]
    tris, i = [], 0
    while i < len(flat):
        k = int(flat[i]); idx = flat[i + 1:i + 1 + k]
        tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, k - 1)]
        i += 1 + k
    T = np.asarray(tris)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    keep = area > 1e-12
    A, B, C, area = A[keep], B[keep], C[keep], area[keep]
    sel = rng.choice(len(A), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return A[sel] + r1[:, None] * (B[sel] - A[sel]) + r2[:, None] * (C[sel] - A[sel])


def symmetry_tolerance(bodies: dict) -> float:
    lo, hi = bbox_of(bodies)
    return round(max(0.05, 5e-4 * float((hi - lo).max())), 4)


def mirror_deviation(bodies: dict, plane: str, n: int = 1500) -> dict:
    """How far the model is from its own mirror image across `plane` ("YZ": left/right, "XZ", "XY"), in mm.

    The plane sits at the middle of the model's bounding box on that axis. Each sampled surface point is mirrored
    and SMLib returns its exact closest point on the solids; the worst distance is the deviation."""
    ax = AXIS[plane]
    lo, hi = bbox_of(bodies)
    size = float((hi - lo).max())
    mid = float(lo[ax] + hi[ax]) / 2
    chord = max(0.01, size * 2e-4)
    rng = np.random.default_rng(0)
    areas = {k: max(float(b.area()), 1e-9) for k, b in bodies.items()}
    total = sum(areas.values())
    boxes = {k: b.bounding_box(tight=True) for k, b in bodies.items()}
    worst = dict(dev=0.0, at=None, body=None)
    for name, b in bodies.items():
        P = _surface_points(b, chord, max(100, int(n * areas[name] / total)), rng)
        Q = P.copy()
        Q[:, ax] = 2 * mid - Q[:, ax]
        for p, q in zip(P, Q):
            best = math.inf
            for other, ob in bodies.items():
                (x0, y0, z0), (x1, y1, z1) = boxes[other]
                gap = max(x0 - q[0], q[0] - x1, y0 - q[1], q[1] - y1, z0 - q[2], q[2] - z1, 0.0)
                if gap >= best:
                    continue                       # this body's box is already farther than the best so far
                best = min(best, float(u.brep_closest_point(ob, tuple(float(v) for v in q))[1]))
            if best > worst["dev"]:
                worst = dict(dev=best, at=[round(float(v), 2) for v in p], body=name)
    # points sit within one chord of the true surface on each side of the mirror
    dev = max(0.0, worst["dev"] - 2 * chord)
    return dict(dev=round(dev, 4), at=worst["at"], body=worst["body"], plane=plane, mid=round(mid, 3))


def _symmetry(res, chk):
    key = (chk["plane"],)
    cache = getattr(res, "_mirror", None)
    if not cache or cache[0] != key:
        cache = (key, mirror_deviation(res.bodies, chk["plane"]))
        res._mirror = cache
    FACTS.setdefault(chk["image"], {})["mirror"] = cache[1]
    return cache[1]["dev"], []


def _volume_gap(res, chk):
    theirs = FACTS[chk["image"]].get("volume_mm3")
    if not theirs:
        return float("nan"), []
    ours = sum(float(u.compute_volume(b)) for b in res.bodies.values())
    FACTS[chk["image"]]["smlib_volume_mm3"] = ours
    return abs(ours - theirs) / theirs, []


def _fact(res, chk):
    v = FACTS[chk["image"]].get(chk["fact"])
    return (float("nan") if v is None else float(v)), []


MEASURES.update(image_symmetry=_symmetry, image_volume_gap=_volume_gap, image_fact=_fact)
