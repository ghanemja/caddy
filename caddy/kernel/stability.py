"""Static stability of one body, computed exactly from its SMLib solid (no simulation).

The body sits in its authored pose on the lowest surface points (its footprint). It tips over its nearest support
edge when the tilt θ reaches tan θ = d / h, where d is the distance from the centre of mass (projected on the floor)
to that edge and h is the height of the centre of mass above the floor. On a slope it slides when the slope α
reaches tan α = μ. Whichever happens first decides what the object does.

The footprint is the convex hull of the tessellated points within `rest_tol` of the lowest z, so a curved base
(a cone, a fillet) counts as its real contact outline, not just its corners.

Caveats (also in docs/VERIFY.md): the tilt is about the nearest edge of a convex footprint, which is exact for a
rigid body resting on a flat floor; a non-convex footprint is reduced to its hull (slightly optimistic); the
authored pose is taken as the resting pose; SMLib's mass properties use uniform density (only the centroid matters).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import usd_brep as u


def hull2d(pts: np.ndarray) -> list[int]:
    """Indices of the convex hull of 2-D points, counter-clockwise (Andrew's monotone chain)."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    def cross(o, a, b):
        return (pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1]) - (pts[a][1] - pts[o][1]) * (pts[b][0] - pts[o][0])
    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 1e-12:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 1e-12:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def polygon_area(pts: np.ndarray, idx: list[int]) -> float:
    x, y = pts[idx][:, 0], pts[idx][:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass
class Stability:
    com: tuple[float, float, float]        # centre of mass (model units)
    height: float                          # centre of mass above the floor
    tip_deg: float                         # tilt at which the body tips over its nearest support edge
    edge: tuple[float, float]              # the support edge's midpoint (x, y): where it would pivot
    distance: float                        # centre of mass to that edge, on the floor
    footprint_area: float
    inside: bool                           # centre of mass projects inside the footprint
    top: float = 0.0                       # the body's highest point above the floor (for drawing it)
    mu: float | None = None
    slip_deg: float | None = None          # tilt at which it slides on a surface with friction mu

    @property
    def first(self) -> str | None:
        if self.slip_deg is None:
            return "tip"
        return "slide" if self.slip_deg < self.tip_deg else "tip"

    def as_dict(self) -> dict:
        return dict(com=[round(x, 3) for x in self.com], height=round(self.height, 3), tip_deg=round(self.tip_deg, 1),
                    edge=[round(x, 3) for x in self.edge], distance=round(self.distance, 3),
                    footprint_area=round(self.footprint_area, 3), inside=self.inside, top=round(self.top, 3), mu=self.mu,
                    slip_deg=None if self.slip_deg is None else round(self.slip_deg, 1), first=self.first)


def static_stability(brep, mu: float | None = None, rest_tol: float = 0.02, chord: float | None = None) -> Stability:
    """Stability of `brep` resting on its lowest surface, optionally on a surface with friction `mu`."""
    lo, hi = brep.bounding_box(tight=True)
    size = max(h - l for l, h in zip(lo, hi))
    chord = chord or max(0.01, size * 2e-3)
    pts = u.tessellate(brep, chord_height_tolerance=chord, surface_angle_tolerance_deg=8.0,
                       curve_angle_tolerance_deg=8.0).to_numpy_arrays()["points"]
    pts = np.asarray(pts, float)
    z0 = float(pts[:, 2].min())
    base = pts[pts[:, 2] <= z0 + rest_tol * size][:, :2]
    if len(base) < 3:
        raise ValueError("no footprint: the body does not rest on a surface (its lowest points are a line or a point)")
    ring = hull2d(base)
    if len(ring) < 3:
        raise ValueError("no footprint: the lowest points are collinear")
    poly = base[ring]
    com = brep.mass_properties(1.0)["centroid"]
    cx, cy, cz = (float(c) for c in com)
    h = cz - z0
    if h <= 1e-9:
        raise ValueError("centre of mass is on the floor: the body cannot tip")
    # counter-clockwise ring: a point is inside when it is left of every edge; its distance to an edge's line is
    # the cross product over the edge length
    nxt = np.roll(poly, -1, axis=0)
    ex, ey = nxt[:, 0] - poly[:, 0], nxt[:, 1] - poly[:, 1]
    L = np.hypot(ex, ey)
    cross = ex * (cy - poly[:, 1]) - ey * (cx - poly[:, 0])   # >0 when the point is left of the edge
    inside = bool(np.all(cross >= -1e-9))
    dists = np.abs(cross) / np.maximum(L, 1e-12)
    k = int(np.argmin(dists))
    d = float(dists[k])
    mid = (float((poly[k][0] + nxt[k][0]) / 2), float((poly[k][1] + nxt[k][1]) / 2))   # the pivot edge's midpoint
    if not inside:
        tip = 0.0                                     # already over the edge: it falls at rest
    else:
        tip = math.degrees(math.atan2(d, h))
    slip = None if mu is None else math.degrees(math.atan(max(mu, 0.0)))
    return Stability(com=(cx, cy, cz), height=h, tip_deg=tip, edge=mid, distance=d,
                     footprint_area=polygon_area(base, ring), inside=inside, top=float(pts[:, 2].max()) - z0,
                     mu=mu, slip_deg=slip)
