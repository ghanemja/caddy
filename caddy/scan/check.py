"""Scan → CAD, the judging half: how far the CAD is from the scan, measured on SMLib.

- scan → CAD: every sampled scan point to its closest point on the exact solid (NVIDIA usd_brep
  `brep_closest_point`), attributed to the CAD face it lands on (`find_face`): "hex.flat is 0.31 mm inside the scan".
- CAD → scan: points on the CAD surface, ray-cast along the surface normal into the scan mesh (smlibpy's kernel ray
  tracer): catches material the CAD has and the scan doesn't.

Both register as ordinary caddy measures, so the spec, the verdict, "done" refused while it fails, the digest and
the board all work unchanged.
"""
from __future__ import annotations

import numpy as np
import usd_brep as u

from caddy.kernel.measure import MEASURES

SCANS: dict[str, "Scan"] = {}


class Scan:
    """A scan in the axis frame: the mesh (smlibpy) and the points the checks sample."""

    def __init__(self, sid: str, mesh, n_sample: int = 3000):
        self.id, self.mesh = sid, mesh
        V = np.asarray(mesh.vertices, dtype=float)
        T = np.asarray(mesh.triangles)
        # sample by area so big flat triangles count as much as fine curved ones (vertices alone undersample flats)
        A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
        rng = np.random.default_rng(0)
        tri = rng.choice(len(T), size=n_sample, p=A / A.sum())
        a, b = rng.random(n_sample), rng.random(n_sample)
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        P = V[T[tri, 0]] + a[:, None] * (V[T[tri, 1]] - V[T[tri, 0]]) + b[:, None] * (V[T[tri, 2]] - V[T[tri, 0]])
        self.samples = np.vstack([P, V]) if len(V) <= 4000 else P
        SCANS[sid] = self


def _deviation(res, scan: Scan) -> dict:
    cache = getattr(res, "_scan_dev", None)
    if cache and cache["id"] == scan.id:
        return cache
    brep = res.brep
    index = {f: i for i, f in enumerate(res.face_objs)}
    d = np.empty(len(scan.samples)); face = np.empty(len(scan.samples), dtype=int); out = np.empty(len(scan.samples))
    for k, p in enumerate(scan.samples):
        c, dist = u.brep_closest_point(brep, tuple(p))
        cp = (c.x, c.y, c.z)
        d[k] = dist
        f = u.find_face(brep, cp)
        face[k] = index.get(f, -1)
        # outside the CAD (scan has material the CAD lacks)? compare distance from the axis, or along z at the ends
        rp, rc = np.hypot(p[0], p[1]), np.hypot(cp[0], cp[1])
        out[k] = 1.0 if (rp - rc) > abs(p[2] - cp[2]) * 0.5 else (-1.0 if (rc - rp) > abs(p[2] - cp[2]) * 0.5 else 0.0)
    by_face: dict[str, dict] = {}
    for i in range(len(scan.samples)):
        if face[i] < 0:
            continue
        a = res.faces[face[i]].alias
        g = by_face.setdefault(a, dict(face=a, max=0.0, n=0, z=[1e9, -1e9], outside=0, inside=0))
        g["n"] += 1
        g["max"] = max(g["max"], float(d[i]))
        if d[i] > 0.02:
            z = float(scan.samples[i][2]); g["z"] = [min(g["z"][0], z), max(g["z"][1], z)]
            g["outside" if out[i] > 0 else "inside"] += out[i] != 0
    cache = dict(id=scan.id, d=d, face=face, max=float(d.max()), p95=float(np.percentile(d, 95)),
                 mean=float(d.mean()), faces=sorted(by_face.values(), key=lambda g: -g["max"]))
    res._scan_dev = cache
    return cache


def _extra(res, scan: Scan) -> dict:
    cache = getattr(res, "_scan_extra", None)
    if cache and cache["id"] == scan.id:
        return cache
    pb = u.tessellate(res.brep, chord_height_tolerance=0.05, surface_angle_tolerance_deg=10, curve_angle_tolerance_deg=10)
    a = pb.to_numpy_arrays()
    P = a["points"].astype(float)
    flat, tris, i = a["faces"], [], 0
    while i < len(flat):
        n = int(flat[i]); idx = flat[i + 1:i + 1 + n]
        tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, n - 1)]
        i += 1 + n
    T = np.asarray(tris)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-9
    T, fn, area = T[keep], fn[keep], area[keep]
    N = fn / area[:, None]                       # each triangle's own normal: no averaging across sharp edges
    P = P[T].mean(axis=1)                        # triangle centres lie inside one face
    sel = np.random.default_rng(0).choice(len(P), 2500, replace=True, p=area / area.sum())   # by area
    P, N = P[sel], N[sel]
    R = 5.0                                     # cast from just outside the CAD surface back through it
    hits = scan.mesh.raycast_many(P + N * R, -N)
    d = np.array([abs(h.distance - R) if h is not None else R for h in hits])
    cache = dict(id=scan.id, max=float(d.max()), p95=float(np.percentile(d, 95)), points=P, d=d)
    res._scan_extra = cache
    return cache


def scan_deviation(res, chk):
    dev = _deviation(res, SCANS[chk["scan"]])
    tol = chk.get("tol_face", 0.1)
    return dev["max"], [g["face"] for g in dev["faces"] if g["max"] > tol] or [g["face"] for g in dev["faces"][:1]]


def scan_extra(res, chk):
    ex = _extra(res, SCANS[chk["scan"]])
    return ex["max"], []


MEASURES.update(scan_deviation=scan_deviation, scan_extra=scan_extra)


def describe(res, scan: Scan, tol: float) -> str:
    """For the model: which CAD faces miss the scan, by how much, where, and which way."""
    dev = _deviation(res, scan)
    lines = [f"scan → CAD: max {dev['max']:.3f} mm, 95% within {dev['p95']:.3f} mm, mean {dev['mean']:.3f} mm "
             f"({len(scan.samples)} scan points, exact closest points on the solid)"]
    for g in dev["faces"]:
        if g["max"] <= tol:
            continue
        way = ("the scan is OUTSIDE this face (CAD is missing material here)" if g["outside"] > g["inside"]
               else "the scan is INSIDE this face (CAD has material the part doesn't)" if g["inside"] > g["outside"]
               else "")
        lines.append(f"  {g['face']}: off by up to {g['max']:.3f} mm for z {g['z'][0]:.2f}..{g['z'][1]:.2f} {way}")
    ex = _extra(res, scan)
    lines.append(f"CAD → scan: max {ex['max']:.3f} mm (CAD surface points ray-cast into the scan)")
    return "\n".join(lines)
