"""IR operations on NVIDIA SMLib (pip `usd-brep`): one function per op. Geometry decisions live here.

Primitives sit on the z axis unless given cx/cy; sizes are diameters; mm and degrees."""
from __future__ import annotations

import math
import os
from pathlib import Path

import usd_brep as u

PRIMITIVE_OPS = {"cone", "cylinder", "slab", "hex_prism", "double_cone", "box", "stock", "prism", "skin", "wing"}
MOVE_OPS = {"move"}
BOOLEANS = {"union": u.boolean_union, "intersect": u.boolean_intersection, "difference": u.boolean_difference}
PARAM_KEYS = {
    "cone": ["z", "d_base", "d_top", "h"],             # optional cx, cy
    "cylinder": ["z", "d", "h"],                       # optional cx, cy
    "slab": ["z", "width", "h"],
    "hex_prism": ["z", "af", "h"],
    "double_cone": ["z", "d", "h", "angle_deg"],
    "box": ["cx", "cy", "cz", "dx", "dy", "dz"],       # optional rot_z
    "prism": ["z", "n", "d", "h"],                     # optional cx, cy, rot_z; d = circumscribed diameter
    "stock": [],                                       # block field "stock": a part in a stock directory
    "skin": ["n"],
    "wing": ["naca", "root_chord", "tip_chord", "span", "x", "y", "z"],   # optional sweep, dihedral, incidence, twist, side                                     # n sections: z{i}, w{i} half-width, b{i} bottom, t{i} top, k{i} taper; e exponent
    "move": [],                                        # optional rot_z | angle, ax, ay, az, px, py, pz; dx, dy, dz
}
MOVE_KEYS = ("rot_z", "angle", "ax", "ay", "az", "px", "py", "pz", "dx", "dy", "dz")

# Where stock parts (.smb, SMLib's native format) are looked up; add your own directories.
STOCK_PATHS: list[Path] = [Path(p) for p in os.environ.get("SMLQ_STOCK_PATH", "").split(os.pathsep) if p]
_stock_cache: dict[str, u.Brep] = {}


def load_stock(name: str) -> u.Brep:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"bad stock name {name!r}")
    if name not in _stock_cache:
        for d in STOCK_PATHS:
            if (d / f"{name}.smb").exists():
                _stock_cache[name] = u.read_brep_from_file(str(d / f"{name}.smb"))
                break
        else:
            have = sorted({p.stem for d in STOCK_PATHS for p in d.glob("*.smb")})
            raise ValueError(f"no stock part {name!r}; available: {have}")
    return _stock_cache[name].copy()


def make_primitive(op: str, q: dict, stock: str = "") -> u.Brep:
    cx, cy = q.get("cx", 0.0), q.get("cy", 0.0)
    if op == "cone":
        return u.create_cone((cx, cy, q["z"]), q["d_base"] / 2, q["d_top"] / 2, q["h"])
    if op == "cylinder":
        return u.create_cylinder((cx, cy, q["z"]), q["d"] / 2, q["h"])
    if op == "prism":
        poly = u.create_regular_polygon((cx, cy, q["z"]), int(round(q["n"])), q["d"] / 2)
        b = u.linear_sweep(poly, (0, 0, 1), q["h"])
        if q.get("rot_z", 0):
            u.rotate(b, (cx, cy, q["z"]), (0, 0, 1), q["rot_z"])
        return b
    if op == "slab":
        return u.create_box((-60, -q["width"] / 2, q["z"]), 120, q["width"], q["h"])
    if op == "hex_prism":
        rc = q["af"] / 2 / math.cos(math.pi / 6)
        return u.linear_sweep(u.create_regular_polygon((0, 0, q["z"]), 6, rc), (0, 0, 1), q["h"])
    if op == "double_cone":
        h, r = q["h"] / 2, q["d"] / 2
        grow = h * math.tan(math.radians(90 - q["angle_deg"]))
        lo = u.create_cone((0, 0, q["z"]), r, r + grow, h)
        hi = u.create_cone((0, 0, q["z"] + h), r + grow, r, h)
        return u.boolean_union(lo, hi)
    if op == "box":   # centred box, turned rot_z degrees about its own vertical axis
        b = u.create_box((q["cx"] - q["dx"] / 2, q["cy"] - q["dy"] / 2, q["cz"] - q["dz"] / 2), q["dx"], q["dy"], q["dz"])
        if q.get("rot_z", 0):
            u.rotate(b, (q["cx"], q["cy"], q["cz"]), (0, 0, 1), q["rot_z"])
        return b
    if op == "stock":
        return load_stock(stock)
    if op == "skin":
        return make_skin(q)
    if op == "wing":
        return make_wing(q)
    raise ValueError(f"unknown primitive op {op!r}")


SKIN_POINTS = 32


def skin_section(z, hw, yb, yt, k, e):
    """One closed cross-section at station z (on the z axis): a superellipse of half-width hw between heights yb and
    yt, narrowed towards the top by k (0: a plain superellipse, 0.4: a glasshouse 40% narrower at the roof).
    Points start at the widest point on +x and run the same way round in every section, so they skin cleanly."""
    yc, hh, pts = (yb + yt) / 2, (yt - yb) / 2, []
    for j in range(SKIN_POINTS):
        a = 2 * math.pi * j / SKIN_POINTS
        c, s = math.cos(a), math.sin(a)
        v = math.copysign(abs(s) ** (2 / e), s)
        x = math.copysign(abs(c) ** (2 / e), c) * hw * (1 - k * max(0.0, v) ** 1.5)
        pts.append((x, yc + hh * v, z))
    pts.append(pts[0])
    return pts


def make_skin(q: dict) -> u.Brep:
    """A smooth body skinned (lofted, cubic) through n cross-sections along z, with flat end caps."""
    n, e = int(round(q["n"])), q.get("e", 3.0)
    if n < 3:
        raise ValueError("skin needs at least 3 sections")
    zs = [q[f"z{i}"] for i in range(n)]
    if any(b <= a for a, b in zip(zs, zs[1:])):
        raise ValueError("skin sections must go strictly forward along z")
    curves = []
    for i in range(n):
        hw, yb, yt = q[f"w{i}"], q[f"b{i}"], q[f"t{i}"]
        if hw <= 0 or yt <= yb:
            raise ValueError(f"skin section {i}: needs half-width > 0 and top above bottom")
        curves.append(u.create_curve_interp_points(skin_section(zs[i], hw, yb, yt, q.get(f"k{i}", 0.0), e)))
    return u.create_skin_primitive([1] * n, curves, degree=3, cap_ends=True)


def naca4(code: int, n: int = 30):
    """NACA 4-digit section (e.g. 2412: 2% camber at 40% chord, 12% thick), unit chord, cosine-spaced.
    Returns (upper, lower) as lists of (x, y) from the leading edge to the trailing edge."""
    code = f"{int(round(code)):04d}"
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    if not 0.01 <= t <= 0.4:
        raise ValueError(f"NACA {code}: thickness must be 1-40% of the chord")
    up, lo = [], []
    for i in range(n + 1):
        x = (1 - math.cos(math.pi * i / n)) / 2
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        if m and p and x < p:
            yc, dy = m / p ** 2 * (2 * p * x - x * x), 2 * m / p ** 2 * (p - x)
        elif m and p:
            yc, dy = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x), 2 * m / (1 - p) ** 2 * (p - x)
        else:
            yc, dy = 0.0, 0.0
        th = math.atan(dy)
        up.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lo.append((x + yt * math.sin(th), yc - yt * math.cos(th)))
    return up, lo


def wing_section(code, chord, le, inc_deg):
    """Closed airfoil section in the plane x = le[0]: chord along +z from the leading edge le, thickness along +y,
    pitched nose-up by inc_deg about the leading edge. Three curves: upper surface, lower surface, trailing edge."""
    up, lo = naca4(code)
    a = math.radians(inc_deg)

    def P(c, y):
        zz, yy = c * chord, y * chord
        return (le[0], le[1] + yy * math.cos(a) + zz * math.sin(a), le[2] + zz * math.cos(a) - yy * math.sin(a))
    U = [P(c, y) for c, y in reversed(up)]
    L = [P(c, y) for c, y in lo]
    return [u.create_curve_interp_points(U), u.create_curve_interp_points(L), u.create_line_segment(L[-1], U[0])]


def wing_planform(q: dict) -> dict:
    """Root and tip leading edges, chords and incidences of one wing panel (also what the aero model reads)."""
    side = -1.0 if q.get("side", 1) < 0 else 1.0
    b, sw, di = q["span"], math.radians(q.get("sweep", 0.0)), math.radians(q.get("dihedral", 0.0))
    root = (q["x"], q["y"], q["z"])
    tip = (q["x"] + side * b, q["y"] + b * math.tan(di), q["z"] + b * math.tan(sw))
    inc = q.get("incidence", 0.0)
    return dict(root=root, tip=tip, root_chord=q["root_chord"], tip_chord=q["tip_chord"], root_inc=inc,
                tip_inc=inc + q.get("twist", 0.0), naca=int(round(q["naca"])), side=side)


def make_wing(q: dict) -> u.Brep:
    """A straight-tapered wing panel: two NACA sections (root at x, tip one span out along ±x), ruled between them.
    The car/aircraft length runs along +z (flow from z = 0 backwards), y is up."""
    w = wing_planform(q)
    if q["span"] <= 0 or w["root_chord"] <= 0 or w["tip_chord"] <= 0:
        raise ValueError("wing needs span, root_chord and tip_chord > 0")
    c = wing_section(w["naca"], w["root_chord"], w["root"], w["root_inc"]) + \
        wing_section(w["naca"], w["tip_chord"], w["tip"], w["tip_inc"])
    return u.create_skin_primitive([3, 3], c, degree=1, cap_ends=True)


def apply_move(brep: u.Brep, q: dict) -> u.Brep:
    """Rigid move: rot_z about the z axis through the origin, or `angle` about axis (ax, ay, az) through
    (px, py, pz); then translate (dx, dy, dz). Mutates and returns `brep`."""
    if q.get("rot_z", 0):
        u.rotate(brep, (0, 0, 0), (0, 0, 1), q["rot_z"])
    if q.get("angle", 0):
        axis = (q.get("ax", 0.0), q.get("ay", 0.0), q.get("az", 0.0))
        if not any(axis):
            axis = (0.0, 0.0, 1.0)
        u.rotate(brep, (q.get("px", 0.0), q.get("py", 0.0), q.get("pz", 0.0)), axis, q["angle"])
    if q.get("dx", 0) or q.get("dy", 0) or q.get("dz", 0):
        u.translate(brep, (q.get("dx", 0), q.get("dy", 0), q.get("dz", 0)))
    return brep
