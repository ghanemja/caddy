"""Scan → CAD, the measuring half: from a triangle mesh to fitted primitives, every geometric query on SMLib.

Steps (each returns plain data the whiteboard draws):
1. scan     smlibpy Mesh: closed? volume, area (kernel mass properties)
2. axis     principal axes of inertia (kernel): a turned part has two equal moments; the odd one is its axis
3. slices   kernel cross-sections across the axis: per layer, outer loop corner/flat radius and 6-fold symmetry
            (a hex), inner loops (a bore)
4. profile  kernel section through the axis (a meridian cut): the part's half outline, straight runs merged
5. fit      the outline as lines → cylinders, cones, end faces; the hex run from the slices
6. draft    an smlibquery script for those primitives (Claude rewrites it with names and design intent)

Scope: parts turned about one axis, optionally with a hex (nipples, bushings, shafts, spacers, fittings). Arcs
(fillets, grooves) are fitted by the kernel (`Wire.to_arcs`) and reported, but smlibquery has no revolve yet, so
they are drafted as cones.
"""
from __future__ import annotations

import math

import numpy as np
import smlibpy as sm

TOL = 0.05          # mm: straight-run merge tolerance (well above scan noise, well below features)


def _v(p) -> list[float]:
    return [float(p.x), float(p.y), float(p.z)] if hasattr(p, "x") else [float(x) for x in p]


def scan_info(mesh: sm.Mesh) -> dict:
    mp = mesh.mass_properties()
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    return dict(triangles=int(len(mesh.triangles)), vertices=int(len(mesh.vertices)), closed=bool(mesh.is_closed()),
                volume=round(float(mp.volume), 2), area=round(float(mp.area), 2),
                size=[round(float(x), 2) for x in hi - lo])


def find_axis(mesh: sm.Mesh) -> dict:
    """The axis of a turned part from the kernel's inertia tensor: the principal axis whose moment differs from the
    other two (which are equal for anything with 3-fold or higher symmetry about it: round, hex, square-ish)."""
    mp = mesh.mass_properties()
    moments = np.array([float(x) for x in mp.principal_moments])
    axes = np.array([_v(a) for a in mp.principal_axes], dtype=float)
    pairs = [(abs(moments[(i + 1) % 3] - moments[(i + 2) % 3]) / max(moments), i) for i in range(3)]
    sym, i = min(pairs)                      # the two most equal moments → the third axis is the turning axis
    d = axes[i] / np.linalg.norm(axes[i])
    return dict(direction=d.round(6).tolist(), center=_v(mp.center), moments=moments.round(2).tolist(),
                symmetry=round(1 - sym, 4), turned=bool(sym < 0.02))


def to_axis_frame(mesh: sm.Mesh, ax: dict) -> tuple[sm.Mesh, np.ndarray]:
    """Mesh moved so the axis is +z from z = 0. Returns (mesh, 4x4 transform from scan to frame)."""
    d = np.array(ax["direction"]); c = np.array(ax["center"])
    e1 = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    R = np.array([e1, e2, d])
    V = (mesh.vertices - c) @ R.T
    V[:, 2] -= V[:, 2].min()
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ c
    return sm.Mesh(V, mesh.triangles.copy()), M


def _loops(wires) -> list[np.ndarray]:
    out = []
    for w in wires:
        pts = [_v(e.start) for e in w.edges()]
        if len(pts) >= 3:
            out.append(np.array(pts))
    return out


def slices(mesh: sm.Mesh, n: int = 60) -> dict:
    """Kernel cross-sections across the axis. Per layer: z, outer corner/flat radius, 6-fold strength and phase,
    bore radius."""
    zmax = float(mesh.vertices[:, 2].max())
    step = zmax / n
    layers = []
    for k, wires in enumerate(mesh.slices(step, (0, 0, 1))):
        loops = _loops(wires)
        if not loops:
            continue
        z = float(loops[0][:, 2].mean())
        rings = []
        for L in loops:
            r = np.hypot(L[:, 0], L[:, 1]); th = np.arctan2(L[:, 1], L[:, 0])
            # dense resample by angle so 6-fold strength doesn't depend on vertex spacing
            order = np.argsort(th); tt = np.linspace(-math.pi, math.pi, 360, endpoint=False)
            rr = np.interp(tt, th[order], r[order], period=2 * math.pi)
            h6 = np.mean(rr * np.exp(-6j * tt))
            rings.append(dict(r_mean=float(rr.mean()), r_min=float(rr.min()), r_max=float(rr.max()),
                              six=float(abs(h6) / rr.mean()), phase=float(-np.angle(h6) / 6)))   # a corner direction
        rings.sort(key=lambda q: -q["r_mean"])
        outer, inner = rings[0], rings[1:]
        layers.append(dict(z=round(z, 3), r_min=round(outer["r_min"], 3), r_max=round(outer["r_max"], 3),
                           six=round(outer["six"], 4), phase=outer["phase"],
                           bore=round(inner[0]["r_mean"], 3) if inner else None))
    hexes = [L for L in layers if L["six"] > 0.01]
    hx = None
    if hexes:
        af = 2 * float(np.median([L["r_min"] for L in hexes]))
        ph = float(np.angle(np.mean([np.exp(6j * L["phase"]) for L in hexes])) / 6)
        ph = (ph + math.pi / 6) % (math.pi / 3) - math.pi / 6
        hx = dict(z0=hexes[0]["z"], z1=hexes[-1]["z"], af=round(af, 3), phase_deg=round(math.degrees(ph), 2),
                  corner_r=round(af / 2 / math.cos(math.pi / 6), 3),
                  corner_min=round(min(L["r_max"] for L in hexes), 3))
    return dict(step=round(step, 4), layers=layers, hex=hx)


def _rdp(P: np.ndarray, tol: float) -> list[int]:
    """Indices kept by Ramer-Douglas-Peucker on an open polyline."""
    keep = [0, len(P) - 1]
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a; L = np.linalg.norm(ab) or 1e-12
        w = P[i + 1:j] - a
        d = np.abs(ab[0] * w[:, 1] - ab[1] * w[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep.append(m)
            stack += [(i, m), (m, j)]
    return sorted(set(keep))


def profile(mesh: sm.Mesh, phase_deg: float = 0.0) -> dict:
    """The half outline in a plane through the axis (kernel section), as (r, z), simplified to straight runs.
    The plane is turned to a hex flat's normal (phase + 30°) so the hex shows its across-flats size."""
    a = math.radians(phase_deg + 30.0)
    plane = sm.Plane((0, 0, 0), x_dir=(math.cos(a), math.sin(a), 0), z_dir=(-math.sin(a), math.cos(a), 0))
    loops = _loops(mesh.section(plane))
    u = np.array([math.cos(a), math.sin(a), 0.0])
    best = None
    for L in loops:            # the loop on the +u side of the axis (a bored part gives two, one each side)
        rz = np.c_[L @ u, L[:, 2]]
        if rz[:, 0].mean() > 0 and (best is None or len(rz) > len(best)):
            best = rz
    if best is None:
        raise ValueError("the section through the axis is empty")
    if best[:, 0].min() < -1e-6:   # solid part: one loop across the axis → keep the +r half, closed on the axis
        best = best[best[:, 0] >= -1e-6]
    raw = best
    # open the closed loop at its lowest-z, smallest-r point, then simplify
    s = int(np.lexsort((raw[:, 0], raw[:, 1]))[0])
    P = np.vstack([np.roll(raw, -s, axis=0), raw[s:s + 1]])
    far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))      # a closed loop: simplify each half
    keep = _rdp(P[:far + 1], TOL) + [far + k for k in _rdp(P[far:], TOL)]
    pts = P[sorted(set(keep))]
    return dict(raw=raw.round(4).tolist(), points=pts.round(4).tolist(), n_raw=len(raw), n_fit=len(pts) - 1)


def fit(prof: dict, sl: dict) -> dict:
    """Outline runs → primitives. Outward-facing runs (+r) are the outer surface, runs facing ±z are end faces and
    shoulders, inward-facing runs are the bore where the slices found a bore at that height. The hex comes from the
    slices (where the cross-section has 6-fold symmetry), its ends snapped to the measured end faces."""
    P = np.array(prof["points"])
    area = 0.5 * np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])
    ccw = area > 0
    layers = sl["layers"]
    def bore_at(z):
        L = min(layers, key=lambda q: abs(q["z"] - z))
        return L["bore"]
    outer, bore, ends = [], [], []
    for (r0, z0), (r1, z1) in zip(P[:-1], P[1:]):
        dr, dz = r1 - r0, z1 - z0
        L = math.hypot(dr, dz)
        if L < 1e-6 or max(r0, r1) < 1e-3:
            continue
        n = np.array([dz, -dr]) / L if ccw else np.array([-dz, dr]) / L     # outward normal in (r, z)
        if abs(n[1]) > 0.98:
            ends.append(dict(z=round((z0 + z1) / 2, 3), r0=round(min(r0, r1), 3), r1=round(max(r0, r1), 3)))
            continue
        seg = dict(z0=round(min(z0, z1), 3), z1=round(max(z0, z1), 3),
                   d0=round(2 * (r0 if z0 < z1 else r1), 3), d1=round(2 * (r1 if z0 < z1 else r0), 3))
        seg["kind"] = "cylinder" if abs(seg["d0"] - seg["d1"]) < 2 * TOL else "cone"
        if n[0] > 0:
            outer.append(seg)
        else:
            b = bore_at((seg["z0"] + seg["z1"]) / 2)
            (bore if b and abs((seg["d0"] + seg["d1"]) / 4 - b) < 0.5 else ends).append(seg)
    hx = sl.get("hex")
    if hx:
        zs = [e["z"] if "z" in e else (e["z0"] + e["z1"]) / 2 for e in ends] + [q[k] for q in outer for k in ("z0", "z1")]
        snap = lambda z: min(zs, key=lambda q: abs(q - z)) if zs and min(abs(q - z) for q in zs) <= 1.5 * sl["step"] else z
        z0, z1 = snap(hx["z0"]), snap(hx["z1"])
        outer = [q for q in outer if not (z0 - 1e-3 <= (q["z0"] + q["z1"]) / 2 <= z1 + 1e-3)]
        outer.append(dict(kind="hex", z0=round(z0, 3), z1=round(z1, 3), af=hx["af"], phase_deg=hx["phase_deg"],
                          chamfered=bool(hx["corner_min"] < hx["corner_r"] - 3 * TOL)))
    outer.sort(key=lambda q: q["z0"]); bore.sort(key=lambda q: q["z0"])
    for run in (outer, bore):      # neighbours share one boundary (no hairline gaps or overlaps between solids)
        for a, b in zip(run, run[1:]):
            if abs(b["z0"] - a["z1"]) < 0.2:
                m = round((a["z1"] + b["z0"]) / 2, 3)
                a["z1"] = b["z0"] = m
                if "d1" in a and "d0" in b and abs(a["d1"] - b["d0"]) < 4 * TOL:   # no real step: same diameter
                    d = round((a["d1"] + b["d0"]) / 2, 3)
                    a["d1"] = b["d0"] = d
                    for q in (a, b):
                        q["kind"] = "cylinder" if abs(q["d0"] - q["d1"]) < 2 * TOL else "cone"
    top = float(P[:, 1].max())
    for b in bore:                 # a bore that reaches an end goes through it (no film left across the hole)
        if b["z0"] < 0.2:
            b["z0"], b["through_bottom"] = -1.0, True
        if b["z1"] > top - 0.2:
            b["z1"], b["through_top"] = round(top + 1.0, 3), True
    for q in outer:                # the outer stack starts and ends exactly at the part's ends
        q["z0"] = 0.0 if q["z0"] < 0.2 else q["z0"]
        q["z1"] = round(top, 3) if q["z1"] > top - 0.2 else q["z1"]
    return dict(outer=outer, bore=bore, ends=sorted(ends, key=lambda q: q.get("z", q.get("z0", 0))),
                length=round(float(P[:, 1].max()), 3))


def draft(f: dict) -> str:
    """A first smlibquery script for the fitted primitives (numbers as measured; no names or intent yet)."""
    L = ["import smlibquery as sq", "import math", "", "# fitted from the scan (mm); z along the turning axis", ""]
    names = []
    for i, s in enumerate(f["outer"], 1):
        h = round(s["z1"] - s["z0"], 3)
        nm = f"s{i}"
        if s["kind"] == "hex":
            L.append(f'{nm} = (sq.Workplane("XY").workplane(offset={s["z0"]})'
                     f'.polygon(6, {s["af"]} / math.cos(math.radians(30))).extrude({h}).tag("{nm}.flat"))')
        elif s["kind"] == "cylinder":
            L.append(f'{nm} = (sq.Workplane("XY").workplane(offset={s["z0"]})'
                     f'.circle({round((s["d0"] + s["d1"]) / 4, 3)}).extrude({h}).tag("{nm}.side"))')
        else:
            L.append(f'{nm} = (sq.Workplane("XY").workplane(offset={s["z0"]}).circle({round(s["d0"] / 2, 3)})'
                     f'.workplane(offset={h}).circle({round(s["d1"] / 2, 3)}).loft().tag("{nm}.side"))')
        names.append(nm)
    body = names[0] + "".join(f".union({n})" for n in names[1:])
    L.append(f"body = {body}")
    if f["bore"]:
        for j, b in enumerate(f["bore"], 1):
            h = round(b["z1"] - b["z0"], 3)
            if b["kind"] == "cylinder":
                L.append(f'b{j} = (sq.Workplane("XY").workplane(offset={b["z0"]})'
                         f'.circle({round((b["d0"] + b["d1"]) / 4, 3)}).extrude({h}).tag("b{j}.wall"))')
            else:
                L.append(f'b{j} = (sq.Workplane("XY").workplane(offset={b["z0"]}).circle({round(b["d0"] / 2, 3)})'
                         f'.workplane(offset={h}).circle({round(b["d1"] / 2, 3)}).loft().tag("b{j}.wall"))')
        L.append("result = body" + "".join(f".cut(b{j})" for j in range(1, len(f["bore"]) + 1)))
    else:
        L.append("result = body")
    return "\n".join(L) + "\n"


def run(mesh: sm.Mesh) -> tuple[sm.Mesh, list[dict], dict, str]:
    """All measuring steps. Returns (mesh in the axis frame, steps for the board, fit, draft script)."""
    import time
    steps = []
    def step(key, title, data, t0):
        steps.append(dict(key=key, title=title, ms=round((time.time() - t0) * 1000), **data))
    t = time.time(); info = scan_info(mesh); step("scan", "Scan", dict(info=info), t)
    t = time.time(); ax = find_axis(mesh)
    framed, X = to_axis_frame(mesh, ax)
    step("axis", "Axis", dict(axis=ax), t)
    t = time.time(); sl = slices(framed)
    if sl["hex"]:          # turn the part so a hex corner sits at 0°, where polygon() puts one (as in CadQuery)
        framed = framed.rotate(sm.Axis.Z, -sl["hex"]["phase_deg"])
        sl = slices(framed)
    step("slices", "Slices", dict(slices=sl), t)
    t = time.time(); pr = profile(framed, 0.0); step("profile", "Profile", dict(profile=pr), t)
    t = time.time(); f = fit(pr, sl); step("fit", "Primitives", dict(fit=f), t)
    code = draft(f)
    step("draft", "Draft CAD", dict(code=code), time.time())
    return framed, steps, f, code
