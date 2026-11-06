"""Conceptual aerodynamics for any model with wing() features: lift and induced drag from a vortex-lattice model of
the wing planforms, plus friction and form drag from the kernel's exact wetted areas and the model's frontal area.

Axes as everywhere in caddy's car/aircraft models: flow along +z (from the nose backwards), y up, x across (span).
This is a first-order estimate for comparing designs, not CFD:
  - wings: Weissinger-type vortex lattice (horseshoe vortices on each panel's planform; incidence, twist and the
    NACA section's zero-lift angle in the boundary condition), linear up to a stated CLmax, no stall model;
  - lift carried by the body itself and by the wing inside the body is ignored (exposed panels only);
  - parasite drag: turbulent flat-plate friction (Schlichting) x form factor (Raymer) x interference, per component;
  - exposed wheels: a bluff-body drag coefficient on their frontal area outside the body;
  - holes such as lift-fan ducts are counted in wetted area but their flow is not modelled.
Every geometric input (planforms, wetted areas, frontal areas, lengths) comes from the built model.
"""
from __future__ import annotations

import math

import numpy as np
import usd_brep as u

from smlibquery.kernel import naca4, wing_planform

RHO, NU = 1.225, 1.46e-5          # sea level, ISA: kg/m^3, m^2/s
CD_WHEEL = 0.25                   # exposed wheel, on its frontal area outside the body
INTERFERENCE = 1.1                # wing-body
CD_BASE = 0.13                    # blunt base (Hoerner), on the base area
CL_MAX = 1.4                      # clean NACA 4-digit wing, typical; the polar is marked beyond it


def zero_lift_angle(code: int) -> float:
    """Thin-aerofoil zero-lift angle (degrees) of a NACA 4-digit camber line."""
    up, lo = naca4(code, 200)
    xs = np.array([(a[0] + b[0]) / 2 for a, b in zip(up, lo)])
    yc = np.array([(a[1] + b[1]) / 2 for a, b in zip(up, lo)])
    th = np.arccos(np.clip(1 - 2 * xs, -1, 1))
    dy = np.gradient(yc, xs, edge_order=1)
    ok = np.isfinite(dy)
    f = dy[ok] * (np.cos(th[ok]) - 1)
    return math.degrees(-np.trapezoid(f, th[ok]) / math.pi)


def _panels(wings: list[dict], n_span: int = 48, n_chord: int = 4):
    """Lattice: bound vortex at each box's quarter chord, control point at its three-quarter chord (metres)."""
    A, B, C, dx, alpha0, inc, owner, area = [], [], [], [], [], [], [], []
    for k, w in enumerate(wings):
        r, t = np.array(w["root"]) / 1000, np.array(w["tip"]) / 1000
        cr, ct = w["root_chord"] / 1000, w["tip_chord"] / 1000
        a0 = zero_lift_angle(w["naca"])
        for i in range(n_span):        # cosine spacing towards the tip, where the loading changes fastest
            s0, s1 = math.sin(math.pi / 2 * i / n_span), math.sin(math.pi / 2 * (i + 1) / n_span)
            sm = (s0 + s1) / 2
            le0, le1 = r + (t - r) * s0, r + (t - r) * s1
            c0, c1 = cr + (ct - cr) * s0, cr + (ct - cr) * s1
            for j in range(n_chord):
                f0, f1 = j / n_chord, (j + 1) / n_chord
                q = lambda le, c, f: le + np.array([0.0, 0.0, c * f])           # point at chord fraction f
                fb, fc = f0 + 0.25 * (f1 - f0), f0 + 0.75 * (f1 - f0)
                a_, b_ = q(le0, c0, fb), q(le1, c1, fb)
                if a_[0] > b_[0]:
                    a_, b_ = b_, a_          # every bound vortex runs in +x, so circulations share one sign
                A.append(a_); B.append(b_)
                C.append((q(le0, c0, fc) + q(le1, c1, fc)) / 2)
                dx.append(abs(le1[0] - le0[0]))
                inc.append(w["root_inc"] + (w["tip_inc"] - w["root_inc"]) * sm)
                alpha0.append(a0); owner.append(k)
                area.append(abs(le1[0] - le0[0]) * (c0 + c1) / 2 * (f1 - f0))
    return (np.array(A), np.array(B), np.array(C), np.array(dx), np.radians(alpha0), np.radians(inc),
            np.array(owner), np.array(area))


def _segment(P, A, B):
    """Induced velocity at points P (n,3) of unit vortex segments A->B (m,3): (n,m,3). Biot-Savart, cored."""
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = (B - A)[None, :, :]
    cr = np.cross(r1, r2)
    cr2 = np.sum(cr * cr, axis=2)
    n1, n2 = np.linalg.norm(r1, axis=2), np.linalg.norm(r2, axis=2)
    core = 1e-10
    k = (np.sum(r0 * r1, axis=2) / np.maximum(n1, core) - np.sum(r0 * r2, axis=2) / np.maximum(n2, core)) / (4 * np.pi * np.maximum(cr2, core))
    k[cr2 < 1e-12] = 0.0
    return cr * k[:, :, None]


def _horseshoe(P, A, B, far=1e4, trailing_only=False):
    Af, Bf = A + np.array([0, 0, far]), B + np.array([0, 0, far])
    v = _segment(P, Af, A) + _segment(P, B, Bf)          # trailing legs: from far downstream to A, B to far
    if not trailing_only:
        v = v + _segment(P, A, B)
    return v


def vlm_polar(wings: list[dict], alphas_deg) -> dict:
    """CL and induced-drag coefficient over angle of attack, on the exposed wing area."""
    A, B, C, dx, a0, inc, owner, area = _panels(wings)
    S = float(area.sum())
    Ainf = _horseshoe(C, A, B)[:, :, 1]                  # normal (y) wash at control points per unit circulation
    # Trefftz plane: each spanwise strip sheds its chordwise-summed circulation from its two edges far downstream
    nc = 4
    strips = len(A) // nc
    Ea, Eb = A[::nc, :2], B[::nc, :2]                    # strip edges in the (x, y) plane
    Pm = (Ea + Eb) / 2
    def wash(E, sign):                                   # y-velocity at strip centres from 2D vortices at E
        r = Pm[:, None, :] - E[None, :, :]
        r2 = np.maximum(np.sum(r * r, axis=2), 1e-12)
        return sign * r[:, :, 0] / (2 * np.pi * r2)
    Wt = wash(Eb, 1.0) + wash(Ea, -1.0)                   # per unit strip circulation
    dxs = np.abs(Eb[:, 0] - Ea[:, 0])
    out = dict(alpha=[], CL=[], CDi=[], S_ref=S)
    for a in alphas_deg:
        rhs = -np.sin(math.radians(a) + inc - a0)          # unit freestream, small-angle boundary condition
        G = np.linalg.solve(Ainf, rhs)
        cl = 2 * float(np.sum(G * dx)) / S
        Gs = G.reshape(strips, nc).sum(axis=1)
        w = Wt @ Gs
        cdi = -float(np.sum(Gs * w * dxs)) / S
        out["alpha"].append(a); out["CL"].append(cl); out["CDi"].append(cdi)
    b = sum(abs(w["tip"][0] - w["root"][0]) for w in wings) / 1000
    out["span_m"] = b
    out["AR"] = b * b / S if S else 0.0
    return out


def frontal_area(brep_front, brep_behind=None, px=400) -> float:
    """Projected area (m^2) looking along z; with brep_behind, only the part of brep_front outside it."""
    def raster(brep, box, n):
        a = u.tessellate(brep, chord_height_tolerance=max(0.5, (box[2] - box[0]) * 1e-3)).to_numpy_arrays()
        pts, flat = a["points"], a["faces"]
        img = np.zeros((n, n), bool)
        sx, sy = (n - 1) / (box[2] - box[0]), (n - 1) / (box[3] - box[1])
        i = 0
        while i < len(flat):
            m = int(flat[i]); idx = flat[i + 1:i + 1 + m]; i += 1 + m
            for k in range(1, m - 1):
                tri = pts[[idx[0], idx[k], idx[k + 1]]][:, :2]
                X, Y = (tri[:, 0] - box[0]) * sx, (tri[:, 1] - box[1]) * sy
                x0, x1 = int(max(0, np.floor(X.min()))), int(min(n - 1, np.ceil(X.max())))
                y0, y1 = int(max(0, np.floor(Y.min()))), int(min(n - 1, np.ceil(Y.max())))
                if x1 < x0 or y1 < y0:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
                d = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2])
                if abs(d) < 1e-12:
                    continue
                l1 = ((Y[1] - Y[2]) * (gx - X[2]) + (X[2] - X[1]) * (gy - Y[2])) / d
                l2 = ((Y[2] - Y[0]) * (gx - X[2]) + (X[0] - X[2]) * (gy - Y[2])) / d
                img[y0:y1 + 1, x0:x1 + 1] |= (l1 >= -0.02) & (l2 >= -0.02) & (1 - l1 - l2 >= -0.02)
        return img, (box[2] - box[0]) / (n - 1) * (box[3] - box[1]) / (n - 1)
    boxes = [b.bounding_box(tight=True) for b in [brep_front] + ([brep_behind] if brep_behind else [])]
    lo = np.min([b[0] for b in boxes], axis=0); hi = np.max([b[1] for b in boxes], axis=0)
    box = (lo[0], lo[1], hi[0], hi[1])
    f, cell = raster(brep_front, box, px)
    if brep_behind is not None:
        f &= ~raster(brep_behind, box, px)[0]
    return float(f.sum()) * cell / 1e6


def frontal_area_faces(res, body, exclude=(), px=400) -> float:
    """Projected area (m^2) looking along z of one body's faces, leaving out faces made by `exclude` blocks."""
    from caddy.kernel.build import tessellate
    m = tessellate(res)
    P, T, TF = m["points"], m["tris"], m["tri_face"]
    keep = [i for i, f in enumerate(res.faces) if f.body == body and f.block not in exclude]
    keep = np.isin(TF, keep)
    tris = P[T[keep]][:, :, :2]
    if not len(tris):
        return 0.0
    lo, hi = tris.reshape(-1, 2).min(axis=0), tris.reshape(-1, 2).max(axis=0)
    sx, sy = (px - 1) / max(hi[0] - lo[0], 1e-9), (px - 1) / max(hi[1] - lo[1], 1e-9)
    img = np.zeros((px, px), bool)
    for tri in tris:
        X, Y = (tri[:, 0] - lo[0]) * sx, (tri[:, 1] - lo[1]) * sy
        x0, x1 = int(np.floor(X.min())), int(np.ceil(X.max())); y0, y1 = int(np.floor(Y.min())), int(np.ceil(Y.max()))
        d = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2])
        if abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l1 = ((Y[1] - Y[2]) * (gx - X[2]) + (X[2] - X[1]) * (gy - Y[2])) / d
        l2 = ((Y[2] - Y[0]) * (gx - X[2]) + (X[0] - X[2]) * (gy - Y[2])) / d
        img[y0:y1 + 1, x0:x1 + 1] |= (l1 >= -0.02) & (l2 >= -0.02) & (1 - l1 - l2 >= -0.02)
    return float(img.sum()) / (sx * sy) / 1e6


def _cf(re):
    return 0.455 / math.log10(max(re, 1e5)) ** 2.58


def analyse(res, speed=50.0, alphas=None) -> dict:
    """Aero estimate for a built model (caddy Result). Needs at least one wing() block."""
    from caddy.kernel.graph import evaluate_expr
    alphas = list(range(-4, 17)) if alphas is None else alphas
    blocks = {b["id"]: b for b in res.graph["blocks"]}
    wings, wing_ids = [], set()
    for b in res.graph["blocks"]:
        if b["op"] == "wing":
            q = {k: evaluate_expr(e, res.env) for k, e in b["params"].items()}
            w = wing_planform(q); w["id"], w["part"] = b["id"], b["part"]
            wings.append(w); wing_ids.add(b["id"])
    if not wings:
        raise ValueError("no wing() features in this model: nothing generates lift to analyse")
    lattice = list(wings)
    for part in {w["part"] for w in wings}:     # reference wing through the body: join mirrored roots at the centreline
        r = [w for w in wings if w["part"] == part and w["side"] > 0]
        l = [w for w in wings if w["part"] == part and w["side"] < 0]
        for a, b in zip(r, l):
            if abs(a["root"][0] + b["root"][0]) < 1 and a["root"][0] > 1:
                lattice.append(dict(root=(-a["root"][0], a["root"][1], a["root"][2]), tip=a["root"], root_chord=a["root_chord"],
                                    tip_chord=a["root_chord"], root_inc=a["root_inc"], tip_inc=a["root_inc"], naca=a["naca"],
                                    side=1.0, part=part, id=part + "_carry_through"))
    pol = vlm_polar(lattice, alphas)
    main_part = max({w["part"] for w in wings}, key=lambda p_: sum((w["root_chord"] + w["tip_chord"]) * abs(w["tip"][0] - w["root"][0]) for w in lattice if w["part"] == p_))
    mw = [w for w in lattice if w["part"] == main_part]
    xs = [w["root"][0] for w in mw] + [w["tip"][0] for w in mw]
    span_main = (max(xs) - min(xs)) / 1000
    S_main = sum((w["root_chord"] + w["tip_chord"]) / 2 * abs(w["tip"][0] - w["root"][0]) for w in mw) / 1e6
    # coefficients in vlm_polar are on the whole lattice area; restate them on the main wing's reference area
    k_ref = pol["S_ref"] / S_main
    pol["CL"] = [c * k_ref for c in pol["CL"]]; pol["CDi"] = [c * k_ref for c in pol["CDi"]]
    pol["S_ref"], pol["span_m"], pol["AR"] = S_main, span_main, span_main ** 2 / S_main
    S = pol["S_ref"]
    # wetted areas from the kernel: faces made by wing blocks vs the rest, per body
    by = {}
    for f in res.faces:
        kind = blocks.get(f.block, {}).get("part", "body") if f.block in wing_ids else f.body
        by[kind] = by.get(kind, 0.0) + f.area / 1e6
    comps = []
    bodies = res.bodies
    main = next(iter(bodies))
    lo, hi = bodies[main].bounding_box(tight=True)
    L = (hi[2] - lo[2]) / 1000
    A_front = frontal_area_faces(res, main, exclude=wing_ids)
    zmax = hi[2]
    base = sum(f.area for f in res.faces if f.body == main and f.kind == "plane" and f.block not in wing_ids
               and abs(f.centroid[2] - zmax) < 1e-3 * max(L * 1000, 1)) / 1e6
    d_eq = math.sqrt(4 * A_front / math.pi)
    fr = L / d_eq if d_eq else 10.0
    sw_body = sum(a for k, a in by.items() if k == main)
    ff_body = 1 + 60 / fr ** 3 + fr / 400
    comps.append(dict(name=main, kind="body", wetted_m2=sw_body, length_m=L, Re=speed * L / NU, Cf=_cf(speed * L / NU),
                      FF=ff_body, Q=1.0, fineness=fr, frontal_m2=A_front))
    for part in sorted({w["part"] for w in wings}):
        ws = [w for w in wings if w["part"] == part]
        c = sum((w["root_chord"] + w["tip_chord"]) / 2 for w in ws) / len(ws) / 1000
        t = (ws[0]["naca"] % 100) / 100
        ff = 1 + 0.6 / 0.3 * t + 100 * t ** 4
        comps.append(dict(name=part, kind="wing", wetted_m2=by.get(part, 0.0), length_m=c, Re=speed * c / NU,
                          Cf=_cf(speed * c / NU), FF=ff, Q=INTERFERENCE, t_c=t))
    for c in comps:
        c["CD0"] = c["Cf"] * c["FF"] * c["Q"] * c["wetted_m2"] / S
    if base > 0:   # flat tail face: base drag (Hoerner, about 0.13 on the base area for a blunt base)
        comps.append(dict(name=main + " base", kind="base", frontal_m2=base, Cd=CD_BASE, CD0=CD_BASE * base / S))
    extra = []
    for name, brep in bodies.items():          # other bodies (e.g. wheels): bluff drag on what sticks out
        if name == main:
            continue
        a = frontal_area(brep, bodies[main])
        extra.append(dict(name=name, kind="bluff", frontal_m2=a, Cd=CD_WHEEL, CD0=CD_WHEEL * a / S))
    CD0 = sum(c["CD0"] for c in comps + extra)
    q = 0.5 * RHO * speed ** 2
    rows = []
    for a, cl, cdi in zip(pol["alpha"], pol["CL"], pol["CDi"]):
        cd = CD0 + max(cdi, 0.0)
        rows.append(dict(alpha=a, CL=round(cl, 4), CDi=round(cdi, 5), CD=round(cd, 5), LD=round(cl / cd, 3) if cd else 0,
                         lift_N=round(cl * q * S, 1), stalled=cl > CL_MAX))
    rows = [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()} for r in rows]
    ok = [r for r in rows if not r["stalled"]]
    best = max(ok, key=lambda r: r["LD"]) if ok else None
    cla = (pol["CL"][-1] - pol["CL"][0]) / math.radians(pol["alpha"][-1] - pol["alpha"][0])
    e = None
    if best and best["CL"] > 0.05:
        e = best["CL"] ** 2 / (math.pi * pol["AR"] * max(best["CDi"], 1e-9))
    cd_car = (CD0 * S) / A_front if A_front else None      # parasite drag area on the body's frontal area
    clean = lambda v: float(v) if isinstance(v, (np.floating,)) else v
    comps = [{k: clean(v) for k, v in c.items()} for c in comps]; extra = [{k: clean(v) for k, v in c.items()} for c in extra]
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    return dict(speed=speed, rho=RHO, q=q, S_ref=S, frontal_m2=A_front, CdA_m2=float(CD0 * S), Cd_frontal=cd_car, span_m=pol["span_m"], AR=pol["AR"], CL_alpha=cla, e=e,
                CD0=float(CD0), components=comps + extra, polar=rows, best=best, CL_max=CL_MAX,
                wings=[dict(id=w["id"], part=w["part"], naca=f"{w['naca']:04d}", root_chord=w["root_chord"],
                            tip_chord=w["tip_chord"], span=abs(w["tip"][0] - w["root"][0]), root_inc=w["root_inc"],
                            tip_inc=w["tip_inc"], alpha0=zero_lift_angle(w["naca"])) for w in wings],
                notes=["Vortex-lattice wings, carried through the body to the centreline (the usual reference wing); coefficients on the main wing's area; linear lift, no stall model beyond the CLmax mark.",
                       "Parasite drag: turbulent flat-plate friction x form factor x interference per component; wheels as bluff bodies.",
                       "Holes such as lift-fan ducts are in the wetted area but their flow is not modelled. A first-order estimate for comparing designs, not CFD."])
