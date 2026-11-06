"""Provenance (face -> block -> part), triangle -> face mapping, and kernel errors as repair facts.

SMLib has no persistent face ids or attributes, so provenance is ours: each primitive block is copied before the
booleans consume it; afterwards every result face is matched to the primitive whose face it lies on (a face split
by a boolean still lies on its parent's face, so it keeps the label). Ties go to the earliest block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import usd_brep as u

from .ops import Block

ON_FACE_TOL = 1e-4  # mm


class KernelError(RuntimeError):
    """A kernel failure turned into repair facts: which block, which SMLib status, where in the kernel."""

    def __init__(self, block: Block, raw: str, context: dict | None = None):
        self.block_id, self.part, self.op, self.raw = block.id, block.part, block.op, raw
        self.context = context or {}
        m = re.search(r"failed: (SM_\w+) \((-?\d+)\)", raw)
        self.status = m.group(1) if m else "UNKNOWN"
        self.code = int(m.group(2)) if m else None
        t = re.search(r"Kernel trace: (\w+) \[at ([^\]]+)\](?: \(([^)]*)\))?", raw)
        self.trace = dict(name=t.group(1), at=t.group(2), msg=t.group(3)) if t else None
        super().__init__(f"block {self.block_id} ({self.op}, part {self.part}): {self.status}")

    def facts(self) -> dict:
        out = dict(block=self.block_id, part=self.part, op=self.op, status=self.status, code=self.code,
                   trace=self.trace)
        out.update(self.context)
        return out


BOOLEAN_HINT = ("SMLib could not combine these two solids. The usual cause is faces that touch, coincide or meet "
                "tangentially instead of clearly crossing: a root or end face lying exactly on another face, two panels "
                "meeting at the same plane (e.g. both wing roots at x = 0), or a thin edge grazing a curved surface. "
                "Make the tool clearly overlap (bury roots 50 mm or more inside) or clearly stay apart.")


def boolean_context(a, b, inputs) -> dict:
    """Facts about a failed boolean's two inputs, from the kernel's own queries (best effort)."""
    ctx = dict(inputs=list(inputs), hint=BOOLEAN_HINT)
    try:
        rel = u.brep_relationship(a, b)
        ctx["relationship"] = rel.get("relationship")
        ctx["distance_mm"] = round(float(rel.get("distance", 0.0)), 3)
    except Exception:  # the query itself can fail on the same geometry
        pass
    return ctx


@dataclass
class FaceLabel:
    index: int
    block: str
    part: str
    role: str
    kind: str          # plane | cone | cylinder
    area: float
    centroid: tuple
    alias: str = ""
    body: str = "part"


def label_faces(brep: u.Brep, tools: list[tuple[Block, u.Brep]]) -> list[FaceLabel]:
    tool_faces = [(blk, t.faces()) for blk, t in tools]
    labels = []
    for i, f in enumerate(brep.faces()):
        p = f.interior_point()
        owner = None
        if len(tool_faces) == 1:          # a body made from one primitive or stock part: every face is that block's
            owner = tool_faces[0][0]
        for blk, fs in ([] if owner else tool_faces):
            if any(tf.closest_point(p)[1] < ON_FACE_TOL for tf in fs):
                owner = blk
                break
        if owner is None:  # fallback: nearest primitive surface
            owner = min(tool_faces, key=lambda bf: min(tf.closest_point(p)[1] for tf in bf[1]))[0]
        kind = "plane" if f.surface().is_planar() else ("cylinder" if owner.op in ("cylinder",) else "curved" if owner.op == "stock" else "cone")
        if owner.op == "hex_prism" and kind != "plane":
            kind = "plane"
        labels.append(FaceLabel(i, owner.id, owner.part, owner.role, kind, f.area(), tuple(f.centroid())))
    return labels


def _face_at(brep, p):
    """The exact face a triangle came from. A coarse triangle's centre on a curved face sits off the surface, so if
    the lookup misses, snap the point onto the solid's surface first."""
    try:
        return u.find_face(brep, p)
    except RuntimeError:
        q = u.brep_closest_point(brep, p)[0]
        return u.find_face(brep, (q.x, q.y, q.z))


def tessellate(built, chord=0.01, angle=8.0) -> dict:
    """Mesh of every body, with one face index per triangle (mapped back to the exact BRep face)."""
    bodies = getattr(built, "bodies", None) or {"part": built.brep}
    index = {f: k for k, f in enumerate(getattr(built, "face_objs", None) or built.brep.faces())}
    all_pts, all_tris, all_tf, mesh_vol = [], [], [], 0.0
    base = 0
    for brep in bodies.values():
        lo, hi = brep.bounding_box(tight=True)
        size = max(h - l for l, h in zip(lo, hi))
        ch = max(chord, size * 2e-4)     # a fixed 0.01 mm chord is right for a fitting, 33 MB of mesh for a car
        pb = u.tessellate(brep, chord_height_tolerance=ch, surface_angle_tolerance_deg=angle,
                          curve_angle_tolerance_deg=angle)
        a = pb.to_numpy_arrays()
        pts = a["points"].astype(float)
        flat = a["faces"]
        tris, i = [], 0
        while i < len(flat):
            n = int(flat[i])
            idx = flat[i + 1:i + 1 + n]
            tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, n - 1)]
            i += 1 + n
        tris = np.asarray(tris, dtype=np.int64)
        cents = pts[tris].mean(axis=1)
        all_tf.append(np.array([index[_face_at(brep, tuple(float(x) for x in c))] for c in cents], dtype=np.int32))
        all_pts.append(pts)
        all_tris.append(tris + base)
        base += len(pts)
        mesh_vol += u.poly_brep_volume_via_api(pb)
    return dict(points=np.vstack(all_pts), tris=np.vstack(all_tris).astype(np.int32),
                tri_face=np.concatenate(all_tf), mesh_volume=mesh_vol)
