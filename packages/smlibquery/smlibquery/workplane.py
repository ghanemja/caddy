"""smlibquery's Workplane: a CadQuery-style API on NVIDIA SMLib. Every call records a typed op in an IR (see README
"IR") that SMLib builds; nothing is guessed. It is not CadQuery and runs no OpenCascade.

Follows CadQuery's names and defaults where it implements a call (`polygon(n, d)`: d is the circumscribed diameter; `box`,
`rect` centred; `hole()` through all). Unsupported calls raise `SMLQError` with a suggestion.

caddy extension: `.tag("part.role")` also names the part and its faces (side faces "part.role", flat end faces
"part.end"), which is how kernel checks refer to geometry."""
from __future__ import annotations

import re
import sys

from .errors import SMLQError
from .sym import Sym, fmt, is_zero, smax, smin


class Recorder:
    def __init__(self):
        self.blocks: list[dict] = []
        self.n = 0
        self.used: set[str] = set()

    def add(self, op, part, role, params=None, inputs=None, **extra) -> str:
        """part "body" means: not tagged yet (the next .tag() in the chain claims it)."""
        self.n += 1
        bid = f"b{self.n}"
        b = {"id": bid, "op": op, "part": part, "role": role}
        if params is not None:
            b["params"] = {k: fmt(v) for k, v in params.items()}
        if inputs:
            b["inputs"] = list(inputs)
        b.update(extra)
        line = _script_line()
        if line:
            b["line"] = line          # the script line that made this block (for showing code next to geometry)
        self.blocks.append(b)
        return bid

    def get(self, bid):
        return next(b for b in self.blocks if b["id"] == bid)

    def consume(self, bid):
        if bid in self.used:
            raise SMLQError("a solid can be combined only once; build it again or combine in one chain")
        self.used.add(bid)

    def subgraph(self, bid) -> list[dict]:
        """The blocks needed to build `bid`, in order."""
        need, stack = set(), [bid]
        by = {b["id"]: b for b in self.blocks}
        while stack:
            k = stack.pop()
            if k not in need:
                need.add(k)
                stack += by[k].get("inputs", [])
        return [b for b in self.blocks if b["id"] in need]


def _script_line():
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename == "<script>":
            return f.f_lineno
        f = f.f_back
    return None


_ACTIVE: list[Recorder] = []          # the recorder of the script being compiled (else a default one)
_DEFAULT = Recorder()


def rec() -> Recorder:
    return _ACTIVE[-1] if _ACTIVE else _DEFAULT


class Workplane:
    """A Workplane in the style of CadQuery's, on the XY plane, its offsets and the >Z / <Z faces."""

    def __init__(self, inPlane="XY", origin=(0, 0, 0), obj=None, *, _state=None):
        if _state is not None:
            self.__dict__.update(_state)
            return
        if inPlane != "XY":
            raise SMLQError(f'Workplane("{inPlane}") is not supported yet: build on "XY" and orient with .rotate() '
                            "or sq.Location")
        self.rec = rec()
        self.solid = None
        self.z = origin[2]
        self.dir = 1
        self.pts = [(origin[0], origin[1])]
        self.pending = []
        self.loft_stack = []
        self.lo = self.hi = None
        self.chain = []
        self.tagname = None

    # --- internals ------------------------------------------------------------------------------------------
    def _new(self, **changes):
        st = dict(self.__dict__)
        st.update(changes)
        for k in ("pending", "loft_stack", "chain"):
            st[k] = list(st[k])
        return Workplane(_state=st)

    def _part_role(self, role):
        return "body", role          # named later by .tag(), or by the tag in force when the script ends

    def _combine(self, tool, lo, hi, op="union"):
        if self.solid is None:
            return self._new(solid=tool, lo=lo, hi=hi, chain=self.chain + [tool], pending=[], loft_stack=[])
        self.rec.consume(self.solid)
        self.rec.consume(tool)
        bid = self.rec.add(op, "body", op, inputs=[self.solid, tool], inherit=self.tagname)
        nlo, nhi = (smin(self.lo, lo), smax(self.hi, hi)) if op == "union" and None not in (self.lo, lo) else \
            ((self.lo, self.hi) if op != "union" else (None, None))
        return self._new(solid=bid, lo=nlo, hi=nhi, chain=self.chain + [tool, bid], pending=[], loft_stack=[])

    def _prims(self, items, z0, h, role):
        part, r = self._part_role(role)
        ids = []
        for kind, args, (x, y) in items:
            c = {} if (is_zero(x) and is_zero(y)) else {"cx": x, "cy": y}
            if kind == "circle":
                ids.append(self.rec.add("cylinder", part, r, {"z": z0, "d": args[0], "h": h, **c},
                                        inherit=self.tagname))
            elif kind == "rect":
                ids.append(self.rec.add("box", part, r, {"cx": x, "cy": y, "cz": z0 + h / 2, "dx": args[0],
                                                          "dy": args[1], "dz": h}, inherit=self.tagname))
            elif kind == "polygon":
                ids.append(self.rec.add("prism", part, r, {"z": z0, "n": args[0], "d": args[1], "h": h, **c},
                                        inherit=self.tagname))
        for extra in ids[1:]:
            self.rec.consume(ids[0])
            self.rec.consume(extra)
            ids[0] = self.rec.add("union", part, r, inputs=[ids[0], extra], inherit=self.tagname)
        return ids[0]

    # --- sketch -------------------------------------------------------------------------------------------
    def center(self, x, y):
        return self._new(pts=[(px + x, py + y) for px, py in self.pts])

    def moveTo(self, x=0, y=0):
        return self._new(pts=[(x, y)])

    def pushPoints(self, pts):
        return self._new(pts=[(p[0], p[1]) for p in pts])

    def circle(self, radius, forConstruction=False):
        return self._new(pending=self.pending + [("circle", (2 * radius,), p) for p in self.pts])

    def rect(self, xLen, yLen, centered=True, forConstruction=False):
        if centered is not True:
            raise SMLQError("rect(centered=False) is not supported yet; offset with .center(x, y)")
        return self._new(pending=self.pending + [("rect", (xLen, yLen), p) for p in self.pts])

    def polygon(self, nSides, diameter, forConstruction=False, circumscribed=False):
        if circumscribed:
            raise SMLQError("polygon(circumscribed=True) is not supported: pass the circumscribed diameter "
                            "(a hex across flats AF has diameter AF / cos(radians(30)))")
        return self._new(pending=self.pending + [("polygon", (nSides, diameter), p) for p in self.pts])

    # --- workplanes ---------------------------------------------------------------------------------------
    def workplane(self, offset=0, invert=False, centerOption=None, origin=None):
        st = {"z": self.z + self.dir * offset}
        if invert:
            st["dir"] = -self.dir
        if self.pending:
            st["loft_stack"] = self.loft_stack + [(self.z, self.pending)]
            st["pending"] = []
        if origin is not None:
            st["pts"] = [(origin[0], origin[1])]
        return self._new(**st)

    def faces(self, selector=None):
        if self.solid is None:
            raise SMLQError("faces() needs a solid first")
        if selector in (">Z", "<Z"):
            z = self.hi if selector == ">Z" else self.lo
            if z is None:
                raise SMLQError(f'faces("{selector}") after a cut or rotation is not supported yet; '
                                "use .workplane(offset=...) from a known height")
            return self._new(z=z, dir=1 if selector == ">Z" else -1, pts=[(0, 0)])
        raise SMLQError(f'faces("{selector}") is not supported yet: ">Z" and "<Z" are')

    # --- solids -------------------------------------------------------------------------------------------
    def box(self, length, width, height, centered=True, combine=True):
        if centered is not True:
            raise SMLQError("box(centered=False) is not supported yet; use .translate()")
        (x, y), = self.pts[:1] or [(0, 0)]
        part, r = self._part_role("box")
        tool = self.rec.add("box", part, r, {"cx": x, "cy": y, "cz": self.z, "dx": length, "dy": width,
                                             "dz": height}, inherit=self.tagname)
        lo, hi = self.z - height / 2, self.z + height / 2
        return self._combine(tool, lo, hi) if combine else self._new(solid=tool, lo=lo, hi=hi, chain=self.chain + [tool])

    def cylinder(self, height, radius, centered=True, combine=True):
        (x, y), = self.pts[:1] or [(0, 0)]
        part, r = self._part_role("cylinder")
        z0 = self.z - height / 2 if centered else self.z
        c = {} if (is_zero(x) and is_zero(y)) else {"cx": x, "cy": y}
        tool = self.rec.add("cylinder", part, r, {"z": z0, "d": 2 * radius, "h": height, **c}, inherit=self.tagname)
        return self._combine(tool, z0, z0 + height) if combine else \
            self._new(solid=tool, lo=z0, hi=z0 + height, chain=self.chain + [tool])

    def extrude(self, until, combine=True, both=False, taper=None):
        if taper is not None:
            raise SMLQError("extrude(taper=...) is not supported; use .loft() between two circles for a taper")
        if not self.pending:
            raise SMLQError("extrude() needs a sketch (circle, rect or polygon) first")
        z0 = self.z if self.dir > 0 else self.z - until
        if both:
            z0, until = self.z - until, 2 * until
        tool = self._prims(self.pending, z0, until, "extrude")
        if not combine:
            return self._new(solid=tool, lo=z0, hi=z0 + until, chain=self.chain + [tool], pending=[])
        return self._combine(tool, z0, z0 + until)

    def loft(self, combine=True, ruled=False):
        if not self.loft_stack or not self.pending:
            raise SMLQError("loft() needs two sketches on stacked workplanes, e.g. "
                            ".circle(r1).workplane(offset=h).circle(r2).loft()")
        (z0, a), b = self.loft_stack[-1], self.pending
        if len(a) != 1 or len(b) != 1 or a[0][0] != "circle" or b[0][0] != "circle":
            raise SMLQError("loft() is supported between two coaxial circles (a cone) only")
        (x, y) = a[0][2]
        up = self.dir > 0
        lo_z, hi_z = (z0, self.z) if up else (self.z, z0)
        d_lo, d_hi = (a[0][1][0], b[0][1][0]) if up else (b[0][1][0], a[0][1][0])
        part, r = self._part_role("loft")
        c = {} if (is_zero(x) and is_zero(y)) else {"cx": x, "cy": y}
        tool = self.rec.add("cone", part, r, {"z": lo_z, "d_base": d_lo, "d_top": d_hi, "h": hi_z - lo_z, **c},
                            inherit=self.tagname)
        if not combine:
            return self._new(solid=tool, lo=lo_z, hi=hi_z, chain=self.chain + [tool], pending=[], loft_stack=[])
        return self._combine(tool, lo_z, hi_z)

    def skin(self, sections, exponent=3, combine=True):
        """caddy extension: a smooth body lofted through cross-sections along this workplane's z axis.
        sections: [(z, half_width, bottom, top, taper), ...] front to back; each is a rounded superellipse of that
        half-width between heights bottom and top (y), narrowed towards the top by taper (0..0.8)."""
        if len(sections) < 3:
            raise SMLQError("skin() needs at least 3 sections: [(z, half_width, bottom, top, taper), ...]")
        q = {"n": len(sections), "e": exponent}
        for i, sec in enumerate(sections):
            if len(sec) != 5:
                raise SMLQError(f"skin() section {i}: give (z, half_width, bottom, top, taper)")
            for key, v in zip("zwbtk", sec):
                q[f"{key}{i}"] = self.z + v if key == "z" else v
        part, r = self._part_role("skin")
        tool = self.rec.add("skin", part, r, q, inherit=self.tagname)
        lo, hi = q["z0"], q[f"z{len(sections) - 1}"]
        return self._combine(tool, lo, hi) if combine else self._new(solid=tool, lo=lo, hi=hi, chain=self.chain + [tool])

    def wing(self, naca, root_chord, tip_chord, span, at=(0, 0, 0), sweep=0, dihedral=0, incidence=0, twist=0,
             side=1, combine=True):
        """caddy extension: one wing panel with NACA 4-digit sections (naca=2412), root leading edge at `at`,
        reaching `span` out along +x (side=1) or -x (side=-1). Chord runs along +z (flow from the nose backwards),
        y is up. sweep and dihedral in degrees at the leading edge; incidence at the root, twist added at the tip."""
        q = {"naca": naca, "root_chord": root_chord, "tip_chord": tip_chord, "span": span,
             "x": at[0], "y": at[1], "z": self.z + at[2], "sweep": sweep, "dihedral": dihedral,
             "incidence": incidence, "twist": twist, "side": side}
        part, r = self._part_role("wing")
        tool = self.rec.add("wing", part, r, q, inherit=self.tagname)
        return self._combine(tool, None, None) if combine else self._new(solid=tool, lo=None, hi=None, chain=self.chain + [tool])

    def cutBlind(self, depth):
        if not self.pending:
            raise SMLQError("cutBlind() needs a sketch first")
        d = -depth
        z0 = self.z - d if self.dir > 0 else self.z
        return self._combine(self._prims(self.pending, z0, d, "cut"), None, None, "difference")

    def cutThruAll(self):
        if self.solid is None or not self.pending or None in (self.lo, self.hi):
            raise SMLQError("cutThruAll() needs a solid of known height and a sketch")
        tool = self._prims(self.pending, self.lo - 1, self.hi - self.lo + 2, "cut")
        return self._combine(tool, None, None, "difference")

    def hole(self, diameter, depth=None):
        if self.solid is None:
            raise SMLQError("hole() needs a solid first")
        items = [("circle", (diameter,), p) for p in self.pts]
        if depth is None:
            if None in (self.lo, self.hi):
                raise SMLQError("hole() through all needs a solid of known height; give depth=")
            tool = self._prims(items, self.lo - 1, self.hi - self.lo + 2, "hole")
        else:
            tool = self._prims(items, self.z - depth if self.dir > 0 else self.z, depth, "hole")
        return self._combine(tool, None, None, "difference")

    def union(self, other, clean=True):
        return self._bool(other, "union")

    def cut(self, other, clean=True):
        return self._bool(other, "difference")

    def intersect(self, other, clean=True):
        return self._bool(other, "intersect")

    def _bool(self, other, op):
        if not isinstance(other, Workplane) or other.solid is None or self.solid is None:
            raise SMLQError(f"{op} needs two solids")
        self.rec.consume(self.solid)
        self.rec.consume(other.solid)
        bid = self.rec.add(op, "body", op, inputs=[self.solid, other.solid], inherit=self.tagname)
        if op == "union" and None not in (self.lo, self.hi, other.lo, other.hi):
            lo, hi = smin(self.lo, other.lo), smax(self.hi, other.hi)
        elif op == "union":
            lo = hi = None
        else:
            lo, hi = self.lo, self.hi
        return self._new(solid=bid, lo=lo, hi=hi, chain=self.chain + other.chain + [bid], pending=[])

    def translate(self, vec):
        lo = self.lo + vec[2] if self.lo is not None else None
        hi = self.hi + vec[2] if self.hi is not None else None
        return self._move({"dx": vec[0], "dy": vec[1], "dz": vec[2]}, lo, hi)

    def rotate(self, axisStartPoint, axisEndPoint, angleDegrees):
        a = [axisEndPoint[i] - axisStartPoint[i] for i in range(3)]
        about_z = is_zero(a[0]) and is_zero(a[1])
        p = {"angle": angleDegrees, "ax": a[0], "ay": a[1], "az": a[2],
             "px": axisStartPoint[0], "py": axisStartPoint[1], "pz": axisStartPoint[2]}
        return self._move(p, self.lo if about_z else None, self.hi if about_z else None)

    def _move(self, params, lo, hi):
        if self.solid is None:
            raise SMLQError("translate/rotate need a solid")
        self.rec.consume(self.solid)
        keep = {k: v for k, v in params.items() if not is_zero(v)} or {"dx": 0}
        bid = self.rec.add("move", "body", "move", keep, inputs=[self.solid], inherit=self.tagname)
        return self._new(solid=bid, lo=lo, hi=hi, chain=self.chain + [bid])

    def tag(self, name):
        if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?", str(name)):
            raise SMLQError(f'tag "{name}": use "part" or "part.role" in lower case, e.g. "hex.flat"')
        part, _, role = name.partition(".")
        for bid in self.chain + ([self.solid] if self.solid and self.solid not in self.chain else []):
            b = self.rec.get(bid)
            if b["part"] == "body":
                b.pop("inherit", None)
                b["part"] = part
                if role and b["op"] not in ("union", "difference", "intersect", "move"):
                    b["role"] = role
                b.setdefault("aliases", {"side": name if role else f"{part}.side", "cap": f"{part}.end"})
        return self._new(tagname=name)

    # --- results ------------------------------------------------------------------------------------------
    def val(self, values: dict | None = None):
        """Build this solid on SMLib and return the usd_brep Brep."""
        from .build import build
        if self.solid is None:
            raise SMLQError("no solid to build")
        return build({"blocks": self.rec.subgraph(self.solid)}, values or {})["result"]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise SMLQError(f"Workplane.{name}() is not part of smlibquery yet (see the supported calls in its README)")


class Location:
    def __init__(self, t=(0, 0, 0), axis=None, angle=0):
        self.t, self.axis, self.angle = t, axis, angle


class Assembly:
    def __init__(self, obj=None, name=None, loc=None):
        self.items: list[tuple[str, Workplane]] = []
        if obj is not None:
            self.add(obj, name=name, loc=loc)

    def add(self, obj, name=None, loc=None, color=None):
        if not isinstance(obj, Workplane) or obj.solid is None:
            raise SMLQError("Assembly.add() takes a Workplane holding a solid")
        if not name or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise SMLQError('Assembly.add(..., name="lower_case_name") is required: the name is the body name')
        wp = obj
        if loc is not None:
            if loc.axis is not None and not is_zero(loc.angle):
                wp = wp.rotate((0, 0, 0), loc.axis, loc.angle)
            if any(not is_zero(v) for v in loc.t):
                wp = wp.translate(loc.t)
        self.items.append((name, wp))
        return self


class importers:  # noqa: N801
    @staticmethod
    def importStep(name):
        """Stock parts are pre-converted to SMLib's .smb (tools/import_step.py) and referred to by name."""
        stem = re.sub(r"\.(step|stp)$", "", str(name), flags=re.I).split("/")[-1].lower()
        r = rec()
        bid = r.add("stock", stem, "stock", stock=stem, aliases={"side": f"{stem}.surface", "cap": f"{stem}.flat"})
        wp = Workplane()
        return wp._new(solid=bid, chain=[bid])
