"""smlibquery: a CadQuery-style API on NVIDIA SMLib (pip `usd-brep`). Not CadQuery: no OpenCascade inside.

    import smlibquery as sq
    part = sq.Workplane("XY").polygon(6, 34.6).extrude(10).faces(">Z").workplane().hole(14)
    solid = part.val()                 # an exact usd_brep.Brep
    sq.export_usd(solid, "part.usda")

For catalogue families, `compile_script(code, values)` records a script once with catalogue values as tracked
numbers and returns an IR whose parameters are expressions (see README)."""
from .build import build, export_usd
from .errors import SMLQError
from .script import check, compile_script
from .sym import Sym
from .workplane import Assembly, Location, Workplane, importers

__all__ = ["Workplane", "Assembly", "Location", "importers", "compile_script", "check", "build", "export_usd",
           "Sym", "SMLQError"]
__version__ = "0.1.0"
