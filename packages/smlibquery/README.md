# smlibquery: a CadQuery-style API on SMLib

**smlibquery is a Python modelling API on NVIDIA's SMLib kernel** (the open-sourced solid modeller behind
[USD BRep](https://github.com/NVIDIA-Omniverse/usd-brep), pip `usd-brep`). Its calls are named and behave like
CadQuery's, because people and language models already write CadQuery well, but **it is not CadQuery**: there is no
OpenCascade inside, every solid is built, measured and tessellated by SMLib, and it implements a subset of
CadQuery's calls plus a few of its own.

```python
import smlibquery as sq

part = (sq.Workplane("XY")
        .polygon(6, 34.64).extrude(10)        # a hex, 30 mm across flats (d = across corners, as in CadQuery)
        .faces(">Z").workplane().hole(14))     # a through bore

solid = part.val()                             # an exact usd_brep.Brep: 9 faces, closed solid
sq.export_usd(solid, "hex.usda")               # exact B-rep in USD (NVIDIA's BrepArray schema)
```

## Not CadQuery

| | CadQuery | smlibquery |
|---|---|---|
| Kernel | OpenCascade (OCCT) | NVIDIA SMLib (`usd-brep`) |
| Import | `import cadquery as cq` | `import smlibquery as sq` |
| Calls | the full API | a subset with the same names and defaults (table below); anything else is refused with the line and a suggestion |
| Own additions | | `.skin()` and `.wing()` (lofted bodies and NACA wings on SMLib), `.tag("part.role")` naming parts and faces |
| Results | OCCT geometry | SMLib geometry: the same call can give slightly different triangulation, tolerances and edge cases |
| Scripts | run anything Python | checked before they run: no loops, branches, file or network access |

A script written for CadQuery is not guaranteed to run here, and one written for smlibquery does not run in
CadQuery once it uses `skin`, `wing` or `tag`. Scripts saved before the rename that start with
`import cadquery as cq` still compile, to the same SMLib layer.

## Why

- **Models already speak CadQuery.** It is the default code target in LLM-CAD research (CAD-Recode,
  CADCodeVerify, CAD-Coder, BenchCAD), and its kernel is OpenCascade. smlibquery keeps the familiar call names so
  models and people write it fluently, and builds everything on SMLib instead.
- **Raw CadQuery often fails to run** when models write it (P3D-Bench, arXiv 2606.11152: 22–99% valid across
  models). smlibquery checks the script before running it and answers every unsupported call with the line and a
  fix, never a silent approximation.
- **One script, every size.** Compile a script once with catalogue values as *tracked numbers* and the recorded
  operations keep expressions (`hex_af / 2`, not `13.5`), so the same model builds every row of a catalogue.

## Two ways to use it

**Direct** (numbers in, solid out): build with `Workplane`, call `.val()`.

**Compiled for a catalogue**: catalogue columns are names in the script.

```python
code = '''
import smlibquery as sq
body = sq.Workplane("XY").polygon(6, af / cos(radians(30))).extrude(length).tag("hex.flat")
result = body.faces(">Z").workplane().hole(bore).tag("bore.wall")
'''
ir = sq.compile_script(code, {"af": 30.0, "length": 10.0, "bore": 14.0})
# ir["blocks"]: [{"id": "hex_flat", "op": "prism", "params": {"z": "0", "n": "6", "d": "1.154700538 * af",
#                 "h": "length"}, ...}, {"id": "bore_wall", "op": "cylinder", "params": {"z": "-1", "d": "bore",
#                 "h": "length + 2"}}, ...]
small = sq.build(ir, {"af": 22.0, "length": 8.0, "bore": 8.0})["result"]
```

## Supported calls (named and behaving as in CadQuery)

| Call | Notes |
|---|---|
| `Workplane("XY")` | XY only; orient with `.rotate()` or `sq.Location` |
| `.workplane(offset=, invert=, origin=)` | offsets along the current normal |
| `.faces(">Z")`, `.faces("<Z")` | top / bottom of the current solid, then `.workplane()` |
| `.center(x, y)`, `.moveTo(x, y)`, `.pushPoints(pts)` | sketch points |
| `.circle(r)`, `.rect(w, h)`, `.polygon(n, d)` | `rect` centred; `polygon` d = **circumscribed** diameter (as CadQuery) |
| `.extrude(d, combine=, both=)` | |
| `.loft()` | between two coaxial circles on stacked workplanes (a cone) |
| `.box(l, w, h)`, `.cylinder(h, r)` | centred, as CadQuery |
| `.hole(d, depth=None)`, `.cutBlind(-d)`, `.cutThruAll()` | |
| `.union()`, `.cut()`, `.intersect()` | exact SMLib booleans |
| `.translate((x, y, z))`, `.rotate(p0, p1, deg)` | rigid moves |
| `sq.Assembly().add(wp, name=, loc=sq.Location((x,y,z), axis, deg))` | several named bodies |
| `sq.importers.importStep("name")` | a stock part pre-converted to SMLib `.smb` (see below) |
| `math.*`, `sqrt/sin/cos/tan/radians/degrees`, `min/max/abs` | recorded symbolically |

Refused with a message: other planes, `fillet`/`chamfer` (SMLib's edge fillets aren't reliable enough yet; build a
chamfer as an intersection with a loft), sweeps, splines, `if`/loops on catalogue values (one script must hold for
every size), imports other than `smlibquery`/`math`, and private or dunder names.

## smlibquery's own calls (not in CadQuery)

| Call | What SMLib builds |
|---|---|
| `.skin([(z, half_width, bottom, top, taper), ...], exponent=3)` | a smooth body lofted (cubic) through rounded cross-sections along z: car bodies, fuselages, hulls |
| `.wing(naca, root_chord, tip_chord, span, at=(x, y, z), sweep=, dihedral=, incidence=, twist=, side=±1)` | a NACA 4-digit wing panel lofted between root and tip sections; chord along +z, span along ±x |
| `.tag("part.role")` | names the part and its faces: side faces `part.role`, flat end faces `part.end`, so a checker (e.g. [caddy](../../README.md)) can measure named geometry such as `hex.flat` |

## The intermediate representation

A list of blocks, each `{"id", "op", "part", "role", "params": {name: expression}, "inputs": [...]}`; ops are
`cone`, `cylinder`, `prism`, `box`, `stock`, `skin`, `wing`, `union`, `difference`, `intersect`, `move` (see `kernel.py`).
Expressions use a whitelisted arithmetic language (`expr.py`). `build(ir, values)` evaluates it on SMLib.

## Stock parts (STEP)

NVIDIA's pip package has no STEP reader. Convert once with smlibpy (AWV Informatik, MIT + Apache-2.0, bundles
SMLib with a STEP reader) into SMLib's native `.smb`, then point `SMLQ_STOCK_PATH` (or `smlibquery.kernel.
STOCK_PATHS`) at the folder. caddy's `tools/import_step.py` does this.

## Install

Python 3.12 (NVIDIA publishes `usd-brep` for Linux x86_64/aarch64 and Windows; no macOS: use Docker there).

```bash
pip install ./packages/smlibquery
```

```bash
pytest packages/smlibquery/tests
```

Apache-2.0 (as usd-brep). Not affiliated with CadQuery or NVIDIA. smlibquery borrows CadQuery's call names for
familiarity; it shares no code with CadQuery and does not use OpenCascade.
