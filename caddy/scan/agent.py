"""What Claude reads for scan → CAD: the kernel's measurements of the scan, the draft, and the deviation report."""
from __future__ import annotations

import json

SYSTEM = """You turn a 3D scan of a turned part into editable parametric CAD as an smlibquery script (a CadQuery-style \
API on NVIDIA's SMLib kernel; not CadQuery itself). caddy has already measured \
the scan on SMLib (its axis, cross-sections across the axis, and the outline in a plane through the axis), fitted \
cylinders, cones and a hex to it, and drafted a script. The kernel then compares every scan point with your solid. \
You never judge the fit yourself: the kernel's checks do.

YOUR JOB
1. Rewrite the draft as a script an engineer would keep:
   - name every feature by what it is: `.tag("part.role")`, e.g. "inlet.thread", "hex.flat", "flange.rim", \
"bore.wall", "counterbore.wall" (side faces get "part.role", flat ends "part.end");
   - put the dimensions in top-level named values (`flange_d = 44.0`); these become the design's parameters;
   - round each measured value to the design value it obviously is (43.976 → 44, 21.988 → 22, 27.026 → 27) only \
where the scan supports it, and keep expressions that relate features (`hex_len = 9`, `z_hex = inlet_len`);
   - model what the measurements show that the draft misses. A hex whose corner radius (`corner_min`) is smaller \
than a sharp hex's (`corner_r`) is chamfered: intersect the prism with a double cone made of two lofts (wide at the \
middle of the hex, `corner_min` at each end), e.g. `hexp = hexp.intersect(c1.union(c2))`.
2. build_cad once. Read the kernel's deviation report: it names the faces that miss the scan, by how much, where, \
and which way. Fix them with edit_cad (small exact replacements).
3. finish when matches_scan and no_extra_material both pass.

SCRIPT
- `import smlibquery as sq` and `import math`. Assign the final Workplane to `result`. Sizes in mm; z is the turning \
axis, the part sits on z = 0; a hex has a corner on +x (polygon's default, as in CadQuery).
- SUPPORTED (same names and defaults as CadQuery): Workplane("XY"), .workplane(offset=), .circle(r), .polygon(n, d) (d = CIRCUMSCRIBED: \
hex across flats AF needs d = AF / math.cos(math.radians(30))), .extrude(h), .loft() between two coaxial circles \
(a cone), .hole(d), .union, .cut, .intersect, .translate, .rotate. NOT supported: fillet/chamfer (use lofts), \
revolve, sweeps, splines.
Keep messages short."""


def task(steps: list[dict], draft: str, report: str) -> str:
    by = {s["key"]: s for s in steps}
    sl = by["slices"]["slices"]
    f = by["fit"]["fit"]
    layers = sl["layers"]
    pick = layers[:: max(1, len(layers) // 12)]
    return (
        "Measured on the scan (mm, axis frame):\n"
        f"- scan: {json.dumps(by['scan']['info'])}\n"
        f"- axis: principal moments {by['axis']['axis']['moments']} → turned part: {by['axis']['axis']['turned']}\n"
        f"- hex (from the cross-sections): {json.dumps(sl['hex'])}\n"
        f"- cross-sections (every few): " + "; ".join(
            f"z {L['z']}: r {L['r_min']}..{L['r_max']}" + (f" bore r {L['bore']}" if L["bore"] else "") for L in pick) + "\n"
        f"- outline through the axis, simplified: {by['profile']['profile']['n_raw']} points → "
        f"{by['profile']['profile']['n_fit']} straight runs\n"
        f"- fitted primitives: {json.dumps({k: f[k] for k in ('outer', 'bore', 'length')})}\n\n"
        f"The draft script (numbers as measured):\n```python\n{draft}```\n\n"
        f"The kernel's report on the draft:\n{report}\n\n"
        "Write the engineer's version of this part with build_cad, then fix what the kernel reports and finish."
    )
