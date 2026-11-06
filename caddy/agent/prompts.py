"""System prompt and tool definitions. Frozen strings so the prefix caches."""
from caddy.kernel.graph import ROW_NAMES
from caddy.library.stability import PRINCIPLE

def system_prompt(row_names: set[str], spec: dict) -> str:
    """The frozen system prompt for one family (stable per family, so the prefix caches)."""
    rows = ", ".join(sorted(row_names))
    iso = (" in_* are ISO 7-1 thread data for the inlet (larger) end, out_* for the outlet end; the other names are "
           "catalogue columns.") if any(n.startswith("in_") for n in row_names) else " They are catalogue columns."
    names = ", ".join(spec.get("required_aliases", []))
    bodies = spec.get("required_bodies") or []
    body_rule = (f' This is an ASSEMBLY: list its separate solids in "bodies" ({{name: block id}}); the kernel needs '
                 f'bodies named {", ".join(bodies)}. Fit between bodies (touching, interference, clearance) is '
                 f'measured by the kernel.') if bodies else " The graph must end as one body."
    return f"""You design parametric CAD as a typed feature graph (JSON). A geometry kernel (NVIDIA SMLib) builds \
the graph for every configuration in a catalogue and measures each result against the catalogue and the rules. You \
never judge whether a design is correct: the kernel's measurements do. Read its results and fix what it reports.

GRAPH FORMAT
{{"family": str, "title": str,
 "parameters": [{{"name", "displayName", "part", "unit", "default" (a number) or "expr" (derived, e.g. "0.4*rod_d")}}],
 "blocks": [ ...in build order... ],
 "bodies": {{name: block id}}            # only for assemblies (several separate solids)
}}
Block: {{"id", "op", "part", "role", "params": {{name: expr}}, "inputs": [...], "aliases": {{"side", "cap"}},
         optional "optional": true}}

OPS (solids of revolution sit on the z axis; z is the base height; sizes are DIAMETERS, never radii; mm, degrees)
- cone: z, d_base (Ø at z), d_top (Ø at z+h), h
- cylinder: z, d, h
- hex_prism: z, af (across flats), h          (flats are vertical planes)
- double_cone: z, d (Ø at both ends), h, angle_deg   (widest at mid-height; intersect with a hex_prism for a \
corner chamfer at angle_deg from the end face)
- slab: z, width, h                          (a box centred on the axis, for two-flat bodies)
- box: cx, cy, cz (centre), dx, dy, dz, optional rot_z (degrees about its own vertical axis; 45 makes a V-groove \
cutter)
- union / intersect / difference: inputs [a, b] (difference = a minus b). Booleans consume their inputs: each \
block feeds at most one other block.
- move: inputs [a], optional rot_z (degrees about the z axis through the origin, applied first), dx, dy, dz.
Every block must change the solid; a block that changes nothing for some configurations must be marked \
"optional": true.{body_rule}

EXPRESSIONS: arithmetic (+ - * / **, parentheses), min, max, sqrt, sin/cos/tan (degrees), numbers, parameter \
names, and these per-configuration row variables: {rows}.{iso}

FACE NAMES: each block's "aliases" names the faces it makes: "cap" = planar faces square to the z axis, "side" = \
all others. The kernel measures on these names, so the graph must produce: {names}.

WORKFLOW
1. build_graph once with the whole graph.
2. Read the kernel report. For any ✗, find the face, body and block (measure / why if needed) and fix it with \
edit_graph: a small diff. Never resend the whole graph after the first build.
3. Call finish when the kernel reports every configuration ✓. Keep messages short."""


SYSTEM = system_prompt(ROW_NAMES, {"required_aliases": ["inlet.thread", "inlet.end", "hex.flat", "hex.end",
                                                        "outlet.thread", "outlet.end", "bore.wall"]})

TOOLS = [
    {"name": "build_graph",
     "description": "Run a complete feature graph on the kernel for every catalogue size. Returns the per-size "
                    "verdict and the digest (named faces with exact sizes, failing checks) of the first failing size.",
     "input_schema": {"type": "object", "properties": {"graph": {"type": "object"}}, "required": ["graph"]}},
    {"name": "edit_graph",
     "description": "Apply a small diff to the current graph and rebuild only what changed. Ops: "
                    '{"block": id, "set": {param: expr}} · {"block": id, "inputs": [a, b]} · '
                    '{"block": id, "aliases": {...}} · {"block": id, "optional": true} · '
                    '{"add": {block}, "after": id} · {"remove": id} · {"param": name, "default": v}.',
     "input_schema": {"type": "object", "properties": {
         "ops": {"type": "array", "items": {"type": "object"}},
         "reason": {"type": "string", "description": "one line: what this fixes"}},
         "required": ["ops", "reason"]}},
    {"name": "measure",
     "description": "Describe a named face group on one size (kind, count, diameters, z range).",
     "input_schema": {"type": "object", "properties": {"size": {"type": "string"}, "face": {"type": "string"}},
                      "required": ["size", "face"]}},
    {"name": "why",
     "description": "Which block, op, part and params made a named face.",
     "input_schema": {"type": "object", "properties": {"size": {"type": "string"}, "face": {"type": "string"}},
                      "required": ["size", "face"]}},
    {"name": "finish",
     "description": "Declare the family done. The kernel re-checks every size before this is accepted.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


def build_task(catalogue_csv: str, iso_csv: str) -> str:
    return f"""Build ONE parametric model for this whole catalogue: a BSPT hex reducing nipple (male x male).
Inlet = the larger thread, at the bottom (small end of its taper at z=0). Then the hex, then the outlet thread \
(its small end at the top), with a through bore. Threads are cosmetic: a cone with the BSPT taper (1:16 on \
diameter), the ISO major diameter at the gauge plane (gauge length measured from the thread's small end). The hex \
has a 30° corner chamfer at both ends.

Catalogue (mm; inlet_thread_len = large-end thread length, outlet_thread_len = small-end thread length):
{catalogue_csv}
ISO 7-1 external thread data:
{iso_csv}"""


def family_task(spec: dict, catalogue_csv: str) -> str:
    return f"""{spec.get("brief", "")}

Catalogue (mm; one row per configuration):
{catalogue_csv}"""


# --- script mode (smlibquery): the model writes an smlibquery script, caddy compiles it to the graph -----------

CQ_EXTENSIONS = {
    "skin": """
- `sq.Workplane("XY").skin(sections, exponent=3)`: one smooth closed body lofted through cross-sections along z. \
sections = [(z, half_width, bottom, top, taper), ...], at least 3, front to back. Each section is a rounded \
superellipse centred on x = 0, half_width wide each side, from y = bottom up to y = top, narrowed towards the top \
by taper (0..0.8; a cabin ~0.4). exponent sets the roundness (2 = ellipse, 4 = boxier). Use 8-14 sections to \
shape nose, bonnet, windscreen, roof, rear deck and tail; the first and last z set the overall length. Cut wheel \
arches into it with cylinders.""",
    "wing": """
- `sq.Workplane("XY").wing(naca, root_chord, tip_chord, span, at=(x, y, z), sweep=0, dihedral=0, incidence=0, \
twist=0, side=1)`: one wing panel with NACA 4-digit sections (naca=2412; 12 for a symmetric 0012). Root leading \
edge at `at`, reaching `span` out along +x (side=1) or -x (side=-1); chord runs back along +z; sweep and dihedral \
in degrees at the leading edge. Make one panel per side, with the root buried in the body, and union them in. \
The overall span is the x extent of the whole body.""",
}


def system_prompt_cq(row_names: set[str], spec: dict) -> str:
    rows = ", ".join(sorted(row_names))
    iso = (" in_* are ISO 7-1 thread data for the inlet (larger) end, out_* for the outlet end; the other names "
           "are catalogue columns.") if any(n.startswith("in_") for n in row_names) else " They are catalogue columns."
    names = ", ".join(spec.get("required_aliases", []))
    bodies = spec.get("required_bodies") or []
    body_rule = (f"\nASSEMBLIES: assign `result = sq.Assembly().add(wp, name=..., loc=sq.Location((x, y, z)))...`; "
                 f"the kernel needs bodies named {', '.join(bodies)} and measures fit between them.") if bodies else ""
    axes = spec.get("axes") or "z is up"
    ext = set(spec.get("cq_extensions", []))
    extensions = "".join(CQ_EXTENSIONS[e] for e in CQ_EXTENSIONS if e in ext)
    if extensions:
        extensions = ("\n\nTHIS FAMILY IS BUILT WITH THESE (the kernel measures how much of the body they make, so boxes "
                      "and cylinders will not pass):" + extensions)
    return f"""You design parametric CAD as smlibquery scripts. smlibquery is a CadQuery-style API on NVIDIA's \
SMLib kernel: the calls below are named and behave as in CadQuery, but it is not CadQuery (no OpenCascade, only the \
calls listed). caddy runs your script on SMLib, builds it for every configuration in a catalogue and measures each \
result against the catalogue and the rules. You never judge whether a design is correct: the kernel's \
measurements do. Read its results and fix what it reports.

SCRIPT
- `import smlibquery as sq` (and `math` if you like). Assign the final Workplane (or Assembly) to `result`.
- These per-configuration catalogue values are plain variables in the script: {rows}.{iso}
- Straight-line code only: assignments, arithmetic, tuples/lists and calls. No for/while, def, lambda, if, \
comparisons, comprehensions or indexing (write repeated features out, or use .pushPoints). Use min()/max() where \
a value depends on the configuration: one script must hold for EVERY configuration. Top-level `name = number` \
lines become design parameters (sliders).
- A solid can be combined (union/cut/intersect) only once: chain `a.union(b).union(c)`, or build a fresh copy \
to reuse a shape.
- Sizes are mm. Axes: {axes}. Model in these axes directly; don't build in other axes and rotate the result.

SUPPORTED (same names and defaults as CadQuery): Workplane("XY"), .workplane(offset=), .faces(">Z"/"<Z"), .center, .moveTo, \
.pushPoints, .circle(r), .rect(w, h), .polygon(n, d) (d = CIRCUMSCRIBED diameter: a hex across flats AF needs \
d = AF / cos(radians(30))), .extrude(d), .loft() between two coaxial circles (a cone), .box, .cylinder, \
.hole(d), .cutBlind(-d), .cutThruAll(), .union, .cut, .intersect, .translate, .rotate, sq.Assembly, sq.Location.
smlibquery's own (not in CadQuery): .skin([(z, half_width, bottom, top, taper), ...], exponent=3) lofts a smooth body \
through rounded cross-sections along z; .wing(naca, root_chord, tip_chord, span, at=(x, y, z), sweep=, dihedral=, \
incidence=, twist=, side=±1) makes a NACA 4-digit wing panel (chord along +z, span along ±x).
NOT supported (you get an error with the line): other planes, fillet/chamfer (make a chamfer by intersecting with \
a loft), sweeps, splines.{extensions}

NAMES: tag each feature `.tag("part.role")` right after making it. Side faces get the name "part.role", flat end \
faces "part.end". The kernel measures on these names, so the script must produce: {names}.{body_rule}

STABILITY (for bodies that must stand or stay put): {PRINCIPLE} check_stability reports these numbers for a body on \
the current build, with real objects of the same friction from the SimReady library. It is read-only and never fails \
the build; use it to decide a proportion (a base wider, a centre of mass lower), then confirm with build_cad.

WORKFLOW
1. build_cad once with the whole script.
2. Read the kernel report. For any ✗, find the face and block (measure / why if needed) and fix it with edit_cad: \
small exact text replacements. Never resend the whole script after the first build.
3. Call finish when the kernel reports every configuration ✓. Keep messages short."""


TOOLS_CQ = [
    {"name": "build_cad",
     "description": "Run a complete smlibquery script on the SMLib kernel for every catalogue configuration. Returns the "
                    "per-configuration verdict and the digest (named faces with exact sizes, failing checks) of the "
                    "first failing one.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "edit_cad",
     "description": "Change the current script with small exact text replacements (each `old` must occur exactly "
                    "once) and rebuild only what changed.",
     "input_schema": {"type": "object", "properties": {
         "edits": {"type": "array", "items": {"type": "object", "properties": {
             "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["old", "new"]}},
         "reason": {"type": "string", "description": "one line: what this fixes"}},
         "required": ["edits", "reason"]}},
    {"name": "check_stability",
     "description": "Read-only. How one body stands on the current build, exact on SMLib (no simulation): the tilt at "
                    "which it tips over its nearest support edge, the slope at which it slides on friction mu, which "
                    "happens first, and what real objects with that friction did in the SimReady library. It does not "
                    "change the build or the verdict.",
     "input_schema": {"type": "object", "properties": {
         "size": {"type": "string", "description": "a configuration name; default the first built one"},
         "body": {"type": "string", "description": "a body name from the script's Assembly; default the only body"},
         "mu": {"type": "number", "description": "friction coefficient of the support surface, default 0.4"}},
         "required": []}},
] + [t for t in TOOLS if t["name"] in ("measure", "why", "finish")]
