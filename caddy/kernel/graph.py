"""The feature graph: CAD as data (JSON), evaluated on SMLib.

- Schema-validated JSON: parameters (bound to catalogue columns), blocks (typed ops with expression params, a part
  and a role, face aliases), checks (catalogue columns and rules, measured by the kernel).
- Expressions are a tiny safe arithmetic language over parameter names (no free-form code).
- The evaluator caches every block by the hash of its op, resolved params and inputs, so an edit rebuilds only the
  blocks downstream of what changed.
- Every block must change the solid ("feature exists"), every face gets a block, part and stable alias.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator
import time
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import usd_brep as u

from .build import KernelError, boolean_context, label_faces
from smlibquery import expr as _expr
from smlibquery.errors import SMLQError

from .ops import BOOLEANS, MOVE_KEYS, MOVE_OPS, PARAM_KEYS, PRIMITIVE_OPS, Block, apply_move, make_primitive

FAMILIES = Path(__file__).resolve().parents[2] / "data" / "families"

# Row variables available to every expression (see catalogue.values_for).
ROW_NAMES = {"overall_len", "hex_af", "hex_len", "inlet_thread_len", "outlet_thread_len", "bore_d"} | {
    p + k for p in ("in_", "out_") for k in ("major_d", "gauge_len", "pitch", "tpi", "useful_thread_min")}

SCHEMA = {
    "type": "object",
    "required": ["family", "parameters", "blocks"],
    "properties": {
        "family": {"type": "string"},
        "title": {"type": "string"},
        "parameters": {"type": "array", "items": {
            "type": "object", "required": ["name", "part"],
            "properties": {"name": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"}, "part": {"type": "string"},
                           "column": {"type": "string"}, "default": {"type": "number"}}}},
        "blocks": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["id", "op", "part", "role"],
            "properties": {
                "id": {"type": "string", "pattern": "^[a-z_][a-z0-9_]*$"},
                "op": {"enum": sorted(PRIMITIVE_OPS | set(BOOLEANS) | MOVE_OPS)},
                "part": {"type": "string"}, "role": {"type": "string"},
                "params": {"type": "object", "additionalProperties": {"type": ["string", "number"]}},
                "inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                "aliases": {"type": "object", "properties": {"side": {"type": "string"}, "cap": {"type": "string"}}},
                "optional": {"type": "boolean"},
                "stock": {"type": "string"},
                "line": {"type": "integer"},
            }}},
        "checks": {"type": "array", "items": {"type": "object", "required": ["id", "measure"]}},
        "bodies": {"type": "object", "additionalProperties": {"type": "string"}},
    },
}


class GraphError(ValueError):
    """A graph that can't be run (schema, unknown name, bad wiring). Carries a block id when there is one."""

    def __init__(self, msg: str, block: str | None = None):
        self.block = block
        super().__init__(f"{block}: {msg}" if block else msg)


# --- expressions (smlibquery's language; errors surface as GraphError here) -----------------------------------

_FUN = _expr.FUNCTIONS


def names_in(expr) -> set[str]:
    return _expr.names_in(expr)


def evaluate_expr(expr, env: dict[str, float]) -> float:
    try:
        return _expr.evaluate(expr, env)
    except SMLQError as e:
        raise GraphError(str(e)) from None


# --- graph ---------------------------------------------------------------------------------------------------

def load(family: str = "bspt_nipple") -> dict:
    """The reference graph (hand-written during the build; the agent writes its own)."""
    from .catalogue import rows, values_for
    g = json.loads((FAMILIES / f"{family}.graph.json").read_text())
    validate(g, set(values_for(rows(family=family)[0])) | ROW_NAMES)
    return g


def load_spec(family: str = "bspt_nipple") -> dict:
    """The spec the kernel checks against: catalogue columns, ISO values, rules, required face names.
    Owned by caddy, never written by the model."""
    return json.loads((FAMILIES / f"{family}.checks.json").read_text())


def validate(g: dict, row_names: set[str] | None = None) -> None:
    """Schema + wiring. row_names: the catalogue's variables (default: the nipple family's)."""
    row_names = ROW_NAMES if row_names is None else row_names
    try:
        jsonschema.validate(g, SCHEMA)
    except jsonschema.ValidationError as e:
        raise GraphError(f"schema: {e.message} at {'/'.join(map(str, e.absolute_path))}") from e
    seen, pnames = set(), {p["name"] for p in g["parameters"]}
    consumed: set[str] = set()
    for b in g["blocks"]:
        if b["id"] in seen:
            raise GraphError("duplicate block id", b["id"])
        if b["op"] == "stock" and not b.get("stock"):
            raise GraphError('stock blocks need "stock": <part name>', b["id"])
        if b["op"] in PRIMITIVE_OPS:
            missing = set(PARAM_KEYS[b["op"]]) - set(b.get("params", {}))
            if missing:
                raise GraphError(f"{b['op']} needs params {sorted(missing)}", b["id"])
            for k, e in b.get("params", {}).items():
                try:
                    bad = names_in(e) - pnames - row_names
                except SyntaxError as err:
                    raise GraphError(f"param {k}: bad expression {e!r} ({err.msg})", b["id"]) from err
                if bad:
                    raise GraphError(f"param {k} uses unknown names {sorted(bad)}", b["id"])
        else:
            if b["op"] in MOVE_OPS:
                for k, e in b.get("params", {}).items():
                    if k not in MOVE_KEYS:
                        raise GraphError(f"move takes {', '.join(MOVE_KEYS)}; got {k!r}", b["id"])
                    bad = names_in(e) - pnames - row_names
                    if bad:
                        raise GraphError(f"param {k} uses unknown names {sorted(bad)}", b["id"])
            for i in b.get("inputs", []):
                if i not in seen:
                    raise GraphError(f"input {i!r} is not an earlier block", b["id"])
                if i in consumed:
                    raise GraphError(f"input {i!r} is already used by another boolean", b["id"])
                consumed.add(i)
            n_in = 1 if b["op"] in MOVE_OPS else 2
            if len(b.get("inputs", [])) != n_in:
                raise GraphError(f"{b['op']} takes exactly {n_in} input{'s' if n_in > 1 else ''}", b["id"])
        seen.add(b["id"])
    for name, bid in (g.get("bodies") or {}).items():
        if bid not in seen:
            raise GraphError(f"body {name!r} names unknown block {bid!r}")


def env_for(g: dict, values: dict[str, float]) -> dict[str, float]:
    """Row values, then parameter defaults, then derived parameters ("expr": an expression of the others)."""
    env = dict(values)
    env.update({p["name"]: p["default"] for p in g["parameters"]
                if "default" in p and "expr" not in p and p["name"] not in values})
    for p in g["parameters"]:
        if "expr" in p and p["name"] not in values:
            env[p["name"]] = evaluate_expr(p["expr"], env)
    missing = {p["name"] for p in g["parameters"]} - set(env)
    if missing:
        raise GraphError(f"no value for parameters {sorted(missing)}")
    return env


# --- evaluator -----------------------------------------------------------------------------------------------

@dataclass
class Result:
    brep: u.Brep           # the first body (the only one for a single part)
    graph: dict
    env: dict
    faces: list            # FaceLabel with .alias and .body; .index is into face_objs
    steps: list[dict]      # per block: rebuilt?, volume, faces, feature_exists, ms
    rebuilt: list[str] = field(default_factory=list)
    row_env: dict = field(default_factory=dict)   # the catalogue row's variables only (what checks see)
    bodies: dict = field(default_factory=dict)    # body name -> Brep (assemblies have several)
    face_objs: list = field(default_factory=list) # kernel Face objects, parallel to faces

    def by_alias(self, prefix: str) -> list:
        return [f for f in self.faces if f.alias == prefix or f.alias.startswith(prefix + "[")]


class Evaluator:
    """Caches block results by content hash; re-running after an edit only rebuilds what changed downstream."""

    def __init__(self):
        self.cache: dict[str, tuple[u.Brep, dict]] = {}

    def run(self, g: dict, values: dict[str, float]) -> Result:
        validate(g, set(values) | ROW_NAMES)
        env = env_for(g, values)
        keys: dict[str, str] = {}
        live: dict[str, u.Brep] = {}
        tools: list = []                 # [Block, untouched copy] per primitive, moved along with its body
        owners: dict[str, list[int]] = {}  # live body -> the tools it was made from
        steps, rebuilt = [], []
        vols: dict[str, float | None] = {}   # live body -> its volume, measured only when needed (slow on skinned bodies)
        blocks = {b["id"]: b for b in g["blocks"]}
        for b in g["blocks"]:
            blk = Block(b["id"], b["op"], b["part"], b["role"], inputs=list(b.get("inputs", [])), stock=b.get("stock", ""))
            if b["op"] in PRIMITIVE_OPS or b["op"] in MOVE_OPS:
                blk.params = {k: evaluate_expr(e, env) for k, e in b.get("params", {}).items()}
            key = hashlib.sha1(json.dumps([b["op"], blk.params, blk.stock, [keys[i] for i in blk.inputs]],
                                          sort_keys=True).encode()).hexdigest()[:16]
            keys[b["id"]] = key
            t = time.perf_counter()
            hit = key in self.cache
            if hit:
                brep, info = self.cache[key]
                brep = brep.copy()
            else:
                try:
                    if blk.op in PRIMITIVE_OPS:
                        brep = make_primitive(blk)
                        before = None
                    elif blk.op in MOVE_OPS:
                        brep = apply_move(live[blk.inputs[0]].copy(), blk.params)
                        before = "move"
                    else:
                        a, c = live[blk.inputs[0]], live[blk.inputs[1]]
                        before = (blk.inputs[0], a.face_count())
                        brep = BOOLEANS[blk.op](a.copy(), c.copy())
                except RuntimeError as e:
                    ctx = boolean_context(live[blk.inputs[0]], live[blk.inputs[1]], blk.inputs) \
                        if blk.op in BOOLEANS and len(blk.inputs) == 2 else None
                    raise KernelError(blk, str(e), ctx) from e
                nf = brep.face_count()
                if before is None:
                    vol = brep.volume()
                    exists = vol > 1e-9
                elif before == "move":
                    vol = vols[blk.inputs[0]]     # a rigid move keeps the solid (and its volume) as it was
                    exists = True
                elif nf != before[1]:
                    vol, exists = None, True      # the boolean changed the faces: it did something
                else:                             # same face count: only the volume can tell
                    vol = brep.volume()
                    va = vols[before[0]] if vols[before[0]] is not None else live[before[0]].volume()
                    exists = abs(vol - va) > 1e-6 * max(va, 1.0)
                info = dict(volume=vol, faces=nf, feature_exists=bool(exists), optional=bool(b.get("optional")))
                self.cache[key] = (brep.copy(), info)
                rebuilt.append(b["id"])
            if blk.op in PRIMITIVE_OPS:
                tools.append([blk, brep.copy()])
                owners[b["id"]] = [len(tools) - 1]
            elif blk.op in MOVE_OPS:
                owners[b["id"]] = owners.pop(blk.inputs[0])
                for ti in owners[b["id"]]:          # provenance follows the body
                    apply_move(tools[ti][1], blk.params)
            else:
                owners[b["id"]] = owners.pop(blk.inputs[0]) + owners.pop(blk.inputs[1])
            for i in blk.inputs:
                live.pop(i)
            live[b["id"]] = brep
            vols[b["id"]] = info["volume"]
            steps.append(dict(block=b["id"], part=b["part"], op=b["op"], cached=hit,
                              ms=round((time.perf_counter() - t) * 1000, 1), **info))
        named = g.get("bodies") or {}
        if named:
            extra = set(live) - set(named.values())
            if extra:
                raise GraphError(f"graph leaves bodies not listed in 'bodies': {sorted(extra)}")
            body_ids = dict(named)
        else:
            if len(live) != 1:
                raise GraphError(f"graph leaves {len(live)} separate bodies: {sorted(live)} (list them in 'bodies')")
            body_ids = {"part": next(iter(live))}
        faces, face_objs, bodies = [], [], {}
        for name, bid in body_ids.items():
            brep = live[bid]
            bodies[name] = brep
            fl = label_faces(brep, [tuple(tools[i]) for i in owners[bid]])
            for f, fo in zip(fl, brep.faces()):
                f.index = len(face_objs)
                f.body = name
                faces.append(f)
                face_objs.append(fo)
        assign_aliases(face_objs, faces, blocks)
        first = next(iter(bodies.values()))
        return Result(first, g, env, faces, steps, rebuilt, row_env=dict(values), bodies=bodies, face_objs=face_objs)


def assign_aliases(face_objs: list, faces: list, blocks: dict) -> None:
    groups: dict[str, list] = {}
    for f, bf in zip(faces, face_objs):
        (_, _, z0), (_, _, z1) = bf.bounding_box(tight=True)
        cap = f.kind == "plane" and abs(z1 - z0) < 1e-6   # planar face square to the axis
        al = blocks[f.block].get("aliases", {})
        base = al.get("cap" if cap else "side") or f"{f.part}.{f.role}"
        groups.setdefault(base, []).append(f)
    for base, fs in groups.items():
        fs.sort(key=lambda f: (round(f.centroid[2], 3), round(math.atan2(f.centroid[1], f.centroid[0]) % (2 * math.pi), 3)))
        for i, f in enumerate(fs):
            f.alias = base if len(fs) == 1 else f"{base}[{i}]"


def apply_edit(g: dict, ops: list[dict], row_names: set[str] | None = None) -> dict:
    """A graph diff. Ops:
      {"block": id, "set": {param: expr}}            change block params
      {"block": id, "inputs": [a, b]}                rewire a boolean
      {"add": {...block...}, "after": id}            insert a block
      {"remove": id}                                 delete a block
      {"param": name, "default": v}                  change a parameter's default
    Returns a new, validated graph."""
    g2 = copy.deepcopy(g)
    params = {p["name"]: p for p in g2["parameters"]}
    for op in ops:
        blocks = {b["id"]: b for b in g2["blocks"]}
        if "add" in op:
            ids = [b["id"] for b in g2["blocks"]]
            if op.get("after") not in ids:
                raise GraphError(f"'after' must name an existing block, got {op.get('after')!r}", op["add"].get("id"))
            g2["blocks"].insert(ids.index(op["after"]) + 1, copy.deepcopy(op["add"]))
        elif "remove" in op:
            if op["remove"] not in blocks:
                raise GraphError("no such block", op["remove"])
            g2["blocks"] = [b for b in g2["blocks"] if b["id"] != op["remove"]]
        elif "block" in op:
            if op["block"] not in blocks:
                raise GraphError("no such block", op["block"])
            if "set" in op:
                blocks[op["block"]].setdefault("params", {}).update(op["set"])
            if "inputs" in op:
                blocks[op["block"]]["inputs"] = list(op["inputs"])
            for k in ("aliases", "optional", "part", "role"):
                if k in op:
                    blocks[op["block"]][k] = op[k]
        elif "param" in op:
            if op["param"] not in params:
                raise GraphError(f"no such parameter {op['param']!r}")
            upd = {k: v for k, v in op.items() if k != "param"}
            if isinstance(upd.get("default"), str):        # an expression given as a default: treat it as derived
                upd["expr"] = upd.pop("default")
            if "expr" in upd:
                params[op["param"]].pop("default", None)
            if "default" in upd:
                params[op["param"]].pop("expr", None)
            params[op["param"]].update(upd)
        else:
            raise GraphError(f"unknown edit {op}")
    validate(g2, row_names)
    return g2


def diff_size(ops: list[dict]) -> int:
    """Characters of the diff as JSON: the 'output tokens' proxy for an edit."""
    return len(json.dumps(ops, separators=(",", ":")))
