"""Compile an smlibquery script (CadQuery-style syntax, built on SMLib) into the IR, symbolically, with catalogue
values as tracked numbers.

The script is checked as an AST first and executed with no builtins:
  allowed  : `import smlibquery as sq`, `import math`, assignments, calls, attribute access, arithmetic, tuples/lists
  refused  : other imports, loops, branches, comprehensions, lambdas, dunder/private names, file or network access
"""
from __future__ import annotations

import ast
import re
import threading

from . import sym, workplane as wpm
from .errors import SMLQError
from .sym import Sym

_ALLOWED = (ast.Module, ast.Assign, ast.Expr, ast.Call, ast.Attribute, ast.Name, ast.Constant, ast.BinOp,
            ast.UnaryOp, ast.Tuple, ast.List, ast.keyword, ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Import, ast.ImportFrom, ast.alias)


class _sq:  # noqa: N801  what `import smlibquery as sq` gives a script
    Workplane = wpm.Workplane
    Assembly = wpm.Assembly
    Location = wpm.Location
    importers = wpm.importers


def check(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SMLQError(f"line {e.lineno}: syntax error: {e.msg}") from None
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if not isinstance(node, _ALLOWED):
            raise SMLQError(f"line {line}: {type(node).__name__} is not allowed (smlibquery calls, arithmetic and "
                            "assignments only; one script must hold for every catalogue size)")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(m not in ("smlibquery", "math", LEGACY) for m in mods):
                raise SMLQError(f"line {line}: only `import smlibquery as sq` and `import math` are allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SMLQError(f"line {line}: private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise SMLQError(f"line {line}: names starting with _ are not allowed")
    return tree


# Scripts saved before the rename import `cadquery as cq`; they still compile (to the same SMLib layer), but smlibquery
# is not CadQuery and new scripts use `import smlibquery as sq`.
LEGACY = "cadquery"

_LOCK = threading.Lock()   # compiling uses module state (active recorder, value signs); scripts compile in ms


def compile_script(code: str, values: dict[str, float] | None = None, parameters: list[dict] | None = None,
                   family: str = "script", title: str = "") -> dict:
    with _LOCK:
        return _compile(code, values, parameters, family, title)


def _compile(code, values, parameters, family, title) -> dict:
    """Run `code` against the CadQuery subset and return the IR graph:
    {"blocks": [...], "parameters": [...], "bodies": {name: block id}?}. Parameters in blocks are expressions
    over the names in `values` (catalogue columns) and `parameters` (design parameters with defaults)."""
    tree = check(code)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    # top-level `name = <number>` lines are design parameters: they stay named in expressions (and get sliders)
    design = []
    for n in list(tree.body):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and _number(n.value) is not None and n.targets[0].id not in (values or {})):
            design.append({"name": n.targets[0].id, "default": _number(n.value), "part": "design",
                           "line": n.lineno})
            tree.body.remove(n)
    parameters = list(parameters or []) + design
    builtins = {"min": sym.smin, "max": sym.smax, "abs": abs, "float": float, "int": int}
    ns = {"__builtins__": builtins, "sq": _sq, "smlibquery": _sq, "cq": _sq, LEGACY: _sq, "math": sym.math, "pi": sym.math.pi,
          "sqrt": sym.math.sqrt, "sin": sym.math.sin, "cos": sym.math.cos, "tan": sym.math.tan,
          "radians": sym.math.radians, "degrees": sym.math.degrees}
    ns.update({k: Sym(v, k) for k, v in (values or {}).items()})
    sym._SIGNS.clear()
    sym._SIGNS.update({k: float(v) for k, v in (values or {}).items()})
    sym._SIGNS.update({p["name"]: float(p["default"]) for p in parameters or [] if "default" in p})
    for p in parameters or []:
        if "default" in p:
            ns[p["name"]] = Sym(p["default"], p["name"])
    rec = wpm.Recorder()
    wpm._ACTIVE.append(rec)
    try:
        exec(compile(tree, "<script>", "exec"), ns)          # noqa: S102 (checked AST, no builtins)
    except SMLQError as e:
        raise SMLQError(f"line {_line(e)}: {e}") from None
    except (TypeError, ValueError, ZeroDivisionError, AttributeError, NameError, IndexError, KeyError) as e:
        raise SMLQError(f"line {_line(e)}: {type(e).__name__}: {e}") from None
    finally:
        wpm._ACTIVE.pop()
    result = ns.get("result")
    if isinstance(result, wpm.Workplane) and result.solid is not None:
        finals, bodies = [result.solid], None
    elif isinstance(result, wpm.Assembly) and result.items:
        bodies = {name: wp.solid for name, wp in result.items}
        finals = list(bodies.values())
    else:
        raise SMLQError("assign the final Workplane (or sq.Assembly) to `result`")
    # drop solids that were made but never used in the result
    keep = set()
    for f in finals:
        keep |= {b["id"] for b in rec.subgraph(f)}
    rec.blocks = [b for b in rec.blocks if b["id"] in keep]
    _settle_parts(rec.blocks)
    ren = _name_blocks(rec.blocks)
    from .expr import names_in
    for p in parameters:              # a design parameter belongs to the first part that uses it
        if p.get("part") == "design":
            p["part"] = next((b["part"] for b in rec.blocks
                              if any(p["name"] in names_in(e) for e in b.get("params", {}).values())), "design")
    g = {"family": family, "title": title, "parameters": list(parameters or []), "blocks": rec.blocks,
         "source": "cadquery"}
    if bodies:
        g["bodies"] = {k: ren[v] for k, v in bodies.items()}
    return g


def _number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _number(node.operand)
        return -v if v is not None else None
    return None


def _settle_parts(blocks):
    """Blocks no .tag() claimed take the tag that was in force when they were made (else stay "body")."""
    for b in blocks:
        tag = b.pop("inherit", None)
        if b["part"] == "body" and tag:
            part, _, role = tag.partition(".")
            b["part"] = part
            if role and b["op"] not in ("union", "difference", "intersect", "move"):
                b["role"] = role
            b.setdefault("aliases", {"side": tag if role else f"{part}.side", "cap": f"{part}.end"})


def _name_blocks(blocks) -> dict:
    """Readable, stable ids from part and role: hex_flat, hex_flat_2, inlet_thread ..."""
    seen, ren = {}, {}
    for b in blocks:
        base = b["part"] if b["role"] in (b["part"], "stock") else f"{b['part']}_{b['role']}"
        base = re.sub(r"[^a-z0-9_]", "_", base.lower()) or "block"
        seen[base] = seen.get(base, 0) + 1
        ren[b["id"]] = base if seen[base] == 1 else f"{base}_{seen[base]}"
    for b in blocks:
        b["id"] = ren[b["id"]]
        if "inputs" in b:
            b["inputs"] = [ren[i] for i in b["inputs"]]
    return ren


def _line(e) -> str:
    tb, line = e.__traceback__, "?"
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<script>":
            line = tb.tb_lineno
        tb = tb.tb_next
    return str(line)
