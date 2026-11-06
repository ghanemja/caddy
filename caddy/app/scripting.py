"""Visual scripting: the board edits the smlibquery script, not a copy of it.

`fields(script)` finds the editable values in the script: the arguments of each smlibquery call (`.circle(r0)`,
`.extrude(hex_len)`, `offset=z_hex0`) and the top-level values (`chamfer_deg = 30`). Each is a span of one line.
`splice(script, field, text)` replaces that span; the session recompiles and only changed blocks rebuild.
"""
from __future__ import annotations

import ast


def _seg(lines: list[str], node) -> tuple[int, int, int] | None:
    if node.lineno != node.end_lineno:
        return None                       # only single-line values are edited in place
    return node.lineno, node.col_offset, node.end_col_offset


def _span(lines, line, c0, c1) -> str:
    # ast offsets are UTF-8 bytes; convert to str columns
    b = lines[line - 1].encode()
    return b[c0:c1].decode()


def _col(lines, line, byte_col) -> int:
    return len(lines[line - 1].encode()[:byte_col].decode())


def fields(script: str | None) -> list[dict]:
    if not script:
        return []
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []
    lines = script.split("\n")
    out = []

    def add(node, kind, label, call=None):
        s = _seg(lines, node)
        if s is None:
            return
        line, b0, b1 = s
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return                        # tags, workplane names, selectors: not values
        out.append(dict(line=line, start=_col(lines, line, b0), end=_col(lines, line, b1), kind=kind, label=label,
                        call=call, text=_span(lines, line, b0, b1)))

    for st in tree.body:                  # top-level values: name = expression (not call chains)
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                and not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and n.func.attr not in ("cos", "sin", "tan", "radians", "degrees", "sqrt")
                            for n in ast.walk(st.value)):
            add(st.value, "value", st.targets[0].id)
    for node in ast.walk(tree):           # arguments of smlibquery calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr not in (
                "tag", "Workplane", "cos", "sin", "tan", "radians", "degrees", "sqrt", "importStep", "add",
                "union", "cut", "intersect"):
            for i, a in enumerate(node.args):
                add(a, "arg", f"{node.func.attr}[{i}]", node.func.attr)
            for k in node.keywords:
                if k.arg:
                    add(k.value, "arg", f"{node.func.attr}.{k.arg}", node.func.attr)
    out.sort(key=lambda f: (f["line"], f["start"]))
    return out


def attach(script: str | None, block_lines: dict[str, int]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Give each feature the fields of the statement that made it: a call's arguments flow down the chain to the
    block it creates (workplane offset → circle → loft), so a field goes to the next block line in its statement.
    Returns ({block id: fields}, top-level values that belong to no block)."""
    fs = fields(script)
    if not fs:
        return {}, []
    tree = ast.parse(script)
    spans = [(st.lineno, st.end_lineno) for st in tree.body]
    by_line: dict[int, list[str]] = {}
    for b, ln in block_lines.items():
        if ln:
            by_line.setdefault(ln, []).append(b)
    out: dict[str, list[dict]] = {}
    values = []
    for f in fs:
        lo, hi = next(((a, z) for a, z in spans if a <= f["line"] <= z), (f["line"], f["line"]))
        cand = sorted(ln for ln in by_line if lo <= ln <= hi)
        after = [ln for ln in cand if ln >= f["line"]]
        ln = after[0] if after else (cand[-1] if cand else None)
        if ln is None or f["kind"] == "value":
            values.append(f)
        else:
            out.setdefault(by_line[ln][0], []).append(f)
    return out, values


def splice(script: str, line: int, start: int, end: int, text: str, expect: str | None = None) -> tuple[str, str, str]:
    """Replace columns start:end of a line. Returns (new script, old line, new line)."""
    lines = script.split("\n")
    if not (1 <= line <= len(lines)):
        raise ValueError(f"no line {line}")
    old = lines[line - 1]
    if expect is not None and old[start:end] != expect:
        raise ValueError("the script changed under this edit; refresh and try again")
    if "\n" in text:
        raise ValueError("one line at a time")
    lines[line - 1] = old[:start] + text + old[end:]
    return "\n".join(lines), old, lines[line - 1]
