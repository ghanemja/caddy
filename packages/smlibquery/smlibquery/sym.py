"""Tracked numbers: a float that remembers the expression it came from.

Inside a compiled script, catalogue columns and design parameters are `Sym`s, so `hex_af / 2` records the text
"hex_af / 2" alongside the value. That keeps one script valid for every catalogue row."""
from __future__ import annotations

import math as _math
import re

from .errors import SMLQError


def fmt(x) -> str:
    if isinstance(x, Sym):
        return x.t
    x = float(x)
    return str(int(x)) if x == int(x) and abs(x) < 1e12 else format(x, ".10g")


def _paren(t: str) -> str:
    return t if re.fullmatch(r"-?[A-Za-z0-9_.]+|[A-Za-z_]+\(.*\)", t) and t.count("(") == t.count(")") else f"({t})"


class Sym(float):
    """value + the expression text; linear expressions (sums of names times numbers) are kept as a
    {name: coefficient} map so they print simplified: 2 * (x / 2) -> x, (a + b) - a -> b."""

    def __new__(cls, value, text=None, lin=None):
        o = float.__new__(cls, value)
        if lin is None and text is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            lin = ({text: 1.0}, 0.0)
        o.lin = lin
        o.t = _lin_text(lin) if lin is not None else (text if text is not None else fmt(float(value)))
        return o

    def _bin(self, other, op, fn, rev=False):
        if not isinstance(other, (int, float)):
            return NotImplemented
        a, b = (other, self) if rev else (self, other)
        la, lb = _lin(a), _lin(b)
        lin = None
        if la is not None and lb is not None:
            if op in "+-":
                s = 1.0 if op == "+" else -1.0
                terms = dict(la[0])
                for k, c in lb[0].items():
                    terms[k] = terms.get(k, 0.0) + s * c
                lin = ({k: c for k, c in terms.items() if abs(c) > 1e-12}, la[1] + s * lb[1])
            elif op == "*" and (not la[0] or not lb[0]):
                (k_, c_), sc = (lb, la[1]) if not la[0] else (la, lb[1])
                lin = ({k: c * sc for k, c in k_.items()}, c_ * sc)
            elif op == "/" and not lb[0] and lb[1] != 0:
                lin = ({k: c / lb[1] for k, c in la[0].items()}, la[1] / lb[1])
        v = fn(float(a), float(b))
        if lin is not None:
            return Sym(v, lin=lin) if lin[0] else float(v)
        return Sym(v, f"{_paren(fmt(a))} {op} {_paren(fmt(b))}")


    # identities keep recorded expressions readable
    def __add__(self, o): return self if _is(o, 0) else self._bin(o, "+", lambda a, b: a + b)
    def __radd__(self, o): return self if _is(o, 0) else self._bin(o, "+", lambda a, b: a + b, True)
    def __sub__(self, o): return self if _is(o, 0) else self._bin(o, "-", lambda a, b: a - b)
    def __rsub__(self, o): return -self if _is(o, 0) else self._bin(o, "-", lambda a, b: a - b, True)
    def __mul__(self, o): return self if _is(o, 1) else self._bin(o, "*", lambda a, b: a * b)
    def __rmul__(self, o): return self if _is(o, 1) else self._bin(o, "*", lambda a, b: a * b, True)
    def __truediv__(self, o): return self if _is(o, 1) else self._bin(o, "/", lambda a, b: a / b)
    def __rtruediv__(self, o): return self._bin(o, "/", lambda a, b: a / b, True)
    def __pow__(self, o): return self._bin(o, "**", lambda a, b: a ** b)
    def __neg__(self):
        if self.lin is not None:
            return Sym(-float(self), lin=({k: -c for k, c in self.lin[0].items()}, -self.lin[1]))
        return Sym(-float(self), f"-{_paren(self.t)}")
    def __pos__(self): return self
    def __abs__(self): return Sym(abs(float(self)), f"abs({self.t})")

    def _no_branch(self, *_):
        raise SMLQError("no if/else or comparisons on catalogue values: one script must hold for every size "
                        "(use min(), max() or abs())")

    __lt__ = __le__ = __gt__ = __ge__ = __bool__ = _no_branch
    __hash__ = float.__hash__


def _lin(x):
    if isinstance(x, Sym):
        return x.lin
    return ({}, float(x))


def _num_text(c: float) -> str:
    return fmt(round(c, 10))


def _lin_text(lin) -> str:
    terms, const = lin
    out = []
    for k, c in terms.items():
        sign = "-" if c < 0 else "+"
        a = abs(c)
        inv = 1 / a if a else 0
        if abs(a - 1) < 1e-12:
            t = k
        elif abs(inv - round(inv)) < 1e-9 and round(inv) > 1:
            t = f"{k} / {int(round(inv))}"
        else:
            t = f"{_num_text(a)} * {k}"
        out.append((sign, t))
    if abs(const) > 1e-12 or not out:
        out.append(("-" if const < 0 else "+", _num_text(abs(const))))
    text = " ".join(f"{sg} {t}" for sg, t in out).strip()
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def _is(x, v) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, Sym) and float(x) == v


def is_zero(x) -> bool:
    return not isinstance(x, Sym) and float(x) == 0.0


def _le(a, b) -> bool:
    """True when a <= b for every row: b - a is linear with non-negative coefficients on names that are
    non-negative (lengths) and a non-negative constant. Assumes a name keeps its sign across catalogue rows."""
    la, lb = _lin(a), _lin(b)
    if la is None or lb is None:
        return False
    d = (b - a) if isinstance(b, Sym) or isinstance(a, Sym) else b - a
    if not isinstance(d, Sym):
        return float(d) >= 0
    terms, const = d.lin if d.lin is not None else (None, None)
    if terms is None:
        return False
    vals = _SIGNS
    return const >= -1e-12 and all(c >= 0 and vals.get(k, -1) >= 0 for k, c in terms.items())


_SIGNS: dict[str, float] = {}       # current values of the names (set by compile_script), for _le


def _extreme(args, pick_min):
    flat = []
    for a in args:
        flat += list(getattr(a, "args_" + ("min" if pick_min else "max"), [a]))
    keep = []
    for i, a in enumerate(flat):    # drop arguments another argument provably dominates (ties: keep the first)
        if any(j != i and (_le(o, a) if pick_min else _le(a, o)) and not (_le(a, o) and _le(o, a) and j > i)
               for j, o in enumerate(flat)):
            continue
        if any(fmt(a) == fmt(k) for k in keep):
            continue
        keep.append(a)
    if len(keep) == 1:
        return keep[0]
    if not any(isinstance(a, Sym) for a in keep):
        return (min if pick_min else max)(keep)
    name = "min" if pick_min else "max"
    v = (min if pick_min else max)(float(a) for a in keep)
    out = Sym(v, f"{name}({', '.join(fmt(a) for a in keep)})")
    setattr(out, "args_" + name, keep)
    return out


def smin(*args):
    return _extreme(args, True)


def smax(*args):
    return _extreme(args, False)


def _fn(name, fn, expr_name=None):
    def f(*args):
        if not any(isinstance(a, Sym) for a in args):
            return fn(*args)
        return Sym(fn(*[float(a) for a in args]), f"{expr_name or name}({', '.join(fmt(a) for a in args)})")
    f.__name__ = name
    return f


class math:  # noqa: N801  (shadows `math` inside scripts: Python's radian trig, recorded symbolically)
    pi = _math.pi
    sqrt = staticmethod(_fn("sqrt", _math.sqrt))
    sin = staticmethod(_fn("sin", _math.sin, "rsin"))
    cos = staticmethod(_fn("cos", _math.cos, "rcos"))
    tan = staticmethod(_fn("tan", _math.tan, "rtan"))
    radians = staticmethod(_fn("radians", _math.radians))
    degrees = staticmethod(_fn("degrees", _math.degrees))
