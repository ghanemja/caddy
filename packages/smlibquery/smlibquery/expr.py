"""The expression language of the IR: arithmetic over named values (catalogue columns, design parameters).

Whitelisted AST only: numbers, names, + - * / **, unary -, and a few functions. sin/cos/tan take degrees;
rsin/rcos/rtan take radians (what Python's math.* records as)."""
from __future__ import annotations

import ast
import math
import operator

from .errors import SMLQError

BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow}
FUNCTIONS = {"min": min, "max": max, "sqrt": math.sqrt, "abs": abs, "degrees": math.degrees,
             "radians": math.radians, "rsin": math.sin, "rcos": math.cos, "rtan": math.tan,
             "tan": lambda d: math.tan(math.radians(d)), "cos": lambda d: math.cos(math.radians(d)),
             "sin": lambda d: math.sin(math.radians(d))}
CONSTANTS = {"pi": math.pi}


def names_in(expr) -> set[str]:
    if isinstance(expr, (int, float)):
        return set()
    return ({n.id for n in ast.walk(ast.parse(str(expr), mode="eval")) if isinstance(n, ast.Name)}
            - set(FUNCTIONS) - set(CONSTANTS))


def evaluate(expr, env: dict[str, float]) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in env:
                return float(env[n.id])
            if n.id in CONSTANTS:
                return CONSTANTS[n.id]
            raise SMLQError(f"unknown name {n.id!r} in {expr!r}")
        if isinstance(n, ast.BinOp) and type(n.op) in BIN:
            return BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS:
            return float(FUNCTIONS[n.func.id](*[ev(a) for a in n.args]))
        raise SMLQError(f"not allowed in an expression: {ast.dump(n)[:60]}")

    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as e:
        raise SMLQError(f"bad expression {expr!r}: {e.msg}") from None
    return ev(tree)
