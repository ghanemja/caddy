"""Kernel operations: re-exported from smlibquery (caddy's CadQuery-style API on NVIDIA SMLib), plus caddy's
Block record. One copy of the geometry code, shared by the standalone package and caddy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import smlibquery.kernel as K
from smlibquery.kernel import BOOLEANS, MOVE_KEYS, MOVE_OPS, PARAM_KEYS, PRIMITIVE_OPS, apply_move, load_stock  # noqa: F401

STOCK_DIR = Path(__file__).resolve().parents[2] / "data" / "stock"
if STOCK_DIR not in K.STOCK_PATHS:
    K.STOCK_PATHS.append(STOCK_DIR)


@dataclass
class Block:
    id: str
    op: str
    part: str
    role: str
    params: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    stock: str = ""

    def to_dict(self):
        return asdict(self)


def make_primitive(blk: Block):
    return K.make_primitive(blk.op, blk.params, blk.stock)
