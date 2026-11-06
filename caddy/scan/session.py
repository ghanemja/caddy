"""A caddy session whose spec is a scan: the same build → check → repair loop, judged against the scanned part."""
from __future__ import annotations

from caddy.agent.session import Session
from caddy.kernel import graph as G

from .check import Scan, describe


class ScanSession(Session):
    def __init__(self, scan: Scan, title: str = "Scanned part", tol: float = 0.1, material: dict | None = None):
        self.family = "scan"
        self.scan, self.tol = scan, tol
        self.spec = dict(family="scan", title=title, size_name="scan", required_aliases=[],
                         material=material or {"name": "steel", "density_g_cm3": 7.85},
                         checks=[dict(id="matches_scan", measure="scan_deviation", scan=scan.id, op="<=",
                                      expect=str(tol), tol=0.0,
                                      rule=f"every point of the scan lies within {tol:g} mm of the model"),
                                 dict(id="no_extra_material", measure="scan_extra", scan=scan.id, op="<=",
                                      expect=str(tol), tol=0.0,
                                      rule=f"no surface of the model is more than {tol:g} mm from the scan")])
        self.rows = [dict(size="scan")]
        self.row_names = set(G.ROW_NAMES)
        self.ev = G.Evaluator()
        self.graph = None
        self.reports = []
        self.versions = []
        self.locked = set()
        self.script = None

    def report(self) -> str:
        out = super().report()
        r = self.reports[0] if self.reports else None
        if r is not None and r.result is not None:
            out += "\n\n" + describe(r.result, self.scan, self.tol)
        return out
