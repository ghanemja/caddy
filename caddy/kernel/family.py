"""Build a whole family: every catalogue row through one graph, each measured against its row."""
from __future__ import annotations

from dataclasses import dataclass

from .build import KernelError
from .catalogue import rows as catalogue_rows, size_name, values_for
from .digest import digest
from .graph import Evaluator, GraphError, Result, load_spec
from .measure import required_aliases, run_checks, smoke


@dataclass
class SizeReport:
    size: str
    row: dict
    result: Result | None
    checks: list[dict]
    smoke: list[dict]
    error: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(c["ok"] for c in self.checks) and all(s["ok"] for s in self.smoke)

    def digest(self) -> str:
        if self.result is None:
            return f"size {self.size}: BUILD FAILED {self.error}"
        return digest(self.result, self.checks, self.smoke, self.size)


def build_size(ev: Evaluator, g: dict, row: dict, spec: dict | None = None) -> SizeReport:
    spec = spec or load_spec()
    size = size_name(row, spec.get("size_name"))
    try:
        res = ev.run(g, values_for(row))
    except KernelError as e:
        return SizeReport(size, row, None, [], [], error=e.facts())
    except GraphError as e:
        return SizeReport(size, row, None, [], [], error=dict(block=e.block, status="GRAPH_ERROR", message=str(e)))
    except ValueError as e:      # geometry the kernel layer refuses (e.g. skin sections out of order along z)
        return SizeReport(size, row, None, [], [], error=dict(block=None, status="BUILD_ERROR", message=str(e)))
    named = required_aliases(res, spec.get("required_aliases", []))
    named += [dict(id=f"body:{b}", ok=b in res.bodies) for b in spec.get("required_bodies", [])]
    checks = run_checks(res, row, spec["checks"]) if all(n["ok"] for n in named) else []
    return SizeReport(size, row, res, checks, named + smoke(res))


def build_family(g: dict, rows: list[dict] | None = None, ev: Evaluator | None = None,
                 spec: dict | None = None) -> list[SizeReport]:
    ev, spec = ev or Evaluator(), spec or load_spec()
    return [build_size(ev, g, r, spec) for r in (rows or catalogue_rows())]
