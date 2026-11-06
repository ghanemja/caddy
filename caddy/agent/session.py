"""A design session: the current graph, the kernel, and the tool implementations the model calls.

Tool results are text the model reads: a one-line-per-size family table plus the digest of the first failing size.
Pass/fail always comes from the kernel and the catalogue (VERIFY.md: the player doesn't referee).
"""
from __future__ import annotations

import copy
import hashlib
import json

import smlibquery

from caddy.kernel import graph as G
from caddy.kernel.catalogue import rows as catalogue_rows, values_for
from caddy.kernel.digest import describe_group, why as why_face
from caddy.kernel.family import SizeReport, build_family
from caddy.kernel.stability import static_stability
from caddy.library.stability import similar_friction


def graph_hash(g: dict | None) -> str | None:
    """Identity of a graph's geometry (script line numbers are display metadata, not part of it)."""
    if g is None:
        return None
    g = {"blocks": [{k: v for k, v in b.items() if k != "line"} for b in g["blocks"]],
         "parameters": [{k: v for k, v in p.items() if k != "line"} for p in g.get("parameters", [])],
         "bodies": g.get("bodies")}
    return hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()[:10]


class Session:
    def __init__(self, family: str = "bspt_nipple", graph: dict | None = None, rows: list[dict] | None = None):
        self.family = family
        self.spec = G.load_spec(family)
        self.rows = rows or catalogue_rows(family=family)
        self.row_names = set(values_for(self.rows[0])) | G.ROW_NAMES
        self.ev = G.Evaluator()
        self.graph: dict | None = None
        self.reports: list[SizeReport] = []
        self.last_stab: dict | None = None
        self.versions: list[dict] = []          # every graph the kernel ran, in order (for replay)
        self.locked: set[str] = set()           # blocks the engineer marked "keep this": edits to them are refused
        self.script: str | None = None          # the smlibquery script, when the model writes one
        if graph is not None:
            self._run(graph)

    # --- state -------------------------------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        return bool(self.reports) and all(r.ok for r in self.reports)

    def failing(self) -> list[SizeReport]:
        return [r for r in self.reports if not r.ok]

    def _run(self, g: dict) -> None:
        G.validate(g, self.row_names)
        self.graph = g
        self.reports = build_family(g, self.rows, self.ev, self.spec)
        self.versions.append(copy.deepcopy(g))

    def _size(self, size: str) -> SizeReport:
        key = size.replace('"', "").replace("×", "x").replace(" ", "").replace("Ø", "")
        for r in self.reports:
            if r.size.replace(" ", "").replace("Ø", "") == key:
                return r
        raise G.GraphError(f"no size {size!r}; sizes: {', '.join(r.size for r in self.reports)}")

    # --- what the model reads ----------------------------------------------------------------------------
    def family_table(self) -> str:
        n_ok = sum(r.ok for r in self.reports)
        lines = [f"kernel built {len(self.reports)} sizes: {n_ok} ✓ · {len(self.reports) - n_ok} ✗"]
        for r in self.reports:
            if r.error:
                lines.append(f"  {r.size:10} ✗ BUILD FAILED {json.dumps(r.error)}")
            elif r.ok:
                lines.append(f"  {r.size:10} ✓")
            else:
                bad = [c["id"] for c in r.checks if not c["ok"]] + [s["id"] for s in r.smoke if not s["ok"]]
                lines.append(f"  {r.size:10} ✗ {', '.join(bad)}")
        return "\n".join(lines)

    def report(self) -> str:
        out = [self.family_table()]
        bad = self.failing()
        if bad:
            first = next((r for r in bad if r.result is not None), bad[0])
            out += ["", f"digest of {first.size}:", first.digest()]
        elif self.reports:
            out += ["", f"digest of {self.reports[0].size}:", self.reports[0].digest()]
        return "\n".join(out)

    # --- tools -------------------------------------------------------------------------------------------
    def build_graph(self, graph: dict) -> str:
        if self.locked and self.graph is not None:
            old = {b["id"]: b for b in self.graph["blocks"]}
            new = {b["id"]: b for b in graph.get("blocks", [])}
            hit = {k for k in self.locked if k in old and old[k] != new.get(k)}
            if hit:
                raise G.GraphError(f"blocks {sorted(hit)} are locked by the engineer (keep this); keep them unchanged")
        self._run(graph)
        return self.report()

    # --- smlibquery scripts: the language the model writes ------------------------------------------------
    def compile(self, code: str) -> dict:
        """smlibquery script -> feature graph, with catalogue values as tracked numbers (expressions kept)."""
        try:
            g = smlibquery.compile_script(code, values_for(self.rows[0]), family=self.spec.get("family", self.family),
                                          title=self.spec.get("title", ""))
        except smlibquery.SMLQError as e:
            raise G.GraphError(f"script: {e}") from None
        return g

    def _check_locks(self, g: dict) -> None:
        if self.locked and self.graph is not None:
            old = {b["id"]: b for b in self.graph["blocks"]}
            new = {b["id"]: b for b in g["blocks"]}
            hit = {k for k in self.locked if k in old and old[k] != new.get(k)}
            if hit:
                raise G.GraphError(f"blocks {sorted(hit)} are locked by the engineer (keep this); keep them unchanged")

    def build_cad(self, code: str) -> str:
        g = self.compile(code)
        self._check_locks(g)
        self.script = code
        self._run(g)
        return self.report()

    def edit_cad(self, edits: list[dict]) -> str:
        """Small edits to the script: [{"old": exact text, "new": replacement}] (each `old` must occur once)."""
        if self.script is None:
            raise G.GraphError("no script yet: call build_cad first")
        code = self.script
        for e in edits:
            old, new = e.get("old", ""), e.get("new", "")
            n = code.count(old) if old else 0
            if n != 1:
                raise G.GraphError(f"edit 'old' text must occur exactly once in the script (found {n}): {old[:60]!r}")
            code = code.replace(old, new)
        g = self.compile(code)
        self._check_locks(g)
        self.script = code
        self._run(g)
        rebuilt = sorted({b for r in self.reports if r.result for b in r.result.rebuilt})
        return f"rebuilt blocks: {', '.join(rebuilt) or 'none (all cached)'}\n" + self.report()

    def touched(self, ops: list[dict]) -> set[str]:
        """Blocks a diff would change or remove."""
        out = set()
        for op in ops:
            out.update(x for x in (op.get("block"), op.get("remove")) if x)
        return out

    def edit_graph(self, ops: list[dict]) -> str:
        if self.graph is None:
            raise G.GraphError("no graph yet: call build_graph first")
        hit = self.touched(ops) & self.locked
        if hit:
            raise G.GraphError(f"blocks {sorted(hit)} are locked by the engineer (keep this); change something else")
        g2 = G.apply_edit(self.graph, ops, self.row_names)
        self._run(g2)
        rebuilt = sorted({b for r in self.reports if r.result for b in r.result.rebuilt})
        return f"rebuilt blocks: {', '.join(rebuilt) or 'none (all cached)'}\n" + self.report()

    def measure(self, size: str, face: str) -> str:
        r = self._size(size)
        if r.result is None:
            return f"{r.size} did not build: {json.dumps(r.error)}"
        labels = r.result.by_alias(face)
        if not labels:
            return f"no face named {face!r} in {r.size}; names: {sorted({f.alias.split('[')[0] for f in r.result.faces})}"
        return describe_group(r.result, face, labels)

    def stability(self, size: str | None = None, body: str | None = None, mu: float = 0.4) -> str:
        """Read-only: how one body stands on the current build (tip and slide angles, exact on SMLib) and what real
        objects with the same friction did in the library. Never changes the build or the verdict."""
        r = self._size(size) if size else next((x for x in self.reports if x.result), None)
        if r is None or r.result is None:
            raise G.GraphError("nothing built yet: call build_cad first")
        bodies = r.result.bodies
        name = body or (next(iter(bodies)) if len(bodies) == 1 else None)
        if name not in bodies:
            raise G.GraphError(f"no body {body!r} in {r.size}; bodies: {sorted(bodies)}")
        st = static_stability(bodies[name], mu=mu)
        lib = similar_friction(mu)
        self.last_stab = dict(size=r.size, body=name, **st.as_dict(), library=lib)
        ex = ", ".join(f"{e['name']} {e['slide_deg']}°" for e in lib["examples"][:3])
        return (f"{r.size} · {name}: centre of mass {st.height:.1f} mm above the floor. It tips over its nearest edge "
                f"at {st.tip_deg:.1f}° (the edge is {st.distance:.1f} mm from the centre of mass). At μ {mu:g} it slides "
                f"at {st.slip_deg:.1f}°. First to go: {st.first}. Library, objects with μ≈{mu:g} (n={lib['n']}): "
                f"slid at a median {lib['library_median_slide_deg']}° (10–90%: {lib['library_range_deg'][0]}–"
                f"{lib['library_range_deg'][1]}°), e.g. {ex}.")

    def why(self, size: str, face: str) -> str:
        r = self._size(size)
        return why_face(r.result, face) if r.result else f"{r.size} did not build"

    def explain(self) -> dict:
        """What the whiteboard draws for a step (never sent to the model): every check with its measured and target
        value and how many configurations pass it, the kernel's build failure with its reason, and a compact
        feature list so consecutive steps can be diffed (added, changed, removed)."""
        import hashlib
        per = {}
        for r in self.reports:
            for c in r.checks:
                k = per.setdefault(c["id"], dict(id=c["id"], ok=0, n=0, rule=c.get("rule"), op=c.get("op", "==")))
                k["ok"] += bool(c["ok"]); k["n"] += 1
                if not c["ok"] and "measured" not in k:      # the first failing configuration explains the ✗
                    k.update(measured=c["measured"], expected=c["expected"], size=r.size)
        if self.reports:
            for c in self.reports[0].checks:                  # otherwise show the first configuration's numbers
                per[c["id"]].setdefault("measured", c["measured"]); per[c["id"]].setdefault("expected", c["expected"])
        err = next((dict(size=r.size, **r.error) for r in self.reports if r.error), None)
        blocks = [[b["id"], b["op"], b["part"], hashlib.sha1(json.dumps(b.get("params", {}), sort_keys=True).encode()).hexdigest()[:8]]
                  for b in (self.graph or {}).get("blocks", [])]
        return dict(checks=list(per.values()), error=err, blocks=blocks)

    def verdict(self) -> dict:
        """The kernel's verdict on the current graph: what 'done' is measured against."""
        return dict(ok=self.ok, sizes=len(self.reports), failing=[r.size for r in self.failing()],
                    failing_checks=sorted({c["id"] for r in self.failing() for c in r.checks if not c["ok"]}
                                          | {s["id"] for r in self.failing() for s in r.smoke if not s["ok"]}
                                          | ({"build"} if any(r.error for r in self.reports) else set())),
                    graph=graph_hash(self.graph))
