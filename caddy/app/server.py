"""caddy web app: one page (chat · viewer · board · parts/sliders · size table), FastAPI + vanilla JS + three.js.

Modes for the agent: "live" (Claude via the API; needs ANTHROPIC_API_KEY) and "replay" (streams a logged run
through the real kernel with no model call, so the page works without a key).
Sliders never call the AI: they rebuild one size on the kernel and re-measure it.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import usd_brep as u
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from caddy.agent.loop import Agent, catalogue_task
from caddy.agent.session import Session, graph_hash
from caddy.kernel import graph as G
from caddy.kernel.build import tessellate
from caddy.kernel.catalogue import DATA
from caddy.kernel.family import build_size
from caddy.app import intent, scripting
from caddy.kernel.annotate import dimensions
from caddy.solidcraft.mcp import McpError, find_cli

STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "docs" / "runs"
FAMILIES = {   # id -> title, recorded run to replay
    "bspt_nipple": dict(title="BSPT reducing nipple", replay="nipple_cq_replay.jsonl"),
    "gripper": dict(title="SO-101 rod gripper (assembly)", replay="gripper_repair.jsonl"),
    "roadster": dict(title="Roadster-style coupe (from photos)", replay="roadster.jsonl", script="roadster.sq.py"),
    "flying_car": dict(title="Flying-car concept (from the coupe)", replay="flying_car.jsonl", script="flying_car.sq.py",
                       start_from="roadster.sq.py"),   # Build with Claude turns the coupe into the flying car
    "scan_nipple": dict(title="Scan → CAD: hex nipple", replay="scan_nipple.jsonl", scan=True),
    "scan_bushing": dict(title="Scan → CAD: flanged bushing", replay="scan_bushing.jsonl", scan=True),
}


def is_scan(fam: str) -> bool:
    return bool(FAMILIES.get(fam, {}).get("scan"))


def is_image(fam: str) -> bool:
    return fam == "image"


def new_session(fam: str, graph: dict | None = None):
    if is_image(fam):      # an image session needs its picture: it is only ever made by /api/image/start
        raise HTTPException(400, "image → CAD starts from a picture: drop one on New")
    if is_scan(fam):
        from caddy.scan.scans import new_session as scan_session
        return scan_session(fam)[0]
    return Session(family=fam, graph=graph)

app = FastAPI(title="caddy")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Hub:
    """Shared state: the session, the event log the page streams, and the diff-approval gate."""

    def __init__(self):
        self.lock = threading.RLock()
        self.family = "bspt_nipple"
        self.reset()

    def reset(self, family: str | None = None, session=None):
        """A fresh session for `family`. `session` hands one in ready-made (image → CAD: it owns a SolidCraft
        process, so it is built by the caller and closed here when it is replaced)."""
        with self.lock:
            old = getattr(self, "session", None)
            self.family = family or self.family
            self.session = session if session is not None else new_session(self.family)
            if old is not None and old is not self.session and hasattr(old, "close"):
                old.close()
            self.events: list[dict] = []
            self.running = False
            self.mesh_cache: dict[tuple, dict] = {}
            self.approval = False
            self.pending: dict | None = None
            self.decision = threading.Event()
            self.undo: list[str] = []          # scripts before each edit made on the board
            self.last_edit: dict | None = None
            self.dims_cache: dict[tuple, dict] = {}
            self.approved = True

    def emit(self, ev: dict):
        with self.lock:
            ev = {**ev, "seq": len(self.events)}
            self.events.append(ev)


HUB = Hub()


# --- state for the page ----------------------------------------------------------------------------------

def size_rows(s: Session) -> list[dict]:
    out = []
    for r in s.reports:
        out.append(dict(size=r.size, ok=r.ok, error=r.error,
                        checks=[{k: c[k] for k in ("id", "ok", "measured", "expected", "delta", "part", "faces",
                                                   "source", "column", "rule", "op")} for c in r.checks],
                        smoke=[s_ for s_ in r.smoke if not s_["ok"] or s_.get("warn")]))
    return out


def block_status(s: Session) -> list[dict]:
    """Per block: part, op, inputs, and whether it is implicated in a failure on any size."""
    if not s.graph:
        return []
    failing_faces = {}
    for r in s.reports:
        if r.result is None:
            continue
        for c in r.checks:
            if not c["ok"]:
                for a in c["faces"]:
                    failing_faces.setdefault(a, set()).add(r.size)
    bad_blocks: dict[str, set] = {}
    for r in s.reports:
        if r.result is None:
            if r.error and r.error.get("block"):
                bad_blocks.setdefault(r.error["block"], set()).add(r.size)
            continue
        for f in r.result.faces:
            if f.alias in failing_faces:
                bad_blocks.setdefault(f.block, set()).update(failing_faces[f.alias])
        for st in r.result.steps:
            if not st["feature_exists"] and not st.get("optional"):
                bad_blocks.setdefault(st["block"], set()).add(r.size)
    out = []
    for b in s.graph["blocks"]:
        out.append(dict(id=b["id"], op=b["op"], part=b["part"], role=b["role"], inputs=b.get("inputs", []), line=b.get("line"),
                        params=b.get("params", {}), optional=b.get("optional", False),
                        failing=sorted(bad_blocks.get(b["id"], set()))))
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    with HUB.lock:
        s = HUB.session
        g = s.graph
        blocks = block_status(s)
        tree = intent.feature_tree(s, blocks)
        order = sorted(blocks, key=lambda b: b["op"] in intent.BOOKKEEPING)
        by_block, values = scripting.attach(s.script, {b["id"]: b.get("line") for b in order})
        for p in tree:
            for f in p["features"] + p["bookkeeping"]:
                f["fields"] = by_block.get(f["id"], [])
        scan = dict(steps=s.scan_steps, tol=s.tol) if is_scan(HUB.family) else None
        image = dict(hint=s.hint, declared=s.declared, parameters=s.sc_parameters,
                     saved={k: v.name for k, v in s.saved.items()}, dead=s.facts.get("dead", []),
                     overlap=s.facts.get("overlap")) if is_image(HUB.family) else None
        return dict(values=values, last_edit=HUB.last_edit, can_undo=bool(HUB.undo), scan=scan, image=image,
            solidcraft=bool(find_cli()),
            graph=g, graph_hash=graph_hash(g), versions=len(s.versions),
            parameters=(g or {}).get("parameters", []), blocks=blocks,
            tree=tree, constraints=intent.constraints(s),
            parts=sorted({b["part"] for b in (g or {}).get("blocks", [])}, key=_part_order),
            sizes=size_rows(s), check_ids=[c["id"] for c in s.spec["checks"]],
            spec=s.spec, verdict=s.verdict() if s.reports else None,
            running=HUB.running, approval=HUB.approval, pending=HUB.pending, locked=sorted(s.locked),
            script=s.script, lang="cadquery" if s.script else ("json" if g else None),
            rows=[dict(r) for r in s.rows], family=HUB.family,
            families=[dict(id=k, title=v["title"], replay=(RUNS_DIR / v["replay"]).exists()) for k, v in FAMILIES.items()],
            key=bool(os.environ.get("ANTHROPIC_API_KEY") or (ROOT / ".env").exists()))


def _part_order(p):
    order = ["inlet", "hex", "outlet", "bore"]
    return (order.index(p) if p in order else 99, p)


def mesh_payload(res) -> dict:
    m = tessellate(res)
    faces = [dict(alias=f.alias, part=f.part, block=f.block, kind=f.kind, area=round(f.area, 3), body=f.body)
             for f in res.faces]
    boxes = [b.bounding_box(tight=True) for b in res.bodies.values()]
    bbox = [[min(b[0][i] for b in boxes) for i in range(3)], [max(b[1][i] for b in boxes) for i in range(3)]]
    return dict(points=np.round(m["points"], 4).ravel().tolist(), tris=m["tris"].ravel().tolist(),
                tri_face=m["tri_face"].tolist(), faces=faces,
                bbox=bbox)


@app.get("/api/mesh/{i}")
def mesh(i: int):
    with HUB.lock:
        s = HUB.session
        if not (0 <= i < len(s.reports)) or s.reports[i].result is None:
            raise HTTPException(404, "no such built size")
        key = (graph_hash(s.graph), i)
        if key not in HUB.mesh_cache:
            HUB.mesh_cache[key] = mesh_payload(s.reports[i].result)
        return JSONResponse(HUB.mesh_cache[key])


class Preview(BaseModel):
    size: int
    values: dict[str, float]


@app.post("/api/preview")
def preview(p: Preview):
    """Slider move: rebuild one size with overridden row values and re-measure. No AI."""
    with HUB.lock:
        s = HUB.session
        if s.graph is None:
            raise HTTPException(400, "no graph yet")
        row = dict(s.rows[p.size])
        params = {q["name"] for q in s.graph.get("parameters", [])}
        row.update({k: float(v) for k, v in p.values.items() if k in row or k in params})
        if is_image(HUB.family) and HUB.running:      # the what-if would run inside the LLM's SolidCraft session
            raise HTTPException(409, "The LLM is working; try again when the run ends")
        rep = s.trial(p.values) if is_image(HUB.family) else build_size(s.ev, s.graph, row, s.spec)
        HUB.last_preview = ((graph_hash(s.graph), p.size, tuple(sorted(p.values.items()))), rep)
        if rep.result is None:
            return dict(ok=False, error=rep.error)
        out = mesh_payload(rep.result)
        out.update(ok=rep.ok, checks=[{k: c[k] for k in ("id", "ok", "measured", "expected", "part", "op")}
                                      for c in rep.checks], rebuilt=rep.result.rebuilt)
        return out


@app.post("/api/predict")
def predict(p: Preview):
    """A slider change broke a check: predict which other design value fixes it, then confirm on the kernel.

    Cheap first, like a draft render before the full one: one probe build per candidate value gives the slope of
    the broken measurement, a linear step predicts the value that restores it, and a single full build verifies
    the whole proposal against every check. No AI."""
    import math, re, time
    t0 = time.time()
    with HUB.lock:
        s = HUB.session
        if s.graph is None:
            raise HTTPException(400, "no graph yet")
        if is_image(HUB.family) and HUB.running:
            raise HTTPException(409, "The LLM is working; try again when the run ends")
        base_row = dict(s.rows[p.size])
        design = {q["name"]: float(q["default"]) for q in s.graph.get("parameters", []) if isinstance(q.get("default"), (int, float))}
        cur = {**design, **{k: float(v) for k, v in base_row.items() if isinstance(v, (int, float))}}

        def run(vals):
            if is_image(HUB.family):       # SolidCraft rebuilds the what-if; SMLib judges it; the design is put back
                return s.trial(vals)
            row = dict(base_row); row.update({k: float(v) for k, v in vals.items()})
            return build_size(s.ev, s.graph, row, s.spec)

        before = {c["id"]: c["ok"] for c in s.reports[p.size].checks}
        tb = time.time()
        key, rep = getattr(HUB, "last_preview", (None, None))
        if key != (graph_hash(s.graph), p.size, tuple(sorted(p.values.items()))):
            rep = run(p.values)      # the slider preview usually built exactly this already
        t_build = max(time.time() - tb, 0.05)
        if rep.result is None:
            return dict(ok=False, error="the kernel could not build that value", fixes=[])
        broken = [c for c in rep.checks if not c["ok"] and before.get(c["id"], False) and c.get("measured") is not None]
        if not broken:
            return dict(ok=True, broken=[], fixes=[], seconds=round(time.time() - t0, 1))
        # free prediction from the two builds we already have (before and after the change): where each broken
        # check crosses its limit, if one value moved
        base_m = {c["id"]: c.get("measured") for c in s.reports[p.size].checks}
        limits = []
        if len(p.values) == 1:
            (pn, pv), = p.values.items()
            pv0 = cur.get(pn)
            for c in broken:
                m0_, m1_ = base_m.get(c["id"]), c["measured"]
                if pv0 is None or m0_ is None or abs(m1_ - m0_) < 1e-9 or c["op"] not in (">=", "<="):
                    continue
                lim = pv0 + (c["expected"] - m0_) * (pv - pv0) / (m1_ - m0_)
                if math.isfinite(lim):
                    limits.append(dict(check=c["id"], param=pn, limit=float(f"{lim:.4g}"), side="min" if lim < pv0 else "max"))
            # the straight line is a guess (a gap stops at 0, for one): confirm a value just inside it on the kernel,
            # halving back toward the old value while it still breaks something, within a small time budget
            if limits and pv0 is not None:
                def passes(v):
                    r = run({pn: v})
                    return r.result is not None and not any(not c["ok"] and before.get(c["id"], False) for c in r.checks)
                q = 0.1 if max(abs(pv), abs(pv0)) >= 10 else 0.01
                snap = lambda v: round((math.floor(v / q) if pv > pv0 else math.ceil(v / q)) * q, 4)   # round toward the old value
                lim = min((l["limit"] for l in limits), key=lambda x: abs(x - pv0))
                lo, hi = pv0, pv
                cand = snap(lim - (lim - pv0) * 0.02) if (lim - pv0) * (pv - pv0) > 0 else snap((pv0 + pv) / 2)
                if (cand - pv0) * (pv - cand) <= 0:
                    cand = snap((pv0 + pv) / 2)
                safe = None
                for _ in range(5):
                    if cand == lo or time.time() - t0 + t_build > 60:
                        break
                    tb = time.time(); good = passes(cand); t_build = max(t_build, time.time() - tb)
                    if good:
                        safe = cand; break
                    hi = cand; cand = snap((lo + hi) / 2)
                for l in limits:
                    l["safe"] = safe
        target = next((c for c in broken if c["op"] != "=="), broken[0])
        want = target["expected"] + {">=": 1, "<=": -1}.get(target["op"], 0) * max(target.get("tol", 0.01) * 2, abs(target["expected"]) * 0.02)
        # candidates: other design values feeding blocks of the broken check's part
        uses = {}
        for b in s.graph["blocks"]:
            for e in (b.get("params") or {}).values():
                for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(e)):
                    uses.setdefault(n, set()).add(b.get("part"))
        cands = [n for n in design if n not in p.values and target.get("part") in uses.get(n, ())]
        cands += [n for n in design if n not in p.values and n not in cands and uses.get(n)]
        m0 = target["measured"]
        fixes = []
        for n in cands[:3]:
            if len(fixes) >= 2 or time.time() - t0 + 2 * t_build > 60:
                break
            v0 = cur[n]; h = max(abs(v0) * 0.05, 0.5 if abs(v0) >= 5 else 0.01)
            probe = run({**p.values, n: v0 + h})
            pc = next((c for c in probe.checks if c["id"] == target["id"]), None) if probe.result else None
            if not pc or pc.get("measured") is None or abs(pc["measured"] - m0) < 1e-9:
                continue
            slope = (pc["measured"] - m0) / h
            v = v0 + (want - m0) / slope
            if not math.isfinite(v) or (v0 > 0 and v <= 0):
                continue
            v = float(f"{v:.4g}")
            ver = run({**p.values, n: v})
            if ver.result is None:
                continue
            bad = [c["id"] for c in ver.checks if not c["ok"]]
            if len(bad) < len([c for c in rep.checks if not c["ok"]]):      # only keep a guess that actually helps
                fixes.append(dict(param=n, now=v0, to=v, verified=ver.ok, still_failing=bad,
                                  measured=next((c["measured"] for c in ver.checks if c["id"] == target["id"]), None)))
            if ver.ok:
                break
        return dict(ok=True, broken=[{k: c.get(k) for k in ("id", "measured", "expected", "op", "part")} for c in broken],
                    fixes=fixes, limits=limits, seconds=round(time.time() - t0, 1))


@app.get("/api/digest/{i}", response_class=PlainTextResponse)
def digest(i: int):
    with HUB.lock:
        return HUB.session.reports[i].digest()


@app.get("/api/export/{i}")
def export_usd(i: int):
    with HUB.lock:
        r = HUB.session.reports[i]
        if r.result is None:
            raise HTTPException(404)
        path = Path(tempfile.gettempdir()) / f"caddy_{r.size.replace('/', '-').replace(' ', '')}.usda"
        u.usd.export_breps([b.copy() for b in r.result.bodies.values()], str(path))
    return FileResponse(path, filename=path.name, media_type="text/plain")


# --- agent -----------------------------------------------------------------------------------------------

class Start(BaseModel):
    mode: str = "live"          # live | replay
    family: str | None = None   # switch family (starts fresh)
    task: str | None = None
    approval: bool = False
    speed: float = 1.0          # replay speed-up
    keep: bool = False          # continue from the current graph (a follow-up request)
    refs: list[dict] = []       # pieces of the workflow the engineer pointed at (dragged into the chat or a sticky)


class ApprovingSession(Session):
    """Session whose edits wait for the engineer's approval when the gate is on."""

    def edit_cad(self, edits):
        if HUB.approval and not self._approve(dict(edits=edits)):
            return "the engineer rejected this edit; propose a different fix\n" + self.family_table()
        return super().edit_cad(edits)

    def _approve(self, proposal) -> bool:
        with HUB.lock:
            HUB.pending = proposal
            HUB.decision.clear()
        HUB.emit(dict(type="proposal", **proposal))
        HUB.decision.wait(timeout=300)
        with HUB.lock:
            HUB.pending = None
            ok = HUB.approved
        HUB.emit(dict(type="decision", approved=ok))
        return ok

    def edit_graph(self, ops):
        if HUB.approval:
            with HUB.lock:
                HUB.pending = dict(ops=ops)
                HUB.decision.clear()
            HUB.emit(dict(type="proposal", ops=ops))
            HUB.decision.wait(timeout=300)
            with HUB.lock:
                HUB.pending = None
                ok = HUB.approved
            HUB.emit(dict(type="decision", approved=ok))
            if not ok:
                return "the engineer rejected this edit; propose a different fix\n" + self.family_table()
        return super().edit_graph(ops)


def _run_live(task: str):
    try:
        lang = "cadquery" if (HUB.session.script or HUB.session.graph is None) else "json"
        a = Agent(HUB.session, on_event=HUB.emit, lang=lang)
        a.run(task)
    except Exception as e:  # surface API problems on the page
        HUB.emit(dict(type="error", message=f"{type(e).__name__}: {e}"))
    finally:
        HUB.running = False


def _scan_prelude(s) -> str:
    """The kernel-only part of scan → CAD, shown step by step: measuring the scan, then caddy's draft."""
    for st in s.scan_steps:
        HUB.emit(dict(type="scan_step", key=st["key"], title=st["title"], ms=st["ms"]))
        time.sleep(0.35)
    out = s.build_cad(s.scan_draft)
    HUB.emit(dict(type="tool", name="build_cad", input={"code": s.scan_draft}, error=False, reference=True,
                  result=out, kernel=s.verdict(), explain=s.explain()))
    return out


def _run_scan_live():
    """Scan → CAD with Claude: the measuring steps and caddy's draft first (kernel only), then Claude."""
    try:
        from caddy.scan import agent as scan_agent
        s = HUB.session
        out = _scan_prelude(s)
        a = Agent(s, on_event=HUB.emit, lang="cadquery", system=scan_agent.SYSTEM)
        a.run(scan_agent.task(s.scan_steps, s.scan_draft, out))
    except Exception as e:
        HUB.emit(dict(type="error", message=f"{type(e).__name__}: {e}"))
    finally:
        HUB.running = False


def _start_task(fam: str, code: str, out: str) -> str:
    """The task for a live run that starts from an existing design instead of a blank page."""
    from caddy.agent.loop import catalogue_task
    return ("A design is already built on the kernel: the script and its kernel report are below. Turn it into what "
            "the brief describes. Keep what already serves the brief (the skinned body, wheels and arches) and add "
            "the rest; edit it with edit_cad (small exact replacements; never resend the whole script). Finish when "
            "the kernel reports every check ✓.\n\n" + catalogue_task(fam)
            + f"\n\nThe current smlibquery script:\n```python\n{code}\n```\n\n" + out)


def _run_from(fam: str, code: str):
    """Build the starting design (kernel only), then run Claude on it."""
    try:
        out = HUB.session.build_cad(code)
        HUB.emit(dict(type="tool", name="build_cad", input={"code": code}, error=False, reference=True,
                      result=out, kernel=HUB.session.verdict(), explain=HUB.session.explain()))
        a = Agent(HUB.session, on_event=HUB.emit, lang="cadquery")
        a.run(_start_task(fam, code, out))
    except Exception as e:  # surface API problems on the page
        HUB.emit(dict(type="error", message=f"{type(e).__name__}: {e}"))
    finally:
        HUB.running = False


def _run_replay(path: Path, speed: float):
    """Re-run a logged agent session on the kernel, with the original pacing (sped up)."""
    try:
        events = [json.loads(line) for line in open(path)]
        if is_scan(HUB.family):
            _scan_prelude(HUB.session)
        start_from = FAMILIES[HUB.family].get("start_from")
        if start_from:            # the recorded run modified this design: build it first, as the live run did
            code = (DATA / "families" / start_from).read_text()
            out = HUB.session.build_cad(code)
            HUB.emit(dict(type="tool", name="build_cad", input={"code": code}, error=False, reference=True,
                          result=out, kernel=HUB.session.verdict(), explain=HUB.session.explain(), replayed=True))
        shim = Agent.__new__(Agent)
        shim.s = HUB.session
        last_t = events[0]["t"]
        for e in events:
            time.sleep(max(0.0, min(4.0, (e["t"] - last_t) / speed)))
            last_t = e["t"]
            if e["type"] == "tool" and e["name"] in ("build_graph", "edit_graph", "build_cad", "edit_cad", "finish", "measure", "why", "check_stability"):
                out, err = (HUB.session.edit_graph(e["input"]["ops"]), False) if e["name"] == "edit_graph" \
                    else Agent._call_tool(shim, e["name"], e["input"])
                e = {**e, "result": out, "error": err, "kernel": HUB.session.verdict(), "explain": HUB.session.explain(),
                     "replayed": True}
                if e["name"] == "check_stability" and not err:
                    e["stab"] = HUB.session.last_stab     # re-measured on the kernel, not taken from the log
            HUB.emit({**e, "replayed": True})
    except Exception as e:
        HUB.emit(dict(type="error", message=f"{type(e).__name__}: {e}"))
    finally:
        HUB.running = False


@app.post("/api/agent/start")
def start(p: Start):
    if HUB.running:
        raise HTTPException(409, "already running")
    fam = p.family if p.family in FAMILIES else HUB.family
    if is_image(fam):
        return _image_follow_up(p)
    keep = p.keep and HUB.session.graph is not None and fam == HUB.family
    prev = HUB.session.graph if keep else None
    prev_script = HUB.session.script if keep else None
    user_checks = [c for c in HUB.session.spec["checks"] if c.get("by") == "you"] if keep else []
    locked = set(HUB.session.locked) if keep else set()
    HUB.reset(fam)
    HUB.approval = p.approval
    log = RUNS_DIR / FAMILIES[fam]["replay"]
    if p.mode == "replay" and not log.exists():
        HUB.running = False
        raise HTTPException(404, "no recorded run for this family yet: use Build with LLM")
    if p.mode == "replay" and json.loads(open(log).readline()).get("preload_reference"):
        prev = G.load(fam)
    HUB.session = new_session(fam) if is_scan(fam) else ApprovingSession(family=fam, graph=prev)
    HUB.session.locked = locked
    if is_scan(fam) and prev_script:
        HUB.session.build_cad(prev_script)
    HUB.session.script = prev_script
    if user_checks:
        HUB.session.spec["checks"] += user_checks
        if prev is not None:
            HUB.session._run(prev)
    HUB.running = True
    if p.mode == "replay":
        threading.Thread(target=_run_replay, args=(log, p.speed), daemon=True).start()
    elif is_scan(fam) and not keep:
        threading.Thread(target=_run_scan_live, daemon=True).start()
    elif not keep and not p.task and FAMILIES[fam].get("start_from"):
        code = (DATA / "families" / FAMILIES[fam]["start_from"]).read_text()
        threading.Thread(target=_run_from, args=(fam, code), daemon=True).start()
    else:
        task = p.task or catalogue_task(fam)
        if keep:
            refs = "".join(f"\n- {describe_ref(HUB.session, r)}" for r in p.refs)
            task = ("The family graph is already built (the kernel report is below).\n"
                    f"The engineer says: {p.task}"
                    + (f"\nThey are pointing at:{refs}" if refs else "")
                    + (f"\nLocked by the engineer (do not change or remove): {', '.join(sorted(locked))}" if locked else "")
                    + ("\nConstraints the engineer added (the kernel checks them): "
                       + "; ".join(c.get("rule") or c["id"] for c in user_checks) if user_checks else "")
                    + (f"\nThe current smlibquery script:\n```python\n{prev_script}\n```" if prev_script else "")
                    + "\nIf this is a question, answer it in one to three short sentences (use measure / why if you "
                      "need numbers) and do not edit the graph. If it asks for a change, make it with " + ("edit_cad" if prev_script else "edit_graph") + " and "
                      "finish when the kernel reports every size ✓.\n\n" + HUB.session.report())
        threading.Thread(target=_run_live, args=(task,), daemon=True).start()
    return dict(ok=True)


# --- image → CAD (SolidCraft models, SMLib judges) -------------------------------------------------------

class ImageStart(BaseModel):
    image: str                  # base64, or a data: URL
    hint: str = ""              # what it is, known sizes, what the picture hides
    name: str = ""


def _run_image(task: str):
    try:
        from caddy.solidcraft.agent import ImageAgent
        ImageAgent(HUB.session, on_event=HUB.emit).run(task)
    except Exception as e:  # surface API and SolidCraft problems on the page
        HUB.emit(dict(type="error", message=f"{type(e).__name__}: {e}"))
    finally:
        HUB.running = False


@app.post("/api/image/start")
def image_start(p: ImageStart):
    """A picture in: Claude models it in SolidCraft (parametric), SMLib judges every step."""
    import base64
    import binascii
    from caddy.solidcraft.agent import task
    from caddy.solidcraft.session import ImageSession
    if HUB.running:
        raise HTTPException(409, "already running")
    try:
        data = base64.b64decode(p.image.split(",", 1)[1] if p.image.startswith("data:") else p.image, validate=True)
        name = Path(p.name).stem[:60] if p.name else "picture"
        s = ImageSession(data, hint=p.hint, title=f"Image → CAD: {name}")
    except (binascii.Error, ValueError, IndexError) as e:
        raise HTTPException(400, f"not a picture caddy can read: {e}")
    except McpError as e:
        raise HTTPException(503, str(e))
    HUB.reset("image", session=s)
    HUB.running = True
    threading.Thread(target=_run_image, args=(task(p.hint),), daemon=True).start()
    return dict(ok=True)


def _image_follow_up(p: Start):
    """A question or a change request about the image model: the same SolidCraft session, a new conversation."""
    s = HUB.session
    if p.mode == "replay":
        raise HTTPException(404, "an image → CAD run can't be replayed: it needs SolidCraft and the LLM")
    if not p.task:
        raise HTTPException(400, "image → CAD starts from a picture: drop one on New")
    HUB.reset("image", session=s)
    params = ", ".join(f"{q['name']} = {q['value']:g} {q['unit']}".strip() for q in s.sc_parameters
                       if isinstance(q.get("value"), (int, float)))
    task = ("The SolidCraft design for this picture is already built (caddy's kernel report is below).\n"
            f"The engineer says: {p.task}\n"
            f"Parameters: {params or 'none'}\n"
            "If this is a question, answer it in one to three short sentences (inspect_design / measure if you need "
            "numbers) and do not change the design. If it asks for a change, make it with small edits (set_parameter, "
            "timeline.edit, one more feature) and finish when caddy's kernel accepts it.\n\n" + s.report())
    HUB.running = True
    threading.Thread(target=_run_image, args=(task,), daemon=True).start()
    return dict(ok=True)


@app.get("/api/image/file/{kind}")
def image_file(kind: str):
    """The picture, or what an accepted run saved: SolidCraft's parametric design (.sc) and the STEP SMLib judged."""
    s = HUB.session
    if not is_image(HUB.family):
        raise HTTPException(404, "not an image → CAD model")
    path = s.reference if kind == "picture" else s.saved.get(kind)
    if path is None or not Path(path).exists():
        raise HTTPException(404, "nothing saved yet: the kernel has not accepted the model")
    return FileResponse(path, filename=Path(path).name)


def describe_ref(s: Session, r: dict) -> str:
    """Turn a pointed-at piece of the UI into exact facts from the kernel (never the UI's own text)."""
    kind = r.get("kind")
    size_i = r.get("size")
    rep = s.reports[size_i] if isinstance(size_i, int) and 0 <= size_i < len(s.reports) else None
    if kind == "block" and s.graph:
        b = next((b for b in s.graph["blocks"] if b["id"] == r.get("id")), None)
        if b:
            params = ", ".join(f"{k}={v}" for k, v in b.get("params", {}).items()) or f"inputs {b.get('inputs')}"
            return f"block {b['id']} ({b['op']}, part {b['part']}, role {b['role']}): {params}"
    if kind == "part" and s.graph:
        blocks = [b["id"] for b in s.graph["blocks"] if b["part"] == r.get("id")]
        bad = sorted({c["id"] for rp in s.reports for c in rp.checks if not c["ok"] and c["part"] == r.get("id")})
        return f"part {r.get('id')} (blocks {', '.join(blocks)})" + (f"; failing checks: {', '.join(bad)}" if bad else "")
    if kind == "face" and rep and rep.result:
        return f"face {r.get('id')} on size {rep.size}: " + s.why(rep.size, r.get("id")).split(": ", 1)[-1]
    if kind == "check" and rep:
        c = next((c for c in rep.checks if c["id"] == r.get("id")), None)
        if c:
            return (f"check {c['id']} on size {rep.size}: measured {c['measured']} {c['op']} expected "
                    f"{c['expected']} ({'pass' if c['ok'] else 'FAIL'}) on faces {', '.join(c['faces'])}")
    if kind == "size" and rep:
        bad = [c["id"] for c in rep.checks if not c["ok"]]
        return f"size {rep.size}: " + ("all checks pass" if not bad else "failing " + ", ".join(bad))
    return f"{kind} {r.get('id', '')}"


class Lock(BaseModel):
    kind: str          # block | part
    id: str
    on: bool = True


@app.post("/api/lock")
def lock(l: Lock):
    """'Keep this' sticker: lock a block (or every block of a part). Enforced by the session, not the model."""
    with HUB.lock:
        s = HUB.session
        if not s.graph:
            raise HTTPException(400, "no graph yet")
        ids = [l.id] if l.kind == "block" else [b["id"] for b in s.graph["blocks"] if b["part"] == l.id]
        (s.locked.update if l.on else s.locked.difference_update)(ids)
        return dict(locked=sorted(s.locked))


class Words(BaseModel):
    text: str


class CheckIn(BaseModel):
    check: dict


class CheckId(BaseModel):
    id: str


@app.post("/api/constraints/propose")
def propose_constraint(w: Words):
    """Claude turns a sentence into one kernel-measurable constraint; caddy previews it on every configuration."""
    if HUB.running:
        raise HTTPException(409, "agent running")
    import anthropic
    from caddy.agent.loop import load_env
    load_env()
    try:
        out = intent.propose(HUB.session, w.text, anthropic.Anthropic())
    except G.GraphError as e:
        raise HTTPException(400, str(e))
    HUB.emit(dict(type="constraint_proposed", text=w.text, check=out["check"], preview=out.get("preview"),
                  cannot=out.get("cannot")))
    return out


@app.post("/api/constraints/approve")
def approve_constraint(c: CheckIn):
    with HUB.lock:
        try:
            intent.approve(HUB.session, c.check)
        except G.GraphError as e:
            raise HTTPException(400, str(e))
        HUB.mesh_cache.clear()
    HUB.emit(dict(type="constraint_added", check=c.check, kernel=HUB.session.verdict()))
    return dict(ok=True)


@app.post("/api/constraints/remove")
def remove_constraint(c: CheckId):
    with HUB.lock:
        try:
            intent.remove(HUB.session, c.id)
        except G.GraphError as e:
            raise HTTPException(400, str(e))
    HUB.emit(dict(type="constraint_removed", id=c.id, kernel=HUB.session.verdict()))
    return dict(ok=True)


# --- visual scripting: the board edits the script; only changed blocks rebuild (no AI) ----------------------

class ScriptEdit(BaseModel):
    line: int
    start: int
    end: int
    text: str
    expect: str | None = None


def _counts(s: Session) -> dict:
    out: dict[str, list] = {}
    for r in s.reports:
        for c in r.checks:
            o = out.setdefault(c["id"], [0, 0])
            o[0] += c["ok"]
            o[1] += 1
    return out


def _apply_script(code: str, what: dict) -> dict:
    s = HUB.session
    before = _counts(s)
    t0 = time.time()
    s.build_cad(code)
    ms = round((time.time() - t0) * 1000)
    rebuilt = sorted({b for r in s.reports if r.result for b in r.result.rebuilt})
    after = _counts(s)
    changed = [dict(id=k, before=before.get(k), after=v) for k, v in after.items() if before.get(k) != v]
    HUB.last_edit = dict(what, ms=ms, rebuilt=rebuilt, checks=changed, at=time.time())
    return HUB.last_edit


@app.post("/api/script/edit")
def script_edit(e: ScriptEdit):
    with HUB.lock:
        s = HUB.session
        if HUB.running:
            raise HTTPException(409, "The LLM is working; try again when the run ends")
        if not s.script:
            raise HTTPException(400, "no script: run Build with LLM or Replay first")
        try:
            code, old, new = scripting.splice(s.script, e.line, e.start, e.end, e.text.strip(), e.expect)
        except ValueError as err:
            raise HTTPException(400, str(err))
        if code == s.script:
            return dict(ok=True, unchanged=True)
        prev = s.script
        try:
            out = _apply_script(code, dict(line=e.line, old=old, new=new))
        except G.GraphError as err:
            raise HTTPException(400, str(err).replace("script: ", ""))
        HUB.undo.append(prev)
    HUB.emit(dict(type="board_edit", **out, kernel=HUB.session.verdict()))
    return dict(ok=True, **out)


class ParamSet(BaseModel):
    values: dict[str, float]


@app.post("/api/param/set")
def param_set(p: ParamSet):
    """Apply design values to a model that has no script (a hand-written graph such as the gripper sample): each one
    becomes its parameter's new default and every size is rebuilt and re-measured. Catalogue columns are not
    parameters here; they stay preview-only (change the CSV to keep them)."""
    with HUB.lock:
        s = HUB.session
        if HUB.running:
            raise HTTPException(409, "The LLM is working; try again when the run ends")
        if s.graph is None:
            raise HTTPException(400, "no graph yet")
        if s.script:
            raise HTTPException(400, "this model is a script: edit its values with /api/script/edit")
        design = {q["name"]: q for q in s.graph.get("parameters", []) if isinstance(q.get("default"), (int, float))}
        bad = sorted(set(p.values) - set(design))
        if bad:
            raise HTTPException(400, f"not design values: {', '.join(bad)}")
        old = {k: design[k]["default"] for k in p.values}
        before = _counts(s)
        t0 = time.time()
        try:
            if is_image(HUB.family):      # the parameters live in SolidCraft: it rebuilds, SMLib judges
                s.set_values(p.values)
                HUB.mesh_cache.clear(); HUB.dims_cache.clear()
            else:
                s.edit_graph([{"param": k, "default": float(v)} for k, v in p.values.items()])
        except (G.GraphError, McpError) as err:
            raise HTTPException(400, str(err))
        rebuilt = sorted({b for r in s.reports if r.result for b in r.result.rebuilt})
        after = _counts(s)
        changed = [dict(id=k, before=before.get(k), after=v) for k, v in after.items() if before.get(k) != v]
        what = ", ".join(f"{k} = {old[k]:g}" for k in p.values)
        now = ", ".join(f"{k} = {float(v):g}" for k, v in p.values.items())
        HUB.last_edit = dict(line=None, old=what, new=now, ms=round((time.time() - t0) * 1000), rebuilt=rebuilt,
                             checks=changed, at=time.time())
        out = HUB.last_edit
    HUB.emit(dict(type="board_edit", **out, kernel=HUB.session.verdict()))
    return dict(ok=True, **out)


@app.post("/api/script/undo")
def script_undo():
    with HUB.lock:
        if not HUB.undo or HUB.running:
            raise HTTPException(400, "nothing to undo")
        s = HUB.session
        cur = s.script.split("\n")
        prev = HUB.undo.pop()
        line = next((i + 1 for i, (a, b) in enumerate(zip(cur, prev.split("\n"))) if a != b), None)
        out = _apply_script(prev, dict(line=line, old=cur[line - 1] if line else "", new=prev.split("\n")[line - 1] if line else "", undo=True))
    HUB.emit(dict(type="board_edit", **out, kernel=HUB.session.verdict()))
    return dict(ok=True, **out)


@app.get("/api/scan/mesh")
def scan_mesh(which: str = "model"):
    """The scan (axis frame) and every scan vertex's exact distance (kernel) to the current model, or to caddy's
    draft (which=draft) for the before/after."""
    with HUB.lock:
        s = HUB.session
        if not is_scan(HUB.family):
            raise HTTPException(404, "not a scan family")
        key = ("scan", which, graph_hash(s.graph) if which == "model" else "draft")
        if key not in HUB.mesh_cache:
            V = np.asarray(s.scan.mesh.vertices, float)
            T = np.asarray(s.scan.mesh.triangles)
            dev = None
            if which == "draft":
                brep = s.ev.run(s.compile(s.scan_draft), {}).brep
            else:
                r = s.reports[0] if s.reports else None
                brep = r.result.brep if r is not None and r.result is not None else None
            if brep is not None:   # per triangle: its centre and edge midpoints (a scan's vertices sit on edges)
                A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
                probes = np.stack([(A + B + C) / 3, (A + B) / 2, (B + C) / 2, (C + A) / 2], axis=1)
                dev = [round(max(float(u.brep_closest_point(brep, tuple(q))[1]) for q in tri), 4) for tri in probes]
            lo, hi = V.min(axis=0), V.max(axis=0)
            HUB.mesh_cache[key] = dict(points=np.round(V, 4).ravel().tolist(), tris=T.ravel().tolist(),
                                       tdev=dev, tol=s.tol, bbox=[lo.tolist(), hi.tolist()])
        return JSONResponse(HUB.mesh_cache[key])


@app.get("/api/bom/{i}")
def bom(i: int):
    """Parts list for size i: every body, what makes it, exact volume and mass from the kernel."""
    with HUB.lock:
        s = HUB.session
        if not (0 <= i < len(s.reports)) or s.reports[i].result is None:
            raise HTTPException(404, "no such built size")
        res = s.reports[i].result
        mat = s.spec.get("material", {"name": "unspecified", "density_g_cm3": 7.85})
        work = set(s.spec.get("workpiece", []))
        bodies = res.bodies or {"part": res.brep}
        ops: dict[str, set] = {}
        for f in res.faces:
            ops.setdefault(f.body, set()).add(next((b["op"] for b in s.graph["blocks"] if b["id"] == f.block), ""))
        out = []
        for name, b in bodies.items():
            (x0, y0, z0), (x1, y1, z1) = b.bounding_box(tight=True)
            v = float(u.compute_volume(b))
            kind = "workpiece" if name in work else "stock" if ops.get(name) == {"stock"} else "make"
            out.append(dict(name=s.spec.get("title", HUB.family) if name == "part" else name, kind=kind,
                            volume_cm3=round(v / 1000, 2), mass_g=None if kind == "workpiece" else round(v / 1000 * mat["density_g_cm3"], 1),
                            size_mm=[round(x1 - x0, 1), round(y1 - y0, 1), round(z1 - z0, 1)]))
        return dict(size=s.reports[i].size, material=mat, parts=out)


@app.get("/api/dims/{i}")
def dims(i: int):
    """Where each check on size i was measured, as a dimension line on the exact solid."""
    with HUB.lock:
        s = HUB.session
        if not (0 <= i < len(s.reports)) or s.reports[i].result is None:
            raise HTTPException(404, "no such built size")
        key = (graph_hash(s.graph), i, json.dumps(s.spec["checks"], sort_keys=True))
        if key not in HUB.dims_cache:
            HUB.dims_cache[key] = dimensions(s.reports[i].result, s.spec["checks"])
        return JSONResponse(HUB.dims_cache[key])


@app.get("/api/aero/{i}")
def aero(i: int, speed: float = 50.0):
    """Conceptual L/D for size i: vortex-lattice wings + drag build-up from the kernel's exact areas."""
    from caddy.kernel import aero as A
    with HUB.lock:
        s = HUB.session
        if not (0 <= i < len(s.reports)) or s.reports[i].result is None:
            raise HTTPException(404, "no such built size")
        key = ("aero", graph_hash(s.graph), i, round(speed, 2))
        if key not in HUB.dims_cache:
            try:
                HUB.dims_cache[key] = A.analyse(s.reports[i].result, speed=speed)
            except ValueError as e:
                raise HTTPException(404, str(e))
        return JSONResponse(HUB.dims_cache[key])


class Decision(BaseModel):
    approve: bool


@app.post("/api/agent/decide")
def decide(d: Decision):
    HUB.approved = d.approve
    HUB.decision.set()
    return dict(ok=True)


class Ref(BaseModel):
    family: str | None = None


@app.post("/api/reference")
def load_reference(r: Ref | None = None):
    """Load the hand-written reference graph (no agent), e.g. to explore the viewer and sliders."""
    if HUB.running:
        raise HTTPException(409, "agent running")
    fam = r.family if r and r.family in FAMILIES else HUB.family
    HUB.reset(fam)
    if is_scan(fam):     # the "reference" of a scan is caddy's own draft from the measurements (no AI)
        for st in HUB.session.scan_steps:
            HUB.emit(dict(type="scan_step", key=st["key"], title=st["title"], ms=st["ms"]))
        code = HUB.session.scan_draft
        out = HUB.session.build_cad(code)
        HUB.emit(dict(type="tool", name="build_cad", input={"code": code}, error=False, reference=True,
                      result=out, kernel=HUB.session.verdict(), explain=HUB.session.explain()))
        return dict(ok=True)
    script = FAMILIES[fam].get("script")
    if script:            # a family whose reference is an smlibquery script (editable on the board), not a JSON graph
        code = (DATA / "families" / script).read_text()
        out = HUB.session.build_cad(code)
        HUB.emit(dict(type="tool", name="build_cad", input={"code": code}, error=False, reference=True,
                      result=out, kernel=HUB.session.verdict(), explain=HUB.session.explain()))
        return dict(ok=True)
    g = G.load(fam)
    HUB.session.build_graph(g)
    HUB.emit(dict(type="tool", name="build_graph", input={"graph": g}, error=False, reference=True,
                  result=HUB.session.report(), kernel=HUB.session.verdict()))
    return dict(ok=True)


@app.get("/api/events")
def events(after: int = -1):
    def gen():
        n = after + 1
        idle = 0
        while idle < 600:
            with HUB.lock:
                new = HUB.events[n:]
            for e in new:
                yield f"data: {json.dumps(e, default=str)}\n\n"
                n += 1
            if not new:
                idle += 1
                yield ": ping\n\n" if idle % 30 == 0 else ""
                time.sleep(0.2)
            else:
                idle = 0
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/catalogue", response_class=PlainTextResponse)
def catalogue():
    return (DATA / "bspt_nipple.csv").read_text()


def main(host="127.0.0.1", port=8000):
    import uvicorn
    from caddy.kernel.usd_warmup import warm_up
    warm_up()   # load USD once with its harmless "No module named 'pxr'" warnings silenced
    print(f"caddy → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
