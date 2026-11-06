"""Deterministic replay: rebuild every graph version of a logged run with no model call and check the kernel gives
the same verdict at every step (VERIFY.md).

    python -m caddy.replay runs/<id>.jsonl
"""
from __future__ import annotations

import json
import sys

from caddy.agent.loop import Agent
from caddy.agent.session import Session
from caddy.kernel import graph as G


def replay(path: str) -> dict:
    events = [json.loads(line) for line in open(path)]
    start = events[0]
    fam = start.get("family", "bspt_nipple")
    s = Session(family=fam, graph=G.load(fam) if start.get("preload_reference") else None)
    shim = Agent.__new__(Agent)   # reuse the tool dispatcher without a client
    shim.s = s
    steps, mismatches = 0, []
    for e in events:
        if e["type"] != "tool" or e["name"] not in ("build_graph", "edit_graph", "build_cad", "edit_cad", "finish"):
            continue
        out, err = Agent._call_tool(shim, e["name"], e["input"])
        steps += 1
        if err != e["error"] or _v(s.verdict()) != _v(e["kernel"]):
            mismatches.append(dict(round=e["round"], tool=e["name"], logged=e["kernel"], replayed=s.verdict()))
    end = next((e for e in events if e["type"] == "end"), {})
    return dict(steps=steps, mismatches=mismatches, final=s.verdict(), logged_final=end.get("verdict"),
                identical=not mismatches and _v(s.verdict()) == _v(end.get("verdict") or {}))


def _v(verdict: dict) -> dict:
    """What the kernel decided (pass/fail per size and which checks); the graph fingerprint is an internal id."""
    return {k: verdict.get(k) for k in ("ok", "sizes", "failing", "failing_checks")}


if __name__ == "__main__":
    r = replay(sys.argv[1])
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["identical"] else 1)
