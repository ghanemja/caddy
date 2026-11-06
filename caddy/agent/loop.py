"""The agent loop (manual tool-use loop over the Anthropic Messages API).

The model designs; the kernel decides. Every request's token usage and every tool call is logged to
runs/<id>.jsonl, so a run can be replayed with no model call (caddy.replay) and costed.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic

from caddy.kernel import graph as G
from caddy.kernel.catalogue import DATA, values_for
from caddy.kernel.graph import GraphError
from caddy.kernel.build import KernelError

from .prompts import SYSTEM, TOOLS, TOOLS_CQ, build_task, family_task, system_prompt, system_prompt_cq
from .session import Session, graph_hash

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
MODEL = os.environ.get("CADDY_MODEL", "claude-opus-5-5")
PRICES = {  # $ per 1M tokens: input, output, cache read, cache write (5 min)
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Usage:
    requests: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, u) -> None:
        self.requests += 1
        self.input += u.input_tokens
        self.output += u.output_tokens
        self.cache_read += u.cache_read_input_tokens or 0
        self.cache_write += u.cache_creation_input_tokens or 0

    def cost(self, model: str) -> float:
        pi, po, pr, pw = PRICES.get(model, PRICES["claude-opus-5-5"])
        return (self.input * pi + self.output * po + self.cache_read * pr + self.cache_write * pw) / 1e6

    def to_dict(self, model: str) -> dict:
        return dict(requests=self.requests, input=self.input, output=self.output, cache_read=self.cache_read,
                    cache_write=self.cache_write, cost_usd=round(self.cost(model), 4))


@dataclass
class RunResult:
    run_id: str
    ok: bool                       # the kernel's verdict at the end
    claimed_done: bool
    false_done: int                # finish calls while the kernel still showed a failure
    rounds: int                    # model requests
    tool_calls: int
    usage: dict
    seconds: float
    log: Path
    verdict: dict = field(default_factory=dict)


class Agent:
    def __init__(self, session: Session, model: str = MODEL, effort: str = "medium", max_rounds: int = 14,
                 on_event: Callable[[dict], None] | None = None, run_id: str | None = None,
                 system: str | None = None, tools: list[dict] | None = None, lang: str = "cadquery"):
        load_env()
        self.client = anthropic.Anthropic()
        self.s, self.model, self.effort, self.max_rounds = session, model, effort, max_rounds
        self.on_event = on_event or (lambda e: None)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        RUNS.mkdir(exist_ok=True)
        self.log_path = RUNS / f"{self.run_id}.jsonl"
        self.lang = lang
        if lang == "cadquery":
            self.system = system or system_prompt_cq(session.row_names - G.ROW_NAMES | set(values_for(session.rows[0])),
                                                     session.spec)
            self.tools = tools or TOOLS_CQ
        else:
            self.system = system or (SYSTEM if session.family == "bspt_nipple"
                                     else system_prompt(session.row_names, session.spec))
            self.tools = tools or TOOLS
        self.usage = Usage()

    # --- logging -----------------------------------------------------------------------------------------
    def _emit(self, ev: dict) -> None:
        ev = {"t": round(time.time(), 3), **ev}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        self.on_event(ev)

    # --- tools -------------------------------------------------------------------------------------------
    def _call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        s = self.s
        try:
            if name == "build_graph":
                return s.build_graph(args["graph"]), False
            if name == "edit_graph":
                return s.edit_graph(args["ops"]), False
            if name == "build_cad":
                return s.build_cad(args["code"]), False
            if name == "edit_cad":
                return s.edit_cad(args["edits"]), False
            if name == "check_stability":
                return s.stability(args.get("size"), args.get("body"), float(args.get("mu", 0.4))), False
            if name == "measure":
                return s.measure(args["size"], args["face"]), False
            if name == "why":
                return s.why(args["size"], args["face"]), False
            if name == "finish":
                v = s.verdict()
                if v["ok"]:
                    return "accepted: the kernel confirms every size ✓", False
                return ("not accepted: the kernel still reports failures\n" + s.family_table()), True
            return f"unknown tool {name}", True
        except KernelError as e:
            return f"kernel error: {json.dumps(e.facts())}", True
        except GraphError as e:
            return f"graph error: {e}", True
        except (KeyError, TypeError) as e:
            return f"bad tool input: {type(e).__name__}: {e}", True

    # --- loop --------------------------------------------------------------------------------------------
    def _request(self, messages: list):
        kw = dict(model=self.model, max_tokens=16000, system=self.system, tools=self.tools, messages=messages,
                  cache_control={"type": "ephemeral"}, output_config={"effort": self.effort})
        if self.model in ("claude-opus-5-5", "claude-sonnet-5-5"):
            # server-side refusal fallback: a declined request is re-run on a fallback model in the same call
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def run(self, task: str) -> RunResult:
        t0 = time.time()
        messages: list = [{"role": "user", "content": task}]
        claimed, false_done, rounds, calls = False, 0, 0, 0
        self._emit(dict(type="start", run_id=self.run_id, model=self.model, effort=self.effort, family=self.s.family,
                        lang=self.lang,
                        graph=graph_hash(self.s.graph), task=task))
        while rounds < self.max_rounds:
            resp = self._request(messages)
            rounds += 1
            self.usage.add(resp.usage)
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            self._emit(dict(type="response", round=rounds, stop=resp.stop_reason, text=text,
                            usage=resp.usage.model_dump(exclude_none=True)))
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                break
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                break
            results, accepted = [], False
            for b in uses:
                calls += 1
                out, is_err = self._call_tool(b.name, dict(b.input))
                if b.name == "finish":
                    claimed = True
                    false_done += int(is_err)
                    accepted = not is_err
                self._emit(dict(type="tool", round=rounds, name=b.name, input=b.input, error=is_err,
                                result=out, graph=graph_hash(self.s.graph), kernel=self.s.verdict(),
                                explain=self.s.explain(),
                                **({"stab": self.s.last_stab} if b.name == "check_stability" and not is_err else {})))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                                **({"is_error": True} if is_err else {})})
            messages.append({"role": "user", "content": results})
            if accepted:
                break
        v = self.s.verdict()
        res = RunResult(self.run_id, v["ok"], claimed, false_done, rounds, calls, self.usage.to_dict(self.model),
                        round(time.time() - t0, 1), self.log_path, v)
        self._emit(dict(type="end", ok=res.ok, claimed_done=claimed, false_done=false_done, rounds=rounds,
                        tool_calls=calls, usage=res.usage, seconds=res.seconds, verdict=v))
        return res


def catalogue_task(family: str = "bspt_nipple") -> str:
    if family == "bspt_nipple":
        return build_task((DATA / "bspt_nipple.csv").read_text().strip(), (DATA / "iso7_1.csv").read_text().strip())
    from caddy.kernel.catalogue import spec as family_spec
    sp = family_spec(family)
    return family_task(sp, (DATA / sp["catalogue"]).read_text().strip())


if __name__ == "__main__":
    import sys
    fam = os.environ.get("CADDY_FAMILY", "bspt_nipple")
    s = Session(family=fam)
    a = Agent(s, on_event=lambda e: print(
        {"response": lambda: f"[{e['round']}] {e['stop']} {e['text'][:300]}",
         "tool": lambda: f"   → {e['name']} {'ERR ' if e['error'] else ''}{json.dumps(e['input'])[:200]}\n"
                         f"     {e['result'][:400]}",
         "start": lambda: f"run {e['run_id']} model {e['model']}",
         "end": lambda: f"END ok={e['ok']} claimed={e['claimed_done']} false_done={e['false_done']} "
                        f"rounds={e['rounds']} {e['usage']} {e['seconds']}s"}[e["type"]](), flush=True))
    r = a.run(catalogue_task(fam) if len(sys.argv) < 2 else sys.argv[1])
