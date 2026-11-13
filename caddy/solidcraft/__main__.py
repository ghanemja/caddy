"""Picture → parametric CAD from the command line.

    python -m caddy.solidcraft chair.jpg --hint "dining chair, seat 450 mm high" --out out/chair

Needs SolidCraft's `solidcraft-cli` (on PATH, or SOLIDCRAFT_CLI) and ANTHROPIC_API_KEY. Writes into --out:
model.sc (the parametric SolidCraft design), model.step (the solids SMLib judged) and run.jsonl (every step with the
kernel's verdict). Exit code 0 only when caddy's kernel accepted the model.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anthropic

from .agent import ImageAgent, task
from .mcp import McpError
from .session import ImageSession


def _show(e: dict) -> None:
    t = e["type"]
    if t == "start":
        print(f"run {e['run_id']} · {e['model']}", flush=True)
    elif t == "response" and e["text"]:
        print(f"[{e['round']}] {e['text'][:400]}", flush=True)
    elif t == "tool":
        k = e.get("kernel") or {}
        mark = "✗" if e["error"] else "→"
        verdict = "" if not k.get("sizes") else (" · kernel ✓" if k["ok"] else f" · kernel ✗ {', '.join(k['failing_checks'])}")
        print(f"   {mark} {e['name']} {json.dumps(e['input'])[:140]}{verdict}", flush=True)
        if e["error"]:
            print(f"     {e['result'][:500]}", flush=True)
    elif t == "error":
        print(f"error: {e['message']}", flush=True)
    elif t == "end":
        u = e["usage"]
        print(f"{'ACCEPTED' if e['ok'] else 'NOT ACCEPTED'} · {e['rounds']} rounds · {e['tool_calls']} tool calls · "
              f"${u['cost_usd']} · {e['seconds']} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m caddy.solidcraft", description=__doc__.splitlines()[0])
    ap.add_argument("image", help="a PNG or JPEG of one object")
    ap.add_argument("--hint", default="", help="what it is, known sizes, anything the picture doesn't show")
    ap.add_argument("--out", default=None, help="folder for model.sc, model.step and run.jsonl (default: out/<image name>)")
    ap.add_argument("--cli", default=None, help="path to solidcraft-cli (default: SOLIDCRAFT_CLI, then PATH)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--min-overlap", type=float, default=None,
                    help="outline overlap with the picture needed to accept (default 0.7, or CADDY_IMAGE_MIN_OVERLAP)")
    a = ap.parse_args(argv)
    img = Path(a.image)
    out = Path(a.out) if a.out else Path("out") / img.stem
    try:
        s = ImageSession(img.read_bytes(), hint=a.hint, workdir=out.resolve(), cli=a.cli, min_overlap=a.min_overlap,
                         title=f"Image → CAD: {img.stem}")
    except (McpError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        agent = ImageAgent(s, effort=a.effort, max_rounds=a.max_rounds, on_event=_show)
        try:
            res = agent.run(task(a.hint))
        except (anthropic.AnthropicError, TypeError) as e:      # TypeError: the SDK found no credentials
            print(f"error: the LLM call failed ({type(e).__name__}: {e}). Is ANTHROPIC_API_KEY set?", file=sys.stderr)
            return 2
        shutil.copyfile(res.log, out / "run.jsonl")
        print(s.report())
        if res.ok:
            print("\n".join(f"{k}: {v}" for k, v in s.saved.items()))
        else:
            print(f"the last design SMLib judged is in {out / 'model.step'} (not accepted)")
        return 0 if res.ok else 1
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
