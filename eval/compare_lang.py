"""Before/after: Claude writing caddy's JSON graph vs Claude writing CadQuery (smlibquery), same task.

    python -m eval.compare_lang [runs_per_arm]

Task: build the BSPT reducing nipple family from the catalogue (8 sizes). Same model, effort and kernel checks.
Writes eval/results/lang_compare.json and .md; every run's log is kept in runs/ (replayable)."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from caddy.agent.loop import Agent, catalogue_task
from caddy.agent.session import Session

OUT = Path(__file__).parent / "results"


def one(lang: str, i: int, results: list):
    s = Session()
    first = {}

    def on_event(e):
        if e["type"] == "response" and e["round"] == 1:
            first["out_tokens"] = e["usage"].get("output_tokens")
        if e["type"] == "tool" and e["name"] in ("build_graph", "build_cad") and "valid" not in first:
            first["valid"] = not e["error"]
            first["sizes_ok"] = (e["kernel"]["sizes"] - len(e["kernel"]["failing"])) if not e["error"] else 0
        if e["type"] == "tool" and e["name"] in ("edit_graph", "edit_cad") and not e["error"]:
            first.setdefault("edit_chars", []).append(len(json.dumps(e["input"].get("ops") or e["input"].get("edits"))))

    t = time.time()
    try:
        r = Agent(s, on_event=on_event, run_id=f"cmp-{lang}-{i}-{int(t)}", lang=lang).run(catalogue_task())
        results.append(dict(lang=lang, run=i, ok=r.ok, claimed_done=r.claimed_done, false_done=r.false_done,
                            rounds=r.rounds, tool_calls=r.tool_calls, seconds=r.seconds, log=str(r.log),
                            failing=r.verdict.get("failing_checks"), **r.usage, first_build_valid=first.get("valid"),
                            first_build_sizes_ok=first.get("sizes_ok"), first_output_tokens=first.get("out_tokens"),
                            edit_chars=first.get("edit_chars", [])))
    except Exception as e:  # keep going; record the failure
        results.append(dict(lang=lang, run=i, error=f"{type(e).__name__}: {e}"))


def main(n: int = 3):
    results: list = []
    threads = [threading.Thread(target=one, args=(lang, i, results)) for i in range(n) for lang in ("json", "cadquery")]
    for th in threads:
        th.start()
        time.sleep(2)
    for th in threads:
        th.join()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lang_compare.json").write_text(json.dumps(results, indent=1))
    rows = []
    for lang in ("json", "cadquery"):
        rs = [r for r in results if r["lang"] == lang and "error" not in r]
        if not rs:
            continue
        avg = lambda k: sum(r[k] for r in rs) / len(rs)
        rows.append(dict(lang=lang, n=len(rs), ok=sum(r["ok"] for r in rs), false_done=sum(r["false_done"] for r in rs),
                         first_valid=sum(bool(r["first_build_valid"]) for r in rs),
                         first_out=avg("first_output_tokens"), out=avg("output"), cost=avg("cost_usd"),
                         rounds=avg("rounds"), seconds=avg("seconds")))
    md = ["| writes | runs | all sizes ✓ | false done | first build valid | first-build output tok | total output tok | cost | requests | seconds |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['lang']} | {r['n']} | {r['ok']}/{r['n']} | {r['false_done']} | {r['first_valid']}/{r['n']} | "
                  f"{r['first_out']:.0f} | {r['out']:.0f} | ${r['cost']:.3f} | {r['rounds']:.1f} | {r['seconds']:.0f} |")
    (OUT / "lang_compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for r in results:
        print(r.get("lang"), r.get("run"), r.get("ok"), r.get("failing"), r.get("error", ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
