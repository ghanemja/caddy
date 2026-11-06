"""Deterministic replay of a real agent run: same graphs in, same kernel verdict at every step, no model call."""
from pathlib import Path

import pytest

from caddy.replay import replay

LOG = Path(__file__).resolve().parents[1] / "docs" / "runs" / "demo_build_and_repair.jsonl"


def test_demo_run_replays_identically():
    if not Path(LOG).exists():
        pytest.skip("recording not in the repo yet (see docs/TASKS.md)")
    r = replay(str(LOG))
    assert r["identical"], r["mismatches"]
    assert r["final"]["ok"] and r["steps"] >= 4
