"""The assembly family: Adam's SO-101 demo task ("a better gripper for rods, prove it fits, close it on a rod"),
proven by the kernel for every rod size and pose."""
from pathlib import Path

import pytest

from caddy.agent.session import Session
from caddy.kernel import graph as G
from caddy.replay import replay

RUNS = Path(__file__).resolve().parents[1] / "docs" / "runs"


@pytest.fixture(scope="module")
def stock():
    s = Session(family="gripper")
    s.build_graph(G.load("gripper"))
    return s


def failing(s):
    return {r.size: sorted(c["id"] for c in r.checks if not c["ok"]) for r in s.reports if not r.ok}


def test_stock_fingertips_fail_exactly_where_the_geometry_says(stock):
    """A fixed 5 mm 90° groove: the jaws hit each other on small rods and never reach the flanks on big ones."""
    assert len(stock.reports) == 12
    assert failing(stock) == {
        "Ø6 closed": ["jaws_clear"], "Ø8 closed": ["jaws_clear"],
        "Ø16 closed": ["closed_touch_l", "closed_touch_r"], "Ø20 closed": ["closed_touch_l", "closed_touch_r"]}


def test_every_body_is_a_closed_solid_and_moved_faces_keep_their_labels(stock):
    r = stock.reports[3].result
    assert set(r.bodies) == {"rod", "jaw_r", "jaw_l"} and all(b.is_manifold_solid() for b in r.bodies.values())
    assert all(f.block and f.alias for f in r.faces)
    assert r.by_alias("jaw_l.groove") and r.by_alias("jaw_r.groove")       # the mirrored jaw still knows its groove
    assert {f.body for f in r.by_alias("jaw_l.groove")} == {"jaw_l"}


def test_one_line_repair_grips_every_rod():
    s = Session(family="gripper", graph=G.load("gripper"))
    s.edit_graph([{"param": "groove_depth", "default": "0.5*rod_d"}])    # an expression as default is accepted
    assert s.ok, failing(s)


def test_planted_over_close_is_caught_as_interference():
    s = Session(family="gripper", graph=G.load("gripper"))
    s.edit_graph([{"param": "groove_depth", "expr": "0.5*rod_d"}])
    blocks = {b["id"]: b for b in s.graph["blocks"]}
    over = [{"block": k, "set": {"cx": f"({blocks[k]['params']['cx']}) - 0.3"}}       # jaws travel 0.3 mm too far
            for k in ("pad_r", "vee_r", "pad_l", "vee_l")]
    s.edit_graph(over)
    bad = failing(s)
    assert all("no_interference_r" in v and "no_interference_l" in v for k, v in bad.items() if "closed" in k)
    assert len([k for k in bad if "closed" in k]) == 6


@pytest.mark.parametrize("log", ["gripper_build.jsonl", "gripper_repair.jsonl"])
def test_recorded_gripper_runs_replay_identically(log):
    if not Path(RUNS / log).exists():
        pytest.skip("recording not in the repo yet (see docs/TASKS.md)")
    r = replay(str(RUNS / log))
    assert r["identical"], r["mismatches"]
    assert r["final"]["ok"]
