"""caddy scripts (smlibquery, a CadQuery-style API on SMLib): same kernel verdicts as the JSON graph, small edits, replayable runs."""
from pathlib import Path

import pytest

from caddy.agent.session import Session
from caddy.kernel import graph as G
from caddy.kernel.family import build_family
from caddy.replay import replay

HERE = Path(__file__).parent
NIPPLE = (HERE / "fixtures" / "nipple.sq.py").read_text()


def verdicts(reports):
    return [[(c["id"], c["ok"], round(c["measured"] or 0, 3)) for c in r.checks] for r in reports]


def test_script_gives_the_same_verdicts_as_the_json_graph():
    s = Session()
    s.build_cad(NIPPLE)
    assert verdicts(s.reports) == verdicts(build_family(G.load()))
    assert {b["id"] for b in s.graph["blocks"]} >= {"inlet_thread", "hex_flat", "outlet_thread", "bore_wall"}


def test_design_parameter_edit_rebuilds_only_downstream():
    s = Session()
    s.build_cad(NIPPLE)
    assert s.graph["parameters"][0]["name"] == "chamfer_deg"
    out = s.edit_cad([{"old": "chamfer_deg = 30", "new": "chamfer_deg = 25"}])
    rebuilt = out.splitlines()[0]
    assert "hex_chamfer" in rebuilt and "inlet_thread" not in rebuilt and "outlet_thread" not in rebuilt


@pytest.mark.parametrize("edit,msg", [
    ({"old": "hex_af * 1.02 / 2", "new": "x"}, "exactly once"),
    ({"old": ".hole(bore_d)", "new": ".hole(bore_d).fillet(1)"}, "fillet"),
])
def test_bad_edits_are_refused_and_the_model_is_unchanged(edit, msg):
    s = Session()
    s.build_cad(NIPPLE)
    before = s.script
    with pytest.raises(G.GraphError, match=msg):
        s.edit_cad([edit])
    assert s.script == before


def test_recorded_script_run_replays_identically():
    log = HERE.parent / "docs" / "runs" / "nipple_cq_build.jsonl"
    if not Path(log).exists():
        pytest.skip("recording not in the repo yet (see docs/TASKS.md)")
    r = replay(str(log))
    assert r["identical"], r["mismatches"]
