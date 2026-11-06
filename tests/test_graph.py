"""The graph: schema, safe expressions, diffs, downstream-only rebuild, feature-exists, planted faults."""
import json

import pytest

from caddy.kernel import graph as G
from caddy.kernel.catalogue import values_for
from caddy.kernel.family import build_family, build_size

from .conftest import ROWS

REPAIR = json.load(open(G.FAMILIES / "repair_thread_runout.json"))


def test_expressions_are_arithmetic_only():
    assert G.evaluate_expr("a*2 + min(b, 3)", {"a": 1, "b": 5}) == 5
    for bad in ("__import__('os')", "a.b", "[1][0]", "open('x')"):
        with pytest.raises(G.GraphError):
            G.evaluate_expr(bad, {"a": 1})


@pytest.mark.parametrize("edit,msg", [
    ([{"block": "nope", "set": {}}], "no such block"),
    ([{"block": "hex_prism", "set": {"af": "hex_aff"}}], "unknown names"),
    ([{"block": "join_inlet", "inputs": ["inlet_thread", "bore"]}], "earlier block"),
    ([{"remove": "hex_prism"}], "earlier block"),
])
def test_bad_edits_are_rejected_with_the_block(graph, edit, msg):
    with pytest.raises(G.GraphError, match=msg):
        G.apply_edit(graph, edit)


def test_edit_rebuilds_only_downstream(graph):
    ev = G.Evaluator()
    ev.run(graph, values_for(ROWS[2]))
    g2 = G.apply_edit(graph, [{"block": "hex_prism", "set": {"af": "hex_af + 1"}}])
    res = ev.run(g2, values_for(ROWS[2]))
    assert res.rebuilt == ["hex_prism", "hex", "join_inlet", "join_outlet", "cut_bore"]


def test_repair_diff_fixes_every_size_and_is_small(graph):
    g2 = G.apply_edit(graph, REPAIR)
    reps = build_family(g2)
    assert all(r.ok for r in reps), [(r.size, [c["id"] for c in r.checks if not c["ok"]]) for r in reps]
    assert G.diff_size(REPAIR) * 5 < len(json.dumps(g2, separators=(",", ":")))
    assert len(reps[5].result.faces) == 28   # no more slivers on 3/4 x 1/2


# --- bug seeding: planted faults must be caught by the kernel, not by a model -----------------------------

FAULTS = {
    "wrong_column (bore uses hex A/F)": [{"block": "bore", "set": {"d": "hex_af"}}],
    "off_by_one (outlet thread +1 mm)": [{"block": "outlet_thread", "set": {"h": "outlet_thread_len + 1"}}],
    "bore_too_large": [{"block": "bore", "set": {"d": "bore_d * 1.1"}}],
    "threads_dont_cut (bore misses the part)": [{"block": "bore", "set": {"z": "200"}}],
    "radius_not_diameter (inlet)": [{"block": "inlet_thread", "set": {"d_base": "(in_major_d - in_gauge_len/16)/2"}}],
}


@pytest.mark.parametrize("name", list(FAULTS))
def test_planted_fault_is_caught(graph, name):
    rep = build_size(G.Evaluator(), G.apply_edit(graph, FAULTS[name]), ROWS[2])
    assert not rep.ok, f"planted fault {name} went undetected"
