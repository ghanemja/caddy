"""The board's design intent: feature tree and constraints are read off the kernel, and a constraint added in words
is previewed on every size, then enforced by the kernel once approved. No API calls: the proposal is given."""
from pathlib import Path

import pytest

from caddy.agent.session import Session
from caddy.app import intent
from caddy.kernel import graph as G

NIPPLE = (Path(__file__).parent / "fixtures" / "nipple.sq.py").read_text()

# What Claude proposed live for "the hex should be at least 1 mm wider than the inlet thread so a spanner clears it"
HEX_CLEARS = {"id": "you_hex_clears_inlet_thread", "rule": "hex across flats ≥ inlet thread major Ø + 1",
              "measure": "across_flats", "target": "hex.flat", "op": ">=", "expect": "in_major_d + 1", "by": "you"}


@pytest.fixture(scope="module")
def s():
    s = Session()
    s.build_cad(NIPPLE)
    return s


def blocks_of(s):
    return [dict(b, failing=[]) for b in s.graph["blocks"]]


def test_tree_groups_features_by_part_and_traces_every_number(s):
    tree = intent.feature_tree(s, blocks_of(s))
    assert [p["part"] for p in tree] == ["inlet", "hex", "outlet", "bore"]
    feats = {f["id"]: f for p in tree for f in p["features"]}
    book = {f["op"] for p in tree for f in p["bookkeeping"]}
    assert book <= intent.BOOKKEEPING and not any(f["op"] in intent.BOOKKEEPING for f in feats.values())
    kinds = {x["name"]: x["kind"] for x in feats["inlet_thread"]["sources"]}
    assert kinds["in_major_d"] == "iso" and kinds["inlet_thread_len"] == "col"
    assert all(f["line"] for f in feats.values())            # every feature points at its script line


def test_constraints_show_driving_dimensions_and_rules_with_live_status(s):
    cons = {c["id"]: c for c in intent.constraints(s)}
    hex_af = cons["drive:hex_af"]
    assert hex_af["kind"] == "driving" and hex_af["source"] == "col" and hex_af["status"]["applies"] == 8
    rule = next(c for c in cons.values() if c["kind"] == "rule")
    assert rule["status"]["applies"] and rule["features"]


def test_preview_then_approve_then_remove(s):
    before = sum(c["ok"] for r in s.reports for c in r.checks)
    pv = intent.preview(s, HEX_CLEARS)
    assert pv["applies"] == 8 and pv["ok"] == 6                 # the 3/4" sizes are too tight
    assert {r["size"] for r in pv["rows"] if not r["ok"]} == {r.size for r in s.reports if "3/4" in r.size.split(" x ")[0]}
    assert sum(c["ok"] for r in s.reports for c in r.checks) == before   # preview changes nothing

    intent.approve(s, HEX_CLEARS)
    cons = {c["id"]: c for c in intent.constraints(s)}
    assert cons[HEX_CLEARS["id"]]["by"] == "you" and cons[HEX_CLEARS["id"]]["status"]["ok"] == 6
    assert not s.verdict()["ok"]                                # the kernel now enforces it: "done" is refused
    with pytest.raises(G.GraphError, match="already exists"):
        intent.approve(s, HEX_CLEARS)

    intent.remove(s, HEX_CLEARS["id"])
    assert HEX_CLEARS["id"] not in {c["id"] for c in s.spec["checks"]}
    assert sum(c["ok"] for r in s.reports for c in r.checks) == before
    with pytest.raises(G.GraphError, match="only constraints you added"):
        intent.remove(s, s.spec["checks"][0]["id"])


def test_unknown_measure_is_refused(s):
    with pytest.raises(G.GraphError, match="unknown measurement"):
        intent.approve(s, dict(HEX_CLEARS, id="you_x", measure="volume_of_love"))


# --- visual scripting and dimensions ------------------------------------------------------------------------

def test_script_fields_attach_to_the_feature_that_uses_them(s):
    from caddy.app.scripting import attach
    order = sorted(s.graph["blocks"], key=lambda b: b["op"] in intent.BOOKKEEPING)
    by, values = attach(s.script, {b["id"]: b.get("line") for b in order})
    assert [f["text"] for f in by["hex_flat"]][-1] == "hex_len"          # .extrude(hex_len)
    assert [f["text"] for f in by["bore_wall"]] == ["bore_d"]            # .hole(bore_d), not the boolean
    assert values[0]["label"] == "chamfer_deg" and values[0]["text"] == "30"


def test_dimensions_are_drawn_where_the_kernel_measured():
    from caddy.kernel.annotate import dimensions
    s = Session()
    s.build_cad(NIPPLE)
    r = s.reports[5]
    d = dimensions(r.result, s.spec["checks"])
    measured = {c["id"]: c["measured"] for c in r.checks}
    span = lambda k: sum((x - y) ** 2 for x, y in zip(d[k]["a"], d[k]["b"])) ** 0.5
    for k in ("hex_af", "overall_len", "hex_len", "bore_d", "thread_clears_hex"):
        assert abs(span(k) - measured[k]) < 0.01, k


def test_board_edit_changes_the_script_rebuilds_and_undoes():
    from fastapi.testclient import TestClient
    from caddy.app.server import HUB, app
    c = TestClient(app)
    c.post("/api/reference", json={"family": "bspt_nipple"})   # explicit: the hub keeps whatever family ran last
    HUB.session.build_cad(NIPPLE)
    st = c.get("/api/state").json()
    f = next(x for p in st["tree"] for ft in p["features"] if ft["id"] == "bore_wall" for x in ft["fields"])
    r = c.post("/api/script/edit", json=dict(line=f["line"], start=f["start"], end=f["end"], text="bore_d + 1",
                                              expect=f["text"])).json()
    assert r["ok"] and "bore_wall" in r["rebuilt"] and "hex_flat" not in r["rebuilt"]
    assert any(x["id"] == "bore_d" and x["after"][0] == 0 for x in r["checks"])     # the catalogue bore now fails
    assert "hole(bore_d + 1)" in HUB.session.script
    bad = c.post("/api/script/edit", json=dict(line=f["line"], start=f["start"], end=f["start"] + len("bore_d + 1"),
                                                text="bore_d +"))
    assert bad.status_code == 400 and "syntax" in bad.json()["detail"]
    assert c.post("/api/script/undo", json={}).json()["ok"] and "hole(bore_d)" in HUB.session.script


def test_parts_list_comes_from_the_kernel():
    from fastapi.testclient import TestClient
    from caddy.app.server import app
    c = TestClient(app)
    c.post("/api/reference", json={"family": "gripper"})
    b = c.get("/api/bom/0").json()
    kinds = {p["name"]: p["kind"] for p in b["parts"]}
    assert kinds == {"rod": "workpiece", "jaw_r": "make", "jaw_l": "make"}
    jaw = next(p for p in b["parts"] if p["name"] == "jaw_r")
    assert jaw["mass_g"] == round(jaw["volume_cm3"] * b["material"]["density_g_cm3"], 1) > 0
    c.post("/api/reference", json={"family": "bspt_nipple"})
