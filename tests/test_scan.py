"""Scan → CAD: every measuring step on SMLib recovers the part; the kernel judges the CAD against the scan."""
import json
from pathlib import Path

import pytest

from caddy.scan import fit as F
from caddy.scan.scans import BUSHING, measured, new_session

RUNS = Path(__file__).resolve().parents[1] / "docs" / "runs"


def test_bushing_primitives_come_back_from_a_mesh_no_kernel_made():
    """The bushing is meshed by plain numpy and posed at random: the pipeline finds its axis and every feature."""
    framed, steps, f, code = measured("scan_bushing")
    by = {s["key"]: s for s in steps}
    assert by["axis"]["axis"]["turned"] and by["slices"]["slices"]["hex"] is None
    diam = sorted({round((q["d0"] + q["d1"]) / 2) for q in f["outer"] if q["kind"] == "cylinder"})
    assert diam == [28, 44]                                     # shank and flange
    cone = [q for q in f["outer"] if q["kind"] == "cone"]
    assert any(abs(min(q["d0"], q["d1"]) - 20) < 0.05 and abs(max(q["d0"], q["d1"]) - 24) < 0.05 for q in cone)
    assert sorted(round(b["d0"]) for b in f["bore"]) == [12, 18]   # bore and counterbore
    assert abs(f["length"] - max(z for _, z in BUSHING)) < 0.05


def test_nipple_hex_is_found_and_the_draft_misses_only_the_chamfer():
    s, steps, code = new_session("scan_nipple")
    hx = {x["key"]: x for x in steps}["slices"]["slices"]["hex"]
    assert abs(hx["af"] - 27.0) < 0.05 and hx["corner_min"] < hx["corner_r"]
    report = s.build_cad(code)
    c = {x["id"]: x for x in s.reports[0].checks}
    assert not c["matches_scan"]["ok"] and 0.3 < c["matches_scan"]["measured"] < 0.7
    worst = report.split("scan → CAD")[1].splitlines()[1]
    assert ".flat" in worst or ".end" in worst                 # the deviation report points at the hex


def test_draft_of_the_bushing_already_matches_the_scan():
    s, steps, code = new_session("scan_bushing")
    s.build_cad(code)
    assert s.verdict()["ok"]


@pytest.mark.parametrize("name", ["scan_nipple", "scan_bushing"])
def test_recorded_claude_run_replays_to_a_match(name):
    s, steps, code = new_session(name)
    s.build_cad(code)
    for line in open(RUNS / f"{name}.jsonl"):
        e = json.loads(line)
        if e["type"] == "tool" and e["name"] == "build_cad" and not e["error"]:
            s.build_cad(e["input"]["code"])
        elif e["type"] == "tool" and e["name"] == "edit_cad" and not e["error"]:
            s.edit_cad(e["input"]["edits"])
    assert s.verdict()["ok"]
    assert all(".tag(" in s.script for _ in [0]) and "s1." not in s.script   # named by what they are, not s1, s2


def test_scan_api_heatmap():
    from fastapi.testclient import TestClient
    from caddy.app.server import app
    c = TestClient(app)
    assert c.post("/api/reference", json={"family": "scan_bushing"}).json()["ok"]
    st = c.get("/api/state").json()
    assert [x["key"] for x in st["scan"]["steps"]] == ["scan", "axis", "slices", "profile", "fit", "draft"]
    m = c.get("/api/scan/mesh").json()
    assert len(m["tdev"]) * 3 == len(m["tris"]) and max(m["tdev"]) < m["tol"]
    c.post("/api/reference", json={"family": "bspt_nipple"})
