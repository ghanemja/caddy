"""The web app's API: reference graph, state, mesh with per-triangle faces, slider preview (no AI), USD export,
and the replay mode streaming a recorded run through the real kernel."""
import time

from fastapi.testclient import TestClient

from caddy.app.server import HUB, app

client = TestClient(app)


def test_reference_state_mesh_preview_export():
    assert client.post("/api/reference").json()["ok"]
    st = client.get("/api/state").json()
    assert len(st["sizes"]) == 8 and st["parts"] == ["inlet", "hex", "outlet", "bore"]
    assert sum(c["ok"] for s in st["sizes"] for c in s["checks"]) == 70   # 2 real failures on the 3/4" rows
    m = client.get("/api/mesh/5").json()
    assert len(m["tri_face"]) * 3 == len(m["tris"]) and max(m["tri_face"]) < len(m["faces"])
    p = client.post("/api/preview", json={"size": 5, "values": {"hex_af": 28}}).json()
    assert p["ok"] and "hex_prism" in p["rebuilt"] and "outlet_thread" not in p["rebuilt"]
    usd = client.get("/api/export/5").text
    assert usd.startswith("#usda") and "BrepSurfaceConeAPI" in usd


def test_replay_mode_reaches_the_logged_verdict():
    assert client.post("/api/agent/start", json={"mode": "replay", "speed": 1000}).json()["ok"]
    t0 = time.time()
    while HUB.running and time.time() - t0 < 120:
        time.sleep(0.2)
    st = client.get("/api/state").json()
    assert st["verdict"]["ok"] and all(s["ok"] for s in st["sizes"])
    assert any(e["type"] == "end" for e in HUB.events)


def test_pointed_at_pieces_become_kernel_facts():
    """Dragging a block / face / check / part into the chat sends the agent exact kernel facts, not UI text."""
    from caddy.app.server import describe_ref
    client.post("/api/reference")
    s = HUB.session
    assert describe_ref(s, {"kind": "block", "id": "hex_prism"}).startswith("block hex_prism (hex_prism, part hex")
    c = describe_ref(s, {"kind": "check", "id": "thread_clears_hex", "size": 5})
    assert "measured 27.1597 <= expected 27.0 (FAIL)" in c and "inlet.thread" in c
    assert "made by block hex_prism" in describe_ref(s, {"kind": "face", "id": "hex.flat", "size": 5})
    assert "failing checks: thread_clears_hex" in describe_ref(s, {"kind": "part", "id": "inlet"})


def test_keep_this_lock_is_enforced_by_the_session_not_the_model():
    """'Keep this' locks a block: any diff that changes or removes it is refused, whatever the model intended."""
    import pytest
    from caddy.kernel.graph import GraphError
    client.post("/api/reference")
    assert client.post("/api/lock", json={"kind": "block", "id": "hex_prism"}).json()["locked"] == ["hex_prism"]
    s = HUB.session
    with pytest.raises(GraphError, match="locked by the engineer"):
        s.edit_graph([{"block": "hex_prism", "set": {"af": "hex_af + 1"}}])
    with pytest.raises(GraphError, match="locked"):
        s.edit_graph([{"remove": "hex_prism"}])
    s.edit_graph([{"block": "bore", "set": {"d": "bore_d"}}])          # other blocks still editable
    assert client.post("/api/lock", json={"kind": "part", "id": "hex"}).json()["locked"] == ["hex", "hex_chamfer", "hex_prism", "join_inlet"]
    client.post("/api/lock", json={"kind": "part", "id": "hex", "on": False})
    assert client.get("/api/state").json()["locked"] == []


def test_predict_finds_a_kernel_verified_safe_value():
    # gripper, Ø10 closed: a 9 mm groove closes the jaws onto each other; the kernel proposes a depth that keeps the gap
    assert client.post("/api/reference", json={"family": "gripper"}).json()["ok"]
    for _ in range(120):
        if client.get("/api/state").json()["sizes"][5].get("checks"):
            break
        time.sleep(0.5)
    r = client.post("/api/predict", json={"size": 5, "values": {"groove_depth": 9}}).json()
    assert [b["id"] for b in r["broken"]] == ["jaws_clear"]
    lim = r["limits"][0]
    assert lim["param"] == "groove_depth" and lim["side"] == "max" and 5 < lim["safe"] < 9
    ok = client.post("/api/preview", json={"size": 5, "values": {"groove_depth": lim["safe"]}}).json()
    assert all(c["ok"] for c in ok["checks"])


def test_param_set_keeps_a_design_value_on_a_graph_without_a_script():
    # the gripper sample is a hand-written graph: Apply sets the parameter's default and rebuilds every size
    assert client.post("/api/reference", json={"family": "gripper"}).json()["ok"]
    assert not client.get("/api/state").json()["script"]
    r = client.post("/api/param/set", json={"values": {"pad_thick": 14}}).json()
    assert r["ok"] and r["old"] == "pad_thick = 12" and r["new"] == "pad_thick = 14"
    st = client.get("/api/state").json()
    assert next(q for q in st["parameters"] if q["name"] == "pad_thick")["default"] == 14
    assert client.post("/api/param/set", json={"values": {"rod_d": 7}}).status_code == 400   # a catalogue column
