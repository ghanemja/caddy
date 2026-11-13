"""Image → CAD: Claude models in SolidCraft over MCP; SMLib judges what SolidCraft exports.

The first tests need only SMLib. The rest drive the real `solidcraft-cli mcp` and are skipped where SolidCraft isn't
installed (SOLIDCRAFT_CLI or PATH). No test calls the LLM: where a run is needed, a scripted stand-in plays the model,
so what is proven is the loop, the tools and the kernel's verdicts, not Claude's reading of a picture.
"""
import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from smlibquery.kernel import BOOLEANS, make_primitive

from caddy.solidcraft import judge
from caddy.solidcraft.mcp import Mcp, McpError, find_cli

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_mcp.py")]
needs_solidcraft = pytest.mark.skipif(not find_cli(), reason="solidcraft-cli is not installed (set SOLIDCRAFT_CLI)")


def box(cx, cy, cz, dx, dy, dz):
    return make_primitive("box", dict(cx=cx, cy=cy, cz=cz, dx=dx, dy=dy, dz=dz))


def hole(cx, cy):
    return make_primitive("cylinder", dict(cx=cx, cy=cy, z=-1, d=8, h=12))


# --- the symmetry measure, on SMLib alone ------------------------------------------------------------------

def test_a_plate_with_holes_on_both_sides_is_its_own_mirror_image():
    plate = BOOLEANS["difference"](BOOLEANS["difference"](box(0, 0, 5, 100, 60, 10), hole(30, 0)), hole(-30, 0))
    assert judge.mirror_deviation({"plate": plate}, "YZ")["dev"] == 0.0
    assert judge.mirror_deviation({"plate": plate}, "XZ")["dev"] == 0.0


def test_a_hole_on_one_side_only_is_measured_and_located():
    plate = BOOLEANS["difference"](box(0, 0, 5, 100, 60, 10), hole(30, 0))
    m = judge.mirror_deviation({"plate": plate}, "YZ")
    assert 3.5 < m["dev"] <= 5.0                  # the Ø8 wall is up to 4 mm (and 5 mm deep) from any mirrored face
    assert abs(abs(m["at"][0]) - 30) <= 4.1 and abs(m["at"][1]) <= 4.1     # the point named is on that hole's wall
    assert judge.mirror_deviation({"plate": plate}, "XZ")["dev"] == 0.0     # still symmetric front to back
    assert m["dev"] > judge.symmetry_tolerance({"plate": plate})


def test_symmetry_is_judged_across_separate_bodies_and_off_origin():
    legs = {f"leg{i}": box(100 + x, y, 20, 5, 5, 40) for i, (x, y) in enumerate([(40, 25), (-40, 25), (40, -25), (-40, -25)])}
    m = judge.mirror_deviation(legs, "YZ")
    assert m["dev"] == 0.0 and m["mid"] == 100.0         # the plane is the middle of the model, not the world origin
    legs["leg3"] = box(100 - 40, -15, 20, 5, 5, 40)      # one leg moved 10 mm
    assert judge.mirror_deviation(legs, "YZ")["dev"] > 2.0


def test_body_names_become_unique_identifiers():
    taken = set()
    assert [judge.ident(n, taken) for n in ["Leg", "Leg (2)", "leg", "3rd Rail", ""]] == ["leg", "leg_2", "leg_3", "b_3rd_rail", "body"]


# --- the MCP client, against a fake server ------------------------------------------------------------------

def test_mcp_client_speaks_the_protocol_and_skips_noise():
    with Mcp(FAKE, timeout=10) as m:
        assert m.server["name"] == "fake" and m.instructions == "hello"
        assert [t["name"] for t in m.tools()] == ["echo", "picture", "fail", "die"]
        assert m.call("echo", {"text": "hi"}).json() == {"text": "hi"}
        r = m.call("picture")
        assert r.images == [("image/png", "aGk=")] and not r.is_error
        r = m.call("fail")
        assert r.is_error and "operation failed" in r.text          # an engine failure is a result the model can read
        with pytest.raises(McpError, match="unknown tool"):
            m.call("nope")                                           # a protocol error is not


def test_mcp_client_reports_a_server_that_exits_with_its_last_words():
    m = Mcp(FAKE, timeout=10)
    with pytest.raises(McpError, match="out of cheese"):
        m.call("die")
    m.close()


def test_a_missing_solidcraft_is_said_plainly(monkeypatch, tmp_path):
    from caddy.solidcraft.session import ImageSession
    monkeypatch.setenv("SOLIDCRAFT_CLI", str(tmp_path / "nowhere"))
    with pytest.raises(McpError, match="solidcraft-cli was not found"):
        ImageSession(b"\x89PNG\r\n\x1a\n", workdir=tmp_path)
    with pytest.raises(ValueError, match="PNG or a JPEG"):
        ImageSession(b"GIF89a", workdir=tmp_path)


# --- the whole loop on the real SolidCraft ------------------------------------------------------------------

PARAMS = [("top_w", "800 mm"), ("top_d", "500 mm"), ("top_t", "30 mm"), ("h", "450 mm"), ("leg", "40 mm")]
TOP = {"command": "solid.box", "params": {"length": "top_w", "width": "top_d", "height": "top_t",
                                          "center": [0, 0, "h - top_t/2"], "body_name": "Top", "name": "Top"}}
LEG = {"command": "solid.box", "params": {"length": "leg", "width": "leg", "height": "h - top_t",
                                          "corner": ["top_w/2 - leg - 20", "top_d/2 - leg - 20", 0],
                                          "body_name": "Leg", "name": "Leg"}}
ADD = [{"command": "parameters.add", "params": {"name": n, "expression": e}} for n, e in PARAMS]
MIRROR_Y = {"command": "solid.mirror", "params": {"bodies": ["Leg"], "plane": "XZ", "name": "LegsFrontBack"}}
MIRROR_X = {"command": "solid.mirror", "params": {"bodies": ["Leg", "Leg (2)"], "plane": "YZ", "name": "LegsLeftRight"}}
VIEW = {"view": "iso"}


@pytest.fixture(scope="module")
def picture(tmp_path_factory):
    """A picture of a four-legged table: SolidCraft's own render of the model the tests then have to rebuild."""
    d = tmp_path_factory.mktemp("truth")
    with Mcp([find_cli(), "mcp", "--headless", "--allow-path", str(d)]) as m:
        assert not m.call("batch", {"verbose": False, "commands": ADD + [TOP, LEG, MIRROR_Y, MIRROR_X]}).is_error
        r = m.call("screenshot", {**VIEW, "source": "model", "grid": False, "axes": False, "width": 640, "height": 480})
        assert r.images
        return base64.b64decode(r.images[0][1])


@pytest.fixture
def session(picture, tmp_path):
    from caddy.solidcraft.session import ImageSession
    s = ImageSession(picture, hint="a table", workdir=tmp_path)
    yield s
    s.close()


def checks(s):
    return {c["id"]: c for c in s.reports[0].checks}


@needs_solidcraft
def test_the_kernel_refuses_then_accepts_as_the_model_is_repaired(session):
    s = session
    assert not s.verdict()["ok"]
    out, _, err = s.call("finish", {})                              # nothing built
    assert err and "nothing to judge" in out

    # a table with legs on one side only, an unused parameter, and no description of the picture
    unused = {"command": "parameters.add", "params": {"name": "spare", "expression": "7 mm"}}
    out, _, err = s.call("batch", {"verbose": False, "commands": ADD + [unused, TOP, LEG, MIRROR_Y]})
    assert not err and "NOT accepted" in out
    c = checks(s)
    assert c["export_intact"]["ok"] and c["export_intact"]["measured"] < 1e-6      # SMLib reads back the same volume
    assert c["features_healthy"]["ok"] and c["parametric"]["measured"] == 6
    assert not c["symmetric"]["ok"]                                  # no plane declared yet
    assert not c["parameters_drive"]["ok"] and [d["name"] for d in s.facts["dead"]] == ["spare"]
    assert c["matches_picture"]["measured"] is None                  # no view declared: not measured, so not passed

    # "none" needs evidence; a plane is checked on SMLib
    with pytest.raises(Exception, match="assumed mirror-symmetric"):
        s.describe_object(dict(object="table", symmetry_plane="none", picture_view=VIEW))
    out, _, _ = s.call("describe_object", dict(object="four-legged table", symmetry_plane="YZ", picture_view=VIEW,
                                               assumptions="the hidden legs mirror the visible ones"))
    c = checks(s)
    assert not c["symmetric"]["ok"] and c["symmetric"]["measured"] > 300      # the legs are ~700 mm from their mirror
    assert "no mirror partner across YZ" in out and "body leg" in out
    assert c["matches_picture"]["measured"] is not None and not s.verdict()["ok"]

    out, _, err = s.call("finish", {})
    assert err and "not accepted" in out and not s.saved

    # mirror the legs: symmetric now; the spare parameter still fails
    s.call("batch", {"verbose": False, "commands": [MIRROR_X]})
    c = checks(s)
    assert c["symmetric"]["ok"] and c["symmetric"]["measured"] == 0.0
    assert c["matches_picture"]["ok"] and c["matches_picture"]["measured"] > 0.95
    assert s.verdict()["failing_checks"] == ["parameters_drive"]

    s.call("execute", {"command": "parameters.delete", "params": {"name": "spare"}})
    assert s.verdict()["ok"], s.report()
    out, _, err = s.call("finish", {})
    assert not err and "accepted" in out
    assert s.saved["design"].name == "model.sc" and s.saved["design"].stat().st_size > 1000
    assert s.saved["step"].read_text().startswith("ISO-10303-21")

    # the result is parametric: a value change rebuilds in SolidCraft and SMLib measures the new solid
    s.set_parameter("h", "600 mm")
    r = s.reports[0].result
    assert len(r.bodies) == 5 and abs(judge.bbox_of(r.bodies)[1][2] - 600) < 1e-6
    assert checks(s)["symmetric"]["ok"]
    assert not checks(s)["matches_picture"]["ok"] or checks(s)["matches_picture"]["measured"] < 0.99   # taller than the picture


@needs_solidcraft
def test_a_feature_error_and_a_wrong_outline_are_caught(session):
    s = session
    s.call("describe_object", dict(object="table", symmetry_plane="YZ", picture_view=VIEW))
    s.call("batch", {"verbose": False, "commands": ADD + [TOP, LEG, MIRROR_Y, MIRROR_X]})
    assert s.verdict()["ok"]
    s.set_parameter("top_w", "2400 mm")                            # three times as wide as the table in the picture
    c = checks(s)
    assert c["symmetric"]["ok"] and not c["matches_picture"]["ok"]
    assert s.verdict()["failing_checks"] == ["matches_picture"]
    s.set_parameter("top_w", "800 mm")
    assert s.verdict()["ok"]
    out, _, err = s.call("execute", {"command": "solid.fillet", "params": {"edges": [[0, 0, 5000]], "radius": 3}})
    assert err                                                       # SolidCraft's refusal reaches the model as an error
    assert s.verdict()["ok"]                                         # and leaves the design as it was


class Scripted:
    """Stands in for the LLM: plays a fixed list of tool calls, one request each, and records what it was sent."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []
        self.beta = SimpleNamespace(messages=self)
        self.messages = self

    def create(self, **kw):
        self.seen.append(kw)
        text, calls = self.turns.pop(0)
        content = [SimpleNamespace(type="text", text=text)] + [
            SimpleNamespace(type="tool_use", id=f"t{len(self.seen)}_{i}", name=n, input=a) for i, (n, a) in enumerate(calls)]
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0,
                                model_dump=lambda **k: dict(input_tokens=10, output_tokens=5))
        return SimpleNamespace(content=content, usage=usage, stop_reason="tool_use" if calls else "end_turn")


@needs_solidcraft
def test_the_agent_loop_sends_the_picture_and_stops_only_when_the_kernel_accepts(session, monkeypatch, tmp_path):
    from caddy.agent import loop
    from caddy.solidcraft.agent import SYSTEM, ImageAgent, task
    model = Scripted([
        ("It is a table.", [("describe_object", dict(object="table", symmetry_plane="YZ", picture_view=VIEW))]),
        ("Building.", [("batch", {"verbose": False, "commands": ADD + [TOP, LEG, MIRROR_Y]})]),
        ("Done?", [("finish", {})]),                                # two legs are missing: refused
        ("Mirroring.", [("batch", {"verbose": False, "commands": [MIRROR_X]}),
                        ("compare_picture", VIEW), ("screenshot", {**VIEW, "source": "model", "width": 240, "height": 160})]),
        ("Done.", [("finish", {})]),
        ("should never be asked", []),
    ])
    monkeypatch.setattr(loop.anthropic, "Anthropic", lambda: model)
    monkeypatch.setattr(loop, "RUNS", tmp_path / "runs")
    events = []
    res = ImageAgent(session, on_event=events.append).run(task("a table"))
    assert res.ok and res.claimed_done and res.false_done == 1 and res.rounds == 5 and len(model.turns) == 1

    first = model.seen[0]
    assert first["system"] == SYSTEM and "symmetric" in SYSTEM.lower()
    assert first["messages"][0]["content"][0]["type"] == "image"            # the picture goes in with the task
    names = [t["name"] for t in first["tools"]]
    assert {"batch", "execute", "set_parameter", "skill_reference", "describe_object", "compare_picture", "finish"} <= set(names)
    assert not {"open", "save", "export", "new_design"} & set(names)        # files stay caddy's business
    results = model.seen[4]["messages"][8]["content"]                        # what came back from the fourth turn
    assert [r["type"] for r in results] == ["tool_result"] * 3
    assert "caddy's kernel check" in results[0]["content"][0]["text"]        # the verdict rides on the build's result
    assert any(b["type"] == "image" for b in results[1]["content"])          # the overlay
    assert any(b["type"] == "image" for b in results[2]["content"])          # the render

    tools = [e for e in events if e["type"] == "tool"]
    assert [e["name"] for e in tools] == ["describe_object", "batch", "finish", "batch", "compare_picture", "screenshot", "finish"]
    assert tools[2]["error"] and not tools[2]["kernel"]["ok"] and tools[-1]["kernel"]["ok"]
    assert tools[4]["overlay"].startswith("data:image/png;base64,") and "saved" in tools[-1]
    log = [json.loads(line) for line in open(res.log)]
    assert log[0]["type"] == "start" and log[-1]["type"] == "end" and log[-1]["ok"]
    assert "base64" not in json.dumps([e for e in log if e["type"] == "tool" and e["name"] == "screenshot"])


@needs_solidcraft
def test_the_page_api_starts_a_run_from_a_picture_and_changes_values_on_the_kernel(picture, monkeypatch, tmp_path):
    import time
    from fastapi.testclient import TestClient
    from caddy.agent import loop
    from caddy.app import server
    model = Scripted([
        ("", [("describe_object", dict(object="table", symmetry_plane="YZ", picture_view=VIEW))]),
        ("", [("batch", {"verbose": False, "commands": ADD + [TOP, LEG, MIRROR_Y, MIRROR_X]})]),
        ("", [("finish", {})]),
    ])
    monkeypatch.setattr(loop.anthropic, "Anthropic", lambda: model)
    monkeypatch.setattr(loop, "RUNS", tmp_path / "runs")
    c = TestClient(server.app)
    assert c.post("/api/image/start", json=dict(image="not base64 !", name="x.png")).status_code == 400
    url = "data:image/png;base64," + base64.b64encode(picture).decode()
    try:
        assert c.post("/api/image/start", json=dict(image=url, hint="a table", name="table.png")).json() == {"ok": True}
        assert c.post("/api/image/start", json=dict(image=url)).status_code == 409          # one run at a time
        for _ in range(300):
            if not server.HUB.running:
                break
            time.sleep(0.1)
        st = c.get("/api/state").json()
        assert st["family"] == "image" and st["verdict"]["ok"] and st["spec"]["title"] == "Image → CAD: table"
        assert st["image"]["saved"] == {"design": "model.sc", "step": "model.step"}
        assert [p["name"] for p in st["image"]["parameters"]] == [n for n, _ in PARAMS]
        assert {p["name"]: p["default"] for p in st["parameters"]}["h"] == 450       # SolidCraft's parameters, as design values
        assert len(c.get("/api/mesh/0").json()["tris"]) > 0                            # SMLib's mesh of the imported solids
        assert len(c.get("/api/image/file/design").content) > 1000                    # the .sc SolidCraft opens

        # a what-if is rebuilt by SolidCraft, judged on SMLib, and not kept
        pv = c.post("/api/preview", json=dict(size=0, values={"h": 700})).json()
        assert pv["bbox"][1][2] == 700 and c.get("/api/state").json()["image"]["parameters"][3]["value"] == 450
        wide = c.post("/api/preview", json=dict(size=0, values={"top_w": 2400})).json()
        assert not wide["ok"] and [k["id"] for k in wide["checks"] if not k["ok"]] == ["matches_picture"]
        # Apply keeps it: the design, the solids and the saved files follow
        assert c.post("/api/param/set", json=dict(values={"leg": 60})).json()["ok"]
        st = c.get("/api/state").json()
        assert st["image"]["parameters"][4]["value"] == 60 and st["verdict"]["ok"]
        assert c.post("/api/param/set", json=dict(values={"nope": 1})).status_code == 400
        # a sample family can be started afterwards, and the image session's SolidCraft process is closed
        s = server.HUB.session
        assert c.post("/api/reference", json=dict(family="bspt_nipple")).json()["ok"]
        assert s.mcp.p.poll() is not None
    finally:
        server.HUB.running = False
        server.HUB.reset("bspt_nipple")
