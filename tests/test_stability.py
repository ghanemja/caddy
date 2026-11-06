"""Stability: the kernel's tip and slip angles against hand calculations, the library lookup, and the agent tool.

What is NOT tested here, and why (docs/VERIFY.md): the tipping principle is not checked against measured tipping
on real objects, because the library has no geometry to compute tip angles from. The slip principle is checked
against the library's measured slide angles (see test_library_slip_relation).
"""
import math
from pathlib import Path

import numpy as np
import pytest
import usd_brep as u

from caddy.agent.loop import Agent
from caddy.agent.session import Session
from caddy.kernel.stability import hull2d, static_stability
from caddy.library.stability import card, index, similar_friction

HERE = Path(__file__).parent
NIPPLE = (HERE / "fixtures" / "nipple.sq.py").read_text()


def deg(x):
    return math.degrees(math.atan(x))


def test_wide_box_tips_over_its_edge_at_atan_2():
    # 2 x 2 x 1 box: centre of mass 0.5 above the floor, 1 from the nearest edge, so tan(tilt) = 1 / 0.5
    s = static_stability(u.create_box((0, 0, 0), 2, 2, 1), mu=0.4)
    assert s.tip_deg == pytest.approx(deg(2.0), abs=0.01)
    assert s.distance == pytest.approx(1.0, abs=1e-6)
    assert s.height == pytest.approx(0.5, abs=1e-6)
    assert s.inside and s.footprint_area == pytest.approx(4.0, abs=1e-6)
    assert s.top == pytest.approx(1.0, abs=1e-6)     # the body's height, used to draw it on the whiteboard


def test_tall_thin_box_tips_before_it_slides_on_grippy_surface():
    # 1 x 1 x 4: centre of mass 2 up, 0.5 from the edge: tip at atan(0.25) = 14.04 deg; slip at atan(0.4) = 21.8 deg
    s = static_stability(u.create_box((0, 0, 0), 1, 1, 4), mu=0.4)
    assert s.tip_deg == pytest.approx(deg(0.25), abs=0.01)
    assert s.slip_deg == pytest.approx(deg(0.4), abs=0.01)
    assert s.first == "tip"


def test_wide_box_slides_before_it_tips():
    s = static_stability(u.create_box((0, 0, 0), 2, 2, 1), mu=0.4)
    assert s.first == "slide"


def test_cylinder_footprint_is_its_round_contact_not_its_corners():
    # radius 1, height 2: centre of mass 1 up, nearest edge ~1 away (the round base), tilt ~45 deg, area ~pi
    s = static_stability(u.create_cylinder((0, 0, 0), 1.0, 2.0))
    assert s.tip_deg == pytest.approx(45.0, abs=0.3)
    assert s.footprint_area == pytest.approx(math.pi, rel=0.02)


def test_hull_is_counter_clockwise_and_drops_interior_points():
    pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0.5)]
    ring = hull2d(np.array(pts, float))
    assert sorted(pts[i] for i in ring) == [(0, 0), (0, 2), (2, 0), (2, 2)]


def test_library_index_is_built_and_attributed():
    ix = index()
    assert ix["validation"]["n"] == len(ix["assets"]) > 2000
    assert "CC BY 4.0" in ix["source"]


def test_library_slip_relation_holds_on_its_objects():
    # the principle's slip half, measured on 2,166 SimReady objects: slide angle tracks atan(mu)
    v = index()["validation"]
    assert v["corr_slide_vs_atan_mu"] > 0.5
    assert v["median_abs_error_deg"] < 5.0


def test_lookup_returns_nearest_friction_with_its_measured_slides():
    r = similar_friction(0.4)
    assert r["n"] > 0 and len(r["examples"]) == 5
    assert all(abs(e["mu"] - 0.4) <= 0.05 for e in r["examples"])
    assert abs(r["library_median_slide_deg"] - r["slip_deg"]) < 6.0
    assert "not tested" in card()["tip_check"]  # the card names what was and was not tested


def test_agent_tool_reads_a_built_body_and_changes_nothing():
    s = Session()
    s.build_cad(NIPPLE)
    before = s.verdict()
    shim = Agent.__new__(Agent)     # the tool dispatcher without a client, as replay uses it
    shim.s = s
    out, err = Agent._call_tool(shim, "check_stability", {"mu": 0.3})
    assert not err
    assert "First to go:" in out and "μ 0.3" in out and "Library, objects with μ≈0.3" in out
    assert s.last_stab and s.last_stab["first"] in ("tip", "slide") and s.last_stab["mu"] == 0.3
    assert s.verdict() == before    # read-only: the kernel's verdict is unchanged


def test_agent_tool_says_when_there_is_nothing_to_read():
    s = Session()
    shim = Agent.__new__(Agent)
    shim.s = s
    out, err = Agent._call_tool(shim, "check_stability", {})
    assert err and "nothing built" in out
