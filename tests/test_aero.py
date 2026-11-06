"""Aero estimate: the vortex lattice against textbook values, and the flying-car example end to end on the kernel."""
import math

import pytest

from caddy.kernel.aero import analyse, vlm_polar, zero_lift_angle
from smlibquery.kernel import wing_planform


def _wing(half, root=1000, tip=1000, naca=12):
    return [wing_planform(dict(naca=naca, root_chord=root, tip_chord=tip, span=half, x=0, y=0, z=0, side=s)) for s in (1, -1)]


@pytest.mark.parametrize("half", [1000, 3000, 5000])
def test_lift_slope_matches_lifting_surface_theory(half):
    p = vlm_polar(_wing(half), [0, 5])
    ar = p["AR"]
    cla = (p["CL"][1] - p["CL"][0]) / math.radians(5)
    helmbold = 2 * math.pi * ar / (2 + math.sqrt(ar * ar + 4))
    assert abs(cla - helmbold) / helmbold < 0.08


def test_span_efficiency_is_physical():
    for tip in (1000, 400):
        p = vlm_polar(_wing(3000, tip=tip), [5])
        e = p["CL"][0] ** 2 / (math.pi * p["AR"] * p["CDi"][0])
        assert 0.85 < e <= 1.02


def test_zero_lift_angle_of_naca_2412():
    assert abs(zero_lift_angle(2412) - (-2.08)) < 0.1      # published: about -2.1 degrees
    assert abs(zero_lift_angle(12)) < 1e-6                  # symmetric section


def test_flying_car_builds_and_flies():
    from caddy.agent.session import Session
    from caddy.kernel.catalogue import DATA
    s = Session(family="flying_car")
    s.build_cad((DATA / "families" / "flying_car.sq.py").read_text())
    r = s.reports[0]
    assert r.ok, [c for c in r.checks if not c["ok"]]
    a = analyse(r.result, speed=50.0)
    assert 5 < a["best"]["LD"] < 20
    assert 0.15 < a["Cd_frontal"] < 0.45                    # car-like body drag
    assert abs(a["span_m"] - 6.6) < 0.01
