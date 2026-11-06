"""smlibquery: CadQuery-compatible calls build exact SMLib solids; scripts compile to expression IR; anything
outside the subset is refused loudly."""
import math
import re

import pytest

import smlibquery as sq


def vol(wp):
    return wp.val().volume()


def test_box_and_cylinder_match_cadquery_defaults():
    assert vol(sq.Workplane("XY").box(10, 20, 5)) == pytest.approx(1000, rel=1e-6)
    b = sq.Workplane("XY").box(10, 20, 5).val().bounding_box(tight=True)
    assert b[0][2] == pytest.approx(-2.5) and b[1][2] == pytest.approx(2.5)        # centred, like CadQuery
    assert vol(sq.Workplane("XY").cylinder(10, 3)) == pytest.approx(math.pi * 9 * 10, rel=1e-4)


def test_polygon_diameter_is_circumscribed_like_cadquery():
    s = sq.Workplane("XY").polygon(6, 30).extrude(10).val()
    (x0, y0, _), (x1, y1, _) = s.bounding_box(tight=True)
    assert x1 - x0 == pytest.approx(30, abs=1e-6)                     # across corners
    assert y1 - y0 == pytest.approx(30 * math.cos(math.radians(30)), abs=1e-6)   # across flats


def test_faces_workplane_hole_and_loft():
    part = sq.Workplane("XY").box(20, 20, 10).faces(">Z").workplane().hole(6)
    assert vol(part) == pytest.approx(4000 - math.pi * 9 * 10, rel=1e-4)
    cone = sq.Workplane("XY").circle(5).workplane(offset=10).circle(3).loft()
    assert vol(cone) == pytest.approx(math.pi * 10 / 3 * (25 + 15 + 9), rel=1e-4)


def test_booleans_and_moves():
    a = sq.Workplane("XY").box(10, 10, 10)
    b = sq.Workplane("XY").box(10, 10, 10).translate((5, 0, 0))
    assert vol(a.union(b)) == pytest.approx(1500, rel=1e-6)
    a = sq.Workplane("XY").box(10, 10, 10)
    b = sq.Workplane("XY").box(10, 10, 10).translate((5, 0, 0))
    assert vol(a.intersect(b)) == pytest.approx(500, rel=1e-6)
    r = sq.Workplane("XY").box(10, 2, 2).rotate((0, 0, 0), (0, 0, 1), 90).val().bounding_box(tight=True)
    assert r[1][1] - r[0][1] == pytest.approx(10, abs=1e-6)


def test_script_keeps_catalogue_values_as_expressions():
    code = """
import smlibquery as sq
body = sq.Workplane("XY").polygon(6, af / cos(radians(30))).extrude(length).tag("hex.flat")
result = body.faces(">Z").workplane().hole(bore).tag("bore.wall")
"""
    g = sq.compile_script(code, {"af": 30.0, "length": 10.0, "bore": 14.0})
    blocks = {b["id"]: b for b in g["blocks"]}
    assert blocks["hex_flat"]["params"]["h"] == "length"
    assert blocks["bore_wall"]["params"] == {"z": "-1", "d": "bore", "h": "length + 2"}
    assert blocks["hex_flat"]["aliases"]["side"] == "hex.flat"
    small = sq.build(g, {"af": 22.0, "length": 8.0, "bore": 8.0})["result"]
    big = sq.build(g, {"af": 30.0, "length": 10.0, "bore": 14.0})["result"]
    assert small.volume() < big.volume()                       # one script, every size


def test_top_level_numbers_and_math_record_symbolically():
    g = sq.compile_script("import smlibquery as sq\nresult = sq.Workplane().box(2 * a, a + 1, math.sqrt(a))", {"a": 4.0})
    p = g["blocks"][0]["params"]
    assert p["dx"] == "2 * a" and p["dy"] == "a + 1" and p["dz"] == "sqrt(a)"


@pytest.mark.parametrize("code,msg", [
    ("import os", "only `import smlibquery as sq`"),
    ("import cadquery as cq2\nimport numpy", "only `import smlibquery as sq`"),
    ("for i in range(2): pass", "For is not allowed"),
    ("result = sq.Workplane().box(1, 1, 1) if a > 1 else 0", "IfExp"),
    ("result = sq.Workplane().box(1, 1, 1).fillet(0.5)", "fillet() is not part of smlibquery"),
    ("result = sq.Workplane('XZ').box(1, 1, 1)", 'Workplane("XZ")'),
    ("x = a.__class__", "private"),
    ("result = sq.Workplane().polygon(6, 10, circumscribed=True).extrude(1)", "circumscribed"),
    ("result = 3", "assign the final Workplane"),
])
def test_outside_the_subset_is_refused_with_the_line(code, msg):
    with pytest.raises(sq.SMLQError, match=re.escape(msg)):
        sq.compile_script(code, {"a": 2.0})


def test_assembly_names_bodies_and_applies_locations():
    code = """
import smlibquery as sq
rod = sq.Workplane("XY").cylinder(40, d / 2).tag("rod")
pad = sq.Workplane("XY").box(10, 10, 10).tag("pad")
result = sq.Assembly().add(rod, name="rod").add(pad, name="pad", loc=sq.Location((d, 0, 0)))
"""
    g = sq.compile_script(code, {"d": 12.0})
    assert set(g["bodies"]) == {"rod", "pad"}
    bodies = sq.build(g, {"d": 12.0})
    assert bodies["pad"].bounding_box(tight=True)[0][0] == pytest.approx(7.0)


def test_scripts_saved_with_the_old_import_still_compile():
    """Scripts written before the rename (`import cadquery as cq`) build on the same SMLib layer."""
    g = sq.compile_script("import cadquery as cq\nresult = cq.Workplane().box(2, 3, 4)", {})
    assert sq.build(g, {})["result"].volume() == pytest.approx(24, rel=1e-6)
