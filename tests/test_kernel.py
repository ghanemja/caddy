"""Smoke tests, runtime invariants and the spec oracle over every catalogue size (VERIFY.md)."""
import pytest

from caddy.kernel.build import KernelError, tessellate
from caddy.kernel.catalogue import rows, size_name

from .conftest import ROWS

IDS = [size_name(r) for r in ROWS]
PARTS = {"inlet", "hex", "outlet", "bore"}
BREAKS_THROUGH = {"3/4 x 3/8", "3/4 x 1/2"}   # real catalogue case: thread Ø27.16 > hex A/F 27


def test_catalogue_rows_add_up():
    for r in rows():
        assert r["inlet_thread_len"] + r["hex_len"] + r["outlet_thread_len"] == r["overall_len"]


@pytest.mark.parametrize("i", range(len(ROWS)), ids=IDS)
def test_smoke(family, i):
    rep = family[i]
    assert rep.error is None
    assert all(s["ok"] for s in rep.smoke), [s for s in rep.smoke if not s["ok"]]
    assert {f.part for f in rep.result.faces} == PARTS


@pytest.mark.parametrize("i", range(len(ROWS)), ids=IDS)
def test_every_catalogue_cell_matches(family, i):
    rep = family[i]
    bad = [c for c in rep.checks if not c["ok"]]
    if rep.size in BREAKS_THROUGH:
        assert [c["id"] for c in bad] == ["thread_clears_hex"]
        assert bad[0]["measured"] == pytest.approx(27.1597, abs=1e-3)
    else:
        assert bad == []
    assert sum(1 for c in rep.checks if c["column"]) == 6   # every catalogue column measured


@pytest.mark.parametrize("i", range(len(ROWS)), ids=IDS)
def test_aliases(family, i):
    res = family[i].result
    assert len(res.by_alias("hex.flat")) == 6
    assert len(res.by_alias("bore.wall")) == 1 and res.by_alias("bore.wall")[0].kind == "cylinder"
    assert len({f.alias for f in res.faces}) == len(res.faces)   # aliases are unique


def test_every_triangle_maps_to_a_face(family):
    res = family[0].result
    m = tessellate(res)
    assert len(m["tri_face"]) == len(m["tris"])
    assert {res.faces[i].part for i in set(m["tri_face"].tolist())} == PARTS


def test_kernel_error_becomes_repair_facts(graph, ev):
    from caddy.kernel import graph as G
    from caddy.kernel.catalogue import values_for
    g = G.apply_edit(graph, [{"block": "inlet_thread", "set": {"d_base": "-5"}}])
    with pytest.raises(KernelError) as e:
        ev.run(g, values_for(ROWS[0]))
    f = e.value.facts()
    assert f["block"] == "inlet_thread" and f["part"] == "inlet" and f["status"] == "SM_ERR_INVALID_INPUT"
