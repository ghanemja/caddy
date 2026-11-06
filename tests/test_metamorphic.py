"""Metamorphic tests (VERIFY.md): known relations between edits and results, no ground truth, no AI."""
import pytest

from caddy.kernel.catalogue import values_for

from .conftest import ROWS


def face_sig(res, alias, dz=0.0):
    """Area and centroid of a named face; kernel integrals are accurate to ~1e-6 relative, so compare with approx."""
    f = res.by_alias(alias)[0]
    return pytest.approx([f.area, f.centroid[0], f.centroid[1], f.centroid[2] - dz], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("row", ROWS[:4] + ROWS[6:], ids=lambda r: f'{r["inlet"]}x{r["outlet"]}')
def test_hex_length_leaves_thread_faces_unchanged(graph, ev, row):
    v = values_for(row)
    a = ev.run(graph, v)
    b = ev.run(graph, {**v, "hex_len": v["hex_len"] + 3})
    fb = b.by_alias("inlet.thread")[0]
    assert [fb.area, *fb.centroid] == face_sig(a, "inlet.thread")
    fb = b.by_alias("outlet.thread")[0]
    assert [fb.area, fb.centroid[0], fb.centroid[1], fb.centroid[2] - 3] == face_sig(a, "outlet.thread")   # same face, moved up by 3


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f'{r["inlet"]}x{r["outlet"]}')
def test_bigger_hex_never_smaller_part(graph, ev, row):
    v = values_for(row)
    vols = [ev.run(graph, {**v, "hex_af": v["hex_af"] + d}).brep.volume() for d in (0, 1, 2)]
    assert vols[0] < vols[1] < vols[2]


def test_bigger_row_never_gives_smaller_part(family):
    by_inlet = {}
    for r in family:
        by_inlet.setdefault(r.row["inlet"], []).append(r.result.brep.volume())
    order = ["1/4", "3/8", "1/2", "3/4", "1"]
    maxes = [max(by_inlet[k]) for k in order if k in by_inlet]
    assert maxes == sorted(maxes)
