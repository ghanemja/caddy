"""Golden-master digests: an approved digest per size; any change shows as a text diff to approve.
Re-approve after a reviewed change with: pytest --update-golden"""
from pathlib import Path

import pytest

from .conftest import ROWS

GOLDEN = Path(__file__).parent / "golden" / "bspt_nipple"


@pytest.mark.parametrize("i", range(len(ROWS)), ids=[f'{r["inlet"]}x{r["outlet"]}' for r in ROWS])
def test_digest_matches_golden(family, update_golden, i):
    rep = family[i]
    path = GOLDEN / (rep.size.replace("/", "-").replace(" ", "") + ".digest.txt")
    text = rep.digest() + "\n"
    if update_golden or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        pytest.skip("golden written")
    assert text == path.read_text()
