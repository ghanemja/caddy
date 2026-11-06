"""Build the stability library caddy reads: a compact index of SimReady objects with their measured friction and
slide angle, plus the numbers that say how far the slip principle holds on them.

Source: ghanemja/behaviors, out/site/data.json (every asset's simulated outcomes, from NVIDIA's SimReady library,
CC BY 4.0). The geometry (centre of mass, footprint) is NOT in that file, so this script can only test the
friction (slip) principle. The tipping principle is computed exactly by caddy's kernel for each design and is
not fitted here; see docs/VERIFY.md.

    python tools/build_stability_library.py ../behaviors/out/site/data.json data/library/stability_index.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def build(data_path: str, out_path: str) -> dict:
    assets = json.loads(Path(data_path).read_text())["assets"]
    rows = [a for a in assets if a.get("sum") and a.get("mu") is not None]
    mu = np.array([a["mu"] for a in rows], float)
    sl = np.array([a["sum"]["sl"] for a in rows], float)      # measured slide angle on the ramp, degrees
    tp = np.array([a["sum"]["tp"] for a in rows], float)      # measured tip angle, capped at the 50 degree ramp
    arch = [a.get("arch") for a in rows]
    slip = np.degrees(np.arctan(mu))
    err = np.abs(sl - slip)
    report = {
        "n": len(rows),
        "corr_slide_vs_atan_mu": round(float(np.corrcoef(sl, slip)[0, 1]), 3),
        "median_abs_error_deg": round(float(np.median(err)), 2),
        "p90_abs_error_deg": round(float(np.percentile(err, 90)), 2),
        "tip_angle_capped_at_deg": 50.0,
        "share_with_tip_at_cap": round(float(np.mean(tp >= 49.9)), 3),
        "note": ("The measured tip angle reaches the 50 degree test-ramp cap for the share above, and the 'slides "
                 "before it tips' test is true for nearly every object, so it cannot be checked on this data. "
                 "Only the slip relation is tested."),
    }
    index = {
        "source": "ghanemja/behaviors (out/site/data.json) from NVIDIA SimReady, CC BY 4.0",
        "principle": ("A rigid object on a slope slides when the slope exceeds atan(mu), and tips when the slope exceeds "
                      "atan(d / h), where d is the distance from its centre of mass to the nearest support edge and h "
                      "is the height of its centre of mass. Whichever angle is smaller decides."),
        "validation": report,
        # [id, name, friction mu, measured slide angle deg, measured archetype]
        "assets": [[a["id"], a["name"], round(a["mu"], 3), round(a["sum"]["sl"], 1), arch[i]]
                   for i, a in enumerate(rows)],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(index, separators=(",", ":")))
    return report


if __name__ == "__main__":
    rep = build(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=1))
