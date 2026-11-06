"""One-off: convert a STEP file into a caddy stock part (SMLib's native ASCII .smb + a metadata JSON).

NVIDIA's pip usd-brep has no STEP reader, so this uses smlibpy (AWV Informatik, MIT + Apache-2.0), which bundles SMLib
and reads STEP. Run it in its own venv (smlibpy and usd-brep are separate builds of the kernel); caddy itself only
reads the resulting .smb files, so reviewers never need smlibpy.

    python3.12 -m venv .venv-step && .venv-step/bin/pip install smlibpy==0.6.2
    .venv-step/bin/python tools/import_step.py <file.step> <stock_name> --source "<url>" --license "<license>"
"""
import argparse
import hashlib
import json
from pathlib import Path

import smlibpy

STOCK = Path(__file__).resolve().parents[1] / "data" / "stock"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("name")
    ap.add_argument("--source", default="")
    ap.add_argument("--license", default="")
    a = ap.parse_args()
    parts = smlibpy.import_step(a.step)
    parts = parts if isinstance(parts, (list, tuple)) else [parts]
    if len(parts) != 1:
        raise SystemExit(f"expected one solid in {a.step}, got {len(parts)}")
    b = parts[0]._brep
    STOCK.mkdir(parents=True, exist_ok=True)
    smb = STOCK / f"{a.name}.smb"
    b.write_to_file(str(smb), ascii=True)
    meta = dict(name=a.name, source=a.source, license=a.license, step=Path(a.step).name,
                faces=b.face_count(), closed_solid=bool(b.is_manifold_solid()), volume_mm3=round(b.volume(), 1),
                bbox_mm=[[round(x, 3) for x in p] for p in b.bounding_box(tight=True)],
                sha1=hashlib.sha1(smb.read_bytes()).hexdigest())
    (STOCK / f"{a.name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
