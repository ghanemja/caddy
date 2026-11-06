"""Load USD once, quietly.

usd_brep bundles the USD C++ libraries. On first use, USD's script-module loader also tries to import USD's Python
bindings (`pxr`), which are deliberately not installed (usd-core next to usd_brep crashes). Each library then prints a
harmless "No module named 'pxr'" warning from a background thread. We trigger that first load at startup with the
process's stderr pointed at /dev/null, so real exports later print nothing.
"""
from __future__ import annotations

import os
import tempfile
import time

import usd_brep as u

_done = False


def warm_up() -> None:
    global _done
    if _done:
        return
    _done = True
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 2)
            path = os.path.join(tempfile.gettempdir(), "caddy_usd_warmup.usda")
            u.usd.export_brep(u.create_box((0, 0, 0), 1, 1, 1), path)
            time.sleep(0.5)          # the loader prints from a secondary thread
    finally:
        os.dup2(saved, 2)
        os.close(saved)
