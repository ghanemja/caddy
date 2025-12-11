"""
Core application modules
Initialization and setup code.
"""
import os
import sys
import mimetypes
from pathlib import Path

# Setup paths
BACKEND_DIR = Path(__file__).parent.parent.parent.resolve()
ROOT_DIR = BACKEND_DIR.parent

# Add paths, but ensure current directory doesn't interfere with package imports
if '' in sys.path:
    sys.path.remove('')
if str(os.getcwd()) in sys.path:
    sys.path.remove(str(os.getcwd()))

# Add our paths
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# MIME fix for ESM
mimetypes.add_type("application/javascript", ".js")

# Initialize FreeCAD and CadQuery
from app.services.freecad_service import FreeCAD, init_cadquery_with_freecad
import cadquery as cq
cq, exporters, Workplane = init_cadquery_with_freecad(FreeCAD)

# CAD components are no longer loaded at startup
# Users will upload CadQuery models or Python files as needed
# No rover components - removed legacy rover system

# cqparts shim
try:
    from cqparts.utils.geometry import CoordSystem
except Exception:
    class CoordSystem:
        def __sub__(self, other):
            return self
        def __rsub__(self, other):
            return other

cq.Workplane.world_coords = property(lambda self: CoordSystem())
cq.Workplane.local_coords = property(lambda self: CoordSystem())
