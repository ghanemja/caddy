"""
VLM CAD module for semantic geometry analysis and parameter extraction.

This package provides:
- PointNet++ part segmentation for 3D meshes
- VLM-powered semantic parameter extraction
- End-to-end mesh ingestion pipeline
- Parametric mesh deformation for VLM-driven shape editing
"""

__version__ = "0.1.0"

# Export main modules
from . import pointnet_seg
from . import semantics
from . import mesh_deform

__all__ = [
    "pointnet_seg",
    "semantics",
    "mesh_deform",
]

