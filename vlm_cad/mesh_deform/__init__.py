"""
Parametric mesh deformation module for VLM-driven parameter editing.

This module provides tools to deform 3D meshes based on semantic parameters,
enabling VLM-driven shape modifications while respecting part segmentation.
"""

from .deformer import (
    MeshData,
    DeformationConfig,
    ParametricMeshDeformer,
    build_default_deformation_config_for_category,
)

__all__ = [
    "MeshData",
    "DeformationConfig",
    "ParametricMeshDeformer",
    "build_default_deformation_config_for_category",
]

