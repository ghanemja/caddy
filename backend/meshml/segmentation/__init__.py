"""
Segmentation backend abstraction layer.

This module provides a unified interface for different 3D part segmentation backends,
allowing easy switching between PointNet++ and Hunyuan3D-Part (P3-SAM).
"""

from .types import PartSegmentationResult
from .backends import (
    PartSegmentationBackend,
    create_segmentation_backend,
    PointNetSegmentationBackend,
    Hunyuan3DPartSegmentationBackend,
)

__all__ = [
    "PartSegmentationResult",
    "PartSegmentationBackend",
    "create_segmentation_backend",
    "PointNetSegmentationBackend",
    "Hunyuan3DPartSegmentationBackend",
]

