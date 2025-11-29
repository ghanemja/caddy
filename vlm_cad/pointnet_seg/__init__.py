"""
PointNet++ part segmentation module for ShapeNetPart.

This module provides:
- PointNet++ model loading and inference
- Mesh to point cloud conversion
- Part segmentation on point clouds
- Geometric parameter extraction from segmented parts
"""

from .model import PointNet2PartSegWrapper, load_pretrained_model
from .inference import segment_mesh
from .mesh_io import load_mesh_as_point_cloud
from .geometry import (
    compute_part_bounding_boxes,
    axis_extent,
    compute_part_statistics,
)
from .labels import SHAPENETPART_CATEGORY_LABELS, get_label_name

__all__ = [
    "PointNet2PartSegWrapper",
    "load_pretrained_model",
    "segment_mesh",
    "load_mesh_as_point_cloud",
    "compute_part_bounding_boxes",
    "axis_extent",
    "compute_part_statistics",
    "SHAPENETPART_CATEGORY_LABELS",
    "get_label_name",
]

