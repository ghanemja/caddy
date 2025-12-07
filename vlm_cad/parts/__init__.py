"""
Part metadata and labeling module.

This module provides:
- PartInfo: Metadata for individual segmented parts
- PartTable: Collection of parts with per-vertex labels
- JSON export/import for human labeling UI
- Integration with segmentation backends
"""

from .parts import (
    PartInfo,
    PartTable,
    build_part_table_from_segmentation,
    part_table_to_labeling_json,
    apply_labels_from_json,
)

__all__ = [
    "PartInfo",
    "PartTable",
    "build_part_table_from_segmentation",
    "part_table_to_labeling_json",
    "apply_labels_from_json",
]

