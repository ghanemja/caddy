"""
Part metadata and labeling utilities.

This module provides category-agnostic part metadata abstraction
for human labeling and semantic operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class PartInfo:
    """
    Metadata for a single segmented part in a mesh.
    
    This is category-agnostic - works for any type of 3D object.
    """
    part_id: int                     # integer label from segmentation
    name: Optional[str] = None       # human-assigned name, e.g. "backrest", "wheel_front_left"
    description: Optional[str] = None
    
    # Geometry stats (all in world coordinates of the mesh)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    principal_axes: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))  # 3x3 PCA axes
    extents: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))        # length along each PCA axis
    
    # Optional extra flags
    touches_ground: bool = False
    approx_area: Optional[float] = None
    approx_volume: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.centroid, np.ndarray):
            self.centroid = np.array(self.centroid, dtype=np.float32)
        if not isinstance(self.bbox_min, np.ndarray):
            self.bbox_min = np.array(self.bbox_min, dtype=np.float32)
        if not isinstance(self.bbox_max, np.ndarray):
            self.bbox_max = np.array(self.bbox_max, dtype=np.float32)
        if not isinstance(self.principal_axes, np.ndarray):
            self.principal_axes = np.array(self.principal_axes, dtype=np.float32)
        if not isinstance(self.extents, np.ndarray):
            self.extents = np.array(self.extents, dtype=np.float32)


@dataclass
class PartTable:
    """
    A collection of PartInfo plus per-vertex part labels.
    
    This is the main data structure for part metadata and labeling.
    """
    parts: Dict[int, PartInfo]                # part_id -> PartInfo
    vertex_part_labels: np.ndarray            # [N] array of integer labels (same N as vertex count)
    
    def get_part_ids(self) -> List[int]:
        """Get list of all part IDs."""
        return list(self.parts.keys())
    
    def get_named_parts(self) -> Dict[str, PartInfo]:
        """
        Return a mapping from semantic name -> PartInfo for parts with a non-empty name.
        
        Returns:
            Dictionary mapping part name -> PartInfo
        """
        return {p.name: p for p in self.parts.values() if p.name}
    
    def get_part_by_name(self, name: str) -> Optional[PartInfo]:
        """Get PartInfo by semantic name, or None if not found."""
        for part in self.parts.values():
            if part.name == name:
                return part
        return None


def compute_pca_axes_and_extents(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA principal axes and extents for a set of points.
    
    Args:
        points: [N, 3] array of points
        
    Returns:
        Tuple of (principal_axes [3, 3], extents [3])
        - principal_axes: columns are the principal directions (normalized)
        - extents: length along each principal axis
    """
    if len(points) < 3:
        # Degenerate case: return identity axes and default extents
        return np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    
    # Center points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Principal axes (columns are principal directions)
    principal_axes = eigenvectors.astype(np.float32)
    
    # Project points onto each principal axis to get extents
    extents = np.zeros(3, dtype=np.float32)
    for i in range(3):
        projections = np.dot(centered, principal_axes[:, i])
        extents[i] = np.max(projections) - np.min(projections)
    
    return principal_axes, extents


def infer_shape_hint(extents: np.ndarray, threshold_ratio: float = 0.3) -> str:
    """
    Infer a shape hint from extents.
    
    Args:
        extents: [3] array of extents along principal axes
        threshold_ratio: ratio threshold for shape classification
        
    Returns:
        Shape hint string: "long_thin", "flat_plate", "block_like", etc.
    """
    if len(extents) != 3:
        return "unknown"
    
    # Sort extents
    sorted_extents = np.sort(extents)
    max_extent = sorted_extents[2]
    mid_extent = sorted_extents[1]
    min_extent = sorted_extents[0]
    
    # Avoid division by zero
    if max_extent < 1e-6:
        return "point_like"
    
    # Long and thin: one axis much longer than others
    if min_extent / max_extent < threshold_ratio and mid_extent / max_extent < threshold_ratio:
        return "long_thin"
    
    # Flat plate: one axis much smaller than others
    if min_extent / max_extent < threshold_ratio:
        return "flat_plate"
    
    # Block-like: all axes similar
    if min_extent / max_extent > (1.0 - threshold_ratio):
        return "block_like"
    
    # Elongated: one axis longer, but not extreme
    if mid_extent / max_extent < (1.0 - threshold_ratio):
        return "elongated"
    
    return "irregular"


def build_part_table_from_segmentation(
    vertices: np.ndarray,
    part_labels: np.ndarray,
    ground_plane_z: Optional[float] = None,
    epsilon: float = 0.01,
) -> PartTable:
    """
    Construct a PartTable from mesh vertices and per-vertex part labels
    (e.g. from Hunyuan3D-Part segmentation).
    
    Args:
        vertices: [N, 3] array of vertex positions
        part_labels: [N] array of integer part labels
        ground_plane_z: Optional Z coordinate of ground plane
        epsilon: Tolerance for ground plane detection
        
    Returns:
        PartTable with PartInfo for each unique part
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    part_labels = np.asarray(part_labels, dtype=np.int32)
    
    if len(vertices) != len(part_labels):
        raise ValueError(f"Vertices and labels must have same length: {len(vertices)} != {len(part_labels)}")
    
    # Infer ground plane if not provided
    if ground_plane_z is None:
        ground_plane_z = np.min(vertices[:, 2]) + epsilon
    
    # Get unique part IDs
    unique_part_ids = np.unique(part_labels)
    
    parts = {}
    
    for part_id in unique_part_ids:
        # Extract vertices for this part
        mask = part_labels == part_id
        verts_part = vertices[mask]
        
        if len(verts_part) == 0:
            continue
        
        # Compute centroid
        centroid = np.mean(verts_part, axis=0).astype(np.float32)
        
        # Compute bounding box
        bbox_min = np.min(verts_part, axis=0).astype(np.float32)
        bbox_max = np.max(verts_part, axis=0).astype(np.float32)
        
        # Compute PCA axes and extents
        principal_axes, extents = compute_pca_axes_and_extents(verts_part)
        
        # Check if touches ground
        touches_ground = np.min(verts_part[:, 2]) <= (ground_plane_z + epsilon)
        
        # Approximate volume (from bounding box)
        bbox_extent = bbox_max - bbox_min
        approx_volume = np.prod(bbox_extent)
        
        # Approximate surface area (from bounding box)
        if len(bbox_extent) == 3:
            x, y, z = bbox_extent
            approx_area = 2 * (x * y + x * z + y * z)
        else:
            approx_area = None
        
        # Create PartInfo
        part_info = PartInfo(
            part_id=int(part_id),
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            principal_axes=principal_axes,
            extents=extents,
            touches_ground=touches_ground,
            approx_volume=approx_volume,
            approx_area=approx_area,
        )
        
        parts[int(part_id)] = part_info
    
    return PartTable(
        parts=parts,
        vertex_part_labels=part_labels,
    )


def part_table_to_labeling_json(
    part_table: PartTable,
    max_parts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert PartTable to a JSON-serializable dict for a labeling UI.
    
    The UI can iterate over `parts` and highlight each part using its part_id.
    The UI can let users assign `name` and `description`.
    
    Args:
        part_table: PartTable to convert
        max_parts: Optional limit on number of parts to include
        
    Returns:
        JSON-serializable dictionary with part information
    """
    part_list = []
    
    # Sort by part_id for consistent ordering
    sorted_part_ids = sorted(part_table.get_part_ids())
    
    if max_parts is not None:
        sorted_part_ids = sorted_part_ids[:max_parts]
    
    for part_id in sorted_part_ids:
        part_info = part_table.parts[part_id]
        
        # Infer shape hint
        shape_hint = infer_shape_hint(part_info.extents)
        
        # Build part entry - convert all NumPy types to native Python types
        part_entry = {
            "part_id": int(part_info.part_id),
            "provisional_name": f"part_{part_info.part_id}",
            "name": part_info.name,
            "description": part_info.description,
            "centroid": [float(x) for x in part_info.centroid.tolist()],
            "bbox_min": [float(x) for x in part_info.bbox_min.tolist()],
            "bbox_max": [float(x) for x in part_info.bbox_max.tolist()],
            "extents": [float(x) for x in part_info.extents.tolist()],
            "touches_ground": bool(part_info.touches_ground),
            "shape_hint": str(shape_hint),
            "approx_volume": float(part_info.approx_volume) if part_info.approx_volume is not None else None,
            "approx_area": float(part_info.approx_area) if part_info.approx_area is not None else None,
        }
        
        part_list.append(part_entry)
    
    return {
        "parts": part_list,
        "num_parts": len(part_list),
        "num_vertices": len(part_table.vertex_part_labels),
    }


def apply_labels_from_json(
    part_table: PartTable,
    labels_json: Dict[str, Any],
) -> PartTable:
    """
    Given a JSON dict in the format returned by part_table_to_labeling_json,
    update PartInfo.name and PartInfo.description accordingly.
    
    This lets the frontend send user-assigned names/descriptions back to us.
    
    Args:
        part_table: PartTable to update
        labels_json: JSON dict with part labels (from labeling UI)
        
    Returns:
        Updated PartTable (modifies in place, but returns for convenience)
    """
    if "parts" not in labels_json:
        raise ValueError("labels_json must contain 'parts' key")
    
    for part_entry in labels_json["parts"]:
        part_id = part_entry.get("part_id")
        if part_id is None:
            continue
        
        if part_id not in part_table.parts:
            # Skip unknown part IDs
            continue
        
        part_info = part_table.parts[part_id]
        
        # Update name if provided
        if "name" in part_entry and part_entry["name"]:
            part_info.name = str(part_entry["name"]).strip()
        
        # Update description if provided
        if "description" in part_entry:
            part_info.description = str(part_entry["description"]).strip() if part_entry["description"] else None
    
    return part_table

