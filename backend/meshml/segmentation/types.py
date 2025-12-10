"""
Type definitions for part segmentation results.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PartSegmentationResult:
    """
    Result from part segmentation backend.
    
    This dataclass encapsulates the output of any segmentation backend,
    providing a consistent interface for downstream processing.
    """
    # Per-point or per-vertex labels
    # Shape: [N] where N is number of points/vertices
    labels: np.ndarray
    
    # Point cloud coordinates (if available)
    # Shape: [N, 3]
    # Note: For mesh-based segmentation, this may be sampled points
    points: Optional[np.ndarray] = None
    
    # Optional: per-vertex labels for full mesh
    # Shape: [M] where M is number of vertices in original mesh
    # If None, labels are assumed to correspond to points
    vertex_labels: Optional[np.ndarray] = None
    
    # Optional: vertices of the original mesh (for vertex_labels)
    # Shape: [M, 3] where M is number of vertices
    vertices: Optional[np.ndarray] = None
    
    # Optional: per-face labels
    # Shape: [F] where F is number of faces
    face_labels: Optional[np.ndarray] = None
    
    # Optional: logits/probabilities for each point/vertex
    # Shape: [N, num_classes] or [M, num_classes]
    logits: Optional[np.ndarray] = None
    
    # Optional: part meshes or bounding boxes
    # Dictionary mapping part_id -> metadata
    part_metadata: Optional[Dict[int, Dict[str, Any]]] = None
    
    # Metadata
    num_parts: Optional[int] = None
    num_points: Optional[int] = None
    num_vertices: Optional[int] = None
    num_faces: Optional[int] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.labels is None:
            raise ValueError("labels cannot be None")
        
        if self.num_parts is None:
            self.num_parts = len(np.unique(self.labels))
        
        if self.num_points is None and self.points is not None:
            self.num_points = len(self.points)
        
        if self.num_vertices is None and self.vertex_labels is not None:
            self.num_vertices = len(self.vertex_labels)
        
        if self.num_faces is None and self.face_labels is not None:
            self.num_faces = len(self.face_labels)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "labels": self.labels.tolist(),
            "num_parts": self.num_parts,
        }
        
        if self.points is not None:
            result["points"] = self.points.tolist()
            result["num_points"] = self.num_points
        
        if self.vertex_labels is not None:
            result["vertex_labels"] = self.vertex_labels.tolist()
            result["num_vertices"] = self.num_vertices
        
        if self.face_labels is not None:
            result["face_labels"] = self.face_labels.tolist()
            result["num_faces"] = self.num_faces
        
        if self.logits is not None:
            result["logits"] = self.logits.tolist()
        
        if self.part_metadata is not None:
            # Convert numpy arrays in metadata to lists
            metadata_dict = {}
            for part_id, meta in self.part_metadata.items():
                metadata_dict[int(part_id)] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in meta.items()
                }
            result["part_metadata"] = metadata_dict
        
        return result

