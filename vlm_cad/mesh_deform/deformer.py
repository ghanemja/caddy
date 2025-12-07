"""
Parametric mesh deformation module for VLM-driven parameter editing.

This module provides tools to deform 3D meshes based on semantic parameters,
enabling VLM-driven shape modifications while respecting part segmentation.
The deformation is "parametric" in that a small set of numeric parameters
(e.g., wing_span, chord_length, seat_height) control semantic deformations
of the mesh geometry.
"""

from __future__ import annotations  # Defer evaluation of type hints

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

from .utils import compute_pca_axis, normalize_projection, get_world_axis

if TYPE_CHECKING:
    from ..parts.parts import PartTable, PartInfo


@dataclass
class MeshData:
    """Container for mesh geometry data."""
    vertices: np.ndarray  # [N, 3]
    faces: np.ndarray     # [M, 3] integer indices


@dataclass
class DeformationConfig:
    """
    Configuration for how a semantic parameter should deform the mesh.
    
    Attributes:
        affects_parts: list of part label names (e.g. ["left_wing", "right_wing"])
        mode: deformation mode, e.g. "axis_scale", "axis_stretch"
        axis_source: "pca" or "world" (e.g., "x", "y", "z")
    """
    affects_parts: List[str]
    mode: str = "axis_stretch"       # for now, support "axis_stretch"
    axis_source: str = "pca"         # or "x", "y", "z"


class ParametricMeshDeformer:
    """
    Parametric mesh deformer that applies semantic parameter-based deformations.
    
    Given a base mesh, part labels, and a configuration mapping parameters to
    deformation rules, this class can produce deformed meshes based on new
    parameter values.
    """
    
    def __init__(
        self,
        base_mesh: MeshData,
        base_parameters: Dict[str, float],
        part_labels: np.ndarray,
        part_label_names: Dict[int, str],
        config: Dict[str, DeformationConfig],
        part_table: Optional["PartTable"] = None,
        part_info_dict: Optional[Dict[int, "PartInfo"]] = None,
    ):
        """
        Initialize the parametric mesh deformer.
        
        Args:
            base_mesh: original mesh geometry (undeformed)
            base_parameters: semantic param values at baseline (e.g. {"wing_span": 2.0})
            part_labels: [N] array of integer part IDs per vertex
            part_label_names: mapping from part ID -> human-readable name
                (e.g., {0: "fuselage", 1: "left_wing", ...})
            config: mapping from semantic param name -> DeformationConfig
            part_table: Optional PartTable for part metadata
            part_info_dict: Optional dict of part_id -> PartInfo (alternative to part_table)
        """
        self.base_mesh = base_mesh
        self.base_parameters = base_parameters.copy()
        self.part_labels = np.asarray(part_labels)
        self.part_label_names = part_label_names.copy()
        self.config = config.copy()
        
        # Store PartTable or build from part_info_dict
        if part_table is not None:
            self.part_table = part_table
        elif part_info_dict is not None:
            # Create a minimal PartTable from part_info_dict
            from ..parts.parts import PartTable
            self.part_table = PartTable(
                parts=part_info_dict,
                vertex_part_labels=part_labels,
            )
        else:
            self.part_table = None
        
        # Per-part enabled flags (for enable/disable operations)
        self.part_enabled: Dict[int, bool] = {}
        if self.part_table:
            for part_id in self.part_table.get_part_ids():
                self.part_enabled[part_id] = True
        
        # Validate inputs
        if len(self.part_labels) != len(self.base_mesh.vertices):
            raise ValueError(
                f"Part labels length ({len(self.part_labels)}) must match "
                f"vertices length ({len(self.base_mesh.vertices)})"
            )
    
    def deform(
        self,
        new_parameters: Dict[str, float],
        enabled_parts: Optional[Dict[int, bool]] = None,
    ) -> MeshData:
        """
        Deform the mesh based on new parameter values.
        
        Returns a NEW MeshData with deformed vertices based on new_parameters.
        
        Strategy:
        - Start from base_mesh.vertices as the reference.
        - For each parameter in new_parameters:
            - Compare new value to baseline (scale factor or delta).
            - Apply deformations to vertices belonging to the configured parts.
        - Combine all deformations (for now, in a simple additive way).
        
        Args:
            new_parameters: dictionary of parameter name -> new value
            
        Returns:
            New MeshData with deformed vertices
        """
        verts = self.base_mesh.vertices.copy()
        faces = self.base_mesh.faces.copy()
        
        # Update enabled flags if provided
        if enabled_parts is not None:
            self.part_enabled.update(enabled_parts)
        
        # Apply per-part enable/disable: remove faces for disabled parts
        if self.part_enabled:
            enabled_part_ids = {pid for pid, enabled in self.part_enabled.items() if enabled}
            if enabled_part_ids:
                # Filter faces: keep only faces where all vertices belong to enabled parts
                face_mask = np.array([
                    all(self.part_labels[face] in enabled_part_ids for face in face_verts)
                    for face_verts in faces
                ])
                faces = faces[face_mask]
        
        # Handle generic per-part parameters (part_<id>_scale_long, part_<id>_offset_long, etc.)
        # These are category-agnostic and use PartInfo geometry
        for param_name, param_value in new_parameters.items():
            if param_name.startswith("part_") and "_" in param_name:
                # Parse: part_<id>_<operation>
                parts = param_name.split("_")
                if len(parts) >= 3 and parts[0] == "part":
                    try:
                        part_id = int(parts[1])
                        operation = "_".join(parts[2:])  # e.g., "scale_long", "offset_long"
                        
                        if self.part_table and part_id in self.part_table.parts:
                            part_info = self.part_table.parts[part_id]
                            verts = self._apply_generic_part_operation(
                                verts, part_id, operation, param_value, part_info
                            )
                    except (ValueError, IndexError):
                        # Not a part_<id>_<op> parameter, skip
                        pass
        
        # Apply semantic parameter-based deformations
        for param_name, new_value in new_parameters.items():
            if param_name.startswith("part_"):
                # Skip generic part parameters (already handled above)
                continue
                
            if param_name not in self.config:
                # Skip parameters not in config
                continue
            
            if param_name not in self.base_parameters:
                # Skip if baseline value not available
                continue
            
            config = self.config[param_name]
            base_value = self.base_parameters[param_name]
            
            # Apply deformation based on mode
            if config.mode == "axis_stretch":
                verts = self._apply_axis_stretch(
                    verts,
                    param_name,
                    base_value,
                    new_value,
                    config,
                )
            else:
                # Unknown mode - skip
                continue
        
        # Return new mesh data with deformed vertices and filtered faces
        return MeshData(
            vertices=verts,
            faces=faces,
        )
    
    def _apply_axis_stretch(
        self,
        verts: np.ndarray,
        param_name: str,
        base_value: float,
        new_value: float,
        config: DeformationConfig,
    ) -> np.ndarray:
        """
        Apply axis_stretch deformation mode.
        
        Args:
            verts: current vertex positions [N, 3]
            param_name: name of the parameter
            base_value: baseline parameter value
            new_value: new parameter value
            config: deformation configuration
            
        Returns:
            Updated vertex positions [N, 3]
        """
        # Compute scale factor
        if abs(base_value) < 1e-6:
            # Avoid division by zero
            scale = 1.0
        else:
            scale = new_value / base_value
        
        # If scale is 1.0, no deformation needed
        if abs(scale - 1.0) < 1e-6:
            return verts
        
        # Find vertices belonging to affected parts
        affected_vertex_indices = []
        for i, part_id in enumerate(self.part_labels):
            part_name = self.part_label_names.get(part_id, "")
            # Check if this part name matches any in affects_parts
            # Use substring matching for flexibility (e.g., "wing" matches "left_wing")
            for affected_part in config.affects_parts:
                if affected_part.lower() in part_name.lower() or part_name.lower() in affected_part.lower():
                    affected_vertex_indices.append(i)
                    break
        
        if len(affected_vertex_indices) == 0:
            # No vertices found for affected parts - return unchanged
            return verts
        
        affected_vertex_indices = np.array(affected_vertex_indices)
        V_part = verts[affected_vertex_indices]
        
        # Compute deformation axis
        if config.axis_source == "pca":
            axis = compute_pca_axis(V_part)
        elif config.axis_source.lower() in ["x", "y", "z"]:
            axis = get_world_axis(config.axis_source)
        else:
            # Default to PCA
            axis = compute_pca_axis(V_part)
        
        # Project vertices onto the axis to get parameter t
        # Project each vertex onto the axis
        # First, center the points for projection
        mean_part = np.mean(V_part, axis=0)
        V_part_centered = V_part - mean_part
        
        # Project onto axis (centered projections)
        projections_centered = np.dot(V_part_centered, axis)
        
        # Normalize projections to [0, 1] range (t=0 at root, t=1 at tip)
        t = normalize_projection(projections_centered)
        
        # Compute span length along the axis from original positions
        # Project original positions (not centered) to get actual span
        projections_original = np.dot(V_part, axis)
        span_length = np.max(projections_original) - np.min(projections_original)
        if span_length < 1e-6:
            span_length = 1.0  # Avoid division issues
        
        # Apply deformation: tip moves most, root moves least
        # For each affected vertex:
        #   delta = (scale - 1.0) * t * (axis_vector * span_length)
        #   v' = v + delta
        
        # Compute deltas for all affected vertices at once (vectorized)
        deltas = (scale - 1.0) * t[:, np.newaxis] * (axis * span_length)
        
        # Apply deltas to affected vertices
        verts_deformed = verts.copy()
        verts_deformed[affected_vertex_indices] = V_part + deltas
        
        return verts_deformed
    
    def _apply_generic_part_operation(
        self,
        verts: np.ndarray,
        part_id: int,
        operation: str,
        value: float,
        part_info: "PartInfo",
    ) -> np.ndarray:
        """
        Apply a generic, category-agnostic operation to a part.
        
        Uses PartInfo geometry (principal_axes, extents) to perform operations
        without needing semantic knowledge.
        
        Args:
            verts: vertex array [N, 3]
            part_id: part ID to operate on
            operation: operation type (e.g., "scale_long", "offset_long", "offset_up")
            value: operation value
            part_info: PartInfo for the part
            
        Returns:
            Modified vertex array
        """
        # Get vertices belonging to this part
        part_mask = self.part_labels == part_id
        part_vertex_indices = np.where(part_mask)[0]
        
        if len(part_vertex_indices) == 0:
            return verts
        
        verts_part = verts[part_vertex_indices]
        centroid = part_info.centroid
        
        # Get principal axes (columns are the principal directions)
        axes = part_info.principal_axes  # [3, 3]
        extents = part_info.extents  # [3]
        
        # Determine which axis to use based on operation
        if "long" in operation:
            axis_idx = 0  # Longest axis
        elif "short1" in operation or "short" in operation:
            axis_idx = 1  # Second longest
        elif "short2" in operation:
            axis_idx = 2  # Shortest
        elif "up" in operation or "z" in operation:
            # Use Z-up axis (world space)
            axis = np.array([0, 0, 1], dtype=np.float32)
        else:
            # Default to longest axis
            axis_idx = 0
        
        if "up" not in operation and "z" not in operation:
            axis = axes[:, axis_idx]  # Principal axis direction
        
        # Center vertices relative to part centroid
        verts_centered = verts_part - centroid
        
        # Apply operation
        if "scale" in operation:
            # Scale along axis
            projections = np.dot(verts_centered, axis)
            # Normalize to [0, 1] for smooth scaling
            if len(projections) > 0:
                proj_min, proj_max = np.min(projections), np.max(projections)
                if proj_max - proj_min > 1e-6:
                    t = (projections - proj_min) / (proj_max - proj_min)
                else:
                    t = np.zeros_like(projections)
                
                # Scale: tip moves most, root moves least
                scale_factor = value  # value is the scale (e.g., 1.2 for 20% increase)
                deltas = (scale_factor - 1.0) * t[:, np.newaxis] * (axis * extents[axis_idx])
                verts_part = verts_part + deltas
        
        elif "offset" in operation:
            # Translate along axis
            offset = value * axis  # value is the offset distance
            verts_part = verts_part + offset
        
        else:
            # Unknown operation, skip
            return verts
        
        # Apply changes back
        verts_deformed = verts.copy()
        verts_deformed[part_vertex_indices] = verts_part
        
        return verts_deformed


def build_default_deformation_config_for_category(
    category: str,
    part_label_names: Dict[int, str],
) -> Dict[str, DeformationConfig]:
    """
    Build default deformation configuration for a given category.
    
    This is a heuristic, category-specific mapping that suggests which
    semantic parameters should affect which parts.
    
    Args:
        category: object category (e.g., "airplane", "chair", "car")
        part_label_names: mapping from part ID -> name
        
    Returns:
        Dictionary mapping parameter name -> DeformationConfig
    """
    config = {}
    
    # Get all part names (values in part_label_names)
    all_part_names = list(part_label_names.values())
    
    # Helper to find parts by substring
    def find_parts_by_substring(substring: str) -> List[str]:
        """Find part names containing the substring (case-insensitive)."""
        matches = []
        for part_name in all_part_names:
            if substring.lower() in part_name.lower():
                matches.append(part_name)
        return matches
    
    category_lower = category.lower()
    
    if "airplane" in category_lower or "plane" in category_lower:
        # Airplane-specific parameters
        wing_parts = find_parts_by_substring("wing")
        if wing_parts:
            config["wing_span"] = DeformationConfig(
                affects_parts=wing_parts,
                mode="axis_stretch",
                axis_source="pca",
            )
            config["chord_length"] = DeformationConfig(
                affects_parts=wing_parts,
                mode="axis_stretch",
                axis_source="pca",
            )
        
        tail_parts = find_parts_by_substring("tail")
        if tail_parts:
            config["tail_height"] = DeformationConfig(
                affects_parts=tail_parts,
                mode="axis_stretch",
                axis_source="y",  # Typically vertical
            )
    
    elif "chair" in category_lower:
        # Chair-specific parameters
        seat_parts = find_parts_by_substring("seat")
        if seat_parts:
            config["seat_height"] = DeformationConfig(
                affects_parts=seat_parts,
                mode="axis_stretch",
                axis_source="y",  # Typically vertical
            )
        
        back_parts = find_parts_by_substring("back")
        if back_parts:
            config["back_height"] = DeformationConfig(
                affects_parts=back_parts,
                mode="axis_stretch",
                axis_source="y",
            )
        
        leg_parts = find_parts_by_substring("leg")
        if leg_parts:
            config["leg_length"] = DeformationConfig(
                affects_parts=leg_parts,
                mode="axis_stretch",
                axis_source="y",
            )
    
    elif "car" in category_lower:
        # Car-specific parameters
        body_parts = find_parts_by_substring("body")
        if body_parts:
            config["body_length"] = DeformationConfig(
                affects_parts=body_parts,
                mode="axis_stretch",
                axis_source="x",  # Typically front-to-back
            )
        
        wheel_parts = find_parts_by_substring("wheel")
        if wheel_parts:
            config["wheel_size"] = DeformationConfig(
                affects_parts=wheel_parts,
                mode="axis_stretch",
                axis_source="pca",
            )
    
    # For unknown categories, return empty dict
    return config

