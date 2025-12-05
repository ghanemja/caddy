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
from typing import Dict, List, Tuple, Optional
import numpy as np

from .utils import compute_pca_axis, normalize_projection, get_world_axis


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
    ):
        """
        Initialize the parametric mesh deformer.
        
        Args:
            base_mesh: original mesh geometry (undeformed)
            base_parameters: semantic param values at baseline (e.g. {"wing_span": 2.0})
            part_labels: [N] array of integer part IDs per vertex (from PointNet++)
            part_label_names: mapping from part ID -> human-readable name
                (e.g., {0: "fuselage", 1: "left_wing", ...})
            config: mapping from semantic param name -> DeformationConfig
        """
        self.base_mesh = base_mesh
        self.base_parameters = base_parameters.copy()
        self.part_labels = np.asarray(part_labels)
        self.part_label_names = part_label_names.copy()
        self.config = config.copy()
        
        # Validate inputs
        if len(self.part_labels) != len(self.base_mesh.vertices):
            raise ValueError(
                f"Part labels length ({len(self.part_labels)}) must match "
                f"vertices length ({len(self.base_mesh.vertices)})"
            )
    
    def deform(
        self,
        new_parameters: Dict[str, float]
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
        # Start with a copy of the base vertices
        verts = self.base_mesh.vertices.copy()
        
        # Apply deformations for each parameter
        for param_name, new_value in new_parameters.items():
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
        
        # Return new mesh data with deformed vertices
        return MeshData(
            vertices=verts,
            faces=self.base_mesh.faces.copy(),
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

