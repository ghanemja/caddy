"""
Mesh I/O utilities for converting meshes to point clouds.

Uses trimesh for mesh loading and point sampling.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import trimesh


def load_mesh_as_point_cloud(
    path: str | Path,
    num_points: int = 2048,
    normalize: bool = True,
    return_normals: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh (OBJ/STL/etc.) and sample a fixed number of points on its surface.
    
    Args:
        path: path to mesh file (OBJ, STL, PLY, etc.)
        num_points: number of points to sample
        normalize: if True, center at origin and scale to unit sphere/cube
        return_normals: if True, also return surface normals (returns tuple)
        
    Returns:
        pc: np.ndarray of shape [N, 3] with XYZ coordinates
            If return_normals=True, returns tuple (points, normals) where
            points is [N, 3] and normals is [N, 3]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    
    # Load mesh with trimesh
    try:
        mesh = trimesh.load(str(path))
        # Handle scene objects (multiple meshes)
        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in the scene
            mesh = mesh.dump(concatenate=True)
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {path}: {e}")
    
    if not hasattr(mesh, 'vertices'):
        raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")
    
    # Sample points on surface
    # Use trimesh's sample method which samples uniformly on the surface
    points, face_indices = mesh.sample(num_points, return_index=True)
    
    # Get normals if requested
    normals = None
    if return_normals:
        # Compute normals at sampled points
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            # Interpolate vertex normals using barycentric coordinates
            # For simplicity, use face normals
            face_normals = mesh.face_normals
            normals = face_normals[face_indices]
        else:
            # Compute face normals if not available
            face_normals = mesh.face_normals
            normals = face_normals[face_indices]
    
    # Normalize if requested
    if normalize:
        points = normalize_point_cloud(points)
    
    if return_normals:
        return points.astype(np.float32), normals.astype(np.float32)
    return points.astype(np.float32)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud: center at origin and scale to unit sphere.
    
    Args:
        points: point cloud [N, 3]
        
    Returns:
        normalized_points: [N, 3] centered and scaled
    """
    # Center at origin
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Scale to unit sphere (max radius = 1)
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        normalized = centered / max_dist
    else:
        normalized = centered
    
    return normalized


def combine_points_and_normals(
    points: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """
    Combine points and normals into a single array [N, 6].
    
    Args:
        points: [N, 3] XYZ coordinates
        normals: [N, 3] normal vectors
        
    Returns:
        combined: [N, 6] array with [x, y, z, nx, ny, nz]
    """
    return np.concatenate([points, normals], axis=1)

