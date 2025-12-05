"""
Utility functions for mesh deformation.
"""

import numpy as np
from typing import Tuple


def compute_pca_axis(points: np.ndarray) -> np.ndarray:
    """
    Compute the principal direction (unit vector) of maximum variance using PCA.
    
    Args:
        points: point cloud [N, 3]
        
    Returns:
        Unit vector [3] representing the principal axis of maximum variance
    """
    if len(points) < 2:
        # Degenerate case: return default axis
        return np.array([1.0, 0.0, 0.0])
    
    # Center the points
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Get the eigenvector corresponding to the largest eigenvalue
    max_idx = np.argmax(eigenvalues)
    axis = eigenvectors[:, max_idx]
    
    # Normalize to unit vector
    norm = np.linalg.norm(axis)
    if norm > 1e-6:
        axis = axis / norm
    else:
        # Degenerate case: return default axis
        axis = np.array([1.0, 0.0, 0.0])
    
    return axis


def normalize_projection(proj: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D projection array to [0, 1] range.
    
    Handles degenerate cases where all values are the same.
    
    Args:
        proj: 1D array of projection values
        
    Returns:
        Normalized array in [0, 1] range
    """
    proj = np.asarray(proj)
    
    if len(proj) == 0:
        return proj
    
    min_val = np.min(proj)
    max_val = np.max(proj)
    
    # Handle degenerate case: all values are the same
    if abs(max_val - min_val) < 1e-6:
        # Return all zeros (root position)
        return np.zeros_like(proj)
    
    # Normalize to [0, 1]
    normalized = (proj - min_val) / (max_val - min_val)
    
    return normalized


def get_world_axis(axis_name: str) -> np.ndarray:
    """
    Get a world axis vector.
    
    Args:
        axis_name: "x", "y", or "z"
        
    Returns:
        Unit vector [3] for the specified axis
    """
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    return axis_map.get(axis_name.lower(), np.array([1.0, 0.0, 0.0]))

