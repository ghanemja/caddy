"""
Geometry utilities for computing semantic parameters from segmented point clouds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def compute_part_bounding_boxes(
    points: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For each part label, compute an axis-aligned bounding box.
    
    Args:
        points: point cloud [N, 3]
        labels: part labels [N]
        
    Returns:
        Dictionary mapping label_id -> {
            "min": np.array([x_min, y_min, z_min]),
            "max": np.array([x_max, y_max, z_max]),
            "center": np.array([cx, cy, cz]),
            "extent": np.array([dx, dy, dz]),
            "num_points": int
        }
    """
    unique_labels = np.unique(labels)
    bboxes = {}
    
    for label_id in unique_labels:
        mask = labels == label_id
        part_points = points[mask]
        
        if len(part_points) == 0:
            continue
        
        min_xyz = np.min(part_points, axis=0)
        max_xyz = np.max(part_points, axis=0)
        center = (min_xyz + max_xyz) / 2.0
        extent = max_xyz - min_xyz
        
        bboxes[int(label_id)] = {
            "min": min_xyz,
            "max": max_xyz,
            "center": center,
            "extent": extent,
            "num_points": len(part_points),
        }
    
    return bboxes


def axis_extent(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    """
    Return extents along x, y, z axes as a length-3 array.
    
    Args:
        min_xyz: minimum coordinates [3]
        max_xyz: maximum coordinates [3]
        
    Returns:
        extent: [dx, dy, dz]
    """
    return max_xyz - min_xyz


def compute_part_statistics(
    points: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute comprehensive statistics for each part.
    
    Args:
        points: point cloud [N, 3]
        labels: part labels [N]
        
    Returns:
        Dictionary mapping label_id -> {
            "bbox": {...},  # from compute_part_bounding_boxes
            "centroid": np.array([cx, cy, cz]),  # mean of points
            "std": np.array([sx, sy, sz]),  # standard deviation
            "volume_estimate": float,  # approximate volume from bbox
            "surface_area_estimate": float,  # approximate surface area
        }
    """
    bboxes = compute_part_bounding_boxes(points, labels)
    stats = {}
    
    for label_id, bbox in bboxes.items():
        mask = labels == label_id
        part_points = points[mask]
        
        # Centroid (mean)
        centroid = np.mean(part_points, axis=0)
        
        # Standard deviation
        std = np.std(part_points, axis=0)
        
        # Volume estimate (from bounding box)
        extent = bbox["extent"]
        volume_estimate = np.prod(extent)
        
        # Surface area estimate (from bounding box)
        # For a box: 2 * (xy + xz + yz)
        if len(extent) == 3:
            x, y, z = extent
            surface_area_estimate = 2 * (x * y + x * z + y * z)
        else:
            surface_area_estimate = 0.0
        
        stats[int(label_id)] = {
            "bbox": bbox,
            "centroid": centroid,
            "std": std,
            "volume_estimate": volume_estimate,
            "surface_area_estimate": surface_area_estimate,
        }
    
    return stats


def compute_distance_between_parts(
    points: np.ndarray,
    labels: np.ndarray,
    label1: int,
    label2: int,
) -> Optional[float]:
    """
    Compute distance between centers of two parts.
    
    Args:
        points: point cloud [N, 3]
        labels: part labels [N]
        label1: first part label
        label2: second part label
        
    Returns:
        Distance between part centers, or None if either part not found
    """
    bboxes = compute_part_bounding_boxes(points, labels)
    
    if label1 not in bboxes or label2 not in bboxes:
        return None
    
    center1 = bboxes[label1]["center"]
    center2 = bboxes[label2]["center"]
    
    return np.linalg.norm(center1 - center2)


def compute_principal_axis(
    points: np.ndarray,
    labels: np.ndarray,
    label_id: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute principal axis (PCA) for a part.
    
    Args:
        points: point cloud [N, 3]
        labels: part labels [N]
        label_id: part label to analyze
        
    Returns:
        Dictionary with:
            "direction": principal direction [3]
            "eigenvalues": eigenvalues [3] (sorted descending)
            "length": length along principal axis
    """
    mask = labels == label_id
    part_points = points[mask]
    
    if len(part_points) < 3:
        return None
    
    # Center points
    centered = part_points - np.mean(part_points, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Principal direction
    principal_dir = eigenvectors[:, 0]
    
    # Project points onto principal axis to get length
    projections = np.dot(centered, principal_dir)
    length = np.max(projections) - np.min(projections)
    
    return {
        "direction": principal_dir,
        "eigenvalues": eigenvalues,
        "length": length,
    }


def find_largest_parts(
    points: np.ndarray,
    labels: np.ndarray,
    top_k: int = 2,
    metric: str = "volume",
) -> List[Tuple[int, float]]:
    """
    Find the largest parts by some metric.
    
    Args:
        points: point cloud [N, 3]
        labels: part labels [N]
        top_k: number of parts to return
        metric: "volume", "num_points", or "extent_max"
        
    Returns:
        List of (label_id, metric_value) tuples, sorted descending
    """
    stats = compute_part_statistics(points, labels)
    
    scores = []
    for label_id, stat in stats.items():
        if metric == "volume":
            score = stat["volume_estimate"]
        elif metric == "num_points":
            score = stat["bbox"]["num_points"]
        elif metric == "extent_max":
            score = np.max(stat["bbox"]["extent"])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append((label_id, score))
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]

