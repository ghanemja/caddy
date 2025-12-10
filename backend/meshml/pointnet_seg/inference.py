"""
High-level inference API for PointNet++ part segmentation.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from pathlib import Path

from .model import PointNet2PartSegWrapper
from .mesh_io import load_mesh_as_point_cloud


def segment_mesh(
    mesh_path: str | Path,
    model: PointNet2PartSegWrapper,
    device: str | None = None,
    num_points: int = 2048,
    return_logits: bool = False,
) -> Dict[str, Any]:
    """
    High-level helper for mesh segmentation:
      1. Load mesh and sample point cloud.
      2. Run PointNet++ part segmentation.
      3. Return points, labels, and optionally logits.
    
    Args:
        mesh_path: path to mesh file
        model: PointNet2PartSegWrapper instance
        device: device to run inference on (uses model's device if None)
        num_points: number of points to sample from mesh
        return_logits: if True, also return per-point logits
        
    Returns:
        Dictionary with:
            - points: np.ndarray [N, 3] point cloud coordinates
            - labels: np.ndarray [N] integer part labels
            - logits: (optional) np.ndarray [N, num_classes] if return_logits=True
    """
    # Load mesh as point cloud
    points, normals = load_mesh_as_point_cloud(
        mesh_path,
        num_points=num_points,
        normalize=True,
        return_normals=True,
    )
    
    # Combine points and normals if model expects them
    if model.use_normals:
        input_points = np.concatenate([points, normals], axis=1)  # [N, 6]
    else:
        input_points = points  # [N, 3]
    
    # Convert to tensor and add batch dimension
    device = device or model.device
    input_tensor = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0)  # [1, N, C]
    input_tensor = input_tensor.to(device)
    
    # Run inference
    model.model.eval()
    with torch.no_grad():
        # Model wrapper's forward expects (points, return_labels)
        # It handles cls_label internally
        logits = model.forward(input_tensor, return_labels=False)  # [1, N, num_classes]
        labels = torch.argmax(logits, dim=-1)  # [1, N]
    
    # Move to CPU and convert to numpy
    logits_np = logits.squeeze(0).cpu().numpy()  # [N, num_classes]
    labels_np = labels.squeeze(0).cpu().numpy()  # [N]
    
    result = {
        "points": points,
        "labels": labels_np,
    }
    
    if return_logits:
        result["logits"] = logits_np
    
    return result


def segment_point_cloud(
    points: np.ndarray,
    model: PointNet2PartSegWrapper,
    normals: Optional[np.ndarray] = None,
    device: str | None = None,
    return_logits: bool = False,
) -> Dict[str, Any]:
    """
    Segment a point cloud directly (without loading from mesh).
    
    Args:
        points: point cloud [N, 3] or [B, N, 3]
        model: PointNet2PartSegWrapper instance
        normals: optional normals [N, 3] or [B, N, 3]
        device: device to run inference on
        return_logits: if True, also return logits
        
    Returns:
        Dictionary with points, labels, and optionally logits
    """
    points = np.asarray(points)
    if points.ndim == 2:
        points = points[np.newaxis, :, :]  # Add batch dimension
    
    # Combine with normals if provided and model expects them
    if model.use_normals:
        if normals is None:
            raise ValueError("Model expects normals but none provided")
        normals = np.asarray(normals)
        if normals.ndim == 2:
            normals = normals[np.newaxis, :, :]
        input_points = np.concatenate([points, normals], axis=-1)  # [B, N, 6]
    else:
        input_points = points  # [B, N, 3]
    
    # Convert to tensor
    device = device or model.device
    input_tensor = torch.tensor(input_points, dtype=torch.float32).to(device)
    
    # Run inference
    model.model.eval()
    with torch.no_grad():
        # Model wrapper's forward expects (points, return_labels)
        # It handles cls_label internally
        logits = model.forward(input_tensor, return_labels=False)
        labels = torch.argmax(logits, dim=-1)
    
    # Convert to numpy
    logits_np = logits.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Remove batch dimension if single sample
    if points.shape[0] == 1:
        points = points.squeeze(0)
        labels_np = labels_np.squeeze(0)
        if return_logits:
            logits_np = logits_np.squeeze(0)
    
    result = {
        "points": points,
        "labels": labels_np,
    }
    
    if return_logits:
        result["logits"] = logits_np
    
    return result

