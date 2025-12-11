"""
Segmentation backend implementations.

This module provides:
- PartSegmentationBackend: Protocol/base class for segmentation backends
- PointNetSegmentationBackend: Wrapper for existing PointNet++ implementation
- Hunyuan3DPartSegmentationBackend: Implementation using Hunyuan3D-Part P3-SAM
- create_segmentation_backend: Factory function to create backends
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Protocol, runtime_checkable
import numpy as np

from .types import PartSegmentationResult


@runtime_checkable
class PartSegmentationBackend(Protocol):
    """
    Protocol for part segmentation backends.

    All segmentation backends must implement the segment() method
    that takes a mesh path and returns a PartSegmentationResult.
    """

    def segment(
        self, mesh_path: Path | str, *, num_points: Optional[int] = None, **kwargs
    ) -> PartSegmentationResult:
        """
        Segment a mesh into parts.

        Args:
            mesh_path: Path to mesh file (STL, OBJ, PLY, GLB, etc.)
            num_points: Optional number of points to sample (for point-based backends)
            **kwargs: Additional backend-specific parameters

        Returns:
            PartSegmentationResult with labels and metadata
        """
        ...


class PointNetSegmentationBackend:
    """
    Wrapper for existing PointNet++ segmentation implementation.

    This backend preserves the current PointNet++ behavior while
    conforming to the PartSegmentationBackend interface.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[str] = None,
        num_classes: int = 50,
        use_normals: bool = True,
        **kwargs,
    ):
        """
        Initialize PointNet++ backend.

        Args:
            checkpoint_path: Path to PointNet++ checkpoint (.pth file)
            device: 'cuda' or 'cpu' (auto-detected if None)
            num_classes: Number of part classes (default: 50 for ShapeNetPart)
            use_normals: Whether to use normals (default: True)
        """
        import torch
        from ..pointnet_seg.model import load_pretrained_model

        if checkpoint_path is None:
            # Try default location
            default_path = (
                Path(__file__).parent.parent.parent
                / "checkpoints"
                / "pointnet2"
                / "pointnet2_part_seg_msg.pth"
            )
            checkpoint_path = os.environ.get("POINTNET2_CHECKPOINT", str(default_path))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = load_pretrained_model(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            use_normals=use_normals,
            device=device,
        )
        self.use_normals = use_normals

    def segment(
        self,
        mesh_path: Path | str,
        *,
        num_points: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ) -> PartSegmentationResult:
        """
        Segment mesh using PointNet++.

        Args:
            mesh_path: Path to mesh file
            num_points: Number of points to sample (default: 2048)
            return_logits: Whether to return logits
            **kwargs: Additional parameters (ignored for PointNet)

        Returns:
            PartSegmentationResult
        """
        from ..pointnet_seg.inference import segment_mesh

        if num_points is None:
            num_points = 2048

        # Call existing PointNet segmentation
        result_dict = segment_mesh(
            mesh_path,
            self.model,
            device=self.device,
            num_points=num_points,
            return_logits=return_logits,
        )

        # Convert to PartSegmentationResult
        return PartSegmentationResult(
            labels=result_dict["labels"],
            points=result_dict["points"],
            logits=result_dict.get("logits"),
            num_points=len(result_dict["points"]),
        )


class Hunyuan3DPartSegmentationBackend:
    """
    Backend using Tencent's Hunyuan3D-Part P3-SAM model.

    This backend uses the P3-SAM segmentation model from:
    https://github.com/Tencent-Hunyuan/Hunyuan3D-Part
    """

    def __init__(
        self,
        model_ckpt_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Hunyuan3D-Part backend.

        Args:
            model_ckpt_dir: Directory containing P3-SAM checkpoint
                           (default: checks backend/checkpoints/Hunyuan3D-Part/p3sam/)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print(
                    "[Hunyuan3D] Warning: Running on CPU. GPU recommended for better performance."
                )

        self.device = device

        # Set default checkpoint directory if not provided
        if model_ckpt_dir is None:
            # Try cloned Hunyuan3D-Part repository first
            cloned_path = (
                Path(__file__).parent.parent.parent / "checkpoints" / "Hunyuan3D-Part"
            )
            if cloned_path.exists() and (cloned_path / "model").exists():
                model_ckpt_dir = str(cloned_path / "model")
                print(
                    f"[Hunyuan3D] Using cloned Hunyuan3D-Part repository: {model_ckpt_dir}"
                )
            else:
                # Fallback to default local location
                default_path = (
                    Path(__file__).parent.parent.parent / "checkpoints" / "partseg"
                )
                if default_path.exists() and any(default_path.iterdir()):
                    model_ckpt_dir = str(default_path)
                    print(
                        f"[Hunyuan3D] Using local checkpoint directory: {model_ckpt_dir}"
                    )

        self.model_ckpt_dir = model_ckpt_dir
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the P3-SAM model from local files only."""
        if self._model is not None:
            return

        try:
            import torch
            import trimesh
            import numpy as np
            import sys

            print("[Hunyuan3D] Loading P3-SAM model from local files only...")

            # Find P3-SAM code directory
            p3sam_code_path = (
                Path(__file__).parent.parent.parent
                / "checkpoints"
                / "Hunyuan3D-Part-code"
                / "P3-SAM"
            )

            if not p3sam_code_path.exists():
                raise RuntimeError(
                    f"P3-SAM code not found at: {p3sam_code_path}\n"
                    "Please ensure the Hunyuan3D-Part repository is cloned:\n"
                    "  cd backend/checkpoints && git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git Hunyuan3D-Part-code"
                )

            # Add paths for imports
            if str(p3sam_code_path) not in sys.path:
                sys.path.insert(0, str(p3sam_code_path))

            # Add XPart/partgen to path (needed for sonata import)
            xpart_path = p3sam_code_path.parent / "XPart" / "partgen"
            if xpart_path.exists() and str(xpart_path.parent) not in sys.path:
                sys.path.insert(0, str(xpart_path.parent))

            # Add demo directory to path
            demo_path = p3sam_code_path / "demo"
            if demo_path.exists() and str(demo_path) not in sys.path:
                sys.path.insert(0, str(demo_path))

            print(f"[Hunyuan3D] Using P3-SAM code from: {p3sam_code_path}")

            # Import P3-SAM AutoMask class
            try:
                from auto_mask_no_postprocess import AutoMask

                print("[Hunyuan3D] ✓ Imported AutoMask from local code")
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import P3-SAM AutoMask: {e}\n\n"
                    "Please ensure:\n"
                    "  1. P3-SAM code is cloned: checkpoints/Hunyuan3D-Part-code/P3-SAM/\n"
                    "  2. All dependencies are installed:\n"
                    "     pip install viser fpsample trimesh numba scikit-learn scipy\n"
                    "  3. XPart/partgen directory exists for sonata import"
                ) from e

            # Find checkpoint file
            p3sam_ckpt = (
                Path(__file__).parent.parent.parent
                / "checkpoints"
                / "Hunyuan3D-Part"
                / "p3sam"
                / "p3sam.safetensors"
            )

            if not p3sam_ckpt.exists():
                # Try alternative locations
                alt_paths = [
                    Path(__file__).parent.parent.parent
                    / "checkpoints"
                    / "Hunyuan3D-Part"
                    / "p3sam"
                    / "p3sam.safetensors",
                    Path(__file__).parent.parent.parent
                    / "checkpoints"
                    / "partseg"
                    / "p3sam.safetensors",
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        p3sam_ckpt = alt_path
                        break
                else:
                    raise RuntimeError(
                        f"P3-SAM checkpoint not found. Tried:\n"
                        f"  - {p3sam_ckpt}\n"
                        f"  - {alt_paths[1]}\n\n"
                        "Please ensure p3sam.safetensors is present in checkpoints/Hunyuan3D-Part/p3sam/\n"
                        "You can download it from: https://huggingface.co/tencent/Hunyuan3D-Part"
                    )

            print(f"[Hunyuan3D] Loading checkpoint from: {p3sam_ckpt}")

            # Initialize and load model
            try:
                # Use AutoMask wrapper (handles model initialization)
                self._automask = AutoMask(
                    ckpt_path=str(p3sam_ckpt),
                    point_num=100000,
                    prompt_num=400,
                    threshold=0.95,
                    post_process=False,  # Use no post-process version for speed
                )
                self._model = self._automask  # Store AutoMask as the model
                print("[Hunyuan3D] ✓ Loaded P3-SAM model from local checkpoint")
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to initialize P3-SAM model: {e}\n\n"
                    "Please ensure:\n"
                    "  1. Checkpoint file exists and is valid: checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors\n"
                    "  2. All dependencies are installed (see P3-SAM/README.md)\n"
                    "  3. CUDA is available if using GPU\n"
                    "  4. Sonata model can be loaded (may require network for first-time download)"
                ) from e

        except ImportError as e:
            raise ImportError(
                f"Failed to import required dependencies for Hunyuan3D-Part: {e}\n\n"
                "Install required packages with:\n"
                "  pip install torch trimesh numpy scipy scikit-learn fpsample numba viser\n\n"
                "The Hunyuan3D-Part segmentation backend requires these dependencies."
            ) from e
        except RuntimeError:
            # Re-raise RuntimeErrors (already formatted)
            raise
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load Hunyuan3D-Part P3-SAM model: {e}\n\n"
                "The segmentation backend requires a working Hunyuan3D-Part model.\n\n"
                "Please ensure:\n"
                "  1. P3-SAM code is cloned: checkpoints/Hunyuan3D-Part-code/P3-SAM/\n"
                "  2. Checkpoint file exists: checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors\n"
                "  3. All dependencies are installed\n"
                "  4. GPU/CUDA is available if required\n\n"
                "If you cannot get Hunyuan3D-Part working, switch to PointNet backend:\n"
                "  export SEGMENTATION_BACKEND=pointnet"
            ) from e

    def segment(
        self, mesh_path: Path | str, *, num_points: Optional[int] = None, **kwargs
    ) -> PartSegmentationResult:
        """
        Segment mesh using Hunyuan3D-Part P3-SAM.

        Args:
            mesh_path: Path to mesh file (GLB, PLY, OBJ, etc.)
            num_points: Number of points to sample (for fallback)
            **kwargs: Additional parameters

        Returns:
            PartSegmentationResult
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        import trimesh
        import numpy as np
        import torch

        # Load mesh
        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        # Load and use P3-SAM model - no fallbacks
        self._load_model()

        if self._model is None:
            raise RuntimeError(
                "Hunyuan3D-Part P3-SAM model failed to load. "
                "Cannot perform segmentation without a loaded model."
            )

        print("[Hunyuan3D] Running P3-SAM inference...")

        # Use AutoMask.predict_aabb method
        if hasattr(self._model, "predict_aabb"):
            # AutoMask interface
            try:
                aabb, face_ids, processed_mesh = self._model.predict_aabb(
                    mesh=mesh,
                    point_num=num_points if num_points else 100000,
                    prompt_num=400,
                    threshold=0.95,
                    post_process=False,
                    save_path=None,
                    save_mid_res=False,
                    show_info=False,
                    clean_mesh_flag=True,
                    seed=42,
                    is_parallel=True,
                    prompt_bs=32,
                )

                # face_ids is per-face labels, need to convert to per-vertex
                # Map face labels to vertex labels
                vertex_labels = np.full(len(vertices), -1, dtype=np.int32)
                for face_idx, face in enumerate(mesh.faces):
                    face_label = face_ids[face_idx]
                    if face_label >= 0:  # Valid label
                        for v_idx in face:
                            if vertex_labels[v_idx] == -1:
                                vertex_labels[v_idx] = face_label
                            # If vertex already has a label, keep the first one

                # For vertices without labels, assign from nearest face
                unlabeled = vertex_labels == -1
                if np.any(unlabeled):
                    try:
                        from scipy.spatial import cKDTree

                        # Get face centers
                        face_centers = np.array(
                            [mesh.triangles_center[i] for i in range(len(mesh.faces))]
                        )
                        tree = cKDTree(face_centers)
                        _, nearest_faces = tree.query(vertices[unlabeled], k=1)
                        vertex_labels[unlabeled] = face_ids[nearest_faces]
                    except ImportError:
                        # Fallback: use face labels directly for unlabeled vertices
                        # Assign from first face that contains the vertex
                        for v_idx in np.where(unlabeled)[0]:
                            for face_idx, face in enumerate(mesh.faces):
                                if v_idx in face:
                                    vertex_labels[v_idx] = face_ids[face_idx]
                                    break

                # Ensure all labels are non-negative and contiguous
                unique_labels = np.unique(vertex_labels)
                if len(unique_labels) > 0 and unique_labels[0] < 0:
                    # Remap negative labels
                    label_map = {
                        old: new for new, old in enumerate(unique_labels) if old >= 0
                    }
                    vertex_labels = np.array(
                        [label_map.get(l, 0) if l >= 0 else 0 for l in vertex_labels],
                        dtype=np.int32,
                    )
                    face_ids = np.array(
                        [label_map.get(l, 0) if l >= 0 else 0 for l in face_ids],
                        dtype=np.int32,
                    )

                # Sample points if needed
                if num_points is not None and len(vertex_labels) > num_points:
                    indices = np.random.choice(len(vertices), num_points, replace=False)
                    points = vertices[indices]
                    point_labels = vertex_labels[indices]
                else:
                    points = vertices
                    point_labels = vertex_labels

                unique_labels = np.unique(vertex_labels)
                num_parts = len(unique_labels[unique_labels >= 0])

                print(f"[Hunyuan3D] ✓ P3-SAM segmentation complete: {num_parts} parts")

                return PartSegmentationResult(
                    labels=point_labels,
                    points=points,
                    vertex_labels=vertex_labels,
                    vertices=vertices,
                    num_parts=num_parts,
                    num_points=len(points),
                    num_vertices=len(vertices),
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"P3-SAM inference failed: {e}\n\n"
                    "The model loaded but inference failed. Please check:\n"
                    "  1. Mesh is valid and can be processed\n"
                    "  2. GPU memory is sufficient\n"
                    "  3. All P3-SAM dependencies are installed\n"
                    "  4. Sonata model is available (may download on first use)"
                ) from e
        else:
            raise RuntimeError(
                f"Model does not have predict_aabb method. "
                f"Model type: {type(self._model)}"
            )


def create_segmentation_backend(
    kind: Optional[str] = None, **kwargs
) -> PartSegmentationBackend:
    """
    Factory function to create a segmentation backend.

    Args:
        kind: Backend type - "pointnet", "hunyuan3d", or None (auto-detect)
              If None, uses SEGMENTATION_BACKEND env var or defaults to "pointnet"
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        PartSegmentationBackend instance
    """
    if kind is None:
        kind = os.environ.get("SEGMENTATION_BACKEND", "hunyuan3d").lower()

    if kind == "pointnet":
        return PointNetSegmentationBackend(**kwargs)
    elif (
        kind == "hunyuan3d"
        or kind == "hunyuan"
        or kind == "hunyuan-2.1"
        or kind == "hunyuan2.1"
    ):
        return Hunyuan3DPartSegmentationBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown segmentation backend: {kind}. "
            "Supported backends: 'pointnet', 'hunyuan3d'"
        )
