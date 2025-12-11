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
                           (default: checks backend/checkpoints/partseg, then downloads from HuggingFace)
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
        """Lazy load the P3-SAM model."""
        if self._model is not None:
            return

        try:
            import torch
            import trimesh
            import numpy as np

            # Store torch for use in model loading
            self._torch = torch

            print("[Hunyuan3D] Attempting to load Hunyuan3D-Part P3-SAM model...")

            # Try multiple approaches to load the model
            model_loaded = False

            # Approach 0: Try loading from local checkpoint directory first
            if self.model_ckpt_dir and Path(self.model_ckpt_dir).exists():
                try:
                    from transformers import AutoModel, AutoProcessor

                    print(
                        f"[Hunyuan3D] Trying to load from local checkpoint: {self.model_ckpt_dir}"
                    )
                    ckpt_path = Path(self.model_ckpt_dir)

                    # If model_ckpt_dir points to a subdirectory (like "model"), check parent for config
                    parent_path = ckpt_path.parent
                    if (parent_path / "config.json").exists():
                        # Use parent directory as the model root (HuggingFace format)
                        model_root = parent_path
                        print(
                            f"[Hunyuan3D] Using parent directory as model root: {model_root}"
                        )
                    else:
                        model_root = ckpt_path

                    # Check for model files in the checkpoint directory
                    model_files = (
                        list(ckpt_path.glob("*.safetensors"))
                        + list(ckpt_path.glob("*.bin"))
                        + list(ckpt_path.glob("*.pth"))
                    )
                    # Also check parent directory if different
                    if model_root != ckpt_path:
                        model_files.extend(
                            list(model_root.glob("*.safetensors"))
                            + list(model_root.glob("*.bin"))
                            + list(model_root.glob("*.pth"))
                        )
                        # Also check model subdirectory
                        model_subdir = model_root / "model"
                        if model_subdir.exists():
                            model_files.extend(
                                list(model_subdir.glob("*.safetensors"))
                                + list(model_subdir.glob("*.bin"))
                                + list(model_subdir.glob("*.pth"))
                            )
                    config_file = model_root / "config.json"

                    if model_files:
                        print(
                            f"[Hunyuan3D] Found model file(s): {[f.name for f in model_files]}"
                        )

                        # If we have a config.json, try loading as HuggingFace model
                        if config_file.exists():
                            try:
                                # Use model_root (parent) if it has config.json, otherwise use ckpt_path
                                model_path = (
                                    str(model_root)
                                    if model_root != ckpt_path
                                    else str(ckpt_path)
                                )
                                print(f"[Hunyuan3D] Loading model from: {model_path}")
                                print(f"[Hunyuan3D] Config file: {config_file}")
                                model = AutoModel.from_pretrained(
                                    model_path,
                                    trust_remote_code=True,
                                    local_files_only=True,
                                ).to(self.device)
                                model.eval()

                                # Try to load processor if available
                                try:
                                    processor = AutoProcessor.from_pretrained(
                                        model_path,
                                        trust_remote_code=True,
                                        local_files_only=True,
                                    )
                                    self._processor = processor
                                except:
                                    self._processor = None

                                self._model = model
                                model_loaded = True
                                print(
                                    f"[Hunyuan3D] ✓ Loaded model from local checkpoint: {ckpt_path}"
                                )
                            except Exception as e:
                                print(
                                    f"[Hunyuan3D] Failed to load from local checkpoint as HuggingFace model: {e}"
                                )
                                print(
                                    "[Hunyuan3D] Note: config.json may be needed for proper model loading"
                                )
                                print("[Hunyuan3D] Trying other approaches...")
                        else:
                            print(
                                f"[Hunyuan3D] Warning: No config.json found in {ckpt_path}"
                            )
                            print(
                                "[Hunyuan3D] Attempting to download config.json from HuggingFace..."
                            )
                            try:
                                from huggingface_hub import hf_hub_download

                                # Download config.json from HuggingFace
                                config_path = hf_hub_download(
                                    repo_id="tencent/Hunyuan3D-Part",
                                    filename="config.json",
                                    local_dir=str(model_root),
                                    local_dir_use_symlinks=False,
                                )
                                print(
                                    f"[Hunyuan3D] ✓ Downloaded config.json to {config_path}"
                                )

                                # Now try loading again
                                model = AutoModel.from_pretrained(
                                    str(model_root),
                                    trust_remote_code=True,
                                    local_files_only=True,
                                ).to(self.device)
                                model.eval()

                                try:
                                    processor = AutoProcessor.from_pretrained(
                                        str(model_root),
                                        trust_remote_code=True,
                                        local_files_only=True,
                                    )
                                    self._processor = processor
                                except:
                                    self._processor = None

                                self._model = model
                                model_loaded = True
                                print(
                                    f"[Hunyuan3D] ✓ Loaded model from local checkpoint with downloaded config: {ckpt_path}"
                                )
                            except Exception as e:
                                print(
                                    f"[Hunyuan3D] Failed to download config or load model: {e}"
                                )
                                print(
                                    "[Hunyuan3D] Trying HuggingFace full download approach..."
                                )
                except Exception as e:
                    print(f"[Hunyuan3D] Error checking local checkpoint: {e}")

            # Approach 1: Try loading from HuggingFace using transformers
            if not model_loaded:
                try:
                    from transformers import AutoModel, AutoProcessor

                    print("[Hunyuan3D] Trying HuggingFace transformers approach...")

                    # Try to find a Hunyuan model on HuggingFace
                    model_name = os.environ.get(
                        "HUNYUAN3D_MODEL_NAME", "tencent/Hunyuan3D-Part"
                    )

                    try:
                        # Try to load model with trust_remote_code (required for custom model classes)
                        print(f"[Hunyuan3D] Loading from HuggingFace: {model_name}")
                        model = AutoModel.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=(
                                self._torch.float16
                                if self.device == "cuda"
                                else self._torch.float32
                            ),
                        ).to(self.device)
                        model.eval()

                        # Try to load processor if available
                        try:
                            processor = AutoProcessor.from_pretrained(
                                model_name, trust_remote_code=True
                            )
                            self._processor = processor
                        except Exception as proc_e:
                            print(f"[Hunyuan3D] Processor not available: {proc_e}")
                            self._processor = None

                        self._model = model
                        model_loaded = True
                        print(
                            f"[Hunyuan3D] ✓ Loaded model from HuggingFace: {model_name}"
                        )
                    except Exception as e:
                        print(f"[Hunyuan3D] Transformers approach failed: {e}")
                        import traceback

                        traceback.print_exc()

                except ImportError as e:
                    print(f"[Hunyuan3D] transformers not available: {e}")

            # Approach 2: Try using P3-SAM from cloned repository
            if not model_loaded:
                try:
                    # Check for cloned P3-SAM code
                    p3sam_code_path = (
                        Path(__file__).parent.parent.parent
                        / "checkpoints"
                        / "Hunyuan3D-Part-code"
                        / "P3-SAM"
                    )

                    if p3sam_code_path.exists():
                        import sys

                        # Add P3-SAM to path
                        if str(p3sam_code_path) not in sys.path:
                            sys.path.insert(0, str(p3sam_code_path))

                        print("[Hunyuan3D] Found P3-SAM code, attempting to import...")
                        # Try importing P3-SAM
                        try:
                            from p3sam import P3SAM

                            # Get checkpoint path
                            p3sam_ckpt = (
                                Path(__file__).parent.parent.parent
                                / "checkpoints"
                                / "Hunyuan3D-Part"
                                / "p3sam"
                                / "p3sam.safetensors"
                            )

                            if not p3sam_ckpt.exists():
                                # Try alternative location
                                p3sam_ckpt = model_root / "p3sam" / "p3sam.safetensors"

                            if p3sam_ckpt.exists():
                                print(
                                    f"[Hunyuan3D] Initializing P3-SAM with checkpoint: {p3sam_ckpt}"
                                )
                                self._model = P3SAM(
                                    model_path=str(p3sam_ckpt), device=self.device
                                )
                                model_loaded = True
                                print("[Hunyuan3D] ✓ Loaded P3-SAM model from code")
                            else:
                                print(
                                    f"[Hunyuan3D] P3-SAM checkpoint not found at: {p3sam_ckpt}"
                                )
                        except ImportError as e:
                            print(f"[Hunyuan3D] Failed to import P3-SAM: {e}")
                        except Exception as e:
                            print(f"[Hunyuan3D] Failed to initialize P3-SAM: {e}")
                    else:
                        print(
                            f"[Hunyuan3D] P3-SAM code not found at: {p3sam_code_path}"
                        )
                except Exception as e:
                    print(f"[Hunyuan3D] Error checking for P3-SAM code: {e}")

            # Approach 3: Try downloading from HuggingFace Hub directly
            if not model_loaded:
                try:
                    from huggingface_hub import snapshot_download
                    from transformers import AutoModel, AutoProcessor

                    print("[Hunyuan3D] Trying HuggingFace Hub download...")

                    if self.model_ckpt_dir is None:
                        cache_dir = snapshot_download(
                            repo_id="tencent/Hunyuan3D-Part",
                            repo_type="model",
                            cache_dir=None,
                        )
                        self.model_ckpt_dir = Path(cache_dir)
                    else:
                        self.model_ckpt_dir = Path(self.model_ckpt_dir)

                    print(f"[Hunyuan3D] Model downloaded to: {self.model_ckpt_dir}")
                    # Try loading from downloaded checkpoint
                    model = AutoModel.from_pretrained(
                        str(self.model_ckpt_dir),
                        trust_remote_code=True,
                        local_files_only=True,
                        torch_dtype=(
                            self._torch.float16
                            if self.device == "cuda"
                            else self._torch.float32
                        ),
                    ).to(self.device)
                    model.eval()
                    try:
                        processor = AutoProcessor.from_pretrained(
                            str(self.model_ckpt_dir),
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                        self._processor = processor
                    except Exception as proc_e:
                        print(f"[Hunyuan3D] Processor not available: {proc_e}")
                        self._processor = None
                    self._model = model
                    model_loaded = True
                    print(
                        f"[Hunyuan3D] ✓ Loaded model from HuggingFace Hub: {self.model_ckpt_dir}"
                    )

                except Exception as e:
                    print(f"[Hunyuan3D] HuggingFace Hub download/load failed: {e}")
                    import traceback

                    traceback.print_exc()

            if not model_loaded:
                raise RuntimeError(
                    "CRITICAL: Could not load Hunyuan3D-Part P3-SAM model. "
                    "The segmentation backend requires a working Hunyuan3D-Part model.\n\n"
                    "The model failed to load from all attempted sources:\n"
                    "  1. Local checkpoint directory (checkpoints/Hunyuan3D-Part/)\n"
                    "  2. HuggingFace transformers AutoModel (tencent/Hunyuan3D-Part)\n"
                    "  3. HuggingFace Hub download\n"
                    "  4. P3-SAM code from cloned repository\n\n"
                    "To fix this:\n"
                    "  1. Ensure transformers and huggingface_hub are installed:\n"
                    "     pip install transformers huggingface_hub\n"
                    "  2. Ensure you have network access to download from HuggingFace\n"
                    "  3. The model may require custom code - check the Hunyuan3D-Part repository:\n"
                    "     https://github.com/Tencent-Hunyuan/Hunyuan3D-Part\n"
                    "  4. Alternatively, switch to PointNet backend by setting:\n"
                    "     export SEGMENTATION_BACKEND=pointnet"
                )

        except ImportError as e:
            raise ImportError(
                f"Failed to import required dependencies for Hunyuan3D-Part: {e}\n\n"
                "Install required packages with:\n"
                "  pip install transformers huggingface_hub torch trimesh\n\n"
                "The Hunyuan3D-Part segmentation backend cannot work without these dependencies."
            ) from e
        except RuntimeError:
            # Re-raise RuntimeErrors (already formatted)
            raise
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load Hunyuan3D-Part P3-SAM model: {e}\n\n"
                "The segmentation backend requires a working Hunyuan3D-Part model. "
                "All fallback methods have been removed - the model MUST load successfully.\n\n"
                "Please check:\n"
                "  1. Model files are present in checkpoints/Hunyuan3D-Part/\n"
                "  2. Network access to HuggingFace (if downloading)\n"
                "  3. transformers library is properly installed\n"
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
        with torch.no_grad():
            # Try different model APIs
            if hasattr(self._model, "predict"):
                # Assume model has predict method
                labels = self._model.predict(vertices, faces)
            elif hasattr(self._model, "__call__"):
                # Assume model is callable
                if self._processor is not None:
                    inputs = self._processor(
                        vertices=vertices, faces=faces, return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self._model(**inputs)
                    if hasattr(outputs, "labels"):
                        labels = outputs.labels.cpu().numpy()
                    elif isinstance(outputs, dict) and "labels" in outputs:
                        labels = outputs["labels"].cpu().numpy()
                    elif isinstance(outputs, torch.Tensor):
                        labels = outputs.cpu().numpy()
                    else:
                        raise ValueError(
                            f"Unknown output format from model: {type(outputs)}. "
                            f"Expected tensor or object with 'labels' attribute."
                        )
                else:
                    # Convert to tensor and call
                    verts_tensor = (
                        torch.from_numpy(vertices).float().unsqueeze(0).to(self.device)
                    )
                    faces_tensor = (
                        torch.from_numpy(faces).long().unsqueeze(0).to(self.device)
                    )
                    outputs = self._model(verts_tensor, faces_tensor)
                    if isinstance(outputs, torch.Tensor):
                        labels = outputs.cpu().numpy().flatten()
                    elif hasattr(outputs, "labels"):
                        labels = outputs.labels.cpu().numpy().flatten()
                    elif isinstance(outputs, dict) and "labels" in outputs:
                        labels = outputs["labels"].cpu().numpy().flatten()
                    else:
                        raise ValueError(
                            f"Unknown output format from model: {type(outputs)}. "
                            f"Expected tensor or object with 'labels' attribute."
                        )
            else:
                raise ValueError(
                    "Model doesn't have predict or __call__ method. "
                    f"Model type: {type(self._model)}"
                )

        # Ensure labels are integers and match vertex count
        if labels.dtype != np.int32:
            labels = labels.astype(np.int32)

        # Ensure labels match vertex count
        if len(labels) != len(vertices):
            if len(labels) < len(vertices):
                # Repeat last label or use nearest neighbor
                try:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(vertices[: len(labels)])
                    _, indices = tree.query(vertices[len(labels) :], k=1)
                    extended_labels = np.concatenate([labels, labels[indices]])
                    labels = extended_labels[: len(vertices)]
                except ImportError:
                    # Fallback: repeat last label
                    padding = np.full(
                        len(vertices) - len(labels), labels[-1], dtype=np.int32
                    )
                    labels = np.concatenate([labels, padding])
            else:
                labels = labels[: len(vertices)]

        # Sample points if needed (for compatibility with point-based pipelines)
        if num_points is not None and len(labels) > num_points:
            indices = np.random.choice(len(vertices), num_points, replace=False)
            points = vertices[indices]
            point_labels = labels[indices]
        else:
            points = vertices
            point_labels = labels

        unique_labels = np.unique(labels)
        num_parts = len(unique_labels)

        print(f"[Hunyuan3D] ✓ P3-SAM segmentation complete: {num_parts} parts")

        return PartSegmentationResult(
            labels=point_labels,
            points=points,
            vertex_labels=labels,
            vertices=vertices,
            num_parts=num_parts,
            num_points=len(points),
            num_vertices=len(vertices),
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
