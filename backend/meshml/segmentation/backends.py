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
from typing import Optional, Dict, Any, runtime_checkable, Union

# Protocol is available in Python 3.8+, but use typing_extensions as fallback
try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:
        # Fallback for very old Python - Protocol won't work but we'll try
        Protocol = object
import numpy as np

# Configure PyTorch CUDA memory allocation to reduce fragmentation
# This helps prevent OOM errors by allowing memory segments to expand/contract
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from .types import PartSegmentationResult


@runtime_checkable
class PartSegmentationBackend(Protocol):
    """
    Protocol for part segmentation backends.

    All segmentation backends must implement the segment() method
    that takes a mesh path and returns a PartSegmentationResult.
    """

    def segment(
        self, mesh_path: Union[Path, str], *, num_points: Optional[int] = None, **kwargs
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
        checkpoint_path: Optional[Union[Path, str]] = None,
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
        mesh_path: Union[Path, str],
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
        model_ckpt_dir: Optional[Union[Path, str]] = None,
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
        self._automask = None  # Store AutoMask instance separately

    def clear_gpu_memory(self):
        """Clear GPU memory by unloading model and clearing cache."""
        import torch
        import gc

        if torch.cuda.is_available():
            print("[Hunyuan3D] Clearing GPU memory...")
            before_allocated = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3

            # Delete model references
            if self._automask is not None:
                # Try to move model to CPU first
                try:
                    if hasattr(self._automask, "model") and hasattr(
                        self._automask.model, "cpu"
                    ):
                        self._automask.model.cpu()
                        print("[Hunyuan3D] Moved AutoMask model to CPU")
                except Exception as e:
                    print(f"[Hunyuan3D] Could not move AutoMask to CPU: {e}")
                del self._automask
                self._automask = None

            if self._model is not None:
                try:
                    if hasattr(self._model, "cpu"):
                        self._model.cpu()
                        print("[Hunyuan3D] Moved model to CPU")
                except Exception as e:
                    print(f"[Hunyuan3D] Could not move model to CPU: {e}")
                del self._model
                self._model = None

            # Clear cache and run garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            after_allocated = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3
            freed = before_reserved - after_reserved

            print(f"[Hunyuan3D] GPU memory cleared: freed {freed:.2f} GB")
            print(
                f"[Hunyuan3D] Memory after clear: {after_allocated:.2f} GB allocated, {after_reserved:.2f} GB reserved"
            )

            return {
                "freed_gb": freed,
                "before": {"allocated": before_allocated, "reserved": before_reserved},
                "after": {"allocated": after_allocated, "reserved": after_reserved},
            }
        else:
            print("[Hunyuan3D] CUDA not available, nothing to clear")
            return {"freed_gb": 0}

    def _load_model(self):
        """Lazy load the P3-SAM model from local files only."""
        import torch  # Import at the top so it's available for early return path

        if self._model is not None:
            # OPTIMIZATION 3: Move model back to GPU if it was offloaded to CPU
            if torch.cuda.is_available():
                try:
                    if hasattr(self._model, "cuda"):
                        self._model = self._model.cuda()
                    elif hasattr(self._automask, "model") and hasattr(
                        self._automask.model, "cuda"
                    ):
                        self._automask.model = self._automask.model.cuda()
                except:
                    pass
            return

        try:
            import trimesh
            import numpy as np
            import sys

            print("[Hunyuan3D] Loading P3-SAM model from local files only...")

            # Find P3-SAM code directory
            p3sam_code_path = (
                Path(__file__).parent.parent.parent
                / "checkpoints"
                / "hunyuanpart"
                / "P3-SAM"
            )

            if not p3sam_code_path.exists():
                raise RuntimeError(
                    f"P3-SAM code not found at: {p3sam_code_path}\n"
                    "Please ensure the Hunyuan3D-Part repository is cloned:\n"
                    "  cd backend/checkpoints && git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git hunyuanpart"
                )

            # CRITICAL PATH SETUP: Add paths BEFORE any imports that might trigger model.py import
            # The issue: model.py does 'from utils.misc import smart_load_model' at module level
            # This import happens when model.py is first imported, so paths MUST be set up first

            hunyuan_root = p3sam_code_path.parent
            xpart_partgen_path = hunyuan_root / "XPart" / "partgen"

            if not xpart_partgen_path.exists():
                raise RuntimeError(
                    f"XPart/partgen directory not found at: {xpart_partgen_path}\n"
                    "Please ensure the Hunyuan3D-Part repository is fully cloned."
                )

            # Verify utils directory exists inside partgen (where utils.misc actually lives)
            utils_path = xpart_partgen_path / "utils"
            misc_file = utils_path / "misc.py"
            if not misc_file.exists():
                raise RuntimeError(
                    f"utils/misc.py not found at: {misc_file}\n"
                    f"Expected location: XPart/partgen/utils/misc.py\n"
                    "Please ensure the Hunyuan3D-Part repository is fully cloned."
                )

            # CRITICAL: Add P3-SAM FIRST (before XPart/partgen) for auto_mask_no_postprocess imports
            # There are TWO different utils packages:
            #   1. P3-SAM/utils/ - contains chamfer3D/ (needed FIRST by auto_mask_no_postprocess.py)
            #   2. XPart/partgen/utils/ - contains misc.py (needed by model.py)
            #
            # Strategy: Add P3-SAM at position 0, then XPart/partgen
            # When auto_mask_no_postprocess imports utils.chamfer3D, P3-SAM/utils will be found first
            # When model.py imports utils.misc, XPart/partgen will be found (after clearing cache)

            # Add P3-SAM root to path FIRST (needed for utils.chamfer3D)
            p3sam_code_path_abs = str(p3sam_code_path.resolve())
            if p3sam_code_path_abs not in sys.path:
                sys.path.insert(0, p3sam_code_path_abs)
            print(
                f"[Hunyuan3D] ✓ Added P3-SAM root to sys.path (position 0): {p3sam_code_path_abs}"
            )

            # Add demo directory to path (for auto_mask_no_postprocess import)
            demo_path = p3sam_code_path / "demo"
            if demo_path.exists():
                demo_path_abs = str(demo_path.resolve())
                if demo_path_abs not in sys.path:
                    sys.path.insert(0, demo_path_abs)
                print(
                    f"[Hunyuan3D] ✓ Added demo directory to sys.path: {demo_path_abs}"
                )

            # Add XPart/partgen to path (needed for utils.misc in model.py)
            # Insert AFTER P3-SAM so P3-SAM/utils takes precedence for auto_mask_no_postprocess
            xpart_partgen_str = str(xpart_partgen_path.resolve())
            if xpart_partgen_str not in sys.path:
                # Insert after demo and P3-SAM
                sys.path.insert(2, xpart_partgen_str)
            # Also ensure it's appended (model.py adds it via sys.path.append)
            if xpart_partgen_str not in sys.path:
                sys.path.append(xpart_partgen_str)
            print(f"[Hunyuan3D] ✓ Added XPart/partgen to sys.path: {xpart_partgen_str}")
            print(f"[Hunyuan3D] ✓ Verified utils/misc.py exists at: {misc_file}")

            # Also add repository root to path (some imports may expect it)
            hunyuan_root_str = str(hunyuan_root.resolve())
            if hunyuan_root_str not in sys.path:
                sys.path.insert(0, hunyuan_root_str)
            print(f"[Hunyuan3D] ✓ Added repo root to sys.path: {hunyuan_root_str}")

            print(f"[Hunyuan3D] Using P3-SAM code from: {p3sam_code_path}")
            print(f"[Hunyuan3D] XPart/partgen in path: {xpart_partgen_str}")

            # CRITICAL: Clear utils cache BEFORE verification to avoid stale imports
            # If utils was imported from a previous backend creation, it might be cached
            # with the wrong path, causing the verification to fail
            utils_modules = [
                k for k in list(sys.modules.keys()) if k.startswith("utils")
            ]
            for mod_name in utils_modules:
                del sys.modules[mod_name]
                print(
                    f"[Hunyuan3D] Cleared cached module before verification: {mod_name}"
                )

            # Verify utils.misc can be imported before proceeding
            # IMPORTANT: We temporarily move XPart/partgen to front for this verification
            # since P3-SAM is currently at position 0
            try:
                # Temporarily put XPart/partgen at position 0 to find utils.misc
                if xpart_partgen_str in sys.path:
                    sys.path.remove(xpart_partgen_str)
                sys.path.insert(0, xpart_partgen_str)

                from utils.misc import smart_load_model

                # Restore path order (P3-SAM at 0, then XPart/partgen)
                sys.path.remove(xpart_partgen_str)
                if (
                    p3sam_code_path_abs not in sys.path
                    or sys.path[0] != p3sam_code_path_abs
                ):
                    if p3sam_code_path_abs in sys.path:
                        sys.path.remove(p3sam_code_path_abs)
                    sys.path.insert(0, p3sam_code_path_abs)
                sys.path.insert(2, xpart_partgen_str)

                print(f"[Hunyuan3D] ✓ Verified utils.misc is importable")
            except ImportError as e:
                # This should not happen if paths are set correctly
                print(f"[Hunyuan3D] ✗ ERROR: utils.misc not importable: {e}")
                print(
                    f"[Hunyuan3D] Current sys.path entries containing 'XPart' or 'Hunyuan3D':"
                )
                for p in sys.path:
                    if "XPart" in p or "Hunyuan3D" in p:
                        print(f"  - {p}")
                raise RuntimeError(
                    f"utils.misc cannot be imported even though XPart/partgen is in path.\n"
                    f"Path: {xpart_partgen_str}\n"
                    f"Error: {e}\n"
                    f"Please ensure the Hunyuan3D-Part repository is fully cloned."
                ) from e

            # CRITICAL: Handle TWO different utils packages:
            #   1. XPart/partgen/utils/ - contains misc.py (needed by model.py)
            #   2. P3-SAM/utils/ - contains chamfer3D/ (needed by auto_mask_no_postprocess.py)
            #
            # Strategy:
            # - Keep both paths in sys.path
            # - P3-SAM is at position 0, so utils.chamfer3D will be found from P3-SAM/utils
            # - model.py adds XPart/partgen to sys.path itself (line 7-10 in model.py)
            # - When model.py imports utils.misc, it will use the XPart/partgen it just added
            # - We just need to clear utils cache before importing so Python can resolve correctly

            # Clear ALL utils-related modules from cache to ensure fresh imports
            # This is critical - Python might have cached utils from our verification above
            utils_modules = [
                k for k in list(sys.modules.keys()) if k.startswith("utils")
            ]
            for mod_name in utils_modules:
                del sys.modules[mod_name]
                print(f"[Hunyuan3D] Cleared cached module: {mod_name}")

            # Also clear any cached 'model' module if it exists
            if "model" in sys.modules:
                mod = sys.modules["model"]
                if hasattr(mod, "__file__") and "P3-SAM" in str(mod.__file__):
                    del sys.modules["model"]
                    print(f"[Hunyuan3D] Cleared cached model module")

            # CRITICAL: Reorder paths so XPart/partgen is at position 0 when model.py is imported
            # model.py needs utils.misc from XPart/partgen/utils, so it must be found first
            # We'll reorder paths temporarily for the import, then restore order
            if xpart_partgen_str in sys.path:
                sys.path.remove(xpart_partgen_str)
            sys.path.insert(0, xpart_partgen_str)  # Put XPart/partgen at position 0

            # Ensure P3-SAM is also in path (at position 1) for utils.chamfer3D
            if p3sam_code_path_abs in sys.path:
                if sys.path.index(p3sam_code_path_abs) != 1:
                    sys.path.remove(p3sam_code_path_abs)
                    sys.path.insert(1, p3sam_code_path_abs)
            else:
                sys.path.insert(1, p3sam_code_path_abs)

            print(
                f"[Hunyuan3D] Reordered paths: XPart/partgen at 0, P3-SAM at 1 (for model.py to find utils.misc)"
            )

            # Clear any other cached imports that might interfere
            other_modules_to_clear = []
            for mod_name in list(sys.modules.keys()):
                if any(
                    x in mod_name for x in ["auto_mask", "model"]
                ) and mod_name not in [
                    "sys",
                    "os",
                ]:
                    mod = sys.modules.get(mod_name)
                    if mod and hasattr(mod, "__file__"):
                        mod_file = str(mod.__file__)
                        if (
                            "Hunyuan3D" in mod_file
                            or "P3-SAM" in mod_file
                            or "XPart" in mod_file
                        ):
                            other_modules_to_clear.append(mod_name)

            for mod_name in other_modules_to_clear:
                del sys.modules[mod_name]
                print(f"[Hunyuan3D] Cleared cached module: {mod_name}")

            # Import P3-SAM AutoMask class
            # The challenge: auto_mask_no_postprocess imports model.py (needs utils.misc from XPart/partgen),
            # then imports utils.chamfer3D (needs utils from P3-SAM).
            #
            # Solution: Use importlib to manually handle the import, clearing utils cache between steps
            try:
                import importlib.util

                # Step 1: Import model.py manually (it needs XPart/partgen at position 0)
                # This ensures model.py can import utils.misc correctly
                model_path = p3sam_code_path / "model.py"
                model_spec = importlib.util.spec_from_file_location(
                    "p3sam_model", model_path
                )
                model_module = importlib.util.module_from_spec(model_spec)
                model_spec.loader.exec_module(model_module)
                # Store model module in sys.modules so auto_mask_no_postprocess can import it
                sys.modules["model"] = model_module
                print(
                    f"[Hunyuan3D] ✓ Manually imported model.py (with utils.misc from XPart/partgen)"
                )

                # Step 2: Clear utils cache and reorder paths for utils.chamfer3D import
                # Now we need P3-SAM at position 0 so utils.chamfer3D is found
                utils_modules = [
                    k for k in list(sys.modules.keys()) if k.startswith("utils")
                ]
                for mod_name in utils_modules:
                    del sys.modules[mod_name]
                    print(f"[Hunyuan3D] Cleared cached module: {mod_name}")

                # Reorder paths: P3-SAM at 0, XPart/partgen at 1
                if p3sam_code_path_abs in sys.path:
                    if sys.path[0] != p3sam_code_path_abs:
                        sys.path.remove(p3sam_code_path_abs)
                        sys.path.insert(0, p3sam_code_path_abs)
                if (
                    xpart_partgen_str in sys.path
                    and sys.path.index(xpart_partgen_str) < 2
                ):
                    sys.path.remove(xpart_partgen_str)
                    sys.path.insert(1, xpart_partgen_str)
                print(f"[Hunyuan3D] Reordered paths: P3-SAM at 0 (for utils.chamfer3D)")

                # Step 3: Now import auto_mask_no_postprocess
                # It will import model from sys.modules, and import utils.chamfer3D from P3-SAM
                from auto_mask_no_postprocess import AutoMask

                print("[Hunyuan3D] ✓ Imported AutoMask from local code")
            except ImportError as e:
                missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
                error_msg = str(e)

                # Provide installation instructions
                install_script = str(p3sam_code_path.parent / "install_dependencies.sh")
                install_help = f"\nTo install all dependencies at once, run:\n"
                install_help += f"  cd {p3sam_code_path.parent}\n"
                install_help += f"  bash install_dependencies.sh\n\n"
                install_help += "Or install manually:\n"
                install_help += "  pip install trimesh fpsample numba scipy scikit-learn scikit-image\n"
                install_help += (
                    "  pip install addict omegaconf einops timm viser ninja\n"
                )
                install_help += "  pip install spconv-cu118  # (or spconv-cu121/cu124 based on CUDA version)\n"
                install_help += "  pip install torch-scatter -f https://data.pyg.org/whl/torch-{VERSION}+{CUDA}.html\n"

                raise RuntimeError(
                    f"Failed to import P3-SAM AutoMask: {e}\n\n"
                    f"Missing module: {missing_module}\n\n"
                    "Please ensure:\n"
                    "  1. P3-SAM code is cloned: checkpoints/hunyuanpart/P3-SAM/\n"
                    "  2. XPart/partgen directory exists: checkpoints/hunyuanpart/XPart/partgen/\n"
                    f"  3. All dependencies are installed:{install_help}"
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
                # Clear GPU cache before loading model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    allocated_before = torch.cuda.memory_allocated() / 1024**3
                    print(
                        f"[Hunyuan3D] GPU memory before model load: {allocated_before:.2f} GB"
                    )

                # Reduce memory usage: lower point_num and prompt_num
                # IMPORTANT: These parameters control how much GPU memory the model
                # allocates at initialization. point_num and prompt_num create buffers
                # that stay in GPU memory as long as the model is loaded.
                # point_num: number of points to sample (lower = less memory)
                # prompt_num: number of prompt points (lower = less memory)
                point_num = int(
                    os.environ.get("P3SAM_POINT_NUM", "20000")
                )  # Default: 20k (reduced from 30k to save more memory)
                prompt_num = int(
                    os.environ.get("P3SAM_PROMPT_NUM", "100")
                )  # Default: 100 (reduced from 150 to save more memory)

                print(
                    f"[Hunyuan3D] Initializing model with point_num={point_num}, prompt_num={prompt_num}"
                )
                print(
                    f"[Hunyuan3D] NOTE: These initialization parameters control model memory footprint (~{point_num//1000}k points, ~{prompt_num} prompts)"
                )

                # Use AutoMask wrapper (handles model initialization)
                self._automask = AutoMask(
                    ckpt_path=str(p3sam_ckpt),
                    point_num=point_num,
                    prompt_num=prompt_num,
                    threshold=0.95,
                    post_process=False,  # Use no post-process version for speed
                )
                self._model = self._automask  # Store AutoMask as the model

                # OPTIMIZATION 1: Note about FP16
                # We use autocast during inference instead of converting the model itself
                # This avoids dtype mismatch issues while still reducing memory during inference
                if torch.cuda.is_available():
                    print(
                        "[Hunyuan3D] Model loaded in FP32. Will use FP16 (autocast) during inference to reduce memory."
                    )
                    # Don't convert model to FP16 directly - causes dtype mismatches
                    # Instead, we use torch.autocast during inference which handles dtype conversion safely

                # Check memory after model load
                if torch.cuda.is_available():
                    allocated_after = torch.cuda.memory_allocated() / 1024**3
                    reserved_after = torch.cuda.memory_reserved() / 1024**3
                    model_memory = allocated_after - allocated_before
                    print(
                        f"[Hunyuan3D] ✓ Loaded P3-SAM model from local checkpoint (point_num={point_num}, prompt_num={prompt_num})"
                    )
                    print(
                        f"[Hunyuan3D] Model memory usage: {model_memory:.2f} GB (Total: {allocated_after:.2f} GB allocated, {reserved_after:.2f} GB reserved)"
                    )
                else:
                    print(
                        f"[Hunyuan3D] ✓ Loaded P3-SAM model from local checkpoint (point_num={point_num}, prompt_num={prompt_num})"
                    )
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
                "  1. P3-SAM code is cloned: checkpoints/hunyuanpart/P3-SAM/\n"
                "  2. Checkpoint file exists: checkpoints/Hunyuan3D-Part/p3sam/p3sam.safetensors\n"
                "  3. All dependencies are installed\n"
                "  4. GPU/CUDA is available if required\n\n"
                "If you cannot get Hunyuan3D-Part working, switch to PointNet backend:\n"
                "  export SEGMENTATION_BACKEND=pointnet"
            ) from e

    def segment(
        self, mesh_path: Union[Path, str], *, num_points: Optional[int] = None, **kwargs
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

        # OPTIMIZATION 2: Aggressive garbage collection before inference
        if torch.cuda.is_available():
            import gc

            # Multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Force Python to release memory
            gc.collect()

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
            )
            print(
                f"[Hunyuan3D] GPU memory before inference: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, ~{free_memory:.2f} GB free"
            )

        # Use AutoMask.predict_aabb method
        if hasattr(self._model, "predict_aabb"):
            # AutoMask interface
            # Use reduced memory parameters to avoid OOM errors
            # These can be overridden via environment variables
            inference_point_num = int(
                os.environ.get("P3SAM_INFERENCE_POINT_NUM", "15000")
            )  # Default: 15k (ultra-low to avoid OOM)
            inference_prompt_num = int(
                os.environ.get("P3SAM_INFERENCE_PROMPT_NUM", "75")
            )  # Default: 75 (ultra-low to avoid OOM)
            prompt_bs = int(
                os.environ.get("P3SAM_PROMPT_BS", "4")
            )  # Default: 4 (ultra-low to avoid OOM)

            # Dynamically adjust parameters based on available GPU memory
            if torch.cuda.is_available():
                free_memory_gb = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                    - reserved
                )

                # Check if we have enough memory for inference (need at least 3 GB free)
                if free_memory_gb < 3.0:
                    # If free memory is critically low (< 3 GB), reduce parameters aggressively
                    print(
                        f"[Hunyuan3D] ⚠⚠ CRITICAL: Very low GPU memory ({free_memory_gb:.2f} GB free), using minimal parameters..."
                    )
                    inference_point_num = min(inference_point_num, 10000)  # Cap at 10k
                    inference_prompt_num = min(inference_prompt_num, 50)  # Cap at 50
                    prompt_bs = min(prompt_bs, 2)  # Cap at 2
                    print(
                        f"[Hunyuan3D] Adjusted: point_num={inference_point_num}, prompt_num={inference_prompt_num}, prompt_bs={prompt_bs}"
                    )

                    # Warn if still very low
                    if free_memory_gb < 1.5:
                        print(
                            f"[Hunyuan3D] ⚠⚠⚠ WARNING: Extremely low free memory ({free_memory_gb:.2f} GB). "
                            f"Inference may still fail. Consider:"
                            f"\n  1. Kill and restart server: ./kill_all_gpu_processes.sh && ./start_server.sh"
                            f"\n  2. Use clear_gpu_memory API: curl -X POST http://localhost:5000/api/mesh/clear_gpu_memory"
                            f"\n  3. Lower P3SAM_POINT_NUM/P3SAM_PROMPT_NUM even further in start_server.sh"
                        )
                # If free memory is low (3-5 GB), reduce moderately
                elif free_memory_gb < 5.0:
                    print(
                        f"[Hunyuan3D] ⚠ Low GPU memory ({free_memory_gb:.2f} GB free), using conservative parameters..."
                    )
                    inference_point_num = min(inference_point_num, 12000)  # Cap at 12k
                    inference_prompt_num = min(inference_prompt_num, 60)  # Cap at 60
                    prompt_bs = min(prompt_bs, 3)  # Cap at 3
                    print(
                        f"[Hunyuan3D] Adjusted: point_num={inference_point_num}, prompt_num={inference_prompt_num}, prompt_bs={prompt_bs}"
                    )
                # If free memory is moderate (5-8 GB), slight reduction
                elif free_memory_gb < 8.0:
                    print(
                        f"[Hunyuan3D] Moderate GPU memory ({free_memory_gb:.2f} GB free), using standard parameters..."
                    )
                    # Keep defaults but cap slightly
                    inference_point_num = min(inference_point_num, 15000)
                    inference_prompt_num = min(inference_prompt_num, 75)
                    prompt_bs = min(prompt_bs, 4)

            try:
                # Use smaller parameters and disable parallel processing to save memory
                # Ensure minimum point_num for reliable segmentation (at least 5000)
                final_point_num = num_points if num_points else inference_point_num
                if final_point_num < 5000:
                    print(
                        f"[Hunyuan3D] ⚠ Warning: point_num={final_point_num} is very low. "
                        f"Increasing to minimum 5000 for reliable segmentation."
                    )
                    final_point_num = 5000

                print(
                    f"[Hunyuan3D] Starting inference with: point_num={final_point_num}, prompt_num={inference_prompt_num}, prompt_bs={prompt_bs}, is_parallel=False"
                )

                # OPTIMIZATION 3: Use torch.autocast for automatic mixed precision during inference
                # This uses FP16 for computations where possible, reducing memory
                # Note: We don't convert the model itself to FP16 to avoid dtype mismatch errors
                import torch

                # Try autocast, but if it causes dtype issues, fall back to FP32
                use_autocast = (
                    torch.cuda.is_available()
                    and os.environ.get("P3SAM_USE_AUTOCAST", "1") == "1"
                )

                if use_autocast:
                    try:
                        print(
                            "[Hunyuan3D] Using FP16 autocast during inference to reduce memory..."
                        )
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=True,
                        ):
                            aabb, face_ids, processed_mesh = self._model.predict_aabb(
                                mesh=mesh,
                                point_num=final_point_num,
                                prompt_num=inference_prompt_num,
                                threshold=0.95,
                                post_process=False,
                                save_path=None,
                                save_mid_res=False,
                                show_info=False,
                                clean_mesh_flag=True,
                                seed=42,
                                is_parallel=False,  # Disable parallel processing to reduce memory usage
                                prompt_bs=prompt_bs,  # Reduced batch size for less memory usage
                            )
                    except IndexError as idx_error:
                        # Handle empty results - model ran but produced no valid masks
                        error_str = str(idx_error)
                        if "out of bounds" in error_str or "size 0" in error_str:
                            print(
                                f"[Hunyuan3D] ⚠ Inference completed but produced no valid masks: {error_str}"
                            )
                            print(
                                "[Hunyuan3D] This usually means parameters are too low or threshold too high."
                            )
                            print(
                                f"[Hunyuan3D] Attempting retry with lower threshold (0.85) and higher point_num..."
                            )

                            # Retry with lower threshold and higher point_num
                            retry_point_num = max(
                                final_point_num, 5000
                            )  # At least 5k points
                            retry_threshold = 0.85  # Lower threshold to get more masks

                            try:
                                aabb, face_ids, processed_mesh = (
                                    self._model.predict_aabb(
                                        mesh=mesh,
                                        point_num=retry_point_num,
                                        prompt_num=inference_prompt_num,
                                        threshold=retry_threshold,  # Lower threshold
                                        post_process=False,
                                        save_path=None,
                                        save_mid_res=False,
                                        show_info=False,
                                        clean_mesh_flag=True,
                                        seed=42,
                                        is_parallel=False,
                                        prompt_bs=prompt_bs,
                                    )
                                )
                                print(
                                    f"[Hunyuan3D] ✓ Retry successful with threshold={retry_threshold}, point_num={retry_point_num}"
                                )
                            except Exception as retry_error:
                                raise RuntimeError(
                                    f"P3-SAM inference failed: No valid masks produced even with adjusted parameters.\n\n"
                                    f"Original error: {error_str}\n"
                                    f"Retry error: {retry_error}\n\n"
                                    f"Possible solutions:\n"
                                    f"  1. Increase P3SAM_INFERENCE_POINT_NUM (current: {final_point_num})\n"
                                    f"  2. Increase P3SAM_INFERENCE_PROMPT_NUM (current: {inference_prompt_num})\n"
                                    f"  3. Check if mesh is valid and has clear parts to segment\n"
                                    f"  4. Try a different mesh or simplify the current one"
                                ) from idx_error
                        else:
                            raise
                    except (RuntimeError, TypeError) as autocast_error:
                        # Check if it's a dtype mismatch error
                        error_str = str(autocast_error).lower()
                        if (
                            "dtype" in error_str
                            or "half" in error_str
                            or "float" in error_str
                        ):
                            print(
                                f"[Hunyuan3D] ⚠ Autocast caused dtype mismatch, disabling and using FP32: {autocast_error}"
                            )
                            print(
                                "[Hunyuan3D] To disable autocast permanently, set: export P3SAM_USE_AUTOCAST=0"
                            )
                            # Retry without autocast
                            aabb, face_ids, processed_mesh = self._model.predict_aabb(
                                mesh=mesh,
                                point_num=final_point_num,
                                prompt_num=inference_prompt_num,
                                threshold=0.95,
                                post_process=False,
                                save_path=None,
                                save_mid_res=False,
                                show_info=False,
                                clean_mesh_flag=True,
                                seed=42,
                                is_parallel=False,
                                prompt_bs=prompt_bs,
                            )
                        else:
                            # Re-raise if it's not a dtype error
                            raise
                else:
                    # Autocast disabled, use FP32
                    try:
                        aabb, face_ids, processed_mesh = self._model.predict_aabb(
                            mesh=mesh,
                            point_num=final_point_num,
                            prompt_num=inference_prompt_num,
                            threshold=0.95,
                            post_process=False,
                            save_path=None,
                            save_mid_res=False,
                            show_info=False,
                            clean_mesh_flag=True,
                            seed=42,
                            is_parallel=False,
                            prompt_bs=prompt_bs,
                        )
                    except IndexError as idx_error:
                        # Handle empty results - model ran but produced no valid masks
                        error_str = str(idx_error)
                        if "out of bounds" in error_str or "size 0" in error_str:
                            print(
                                f"[Hunyuan3D] ⚠ Inference completed but produced no valid masks: {error_str}"
                            )
                            print(
                                "[Hunyuan3D] Attempting retry with lower threshold (0.85) and higher point_num..."
                            )

                            # Retry with lower threshold and higher point_num
                            retry_point_num = max(
                                final_point_num, 5000
                            )  # At least 5k points
                            retry_threshold = 0.85  # Lower threshold to get more masks

                            try:
                                aabb, face_ids, processed_mesh = (
                                    self._model.predict_aabb(
                                        mesh=mesh,
                                        point_num=retry_point_num,
                                        prompt_num=inference_prompt_num,
                                        threshold=retry_threshold,  # Lower threshold
                                        post_process=False,
                                        save_path=None,
                                        save_mid_res=False,
                                        show_info=False,
                                        clean_mesh_flag=True,
                                        seed=42,
                                        is_parallel=False,
                                        prompt_bs=prompt_bs,
                                    )
                                )
                                print(
                                    f"[Hunyuan3D] ✓ Retry successful with threshold={retry_threshold}, point_num={retry_point_num}"
                                )
                            except Exception as retry_error:
                                raise RuntimeError(
                                    f"P3-SAM inference failed: No valid masks produced even with adjusted parameters.\n\n"
                                    f"Original error: {error_str}\n"
                                    f"Retry error: {retry_error}\n\n"
                                    f"Possible solutions:\n"
                                    f"  1. Increase P3SAM_INFERENCE_POINT_NUM (current: {final_point_num})\n"
                                    f"  2. Increase P3SAM_INFERENCE_PROMPT_NUM (current: {inference_prompt_num})\n"
                                    f"  3. Check if mesh is valid and has clear parts to segment\n"
                                    f"  4. Try a different mesh or simplify the current one"
                                ) from idx_error
                        else:
                            raise

                # face_ids is per-face labels, need to convert to per-vertex
                # Map face labels to vertex labels
                # Initialize vertex_labels with length matching vertices (always correct size)
                # This ensures vertex_labels count always matches vertices count
                vertex_labels = np.full(len(vertices), -1, dtype=np.int32)

                # Sanity check: ensure we have the same number of vertices as expected
                if len(vertex_labels) != len(vertices):
                    raise ValueError(
                        f"Mismatch between vertex_labels length ({len(vertex_labels)}) "
                        f"and vertices length ({len(vertices)})"
                    )

                # Step 1: Assign labels from faces to vertices
                # For each face, assign its label to all its vertices (if vertex not already labeled)
                for face_idx, face in enumerate(mesh.faces):
                    face_label = face_ids[face_idx]
                    if face_label >= 0:  # Valid label
                        for v_idx in face:
                            if vertex_labels[v_idx] == -1:
                                vertex_labels[v_idx] = face_label
                            # If vertex already has a label, keep the first one (first assignment wins)

                # Step 2: Handle unlabeled vertices by finding nearest face
                # This ensures all vertices get a label, even if they weren't covered by the face assignment
                unlabeled = vertex_labels == -1
                if np.any(unlabeled):
                    try:
                        from scipy.spatial import cKDTree

                        # Get face centers for nearest-neighbor lookup
                        face_centers = np.array(
                            [mesh.triangles_center[i] for i in range(len(mesh.faces))]
                        )
                        tree = cKDTree(face_centers)
                        unlabeled_vertices = vertices[unlabeled]
                        # Ensure unlabeled_vertices is 2D (N, 3) even if only one vertex
                        if unlabeled_vertices.ndim == 1:
                            unlabeled_vertices = unlabeled_vertices.reshape(1, -1)
                        _, nearest_faces = tree.query(unlabeled_vertices, k=1)
                        # Ensure nearest_faces is a 1D array (cKDTree may return 2D for single query)
                        if nearest_faces.ndim > 1:
                            nearest_faces = nearest_faces.flatten()
                        elif np.isscalar(nearest_faces):
                            nearest_faces = np.array([nearest_faces])
                        # Assign labels from nearest faces to unlabeled vertices
                        vertex_labels[unlabeled] = face_ids[nearest_faces]
                    except ImportError:
                        # Fallback: use face labels directly for unlabeled vertices
                        # Assign from first face that contains the vertex
                        for v_idx in np.where(unlabeled)[0]:
                            for face_idx, face in enumerate(mesh.faces):
                                if v_idx in face:
                                    vertex_labels[v_idx] = face_ids[face_idx]
                                    break

                # Step 3: Ensure all labels are non-negative and contiguous
                # Check if there are any negative labels (not just the first one)
                unique_labels = np.unique(vertex_labels)
                has_negative_labels = np.any(unique_labels < 0)
                noise_label = None  # Will be set if negative labels exist

                if has_negative_labels:
                    # Remap labels to ensure they're non-negative and contiguous (0, 1, 2, ...)
                    # CRITICAL: Map negative labels to a noise label that doesn't conflict with valid part IDs
                    # Filter to only positive labels (>= 0), then create mapping to contiguous indices
                    positive_labels = unique_labels[unique_labels >= 0]

                    if len(positive_labels) == 0:
                        # No valid labels - set all to 0 (shouldn't happen in practice)
                        print(
                            "[Hunyuan3D] ⚠ Warning: No positive labels found, setting all to 0"
                        )
                        vertex_labels[:] = 0
                        face_ids = np.zeros_like(face_ids, dtype=np.int32)
                    else:
                        # Create mapping: old_label -> new_contiguous_label (0, 1, 2, ...)
                        # Sort positive labels to ensure deterministic remapping
                        positive_labels_sorted = np.sort(positive_labels)
                        label_map = {
                            old: new for new, old in enumerate(positive_labels_sorted)
                        }

                        # Determine noise label: use max(positive_labels) + 1 to avoid conflicts
                        # This ensures negative labels don't overwrite valid part ID 0
                        noise_label = int(positive_labels_sorted[-1]) + 1

                        # Apply remapping: positive labels get remapped to contiguous indices,
                        # negative labels become noise_label (separate from valid part IDs)
                        def remap_label(l):
                            """Remap a single label: positive -> contiguous index, negative -> noise_label"""
                            if l < 0:
                                return noise_label  # Negative labels become noise_label (doesn't conflict with part IDs)
                            elif l in label_map:
                                return label_map[
                                    l
                                ]  # Valid positive label gets remapped to contiguous index
                            else:
                                # This should never happen if logic is correct, but handle gracefully
                                print(
                                    f"[Hunyuan3D] ⚠ Warning: Label {l} not in remap dictionary, defaulting to noise_label {noise_label}"
                                )
                                return noise_label

                        vertex_labels = np.array(
                            [remap_label(l) for l in vertex_labels], dtype=np.int32
                        )
                        face_ids = np.array(
                            [remap_label(l) for l in face_ids], dtype=np.int32
                        )

                        # Filter out noise_label when counting parts
                        num_valid_parts = len(positive_labels_sorted)
                        print(
                            f"[Hunyuan3D] Remapped labels: {num_valid_parts} positive labels -> contiguous [0, {num_valid_parts-1}], "
                            f"negative labels -> noise_label {noise_label}"
                        )

                # Final sanity check: ensure vertex_labels count still matches vertices count
                # This should always be true, but verify after all processing
                if len(vertex_labels) != len(vertices):
                    raise ValueError(
                        f"After processing, vertex_labels length ({len(vertex_labels)}) "
                        f"does not match vertices length ({len(vertices)})"
                    )

                # Sample points if needed
                if num_points is not None and len(vertex_labels) > num_points:
                    indices = np.random.choice(len(vertices), num_points, replace=False)
                    points = vertices[indices]
                    point_labels = vertex_labels[indices]
                else:
                    points = vertices
                    point_labels = vertex_labels

                # Count unique valid part IDs (exclude noise labels)
                # After remapping with negative labels -> noise_label, valid parts are contiguous [0, num_parts-1]
                unique_labels_final = np.unique(vertex_labels)
                if has_negative_labels and noise_label is not None:
                    # Valid parts are those less than noise_label (which is max positive + 1)
                    # This excludes the noise_label from the part count
                    valid_part_labels = unique_labels_final[
                        unique_labels_final < noise_label
                    ]
                    num_parts = len(valid_part_labels)
                else:
                    # No negative labels were present, all non-negative labels are valid parts
                    num_parts = len(unique_labels_final[unique_labels_final >= 0])

                print(f"[Hunyuan3D] ✓ P3-SAM segmentation complete: {num_parts} parts")

                # OPTIMIZATION 2: Aggressive garbage collection after inference
                if torch.cuda.is_available():
                    # Get memory stats before clearing
                    before_allocated = torch.cuda.memory_allocated() / 1024**3
                    before_reserved = torch.cuda.memory_reserved() / 1024**3

                    # Aggressively clear cache - multiple rounds
                    import gc

                    for _ in range(3):
                        gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()

                    after_allocated = torch.cuda.memory_allocated() / 1024**3
                    after_reserved = torch.cuda.memory_reserved() / 1024**3

                    freed = before_reserved - after_reserved
                    if freed > 0.1:  # Only print if we freed significant memory
                        print(
                            f"[Hunyuan3D] Freed {freed:.2f} GB GPU memory after inference"
                        )

                return PartSegmentationResult(
                    labels=point_labels,
                    points=points,
                    vertex_labels=vertex_labels,
                    vertices=vertices,
                    num_parts=num_parts,
                    num_points=len(points),
                    num_vertices=len(vertices),
                )
            except RuntimeError as e:
                # Check if it's a CUDA OOM error
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                    # Clear cache and provide helpful error message
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import traceback

                    traceback.print_exc()
                    # Try to free more memory before re-raising
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()

                    allocated_after_oom = torch.cuda.memory_allocated() / 1024**3
                    reserved_after_oom = torch.cuda.memory_reserved() / 1024**3
                    total_memory = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )
                    free_memory_gb = total_memory - reserved_after_oom

                    raise RuntimeError(
                        f"P3-SAM inference failed: CUDA out of memory\n\n"
                        f"Error: {e}\n\n"
                        f"Current GPU memory status:\n"
                        f"  Total GPU: {total_memory:.2f} GB\n"
                        f"  Allocated: {allocated_after_oom:.2f} GB ({allocated_after_oom/total_memory*100:.1f}%)\n"
                        f"  Reserved: {reserved_after_oom:.2f} GB ({reserved_after_oom/total_memory*100:.1f}%)\n"
                        f"  Free: {free_memory_gb:.2f} GB ({free_memory_gb/total_memory*100:.1f}%)\n"
                        f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n\n"
                        "IMMEDIATE ACTIONS:\n"
                        "  1. Kill all GPU processes and restart:\n"
                        "     ./kill_all_gpu_processes.sh && ./start_server.sh\n"
                        "     (Current settings should use ~10-12 GB instead of ~18 GB)\n\n"
                        "  2. If still OOM after restart, reduce parameters in start_server.sh:\n"
                        "     export P3SAM_POINT_NUM=10000        # EXTREME: Very low quality\n"
                        "     export P3SAM_PROMPT_NUM=50          # EXTREME: Very low quality\n"
                        "     export P3SAM_INFERENCE_POINT_NUM=10000\n"
                        "     export P3SAM_INFERENCE_PROMPT_NUM=50\n"
                        "     export P3SAM_PROMPT_BS=2\n\n"
                        "  3. Clear GPU memory via API (may help temporarily):\n"
                        "     curl -X POST http://localhost:5000/api/mesh/clear_gpu_memory\n\n"
                        "  4. Check what's using GPU:\n"
                        "     ./check_gpu_memory.sh\n"
                        "     python diagnose_gpu_memory.py\n\n"
                        "NOTE: Current ultra-low defaults are:\n"
                        f"  P3SAM_POINT_NUM={os.environ.get('P3SAM_POINT_NUM', '15000')}\n"
                        f"  P3SAM_PROMPT_NUM={os.environ.get('P3SAM_PROMPT_NUM', '75')}\n"
                        f"  P3SAM_INFERENCE_POINT_NUM={os.environ.get('P3SAM_INFERENCE_POINT_NUM', '15000')}\n"
                        f"  P3SAM_INFERENCE_PROMPT_NUM={os.environ.get('P3SAM_INFERENCE_PROMPT_NUM', '75')}\n"
                        f"  P3SAM_PROMPT_BS={os.environ.get('P3SAM_PROMPT_BS', '4')}"
                    ) from e
                # Re-raise other RuntimeErrors
                raise
            except Exception as e:
                import traceback

                traceback.print_exc()

                # Clear GPU cache on any error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
