#!/usr/bin/env python3
"""
Verification script for P3-SAM setup.
Run this to check if all dependencies and paths are configured correctly.
"""

import sys
import os
from pathlib import Path


def check_paths():
    """Check that all required paths exist."""
    print("=" * 60)
    print("Checking Paths...")
    print("=" * 60)

    script_dir = Path(__file__).parent
    p3sam_path = script_dir / "P3-SAM"
    xpart_path = script_dir / "XPart" / "partgen"
    demo_path = p3sam_path / "demo"

    checks = [
        ("P3-SAM code", p3sam_path),
        ("XPart/partgen", xpart_path),
        ("P3-SAM/demo", demo_path),
        ("utils/misc.py", xpart_path / "utils" / "misc.py"),
        ("model.py", p3sam_path / "model.py"),
        ("auto_mask_no_postprocess.py", demo_path / "auto_mask_no_postprocess.py"),
    ]

    all_ok = True
    for name, path in checks:
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_ok = False

    return all_ok


def check_dependencies():
    """Check that all required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)

    dependencies = [
        "torch",
        "trimesh",
        "fpsample",
        "numba",
        "scipy",
        "sklearn",
        "skimage",
        "addict",
        "omegaconf",
        "einops",
        "timm",
        "viser",
        "ninja",
        "huggingface_hub",
        "safetensors",
    ]

    all_ok = True
    for dep in dependencies:
        try:
            if dep == "sklearn":
                import sklearn
            elif dep == "skimage":
                import skimage
            else:
                __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (not installed)")
            all_ok = False

    # Check optional dependencies
    print("\nOptional dependencies:")
    try:
        import spconv

        print(
            f"✓ spconv ({spconv.__version__ if hasattr(spconv, '__version__') else 'installed'})"
        )
    except ImportError:
        print("✗ spconv (not installed - required for sonata)")
        all_ok = False

    try:
        import torch_scatter

        print(f"✓ torch_scatter")
    except ImportError:
        print("✗ torch_scatter (not installed - required for sonata)")
        all_ok = False

    try:
        import flash_attn

        print(f"✓ flash_attn (optional)")
    except ImportError:
        print("○ flash_attn (optional, not installed - will be disabled)")

    return all_ok


def test_imports():
    """Test that all imports work correctly."""
    print("\n" + "=" * 60)
    print("Testing Imports...")
    print("=" * 60)

    script_dir = Path(__file__).parent
    p3sam_path = script_dir / "P3-SAM"
    xpart_path = script_dir / "XPart" / "partgen"

    # Add paths
    xpart_str = str(xpart_path.resolve())
    if xpart_str not in sys.path:
        sys.path.insert(0, xpart_str)

    p3sam_str = str(p3sam_path.resolve())
    if p3sam_str not in sys.path:
        sys.path.insert(0, p3sam_str)

    demo_path = p3sam_path / "demo"
    if demo_path.exists():
        demo_str = str(demo_path.resolve())
        if demo_str not in sys.path:
            sys.path.insert(0, demo_str)

    all_ok = True

    # Test utils.misc
    try:
        from utils.misc import smart_load_model

        print("✓ utils.misc")
    except Exception as e:
        print(f"✗ utils.misc: {e}")
        all_ok = False

    # Test model import
    try:
        # Import model.py directly
        import importlib.util

        model_path = p3sam_path / "model.py"
        spec = importlib.util.spec_from_file_location("p3sam_model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        print("✓ model.py")
    except Exception as e:
        print(f"✗ model.py: {e}")
        all_ok = False

    # Test AutoMask import
    try:
        from auto_mask_no_postprocess import AutoMask

        print("✓ auto_mask_no_postprocess.AutoMask")
    except Exception as e:
        print(f"✗ auto_mask_no_postprocess: {e}")
        all_ok = False

    return all_ok


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("P3-SAM Setup Verification")
    print("=" * 60 + "\n")

    path_ok = check_paths()
    deps_ok = check_dependencies()
    import_ok = test_imports()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if path_ok and deps_ok and import_ok:
        print("✓✓✓ All checks passed! Setup is correct.")
        return 0
    else:
        print("✗✗✗ Some checks failed. Please fix the issues above.")
        if not path_ok:
            print(
                "\nPath issues: Ensure the Hunyuan3D-Part repository is fully cloned."
            )
        if not deps_ok:
            print(
                "\nDependency issues: Run 'bash install_dependencies.sh' to install all dependencies."
            )
        if not import_ok:
            print(
                "\nImport issues: Check that paths are set correctly and dependencies are installed."
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
