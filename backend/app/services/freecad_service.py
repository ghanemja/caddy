"""
FreeCAD loading service
"""
import os
import sys
import importlib.util


def init_freecad():
    """
    Load FreeCAD module. Tries multiple locations:
    1. Direct import (if available in Python path)
    2. Conda environment lib directory
    3. Extracted AppImage location
    """
    # Try 1: Direct import (conda-installed FreeCAD may work this way)
    try:
        import FreeCAD
        print("[freecad] ✓ Loaded FreeCAD from system/conda")
        return FreeCAD
    except ImportError:
        pass
    
    # Try 2: Load from conda environment's lib directory
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_freecad_path = os.path.join(conda_prefix, "lib", "FreeCAD.so")
        if os.path.exists(conda_freecad_path):
            try:
                # Add conda lib directory to sys.path for FreeCAD dependencies (at end to avoid conflicts)
                conda_lib = os.path.join(conda_prefix, "lib")
                if conda_lib not in sys.path:
                    sys.path.append(conda_lib)
                spec = importlib.util.spec_from_file_location("FreeCAD", conda_freecad_path)
                FreeCAD = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(FreeCAD)
                sys.modules["FreeCAD"] = FreeCAD
                print(f"[freecad] ✓ Loaded FreeCAD from conda: {conda_freecad_path}")
                return FreeCAD
            except Exception as e:
                print(f"[freecad] ⚠ Failed to load from conda: {e}")
    
    # Try 3: Load from AppImage extraction directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Root level
    appimage_path = os.path.join(
        base_dir, 
        "squashfs-root", "usr", "lib", "FreeCAD.so"
    )
    if os.path.exists(appimage_path):
        try:
            spec = importlib.util.spec_from_file_location("FreeCAD", appimage_path)
            FreeCAD = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(FreeCAD)
            sys.modules["FreeCAD"] = FreeCAD
            print(f"[freecad] ✓ Loaded FreeCAD from AppImage: {appimage_path}")
            return FreeCAD
        except Exception as e:
            print(f"[freecad] ⚠ Failed to load from AppImage: {e}")
    
    # If all methods fail, raise error with helpful message
    raise ImportError(
        "Could not load FreeCAD module. Tried:\n"
        f"  1. Direct import from Python path\n"
        f"  2. Conda environment: {conda_freecad_path if conda_prefix else 'N/A'}\n"
        f"  3. AppImage: {appimage_path}\n"
        "Install FreeCAD with: conda install -c conda-forge freecad"
    )


# Load FreeCAD at module import time
FreeCAD = init_freecad()

