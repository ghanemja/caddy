"""
FreeCAD and CadQuery initialization service
"""
import os
import sys
import importlib.util
from pathlib import Path


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
    conda_freecad_path = None
    if conda_prefix:
        conda_freecad_path = os.path.join(conda_prefix, "lib", "FreeCAD.so")
        if os.path.exists(conda_freecad_path):
            try:
                # Add conda lib directory to sys.path for FreeCAD dependencies
                conda_lib = os.path.join(conda_prefix, "lib")
                if conda_lib not in sys.path:
                    sys.path.append(conda_lib)
                
                # Set DYLD_LIBRARY_PATH for macOS (critical for FreeCAD.so to find its dependencies)
                if sys.platform == "darwin":
                    lib_path = conda_lib
                    if "DYLD_LIBRARY_PATH" in os.environ:
                        if lib_path not in os.environ["DYLD_LIBRARY_PATH"]:
                            os.environ["DYLD_LIBRARY_PATH"] = f"{lib_path}:{os.environ['DYLD_LIBRARY_PATH']}"
                    else:
                        os.environ["DYLD_LIBRARY_PATH"] = lib_path
                
                spec = importlib.util.spec_from_file_location("FreeCAD", conda_freecad_path)
                FreeCAD = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(FreeCAD)
                sys.modules["FreeCAD"] = FreeCAD
                print(f"[freecad] ✓ Loaded FreeCAD from conda: {conda_freecad_path}")
                return FreeCAD
            except Exception as e:
                print(f"[freecad] ⚠ Failed to load from conda: {e}")
                print(f"[freecad] ⚠ This might be a Python version compatibility issue")
                print(f"[freecad] ⚠ FreeCAD.so may have been compiled for a different Python version")
                print(f"[freecad] ⚠ Make sure you're using Python from the conda environment: {conda_prefix}/bin/python")
    
    # Try 3: Load from AppImage extraction directory
    base_dir = Path(__file__).parent.parent.parent  # Root level
    appimage_path = base_dir / "squashfs-root" / "usr" / "lib" / "FreeCAD.so"
    if appimage_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("FreeCAD", str(appimage_path))
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


def init_cadquery_with_freecad(freecad_module):
    """
    Initialize CadQuery with FreeCAD and return initialized modules.
    
    Args:
        freecad_module: The loaded FreeCAD module
        
    Returns:
        Tuple of (cadquery, exporters, Workplane)
    """
    from app.core.cadquery_init import init_cadquery
    return init_cadquery(freecad_module)


# Load FreeCAD at module import time
FreeCAD = None
try:
    FreeCAD = init_freecad()
    print("[freecad] ✓ FreeCAD loaded successfully")
except (ImportError, Exception) as e:
    print(f"[freecad] ✗ FreeCAD loading failed: {e}")
    print("[freecad] ✗ Make sure you're using the correct Python environment")
    print("[freecad] ✗ Try: conda activate vlm_optimizer")
    print("[freecad] ✗ Or run: ./start_server.sh (which activates the correct environment)")
    raise

