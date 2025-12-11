"""
CadQuery initialization with FreeCAD
Handles CadQuery setup and compatibility shims for cqparts_fasteners
"""
import types
import cadquery as cq
from cadquery import exporters, Workplane


def init_cadquery(freecad_module):
    """
    Initialize CadQuery with FreeCAD.
    
    Args:
        freecad_module: The loaded FreeCAD module
        
    Returns:
        Tuple of (cadquery module, exporters, Workplane)
    """
    # Create freecad_impl module if it doesn't exist (for cqparts_fasteners compatibility)
    try:
        if not hasattr(cq, 'freecad_impl'):
            # Create the freecad_impl module dynamically
            cq.freecad_impl = types.ModuleType('freecad_impl')
            cq.freecad_impl.FreeCAD = freecad_module
            print("[cadquery] ✓ Created cadquery.freecad_impl and initialized with FreeCAD")
        else:
            # Set FreeCAD in existing freecad_impl module
            cq.freecad_impl.FreeCAD = freecad_module
            print("[cadquery] ✓ CadQuery initialized with FreeCAD")
    except Exception as e:
        print(f"[cadquery] ⚠ Warning: Could not initialize CadQuery with FreeCAD: {e}")
        print("[cadquery] ⚠ Continuing anyway - some features may not work")

    # Make BoxSelector available at cadquery level for backward compatibility
    try:
        if not hasattr(cq, 'BoxSelector'):
            from cadquery import selectors
            if hasattr(selectors, 'BoxSelector'):
                cq.BoxSelector = selectors.BoxSelector
                print("[cadquery] ✓ Made BoxSelector available from cadquery.selectors")
            else:
                print("[cadquery] ⚠ BoxSelector not found in cadquery.selectors")
    except Exception as e:
        print(f"[cadquery] ⚠ Warning: Could not set up BoxSelector compatibility: {e}")
    
    return cq, exporters, Workplane

