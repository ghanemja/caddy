#!/usr/bin/env python
"""
Standalone GLB builder script that runs in the freecad environment.
Called by optim.py to build CAD models without .wrapped issues.
"""

import sys
import os
import io
import warnings

# Suppress all warnings (CadQuery 1.x deprecation warnings)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def build_glb(use_generated=False):
    """Build GLB file using the appropriate robot_base module"""
    try:
        # Suppress stdout during imports to prevent text corruption of binary output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        if use_generated:
            gen_path = os.path.join(os.path.dirname(__file__), "generated", "robot_base_vlm.py")
            if os.path.exists(gen_path):
                print(f"[build_glb] Loading from generated code: {gen_path}", file=sys.stderr)
                import importlib.util
                spec = importlib.util.spec_from_file_location("robot_base_vlm", gen_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'Rover'):
                    Rover = module.Rover
                    print("[build_glb] ✓ Using generated Rover class", file=sys.stderr)
                else:
                    print("[build_glb] Generated code has no Rover, falling back", file=sys.stderr)
                    from robot_base import Rover
            else:
                print("[build_glb] No generated code found, using original", file=sys.stderr)
                from robot_base import Rover
        else:
            from robot_base import Rover
        
        # Build the model
        print("[build_glb] Building Rover assembly...", file=sys.stderr)
        rv = Rover()
        
        # Build with timeout tolerance (some components may fail)
        try:
            rv.build(recursive=True)
        except Exception as e:
            print(f"[build_glb] Warning during build: {e}", file=sys.stderr)
            print("[build_glb] Continuing with partial build...", file=sys.stderr)
        
        # Export to GLB (using same approach as optim.py)
        print("[build_glb] Exporting to GLB...", file=sys.stderr)
        import trimesh
        import tempfile
        
        scene = trimesh.Scene()
        
        def _cq_to_trimesh(shp, tol=0.6):
            """Convert CadQuery shape to trimesh"""
            try:
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    if hasattr(shp, 'val'):
                        shp = shp.val()
                    shp.exportStl(tmp_path)
                    mesh = trimesh.load(tmp_path)
                    return mesh
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception as e:
                print(f"[mesh] STL export failed: {e}", file=sys.stderr)
                return None
        
        def _get_shape(component):
            for attr in ("world_obj", "toCompound", "obj", "to_cadquery", "shape", "local_obj", "make"):
                if hasattr(component, attr):
                    try:
                        v = getattr(component, attr)
                        shp = v() if callable(v) else v
                        if shp is not None:
                            return shp
                    except:
                        pass
            return None
        
        def _iter_components(root):
            comps = getattr(root, "components", None)
            if isinstance(comps, dict):
                return comps.items()
            if comps:
                try:
                    return list(comps)
                except:
                    pass
            return []
        
        def _walk(node, prefix="Rover"):
            shp = _get_shape(node)
            if shp is not None:
                tm = _cq_to_trimesh(shp)
                if tm and not getattr(tm, "is_empty", False):
                    try:
                        scene.add_geometry(tm, node_name=prefix)
                    except:
                        pass
            for child_name, child in _iter_components(node):
                _walk(child, f"{prefix}/{child_name}")
        
        _walk(rv, "Rover")
        
        # Export scene to GLB
        glb_bytes = scene.export(file_type='glb')
        
        # Close devnull and restore stdout for binary output
        devnull = sys.stdout
        sys.stdout = old_stdout
        devnull.close()
        
        # Write to stdout as binary (ONLY binary, no text!)
        sys.stdout.buffer.write(glb_bytes)
        sys.stdout.flush()
        
        # Log to stderr
        print(f"[build_glb] ✓ Generated {len(glb_bytes)} bytes", file=sys.stderr)
        return 0
            
    except Exception as e:
        import traceback
        print(f"[build_glb] ✗ Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == '__main__':
    use_gen = '--generated' in sys.argv
    sys.exit(build_glb(use_generated=use_gen))

