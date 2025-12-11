"""
CadQuery utility functions
Helper functions for working with CadQuery objects and scenes.
"""
import trimesh
import math
import cadquery as cq


def apply_rotation_to_wheels(scene, current_params: dict):
    """
    Apply rotation to wheels in a scene based on current parameters.
    
    Args:
        scene: trimesh.Scene object
        current_params: Dictionary containing 'wheel_rotation' key with [rx, ry, rz]
    """
    rot = current_params.get("wheel_rotation")
    if not rot or not isinstance(rot, (list, tuple)):
        return
    try:
        rx, ry, rz = [float(x) for x in rot]
    except Exception:
        rx = ry = rz = 0.0
    if abs(rx) < 1e-6 and abs(ry) < 1e-6 and abs(rz) < 1e-6:
        return
    R = trimesh.transformations.euler_matrix(
        math.radians(rx), math.radians(ry), math.radians(rz)
    )
    for name, geom in list(scene.geometry.items()):
        if name.lower().startswith("wheel/"):
            g = geom.copy()
            g.apply_transform(R)
            scene.geometry[name] = g


def apply_runtime_params_to_instance(rv, current_params: dict, this_wheel_class):
    """
    Apply runtime parameters from current_params to a rover instance.
    
    Args:
        rv: Rover instance
        current_params: Dictionary of current parameter values
        this_wheel_class: The wheel class to modify
    """
    if current_params.get("wheel_diameter") is not None:
        try:
            setattr(this_wheel_class, "diameter", float(current_params["wheel_diameter"]))
        except Exception:
            pass
    if current_params.get("wheel_width") is not None:
        try:
            setattr(this_wheel_class, "width", float(current_params["wheel_width"]))
        except Exception:
            pass
    if current_params.get("wheels_per_side") is not None:
        try:
            setattr(rv, "wheels_per_side", int(current_params["wheels_per_side"]))
        except Exception:
            pass
    if current_params.get("axle_spacing_mm") is not None:
        try:
            setattr(rv, "axle_spacing_mm", float(current_params["axle_spacing_mm"]))
        except Exception:
            pass
    if current_params.get("wheelbase_span_mm") is not None:
        try:
            setattr(rv, "wheelbase_span_mm", float(current_params["wheelbase_span_mm"]))
        except Exception:
            pass


def patch_cqparts_brittleness():
    """
    Patch cqparts classes to fix brittleness issues.
    Keeps imports consistent with this project layout.
    """
    from electronics import OtherBatt, OtherController, MotorController, type1
    from robot_base import Rover
    from run import CoordSystem

    roots = (Rover, type1, OtherController, MotorController, OtherBatt)

    for cls in roots:
        try:
            if not hasattr(cls, "world_coords"):
                cls.world_coords = CoordSystem()
        except Exception:
            pass

    def _no_constraints(self):
        return []

    for cls in (OtherController, MotorController, OtherBatt, type1, Rover):
        try:
            cls.make_constraints = _no_constraints
        except Exception:
            pass

    def _ob_box(self, x=60, y=30, z=15):
        self.local_obj = cq.Workplane("XY").box(x, y, z)
        return {}

    def _oc_box(self, x=65, y=55, z=12):
        self.local_obj = cq.Workplane("XY").box(x, y, z)
        return {}

    try:
        OtherBatt.make_components = _ob_box
    except Exception:
        pass
    try:
        OtherController.make_components = _oc_box
    except Exception:
        pass

