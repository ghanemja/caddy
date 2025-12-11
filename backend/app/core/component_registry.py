"""
Component Registry
Manages CAD component specifications and registration
"""
from typing import Dict, Optional, List, Any
from dataclasses import dataclass


@dataclass
class ComponentSpec:
    """Specification for a CAD component."""
    cls: type
    add_fn: Optional[callable] = None
    param_map: Optional[Dict[str, str]] = None
    proxy_fn: Optional[callable] = None
    
    def __post_init__(self):
        if self.param_map is None:
            self.param_map = {}


# Global component registry
COMPONENT_REGISTRY: Dict[str, ComponentSpec] = {}


def register_component(key: str, spec: ComponentSpec):
    """Register a component specification."""
    COMPONENT_REGISTRY[key.lower()] = spec


def get_component_spec(key: str) -> Optional[ComponentSpec]:
    """Get a component specification by key."""
    return COMPONENT_REGISTRY.get(key.lower())


class ModelAdapter:
    """
    Adapter for adding components to the model.
    Queues add operations to PENDING_ADDS for later processing.
    """
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def add(self, kind: str, pending_adds: List[dict], **params):
        """
        Queue a component addition.
        
        Args:
            kind: Component type to add
            pending_adds: List to append the add operation to
            **params: Parameters for the component
        """
        pending_adds.append({"kind": kind.lower(), "params": params})


def emit_missing_proxies(scene):
    """Emit proxy geometry for components that have proxy functions."""
    for key, spec in COMPONENT_REGISTRY.items():
        if hasattr(spec, "proxy_fn") and callable(spec.proxy_fn):
            print(f"[proxy] emitting for {key} …")
            spec.proxy_fn(scene)


def proxy_wheels(scene, current_params: dict, cq_to_trimesh_fn, truthy_fn):
    """
    Detailed wheel proxy with rim/spokes/tread, named nodes.
    
    Args:
        scene: trimesh.Scene to add wheels to
        current_params: Dictionary of current parameter values
        cq_to_trimesh_fn: Function to convert CadQuery to trimesh
        truthy_fn: Function to convert values to boolean
    """
    import cadquery as cq
    import math
    
    try:
        n_side = int((current_params.get("wheels_per_side") or 0))
        if n_side <= 0 or truthy_fn(current_params.get("hide_wheels")):
            return
        dia = float(current_params.get("wheel_diameter") or 130.0)
        wid = float(current_params.get("wheel_width") or 40.0)
        span = float(current_params.get("wheelbase_span_mm") or 320.0)
        axle = float(current_params.get("axle_spacing_mm") or 180.0)
        zoff = float(current_params.get("wheel_z_offset_mm") or 0.0)
    except Exception:
        return

    R = dia * 0.5
    rim_thick = max(2.0, min(0.06 * dia, 6.0))
    hub_rad = max(6.0, 0.12 * R)
    spoke_w = max(2.0, 0.06 * R)
    n_spokes = 8
    n_treads = max(12, int(0.6 * R))

    rim = cq.Workplane("XY").circle(R).circle(R - rim_thick).extrude(wid)
    hub = cq.Workplane("XY").circle(hub_rad).extrude(wid)
    spoke_len = (R - rim_thick) - hub_rad
    spoke = (
        cq.Workplane("XY")
        .rect(spoke_len, spoke_w, centered=(True, True))
        .extrude(wid)
        .translate(((hub_rad + (spoke_len / 2.0)), 0, 0))
    )
    spokes = cq.Workplane("XY")
    for i in range(n_spokes):
        spokes = spokes.union(
            spoke.rotate((0, 0, 0), (0, 0, 1), i * (360.0 / n_spokes))
        )
    wheel_solid = rim.union(hub).union(spokes)

    # grooves (best-effort)
    try:
        groove_w = max(1.0, wid * 0.08)
        groove_d = max(0.8, rim_thick * 0.5)
        groove_rad = R - groove_d * 0.5
        grooves = cq.Workplane("XY")
        ang_step = 360.0 / n_treads
        for i in range(n_treads):
            g = (
                cq.Workplane("XY")
                .circle(groove_w * 0.5)
                .extrude(rim_thick * 1.2)
                .translate(
                    (
                        groove_rad * math.cos(math.radians(i * ang_step)),
                        groove_rad * math.sin(math.radians(i * ang_step)),
                        wid * 0.5,
                    )
                )
                .rotate((0, 0, 0), (1, 0, 0), 90)
            )
            grooves = grooves.union(g)
        wheel_solid = wheel_solid.cut(grooves)
    except Exception:
        pass

    wheel_solid = wheel_solid.rotate((0, 0, 0), (0, 1, 0), 90)

    left_x = -span * 0.5
    right_x = +span * 0.5
    z0 = R + zoff

    if n_side == 1:
        ys = [0.0]
    else:
        step = axle / (n_side - 1) if n_side > 1 else 0.0
        ys = [-axle * 0.5 + i * step for i in range(n_side)]

    def add_one(x, y, z, label):
        tm = cq_to_trimesh_fn(wheel_solid.translate((x, y, z)), tol=0.45)
        if tm and not getattr(tm, "is_empty", False):
            scene.add_geometry(
                tm, node_name=f"wheel/{label}", geom_name=f"wheel_geom/{label}"
            )

    for i, y in enumerate(ys, start=1):
        add_one(left_x, y, z0, f"L{i}")
        add_one(right_x, y, z0, f"R{i}")


def apply_params_to_rover(rv, params: dict, current_params: dict, this_wheel_class, rover_class, pan_tilt_class, clean_num_fn):
    """
    Apply parameters to a rover instance.
    
    Args:
        rv: Rover instance
        params: Parameters to apply (dict)
        current_params: Global CURRENT_PARAMS dict to update
        this_wheel_class: Wheel class to modify
        rover_class: Rover class to modify
        pan_tilt_class: PanTilt class to modify
        clean_num_fn: Function to clean numeric values
    """
    if params:
        for k, v in params.items():
            if k in current_params:
                current_params[k] = clean_num_fn(v)
    if current_params["wheel_diameter"] is not None:
        setattr(this_wheel_class, "diameter", float(current_params["wheel_diameter"]))
    if current_params["wheel_width"] is not None:
        setattr(this_wheel_class, "width", float(current_params["wheel_width"]))

    if current_params["wheels_per_side"] is not None:
        wps = int(current_params["wheels_per_side"])
        try:
            setattr(rv, "wheels_per_side", wps)
        except Exception:
            pass
        setattr(rover_class, "wheels_per_side", wps)
        try:
            setattr(this_wheel_class, "count", max(2, 2 * wps))
        except Exception:
            pass

    for k in ("axle_spacing_mm", "wheelbase_span_mm"):
        if current_params[k] is not None:
            val = float(current_params[k])
            try:
                setattr(rv, k, val)
            except Exception:
                pass
            setattr(rover_class, k, val)

    for axis in ("x", "y", "z"):
        key = f"pan_tilt_offset_{axis}"
        if current_params[key] is not None:
            val = float(current_params[key])
            try:
                setattr(rv, key, val)
            except Exception:
                pass
            try:
                setattr(pan_tilt_class, key, val)
            except Exception:
                pass

