"""
Legacy Component Registration
Component registration for the legacy rover system.
Only called if legacy components are available.
"""
from app.core.component_registry import (
    ComponentSpec,
    register_component,
)
from app.core.state import PENDING_ADDS


def register_legacy_components(
    Rover, _ThisWheel, _PanTilt, SensorFork, 
    proxy_wheels_fn, apply_params_to_rover_fn
):
    """
    Register legacy rover components.
    This is only called if the legacy CAD components are available.
    
    Args:
        Rover: Rover class
        _ThisWheel: Wheel class
        _PanTilt: PanTilt class
        SensorFork: SensorFork class
        proxy_wheels_fn: Function to proxy wheels
        apply_params_to_rover_fn: Function to apply params to rover
    """
    if Rover is None or _ThisWheel is None:
        print("[legacy] Skipping component registration - rover components not available")
        return
    
    register_component(
        "wheel",
        ComponentSpec(
            cls=_ThisWheel,
            add_fn=None,
            param_map={"wheel_diameter": "diameter", "wheel_width": "width"},
            proxy_fn=proxy_wheels_fn,
        ),
    )

    register_component(
        "pan_tilt",
        ComponentSpec(
            cls=_PanTilt,
            add_fn=None,
            param_map={
                "pan_tilt_offset_x": "pan_tilt_offset_x",
                "pan_tilt_offset_y": "pan_tilt_offset_y",
                "pan_tilt_offset_z": "pan_tilt_offset_z",
            },
        ),
    )

    register_component(
        "sensor_fork",
        ComponentSpec(
            cls=SensorFork,
            add_fn=lambda adapter, **p: adapter.add("sensor_fork", PENDING_ADDS, **p),
            param_map={
                "width_mm": "width",
                "depth_mm": "depth",
                "height_mm": "height",
                "wall_mm": "wall",
                "hole_diam_mm": "hole_diam",
            },
        ),
    )

    register_component(
        "rover",
        ComponentSpec(
            cls=Rover,
            add_fn=None,
            param_map={"rover_yaw_deg": "rover_yaw_deg"},
        ),
    )
    
    print("[legacy] ✓ Registered legacy rover components")

