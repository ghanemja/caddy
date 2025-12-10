"""
Component aliases for target and action normalization.
"""
from typing import Dict

# Target aliases for component name normalization
TARGET_ALIASES: Dict[str, str] = {
    "motor_controllerboard": "motor_controller_board",
    "motorcontrollerboard": "motor_controller_board",
    "motor controller board": "motor_controller_board",
    "motorcontroller": "motor_controller",
    "motor": "motor_controller",
    "sensorsbase": "sensor_fork",
    "sensor": "sensor_fork",
    "sensors": "sensor_fork",
    "wheels": "wheel",
    "roverbase": "rover",
    "chassis": "rover",
    "base": "rover",
}

# Action aliases for action name normalization
ACTION_ALIASES: Dict[str, str] = {
    "move": "translate",
    "position": "translate",
    "pos": "translate",
    "orientation": "rotate",
    "orient": "rotate",
    "size": "modify",
    "count": "add",
    "wheels_per_side": "modify",
    "scale": "resize",
    "shrink": "resize",
}

