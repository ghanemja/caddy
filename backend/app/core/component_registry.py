"""
Component Registry
Manages CAD component specifications and registration
"""
from typing import Dict, Optional
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

