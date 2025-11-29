"""
VLM client abstraction for vision-language model interactions.

This module defines a protocol for VLM clients and provides a dummy
implementation for testing without API calls.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Protocol, Optional
import json


@dataclass
class VLMImage:
    """Represents an image for VLM input."""
    path: str  # filesystem path or URL to an image


@dataclass
class VLMMessage:
    """Represents a message in a VLM conversation."""
    role: str  # "system" | "user" | "assistant"
    content: str


class VLMClient(Protocol):
    """
    Protocol for VLM clients.
    
    Implementations should provide a complete_json method that calls
    a vision-language model and returns structured JSON responses.
    """
    
    def complete_json(
        self,
        messages: List[VLMMessage],
        images: Optional[List[VLMImage]] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call a VLM and return a JSON-like dict.
        
        Args:
            messages: conversation history
            images: optional list of images to include
            schema_hint: optional description of desired JSON schema
            
        Returns:
            Dictionary parsed from JSON response
        """
        ...


class DummyVLMClient:
    """
    Dummy VLM client for testing without API calls.
    
    Returns hard-coded or semi-random JSON structures matching
    expected schemas for pre-VLM and post-VLM steps.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize dummy client.
        
        Args:
            seed: random seed for reproducible dummy responses
        """
        import random
        if seed is not None:
            random.seed(seed)
        self._seed = seed
    
    def complete_json(
        self,
        messages: List[VLMMessage],
        images: Optional[List[VLMImage]] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return dummy JSON response based on schema_hint or message content.
        
        Args:
            messages: conversation history
            images: optional list of images (ignored in dummy)
            schema_hint: optional schema description
            
        Returns:
            Dummy JSON response
        """
        # Check if this is a pre-VLM or post-VLM call based on messages
        message_text = " ".join([m.content for m in messages]).lower()
        
        # Pre-VLM: category + candidate parameters
        if "classify" in message_text or "category" in message_text or "candidate" in message_text:
            return self._dummy_pre_vlm_response()
        
        # Post-VLM: final parameters
        elif "refine" in message_text or "final" in message_text or "reconcile" in message_text:
            return self._dummy_post_vlm_response()
        
        # Default: generic response
        else:
            return {"status": "ok", "message": "Dummy response"}
    
    def _dummy_pre_vlm_response(self) -> Dict[str, Any]:
        """Return dummy pre-VLM response."""
        # Try to infer category from context (simplified)
        categories = ["Airplane", "Chair", "Car", "Table", "Lamp"]
        category = categories[0]  # Default to Airplane
        
        # Generate candidate parameters based on category
        if category == "Airplane":
            candidate_params = [
                {"name": "wing_span", "description": "Distance between wing tips"},
                {"name": "chord_length", "description": "Average width of the wing"},
                {"name": "fuselage_length", "description": "Length of the main body"},
                {"name": "tail_height", "description": "Height of the tail fin"},
            ]
        elif category == "Chair":
            candidate_params = [
                {"name": "seat_height", "description": "Height of the seat from ground"},
                {"name": "back_height", "description": "Height of the backrest"},
                {"name": "seat_width", "description": "Width of the seat"},
                {"name": "leg_length", "description": "Length of chair legs"},
            ]
        elif category == "Car":
            candidate_params = [
                {"name": "wheelbase", "description": "Distance between front and rear axles"},
                {"name": "length", "description": "Overall length of the vehicle"},
                {"name": "width", "description": "Overall width of the vehicle"},
                {"name": "height", "description": "Overall height of the vehicle"},
            ]
        else:
            candidate_params = [
                {"name": "length", "description": "Main length dimension"},
                {"name": "width", "description": "Main width dimension"},
                {"name": "height", "description": "Main height dimension"},
            ]
        
        return {
            "category": category,
            "candidate_parameters": candidate_params,
        }
    
    def _dummy_post_vlm_response(self) -> Dict[str, Any]:
        """Return dummy post-VLM response."""
        # Generate a few final parameters with dummy values
        final_params = [
            {
                "name": "wing_span",
                "value": 2.5,
                "units": "m",
                "description": "Distance between wing tips",
                "confidence": 0.85,
                "raw_sources": ["bbox_x_length", "wing_segment_0_span"],
            },
            {
                "name": "chord_length",
                "value": 0.3,
                "units": "m",
                "description": "Average width of the wing",
                "confidence": 0.75,
                "raw_sources": ["bbox_y_length", "wing_segment_0_extent_y"],
            },
        ]
        
        return {
            "final_parameters": final_params,
        }

