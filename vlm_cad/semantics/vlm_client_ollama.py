"""
VLM client implementation using Ollama.

This client uses the existing Ollama integration from optim.py.
"""

from typing import List, Dict, Any, Optional
import json
import base64
from pathlib import Path

from .vlm_client import VLMClient, VLMImage, VLMMessage


class OllamaVLMClient:
    """
    VLM client that uses Ollama via the call_vlm() function from optim.py.
    
    This client uses Ollama instead of the fine-tuned HuggingFace model.
    """
    
    def __init__(self):
        """Initialize the client and import the VLM function."""
        try:
            import sys
            import os
            # Add cqparts_bucket to path if needed
            current_dir = os.path.dirname(os.path.abspath(__file__))
            optim_dir = os.path.join(current_dir, "..", "..")
            cqparts_bucket_dir = os.path.join(optim_dir, "cqparts_bucket")
            
            if cqparts_bucket_dir not in sys.path:
                sys.path.insert(0, cqparts_bucket_dir)
            
            from optim import call_vlm
            self._call_vlm = call_vlm
            self._available = True
        except ImportError as e:
            print(f"[OllamaVLMClient] Warning: Could not import optim.call_vlm: {e}")
            self._available = False
            self._call_vlm = None
    
    def _image_path_to_base64(self, image_path: str) -> str:
        """Convert an image file path to base64-encoded string."""
        from PIL import Image
        import io
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and convert to RGB
        img = Image.open(path).convert('RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_b64
    
    def _messages_to_prompt(self, messages: List[VLMMessage]) -> str:
        """Convert VLM messages to a single prompt string."""
        system_parts = []
        user_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "user":
                user_parts.append(msg.content)
            elif msg.role == "assistant":
                user_parts.append(f"Previous response: {msg.content}")
        
        prompt_parts = []
        if system_parts:
            prompt_parts.append("\n\n".join(system_parts))
        if user_parts:
            prompt_parts.append("\n\n".join(user_parts))
        
        return "\n\n".join(prompt_parts)
    
    def complete_json(
        self,
        messages: List[VLMMessage],
        images: Optional[List[VLMImage]] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call Ollama VLM and return a JSON-like dict."""
        if not self._available:
            raise RuntimeError(
                "OllamaVLMClient is not available. "
                "Could not import call_vlm from optim.py."
            )
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Add schema hint if provided
        if schema_hint:
            prompt += f"\n\nExpected JSON schema:\n{schema_hint}\n\nReturn only valid JSON."
        
        # Convert images to base64
        image_data_urls = None
        if images:
            image_b64_list = []
            for img in images:
                try:
                    b64 = self._image_path_to_base64(img.path)
                    image_b64_list.append(b64)
                except Exception as e:
                    print(f"[OllamaVLMClient] Warning: Failed to load image {img.path}: {e}")
                    continue
            
            if image_b64_list:
                image_data_urls = image_b64_list
        
        # Call VLM (Ollama will be used if fine-tuned model is not available)
        try:
            result = self._call_vlm(
                final_prompt=prompt,
                image_data_urls=image_data_urls,
                expect_json=True,
            )
            
            # Extract response
            raw_response = result.get("raw", "")
            
            # Try to parse JSON from response
            json_str = self._extract_json_from_response(raw_response)
            
            # Parse JSON
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                print(f"[OllamaVLMClient] Failed to parse JSON: {e}")
                print(f"[OllamaVLMClient] Raw response: {raw_response[:500]}...")
                return {
                    "error": "Failed to parse JSON response",
                    "raw": raw_response,
                }
        
        except Exception as e:
            print(f"[OllamaVLMClient] Error calling VLM: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from VLM response, handling code fences and extra text."""
        import re
        
        # Look for ```json ... ``` or ``` ... ```
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Look for JSON object directly
        json_obj_pattern = r'\{.*\}'
        match = re.search(json_obj_pattern, response, re.DOTALL)
        if match:
            return match.group(0)
        
        return response.strip()

