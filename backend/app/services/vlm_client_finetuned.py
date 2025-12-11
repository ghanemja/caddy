"""
VLM client implementation using the existing fine-tuned VLM from optim.py.

This client wraps the call_vlm() function from the main application.
Located in backend/app/services/ because it depends on backend-specific optim.py.
"""

from typing import List, Dict, Any, Optional
import json
import base64
from pathlib import Path

# Import VLM client base classes from meshml
import sys
import os
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from meshml.semantics.vlm_client import VLMClient, VLMImage, VLMMessage


class FinetunedVLMClient:
    """
    VLM client that uses the fine-tuned model from optim.py.
    
    This client imports and uses the existing call_vlm() function
    from the main application, ensuring consistency with the rest
    of the codebase.
    """
    
    def __init__(self):
        """Initialize the client and import the VLM function."""
        # Import the VLM function from optim.py
        # We need to handle the case where optim.py might not be importable
        # (e.g., if running as a standalone module)
        try:
            import sys
            import os
            # This file is at backend/app/services/, optim.py is at backend/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go from app/services/ to backend/
            backend_dir = os.path.join(current_dir, "..", "..")
            
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            
            from optim import call_vlm, load_finetuned_model
            self._call_vlm = call_vlm
            self._load_finetuned_model = load_finetuned_model
            self._available = True
        except ImportError as e:
            print(f"[FinetunedVLMClient] Warning: Could not import optim.call_vlm: {e}")
            print("[FinetunedVLMClient] Falling back to dummy client. Make sure optim.py is importable.")
            self._available = False
            self._call_vlm = None
            self._load_finetuned_model = None
    
    def _image_path_to_base64(self, image_path: str) -> str:
        """
        Convert an image file path to base64-encoded string.
        
        Args:
            image_path: path to image file
            
        Returns:
            Base64-encoded image string (without data URL prefix)
        """
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
        """
        Convert VLM messages to a single prompt string.
        
        The call_vlm function will prepend VLM_SYSTEM_PROMPT, so we combine
        system and user messages into the final prompt. The VLM should follow
        the instructions in our system message even with the CAD-specific
        system prompt prepended.
        
        Args:
            messages: list of VLM messages
            
        Returns:
            Combined prompt string
        """
        # Extract system and user messages
        system_parts = []
        user_parts = []
        
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "user":
                user_parts.append(msg.content)
            elif msg.role == "assistant":
                # Include assistant messages for context
                user_parts.append(f"Previous response: {msg.content}")
        
        # Combine: system instructions first, then user request
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
        """
        Call the fine-tuned VLM and return a JSON-like dict.
        
        Args:
            messages: conversation history
            images: optional list of images to include
            schema_hint: optional description of desired JSON schema
            
        Returns:
            Dictionary parsed from JSON response
        """
        if not self._available:
            raise RuntimeError(
                "FinetunedVLMClient is not available. "
                "Could not import call_vlm from optim.py. "
                "Make sure you're running from the correct directory."
            )
        
        # Ensure model is loaded (call_vlm will handle lazy loading if needed)
        # Don't call load_finetuned_model here - it's already loaded or will be loaded by call_vlm
        
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
                    print(f"[FinetunedVLMClient] Warning: Failed to load image {img.path}: {e}")
                    continue
            
            if image_b64_list:
                image_data_urls = image_b64_list
        
        # Call VLM
        try:
            result = self._call_vlm(
                final_prompt=prompt,
                image_data_urls=image_data_urls,
                expect_json=True,
            )
            
            # Extract response
            raw_response = result.get("raw", "")
            
            # Try to parse JSON from response
            # The response might contain markdown code fences or other text
            json_str = self._extract_json_from_response(raw_response)
            
            # Parse JSON
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                print(f"[FinetunedVLMClient] Failed to parse JSON: {e}")
                print(f"[FinetunedVLMClient] Raw response: {raw_response[:500]}...")
                # Return a fallback structure
                return {
                    "error": "Failed to parse JSON response",
                    "raw": raw_response,
                }
        
        except Exception as e:
            print(f"[FinetunedVLMClient] Error calling VLM: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from VLM response, handling code fences and extra text.
        
        Args:
            response: raw VLM response string
            
        Returns:
            JSON string
        """
        # Try to find JSON in code fences
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
        
        # If no JSON found, return the whole response (might be plain JSON)
        return response.strip()

