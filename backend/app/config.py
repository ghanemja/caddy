"""
Application Configuration
"""
import os
from pathlib import Path


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Paths - backend is in backend/ directory
    BACKEND_DIR = Path(__file__).parent.parent  # backend/ directory
    ROOT_DIR = BACKEND_DIR.parent  # Root of project
    ASSETS_DIR = ROOT_DIR / "frontend" / "assets"
    STATIC_FOLDER = ROOT_DIR / "frontend" / "static"
    TEMPLATES_FOLDER = ROOT_DIR / "frontend" / "templates"
    
    # Server settings
    PORT = int(os.environ.get("PORT", "5160"))
    HOST = os.environ.get("HOST", "0.0.0.0")
    
    # VLM Configuration
    USE_FINETUNED_MODEL = os.environ.get("USE_FINETUNED_MODEL", "1") == "1"
    FINETUNED_MODEL_PATH = os.environ.get(
        "FINETUNED_MODEL_PATH",
        str(BACKEND_DIR / "runs" / "onevision_lora_small" / "checkpoint-4")
    )
    OLLAMA_URL = os.environ.get(
        "OLLAMA_URL",
        os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    ).rstrip("/")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava:latest")
    LLAVA_URL = os.environ.get("LLAVA_URL")
    
    # Ensure assets directory exists
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

