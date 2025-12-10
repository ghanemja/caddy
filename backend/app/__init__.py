"""
CAD Optimizer Flask Application Factory
"""
from flask import Flask
import os
import sys
from pathlib import Path

# Add backend directory to path for imports
BACKEND_DIR = Path(__file__).parent.parent  # backend/ directory
ROOT_DIR = BACKEND_DIR.parent  # Root of project
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))


def create_app(config_name=None):
    """
    Create and configure the Flask application.
    
    Uses run.py's app instance and registers blueprints on it.
    This preserves all functionality while allowing gradual migration.
    """
    # Import run.py first - it creates the Flask app and all global state
    # We'll use run's app and register our blueprints on it
    try:
        import run
        app = run.app  # Use run's app instance
    except ImportError:
        # Fallback: create our own app if run.py doesn't exist
        print("[app] Warning: run.py not found, creating new app")
        app = Flask(
            __name__,
            template_folder=str(ROOT_DIR / "frontend" / "templates"),
            static_folder=str(ROOT_DIR / "frontend" / "static"),
            static_url_path="/static"
        )
    
    # Load configuration
    from app.config import Config
    app.config.from_object(Config)
    
    # Initialize FreeCAD (already done by optim.py import, but ensure it's available)
    from app.services.freecad_service import FreeCAD
    
    # Register new blueprints on the app
    # These provide organized routes while preserving all legacy routes
    from app.routes import main, api, vlm, mesh, model
    
    # Register main routes (may override legacy / and /debug)
    app.register_blueprint(main.bp)
    
    # Register API blueprints
    # Some routes are at root level, some have prefixes
    app.register_blueprint(api.bp)  # Routes like /state, /apply, /params
    app.register_blueprint(vlm.bp)  # Routes like /codegen, /vlm, /recommend
    app.register_blueprint(mesh.bp, url_prefix="/api/mesh")
    app.register_blueprint(model.bp, url_prefix="/api/model")
    
    # Make run module available for services (if it was imported)
    try:
        import run
        app._legacy_run = run
    except ImportError:
        pass
    
    return app


# Legacy routes are automatically available since we use run.app
# Blueprints provide organized routes while preserving legacy functionality

