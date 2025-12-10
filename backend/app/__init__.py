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
    
    Uses optim.py's app instance and registers new blueprints on it.
    This preserves all functionality while allowing gradual migration.
    """
    # Import optim first - it creates the Flask app and all global state
    # We'll use optim's app and register our blueprints on it
    try:
        import optim
        app = optim.app  # Use optim's app instance
    except ImportError:
        # Fallback: create our own app if optim.py doesn't exist
        print("[app] Warning: optim.py not found, creating new app")
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
    
    # Register API blueprints with prefixes
    app.register_blueprint(api.bp, url_prefix="/api")
    app.register_blueprint(vlm.bp, url_prefix="/api/vlm")
    app.register_blueprint(mesh.bp, url_prefix="/api/mesh")
    app.register_blueprint(model.bp, url_prefix="/api/model")
    
    # Make optim module available for services (if it was imported)
    try:
        import optim
        app._legacy_optim = optim
    except ImportError:
        pass
    
    return app


# Legacy routes are automatically available since we use optim.app
# No need for separate registration

