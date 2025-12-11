"""
Main routes blueprint
Handles main page routes, demo files, and static files.
"""
from flask import Blueprint, render_template, send_file, send_from_directory, Response, abort
import os

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    """Main viewer interface."""
    return render_template("viewer.html")


@bp.get("/debug")
def debug_viewer():
    """Simple viewer for inspecting GLB output."""
    return render_template("simple_viewer.html")


@bp.route("/demo/rover.png")
@bp.route("/demo/mars_rover.jpg")
def demo_mars_rover():
    """Serve the demo Mars rover image."""
    from run import ASSETS_DIR
    # Try rover.png first (user's file), then fall back to mars_rover.jpg
    demo_paths = [
        os.path.join(ASSETS_DIR, "demo", "rover.png"),
        os.path.join(ASSETS_DIR, "demo", "mars_rover.jpg"),
    ]
    
    for demo_path in demo_paths:
        if os.path.exists(demo_path):
            mimetype = "image/png" if demo_path.endswith(".png") else "image/jpeg"
            return send_file(demo_path, mimetype=mimetype)
    
    # Return a placeholder message if demo image doesn't exist
    return Response(
        "Demo image not found. Please save the Mars rover image to assets/demo/rover.png or assets/demo/mars_rover.jpg",
        status=404,
        mimetype="text/plain"
    )


@bp.route("/demo/curiosity_rover.stl")
def demo_curiosity_rover():
    """Serve the demo Curiosity Rover STL file."""
    from run import ASSETS_DIR
    # Try multiple possible locations
    possible_paths = [
        os.path.join(ASSETS_DIR, "demo", "curiosity_rover.stl"),
        os.path.join(ASSETS_DIR, "demo", "rover.stl"),
        os.path.join(ASSETS_DIR, "demo", "body-small.STL"),
        "/Users/janelleg/Downloads/Curiosity Rover 3D Printed Model/Simplified Curiosity Model (Small)/STL Files/body-small.STL",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return send_file(path, mimetype="model/stl")
    
    # If no file found, return a helpful message
    return Response(
        f"Demo STL file not found. Please place a demo STL file at: {os.path.join(ASSETS_DIR, 'demo', 'curiosity_rover.stl')}",
        status=404,
        mimetype="text/plain"
    )


@bp.route("/static/<path:filename>")
def custom_static(filename):
    """Serve static files."""
    from run import app
    root = app.static_folder
    full = os.path.join(root, filename)
    if not os.path.exists(full):
        abort(404)
    if filename.endswith(".js"):
        return send_from_directory(root, filename, mimetype="application/javascript")
    return send_from_directory(root, filename)

