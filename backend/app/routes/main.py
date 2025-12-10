"""
Main routes blueprint
Handles main page routes.
"""
from flask import Blueprint, render_template

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    """Main viewer interface."""
    return render_template("viewer.html")


@bp.get("/debug")
def debug_viewer():
    """Simple viewer for inspecting GLB output."""
    return render_template("simple_viewer.html")

