"""
Mesh processing routes blueprint
"""
from flask import Blueprint, request, jsonify
import os
import sys

bp = Blueprint("mesh", __name__)

# Import from legacy optim.py for now
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


# Note: Mesh routes are handled by legacy route registration in app/__init__.py
# These routes will be migrated here gradually
# For now, they're registered directly from optim.py to preserve functionality

