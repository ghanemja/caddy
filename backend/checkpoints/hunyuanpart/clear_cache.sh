#!/bin/bash
# Clear Python cache files that might cause import issues

echo "Clearing Python cache files..."

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

echo "✓ Cache cleared!"
echo ""
echo "If you're still seeing import errors, please restart your server."
