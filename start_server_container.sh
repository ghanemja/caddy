#!/bin/bash
# Container-friendly version of start_server.sh
# This version doesn't require conda and runs directly with the container's Python

cd /app/backend

if [ -f "run.py" ]; then
    python run.py
else
    echo "Error: run.py not found"
    exit 1
fi

