# Multi-stage build for CAD Optimizer
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Create necessary directories
RUN mkdir -p /app/frontend/assets /app/backend/checkpoints /app/backend/runs

# Add backend and cqparts_bucket to Python path
ENV PYTHONPATH=/app/backend:/app:/app/backend/cqparts_bucket

# Expose port
EXPOSE 5160

# Health check (requests is in requirements.txt)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5160/api/state', timeout=5)" || exit 1

# Set working directory to backend
WORKDIR /app/backend

# Run the application
CMD ["python", "run.py"]

