# Dockerfile for offline evaluation (no CUDA required).
#
# Reproduces paper figures from pre-collected data.
#
# Usage:
#   docker build -t hw-router -f docker/Dockerfile .
#   docker run hw-router

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY hw_router/ hw_router/
COPY baselines/ baselines/
COPY pipeline/ pipeline/
COPY scripts/ scripts/
COPY analysis/ analysis/
COPY data/ data/
COPY checkpoints/ checkpoints/
COPY configs/ configs/

# Install package (core + visualization, no vLLM)
RUN pip install --no-cache-dir -e ".[viz]"

# Default: reproduce paper figures
CMD ["python", "scripts/reproduce_figures.py"]
