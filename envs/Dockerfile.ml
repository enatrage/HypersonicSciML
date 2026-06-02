FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system utilities for tracking and performance monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install SciML execution dependencies
RUN pip install --no-cache-dir \
    h5py \
    pyyaml \
    pydantic==2.6.4 \
    wandb \
    scipy

# Set environment variables for clean terminal outputs
ENV PYTHONUNBUFFERED=1