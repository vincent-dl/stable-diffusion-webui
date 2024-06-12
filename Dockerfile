# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    curl

# Set working directory
WORKDIR /app
COPY . /app

# Verify CUDA is accessible
ENV NVIDIA_VISIBLE_DEVICES=all

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

RUN echo "Running launch.py with cuda skipped to download core models"

RUN python3 launch.py --listen --skip-torch-cuda-test --build-dockerimage