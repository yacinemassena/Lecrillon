# Mamba VIX Training Environment - CUDA 12.4
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set CUDA architecture list for compilation
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=16

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    ca-certificates \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel ninja packaging

# Install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
RUN pip install \
    numpy \
    pandas \
    pyarrow \
    boto3 \
    tqdm \
    pyyaml \
    einops \
    triton \
    transformers \
    tensorboard

# Set working directory
WORKDIR /workspace

# Copy custom packages
COPY custom_packages/causal-conv1d-sm120 /tmp/causal-conv1d-sm120
COPY custom_packages/mamba_blackwell /tmp/mamba_blackwell

# Build and install causal-conv1d
WORKDIR /tmp/causal-conv1d-sm120
RUN pip install -e . --no-build-isolation --no-deps

# Build and install mamba_ssm
WORKDIR /tmp/mamba_blackwell
RUN pip install -e . --no-build-isolation --no-deps

# Clean up build artifacts
RUN rm -rf /tmp/causal-conv1d-sm120 /tmp/mamba_blackwell

# Set working directory back to /workspace
WORKDIR /workspace

# Copy project files (optional - you can mount this at runtime instead)
# COPY . /workspace

# Set CUDA paths
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Default command
CMD ["/bin/bash"]
