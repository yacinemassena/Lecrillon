# Mamba VIX Training Environment - CUDA 12.8 (sm_120 / RTX 5090 support)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

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

# Install PyTorch nightly with CUDA 12.8 support (required for sm_120 / RTX 5090)
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install core dependencies
RUN pip install \
    numpy \
    pandas \
    polars \
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

# Build and install causal-conv1d (non-editable so it persists after cleanup)
WORKDIR /tmp/causal-conv1d-sm120
RUN pip install . --no-build-isolation --no-deps

# Build and install mamba_ssm (non-editable so it persists after cleanup)
WORKDIR /tmp/mamba_blackwell
RUN pip install . --no-build-isolation --no-deps

# Clean up build artifacts (packages are installed to site-packages, safe to delete source)
RUN rm -rf /tmp/causal-conv1d-sm120 /tmp/mamba_blackwell

# Set working directory back to /workspace
WORKDIR /workspace

# Copy project files (optional - you can mount this at runtime instead)
# COPY . /workspace

# Set CUDA paths
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Default command - keep container running for RunPod
CMD ["sleep", "infinity"]
