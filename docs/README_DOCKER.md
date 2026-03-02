# Docker Deployment Guide for Mamba VIX Training

This guide explains how to build and deploy the Mamba VIX training environment using Docker for RunPod.

## Prerequisites

- Docker installed on your local machine or WSL
- Docker Hub account (free tier is fine)
- NVIDIA GPU with CUDA support (for testing locally)

## Quick Start

### 1. Build the Docker Image

In WSL or Linux terminal:

```bash
cd /mnt/d/Mamba\ v2
bash build_docker.sh
```

This will:
- Build a complete Docker image with PyTorch, Mamba, causal-conv1d, and all dependencies
- Take 15-30 minutes (one-time build)
- Ask if you want to push to Docker Hub

### 2. Push to Docker Hub

When prompted, enter `y` to push to Docker Hub. You'll need to:
1. Log in with your Docker Hub credentials
2. Wait for the upload (image is ~5-8GB)

### 3. Use in RunPod

In RunPod's pod creation:
1. Click **"Container Image"**
2. Enter: `yacinedaoud/mamba-vix-training:latest` (or your Docker Hub username)
3. Set **Container Disk** to at least 20GB
4. Select your GPU template (CUDA 12.4+ recommended)
5. Click **Deploy**

Your pod will start in **seconds** with everything pre-installed!

## Manual Build Steps

If you prefer manual control:

```bash
# Build
docker build -t yacinedaoud/mamba-vix-training:latest .

# Test locally (requires NVIDIA GPU)
docker run --gpus all -it --rm yacinedaoud/mamba-vix-training:latest

# Push to Docker Hub
docker login
docker push yacinedaoud/mamba-vix-training:latest
```

## What's Included

The Docker image contains:
- Ubuntu 22.04 with CUDA 12.4.1 + cuDNN
- Python 3.11
- PyTorch 2.x with CUDA 12.4 support
- Custom `mamba_blackwell` (sm_120 support)
- Custom `causal-conv1d-sm120` (sm_120 support)
- All dependencies: numpy, pandas, boto3, tqdm, einops, triton, transformers, tensorboard

## RunPod Usage

Once your pod starts:

```bash
# Clone your repo
cd /workspace
git clone https://github.com/yacinemassena/Lecrillon.git
cd Lecrillon

# Download data
python download_data.py --year 2024 --data-type both

# Start training
python train.py
```

## Updating the Image

When you update your code or dependencies:

```bash
# Rebuild and push
bash build_docker.sh

# Or manually
docker build -t yacinedaoud/mamba-vix-training:latest .
docker push yacinedaoud/mamba-vix-training:latest
```

RunPod will automatically pull the latest version on next pod creation.

## Troubleshooting

**Build fails during mamba/conv1d compilation:**
- Ensure you have enough RAM (16GB+ recommended)
- Try reducing `MAX_JOBS` in Dockerfile (line 9)

**Image is too large:**
- The image is ~5-8GB compressed, which is normal for CUDA + PyTorch + custom extensions
- Docker Hub free tier supports up to 1 repository with unlimited pulls

**RunPod can't find the image:**
- Make sure the image is public on Docker Hub
- Check the exact image name matches what you pushed

## Alternative: Private Registry

If you prefer not to use Docker Hub:

1. Use GitHub Container Registry (ghcr.io)
2. Use AWS ECR or Google Container Registry
3. Configure RunPod with registry authentication

See RunPod docs for private registry setup.
