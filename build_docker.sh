#!/bin/bash
set -euo pipefail

# Docker image build and push script for Mamba VIX environment
# This builds a complete Docker image with PyTorch, Mamba, and all dependencies

echo "🐳 Building Mamba VIX Docker Image"
echo "===================================="

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-yacinedaoud}"
IMAGE_NAME="mamba-vix-training"
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Get the directory where the script is located
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "📦 Building Docker image: $FULL_IMAGE_NAME"
echo "This will take 15-30 minutes depending on your internet and CPU..."
echo ""

# Build the image
docker build \
    --tag "$FULL_IMAGE_NAME" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "📊 Image size:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""

# Ask if user wants to push
read -p "🚀 Push to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔐 Logging in to Docker Hub..."
    docker login
    
    echo "⬆️  Pushing image to Docker Hub..."
    docker push "$FULL_IMAGE_NAME"
    
    echo ""
    echo "🎉 Image pushed successfully!"
    echo ""
    echo "📋 To use in RunPod, enter this in 'Container Image':"
    echo "   $FULL_IMAGE_NAME"
else
    echo ""
    echo "ℹ️  Skipped push. To push later, run:"
    echo "   docker push $FULL_IMAGE_NAME"
fi

echo ""
echo "🧪 To test locally:"
echo "   docker run --gpus all -it --rm $FULL_IMAGE_NAME"
