#!/bin/bash
set -euo pipefail

echo "🐳 Installing Docker in WSL"
echo "============================"

# Update package list
echo "📦 Updating package list..."
sudo apt-get update

# Install prerequisites
echo "📦 Installing prerequisites..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
echo "🔑 Adding Docker GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo "📦 Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
echo "🐳 Installing Docker Engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
echo "👤 Adding user to docker group..."
sudo usermod -aG docker $USER

# Start Docker service
echo "🚀 Starting Docker service..."
sudo service docker start

echo ""
echo "✅ Docker installed successfully!"
echo ""
echo "⚠️  IMPORTANT: You need to log out and log back in to WSL for group changes to take effect."
echo ""
echo "To verify installation, run:"
echo "  docker --version"
echo "  docker run hello-world"
echo ""
echo "After logging back in, you can build your image with:"
echo "  bash build_docker.sh"
