#!/bin/bash
set -euo pipefail

# Get the directory where the script is located, which is the project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Setting up Mamba VIX Environment on Linux VPS"
echo "=================================================="
echo "This script installs PyTorch + Mamba for CUDA 12.x"

# Helper to avoid duplicate .bashrc lines
append_if_missing() {
    local line="$1"
    local file="$2"
    grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export FORCE_CUDA=1
export MAX_JOBS=$(nproc)

# Detect CUDA installation
if [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.8" ]; then
    CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_HOME="/usr/local/cuda-12.6"
elif [ -d "/usr/local/cuda-12.4" ]; then
    CUDA_HOME="/usr/local/cuda-12.4"
else
    echo "⚠️  CUDA not found in /usr/local/cuda*"
    echo "   Will install CPU-only PyTorch (you can install CUDA later)"
    CUDA_HOME=""
fi

if [ -n "$CUDA_HOME" ]; then
    echo "✅ Found CUDA at: $CUDA_HOME"
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
    export CUDA_HOME=$CUDA_HOME
    
    # Add to bashrc for persistence
    BASHRC="$HOME/.bashrc"
    append_if_missing "export PATH=$CUDA_HOME/bin:\$PATH" "$BASHRC"
    append_if_missing "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" "$BASHRC"
    append_if_missing "export CUDA_HOME=$CUDA_HOME" "$BASHRC"
    append_if_missing 'export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"' "$BASHRC"
    append_if_missing 'export FORCE_CUDA=1' "$BASHRC"
fi

echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    software-properties-common \
    gnupg \
    ca-certificates \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    curl

echo "🐍 Adding deadsnakes PPA for Python 3.11..."
if ! grep -q "^deb .*deadsnakes/ppa" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
fi

echo "🔧 Installing Python 3.11 and build tools..."
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip

echo "🐍 Creating venv in project root..."
VENV_PATH="$PROJECT_DIR/venv"
echo "Project root: $PROJECT_DIR"
echo "Venv path: $VENV_PATH"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "Creating venv..."
    python3.11 -m venv "$VENV_PATH"
    echo "✅ venv created"
else
    echo "✅ venv already exists"
fi

echo "🔌 Activating venv..."
source "$VENV_PATH/bin/activate"
echo "✅ venv activated: $VENV_PATH"

echo "📦 Installing Python packages..."
pip install --upgrade pip setuptools wheel ninja packaging

# Install PyTorch
if [ -n "$CUDA_HOME" ]; then
    echo "🔥 Installing PyTorch with CUDA 12.x support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "💻 Installing PyTorch (CPU-only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "📦 Installing additional packages..."
pip install numpy pandas pyarrow boto3

# Install causal-conv1d (custom sm_120 version)
echo "🔧 Installing causal-conv1d..."
cd "$PROJECT_DIR/custom_packages/causal-conv1d-sm120"
if [ -n "$CUDA_HOME" ]; then
    pip install -e . --no-build-isolation
else
    echo "⚠️  Skipping causal-conv1d (requires CUDA)"
fi

# Install mamba_ssm (custom Blackwell version)
echo "🐍 Installing mamba_ssm..."
cd "$PROJECT_DIR/custom_packages/mamba_blackwell"
if [ -n "$CUDA_HOME" ]; then
    pip install -e . --no-build-isolation
else
    echo "⚠️  Skipping mamba_ssm (requires CUDA)"
fi

cd "$PROJECT_DIR"

# Test environment
echo ""
echo "🧪 Testing environment..."
cat > test_environment.py << 'EOF'
import sys
import torch

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Torch CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    
    try:
        import causal_conv1d
        print(f"causal_conv1d: {causal_conv1d.__file__}")
    except ImportError as e:
        print(f"⚠️  causal_conv1d not available: {e}")
    
    try:
        import mamba_ssm
        print(f"Mamba module 'mamba_ssm': {mamba_ssm.__file__}")
        
        # Quick forward test
        from mamba_ssm import Mamba
        model = Mamba(d_model=64, d_state=16).cuda()
        x = torch.randn(1, 10, 64).cuda()
        y = model(x)
        print("✅ Mamba forward OK")
    except ImportError as e:
        print(f"⚠️  mamba_ssm not available: {e}")
    except Exception as e:
        print(f"⚠️  Mamba test failed: {e}")
    
    print("✅ Environment OK")
else:
    print("⚠️  CUDA not available - CPU-only mode")
    print("✅ Environment OK (CPU)")
EOF

python test_environment.py
rm test_environment.py

echo ""
echo "🎉 VPS Setup Complete!"
echo "🚀 To activate your environment:"
echo "  source venv/bin/activate"
echo "🧪 To download data:"
echo "  pip install boto3"
echo "  python download_data.py --year 2024 --data-type both"
echo "🏃 To run smoke test:"
echo "  python smoke_mamba_only.py --train-steps 20 --val-steps 5 --epochs 1"
