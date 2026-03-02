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

# Detect if running as root (Docker) or regular user
if [ "$EUID" -eq 0 ]; then
    SUDO=""
    echo "📦 Running as root (Docker container detected)"
else
    SUDO="sudo"
fi

# Use /workspace for apt cache if available to avoid filling overlay
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    export APT_CACHE_DIR="/workspace/.apt-cache"
    mkdir -p "$APT_CACHE_DIR/archives/partial"
    echo "📦 Using apt cache: $APT_CACHE_DIR"
    APT_OPTS="-o Dir::Cache::Archives=$APT_CACHE_DIR"
else
    APT_OPTS=""
fi

echo "📦 Installing system dependencies..."
$SUDO apt-get update $APT_OPTS
$SUDO apt-get install -y $APT_OPTS \
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
    $SUDO add-apt-repository ppa:deadsnakes/ppa -y
    $SUDO apt-get update $APT_OPTS
fi

echo "🔧 Installing Python 3.11 and build tools..."
$SUDO apt-get install -y $APT_OPTS \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip

echo "🐍 Creating venv..."
# Use /workspace if available (Docker/VPS with large disk), otherwise project root
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    VENV_PATH="/workspace/venv"
    echo "Using /workspace for venv (large disk detected)"
else
    VENV_PATH="$PROJECT_DIR/venv"
    echo "Using project root for venv"
fi
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

# Set pip cache to /workspace to avoid filling overlay
if [ -d "/workspace" ]; then
    export PIP_CACHE_DIR="/workspace/.pip-cache"
    mkdir -p "$PIP_CACHE_DIR"
    echo "📦 Using pip cache: $PIP_CACHE_DIR"
fi

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
pip install numpy pandas pyarrow boto3 einops triton transformers

# Detect GPU compute capability
GPU_COMPUTE_CAP=""
if [ -n "$CUDA_HOME" ]; then
    echo "🔍 Detecting GPU compute capability..."
    GPU_COMPUTE_CAP=$(python3.11 -c "import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')" 2>/dev/null || echo "")
    echo "GPU compute capability: $GPU_COMPUTE_CAP"
fi

# Install causal-conv1d and mamba_ssm based on GPU
if [ -n "$CUDA_HOME" ] && [ -n "$GPU_COMPUTE_CAP" ]; then
    # Check if RTX 5080/Blackwell (sm_120 = compute capability 12.0)
    if [[ "$GPU_COMPUTE_CAP" == "(12, 0)" ]]; then
        echo "🚀 RTX 5080/Blackwell detected - installing custom sm_120 versions..."
        
        echo "🔧 Installing causal-conv1d (sm_120)..."
        cd "$PROJECT_DIR/custom_packages/causal-conv1d-sm120"
        pip install -e . --no-build-isolation
        
        echo "🐍 Installing mamba_ssm (Blackwell)..."
        cd "$PROJECT_DIR/custom_packages/mamba_blackwell"
        pip install -e . --no-build-isolation
    else
        echo "📦 Standard GPU detected - installing PyPI versions..."
        pip install causal-conv1d>=1.4.0
        pip install mamba-ssm
    fi
else
    echo "⚠️  No CUDA - skipping causal-conv1d and mamba_ssm"
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
echo ""
echo "🎉 VPS Setup Complete!"
echo ""
if [ -n "$GPU_COMPUTE_CAP" ]; then
    echo "GPU: $(python3.11 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    echo "Compute Capability: $GPU_COMPUTE_CAP"
fi
echo ""
echo "🚀 To activate your environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "🧪 To download data:"
if [ -d "/workspace" ]; then
    echo "  cd /workspace"
    echo "  python $PROJECT_DIR/download_data.py --year 2024 --data-type both --stock-dir /workspace/datasets/Stock_Data_1s --vix-dir /workspace/datasets/VIX"
else
    echo "  python download_data.py --year 2024 --data-type both"
fi
echo ""
echo "🏃 To run smoke test:"
echo "  cd $PROJECT_DIR"
echo "  python smoke_mamba_only.py --train-steps 20 --val-steps 5 --epochs 1"
