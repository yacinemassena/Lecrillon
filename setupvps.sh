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
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
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
    append_if_missing 'export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"' "$BASHRC"
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

echo "🧹 Cleaning up pre-installed PyTorch (RunPod compatibility)..."
# Remove any system-wide PyTorch installations that might conflict
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python3.11 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "🐍 Checking for pre-packaged environment in R2..."
# Quick python script to check and download from R2
cat > "$PROJECT_DIR/download_env.py" << 'EOF'
import os
import sys
import boto3
from botocore.config import Config

R2_ACCOUNT_ID = "2a139e9393f803634546ad9d541d37b9"
R2_ACCESS_KEY_ID = "fdfa18bf64b18c61bbee64fda98ca20b"
R2_SECRET_ACCESS_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET_NAME = "europe"
PREFIX = "environments/"

# Determine GPU compute cap (simplified for script)
try:
    import torch
    cap = str(torch.cuda.get_device_capability(0)).replace('(', '').replace(')', '').replace(', ', '')
except:
    cap = "cpu"

tar_filename = f"mamba_venv_linux_{cap}.tar.zst"
object_name = f"{PREFIX}{tar_filename}"
download_path = sys.argv[1]

print(f"Connecting to R2 to check for {tar_filename}...")
s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL,
                  aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                  config=Config(signature_version='s3v4'), region_name='auto')

try:
    s3.head_object(Bucket=BUCKET_NAME, Key=object_name)
    print(f"✅ Pre-packaged environment found! Downloading...")
    s3.download_file(BUCKET_NAME, object_name, download_path)
    print("✅ Download complete!")
    sys.exit(0)
except Exception as e:
    print(f"⚠️  No pre-packaged environment found in R2 ({object_name}).")
    sys.exit(1)
EOF

TAR_FILEPATH="$PROJECT_DIR/downloaded_venv.tar.zst"

# We need boto3 installed temporarily in the system to run the download script
$SUDO pip3 install boto3 --break-system-packages || pip3 install boto3

if python3 "$PROJECT_DIR/download_env.py" "$TAR_FILEPATH"; then
    echo "🗜️ Extracting pre-packaged environment using zstd..."
    
    # Make sure zstd is installed
    if ! command -v zstd &> /dev/null; then
        $SUDO apt-get update $APT_OPTS && $SUDO apt-get install -y $APT_OPTS zstd
    fi
    
    # Extract
    if [ -d "/workspace" ] && [ -w "/workspace" ]; then
        cd /workspace
        tar -I zstd -xf "$TAR_FILEPATH"
        VENV_PATH="/workspace/venv"
    else
        cd "$PROJECT_DIR"
        tar -I zstd -xf "$TAR_FILEPATH"
        VENV_PATH="$PROJECT_DIR/venv"
    fi
    
    rm "$TAR_FILEPATH"
    
    echo "🔧 Relocating venv paths..."
    # Venvs have hardcoded paths in bin/activate and pip shebangs.
    # We update them to the current absolute path.
    for script in "$VENV_PATH"/bin/*; do
        if [ -f "$script" ] && file "$script" | grep -q "text"; then
            sed -i "s|/workspace/venv|$VENV_PATH|g" "$script" 2>/dev/null || true
            sed -i "s|.*Mamba v2/venv|$VENV_PATH|g" "$script" 2>/dev/null || true
        fi
    done
    
    echo "🔌 Activating extracted venv..."
    source "$VENV_PATH/bin/activate"
    echo "✅ Fast setup complete using pre-packaged environment!"
    
    # Clean up script
    rm "$PROJECT_DIR/download_env.py"
else
    # FALLBACK: Build from scratch
    rm "$PROJECT_DIR/download_env.py"
    echo "⚠️ Falling back to full source build..."
    
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
    pip install numpy pandas pyarrow boto3 tqdm einops triton transformers

    # Detect GPU compute capability
    GPU_COMPUTE_CAP=""
    if [ -n "$CUDA_HOME" ]; then
        echo "🔍 Detecting GPU compute capability..."
        GPU_COMPUTE_CAP=$(python3.11 -c "import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')" 2>/dev/null || echo "")
        echo "GPU compute capability: $GPU_COMPUTE_CAP"
    fi

    # Install causal-conv1d and mamba_ssm based on GPU
    if [ -n "$CUDA_HOME" ] && [ -n "$GPU_COMPUTE_CAP" ]; then
        # Always use custom packages for CUDA GPUs (supports sm_80 A100 and sm_120 Blackwell)
        echo "🚀 GPU detected - installing custom mamba packages..."
        
        # Clean any existing PyPI versions first
        pip uninstall -y mamba-ssm causal-conv1d 2>/dev/null || true
        rm -rf "$VENV_PATH/lib/python3.11/site-packages/mamba_ssm"* 2>/dev/null || true
        rm -rf "$VENV_PATH/lib/python3.11/site-packages/selective_scan_cuda"* 2>/dev/null || true
        rm -rf "$VENV_PATH/lib/python3.11/site-packages/causal_conv1d"* 2>/dev/null || true
        
        echo "☁️ Downloading pre-built wheels from Cloudflare R2..."
        WHEELS_DIR="$PROJECT_DIR/downloaded_wheels"
        mkdir -p "$WHEELS_DIR"
        
        # Download wheels using boto3 script
        cat > "$PROJECT_DIR/download_wheels.py" << 'EOF'
import os
import boto3
from botocore.config import Config

print("Downloading wheels from R2...")
R2_ACCOUNT_ID = "2a139e9393f803634546ad9d541d37b9"
R2_ACCESS_KEY_ID = "fdfa18bf64b18c61bbee64fda98ca20b"
R2_SECRET_ACCESS_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET_NAME = "europe"
PREFIX = "custom_mamba_wheels/"
DOWNLOAD_DIR = "downloaded_wheels"

s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL,
                  aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                  config=Config(signature_version='s3v4'), region_name='auto')

response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
if 'Contents' in response:
    for obj in response['Contents']:
        if obj['Key'].endswith('.whl'):
            filename = os.path.basename(obj['Key'])
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            print(f"Downloading {filename}...")
            s3.download_file(BUCKET_NAME, obj['Key'], filepath)
    print("✅ Download complete!")
else:
    print("❌ No wheels found in R2 bucket.")
EOF
        
        python3 "$PROJECT_DIR/download_wheels.py"
        rm "$PROJECT_DIR/download_wheels.py"
        
        # Check if wheels were downloaded
        if ls "$WHEELS_DIR"/*.whl 1> /dev/null 2>&1; then
            echo "📦 Installing downloaded wheels..."
            pip install "$WHEELS_DIR"/*.whl --force-reinstall
            echo "✅ Installed pre-built custom packages!"
        else
            echo "⚠️ No pre-built wheels found in R2. Falling back to source build..."
            echo "🔧 Installing causal-conv1d from custom package..."
            cd "$PROJECT_DIR/custom_packages/causal-conv1d-sm120"
            pip install -e . --no-build-isolation --force-reinstall
            
            echo "🐍 Installing mamba_ssm from custom package..."
            cd "$PROJECT_DIR/custom_packages/mamba_blackwell"
            pip install -e . --no-build-isolation --force-reinstall
            cd "$PROJECT_DIR"
        fi
    else
        echo "⚠️  No CUDA - skipping causal-conv1d and mamba_ssm"
    fi
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
