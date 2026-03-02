#!/bin/bash
set -euo pipefail

# Get the directory where the script is located, which is the project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"


# To restart clean:
# deactivate 2>/dev/null || true
# rm -rf venv
# bash setupenv.sh

echo "🚀 Setting up RTX 5080 + CUDA 12.8 + Mamba Environment in WSL"
echo "=============================================================="
echo "This script installs RTX 5080/Blackwell (sm_120) optimized versions"

# Check if we're in WSL
if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
    echo "❌ This script must be run in WSL!"
    exit 1
fi

# Helper to avoid duplicate .bashrc lines
append_if_missing() {
    local line="$1"
    local file="$2"
    grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

# Set environment variables for RTX 5080 + sm_120
export DEBIAN_FRONTEND=noninteractive
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
export FORCE_CUDA=1
export MAX_JOBS=12

# CUDA Exports for current session
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.8

# Add to bashrc for persistence (idempotent)
BASHRC="$HOME/.bashrc"
append_if_missing 'export PATH=/usr/local/cuda-12.8/bin:$PATH' "$BASHRC"
append_if_missing 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' "$BASHRC"
append_if_missing 'export CUDA_HOME=/usr/local/cuda-12.8' "$BASHRC"
append_if_missing 'export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"' "$BASHRC"
append_if_missing 'export FORCE_CUDA=1' "$BASHRC"
append_if_missing 'export MAX_JOBS=12' "$BASHRC"

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
if ! grep -q "^deb .*deadsnakes/ppa" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
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

echo "⚡ Checking CUDA 12.8 installation..."
if [ ! -d "/usr/local/cuda-12.8" ]; then
    echo "Installing CUDA 12.8..."
    cd /tmp
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-8
    rm cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
else
    echo "✅ CUDA 12.8 already installed."
fi

echo "🐍 creating venv in project root..."
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

# Verify we're in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Failed to activate venv!"
    exit 1
fi
echo "✅ venv activated: $VIRTUAL_ENV"

echo "📦 Installing Python packages..."
python3 -m pip install --upgrade pip setuptools wheel ninja packaging

echo "🔥 Installing PyTorch nightly with CUDA 12.8 + sm_120..."
# Install torch only if not already installed (checking torch import or just pip install is safe as it skips typically)
python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "🌊 Installing causal-conv1d with RTX 5080/sm_120 support..."
# Using: https://github.com/yacinemassena/causal-conv1d-sm120
# We install directly from git to avoid local folder mess if possible, or keep local clone if desired.
# User script had custom_packages. I will keep using it to be safe with rebuilds.
mkdir -p custom_packages
cd custom_packages

if [[ ! -d "causal-conv1d-sm120" ]]; then
    git clone https://github.com/yacinemassena/causal-conv1d-sm120.git
    cd causal-conv1d-sm120
else
    cd causal-conv1d-sm120
    git pull
fi
pip install -e . --no-build-isolation --no-deps

# Verify immediately
python3 - <<'PY'
import importlib, sys
try:
    m = importlib.import_module("causal_conv1d")
    print("✅ causal_conv1d OK:", m.__file__)
except Exception as e:
    print("❌ causal_conv1d import failed:", e)
    sys.exit(1)
PY

cd ..

echo "🐍 Installing mamba-ssm with RTX 5080/Blackwell support..."
# Using: https://github.com/yacinemassena/mamba_blackwell

if [[ ! -d "mamba_blackwell" ]]; then
    git clone https://github.com/yacinemassena/mamba_blackwell.git
    cd mamba_blackwell
else
    cd mamba_blackwell
    git pull
fi

# Install dependencies first (only what mamba_ssm actually needs)
pip install transformers einops triton

# Build mamba-ssm in editable mode
pip install -e . --no-build-isolation --no-deps

# Verify immediately
python3 - <<'PY'
import importlib, pkgutil, sys

mods = sorted([m.name for m in pkgutil.iter_modules() if "mamba" in m.name])
print("Detected mamba-related modules:", mods)

for name in ("mamba_ssm", "mamba_blackwell"):
    try:
        m = importlib.import_module(name)
        print(f"✅ import {name} OK:", m.__file__)
    except Exception as e:
        print(f"⚠️ import {name} failed:", e)

# Require at least one to work
ok = False
for name in ("mamba_ssm", "mamba_blackwell"):
    try:
        importlib.import_module(name)
        ok = True
        break
    except Exception:
        pass

if not ok:
    print("❌ Neither mamba_ssm nor mamba_blackwell is importable after install.")
    sys.exit(1)

print("✅ Mamba import check passed")
PY

cd ../..

echo "📚 Installing additional ML libraries..."
pip install \
    numpy pandas pyarrow tqdm pyyaml boto3 \
    tensorboard

echo ""
echo "🎉 RTX 5080 + CUDA 12.8 + sm_120 Setup Complete!"
echo "🚀 To activate your environment:"
echo "  source venv/bin/activate"
echo "🧪 To verify:"
echo "  python test_environment.py"

# Create the specific test script requested
cat > test_environment.py <<'PY'
import importlib
import torch

def main():
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    assert torch.cuda.is_available(), "CUDA not available"

    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

    # causal conv
    causal = importlib.import_module("causal_conv1d")
    print("causal_conv1d:", getattr(causal, "__file__", "unknown"))

    # mamba module: accept either name
    mamba_mod = None
    for name in ("mamba_ssm", "mamba_blackwell"):
        try:
            mamba_mod = importlib.import_module(name)
            print(f"Mamba module '{name}':", mamba_mod.__file__)
            break
        except Exception as e:
            print(f"Import {name} failed:", e)
    assert mamba_mod is not None, "No Mamba module importable"

    # Try to import Mamba class from whichever module provides it
    Mamba = None
    try:
        from mamba_ssm import Mamba as Mamba
    except Exception:
        try:
            from mamba_blackwell import Mamba as Mamba
        except Exception as e:
            raise RuntimeError("Could not import Mamba class from either module") from e

    x = torch.randn(2, 64, 128, device=dev)
    m = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to(dev)
    y = m(x)
    assert y.shape == x.shape
    print("✅ Mamba forward OK")

    print("✅ Environment OK")

if __name__ == "__main__":
    main()
PY