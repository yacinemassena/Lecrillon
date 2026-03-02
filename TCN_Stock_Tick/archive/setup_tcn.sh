#!/bin/bash
# TCN Pretraining Setup Script for VPS/Cloud Deployment
# Supports Ubuntu/Debian-based systems with CUDA

set -e  # Exit on error

echo "=========================================="
echo "TCN Pretraining Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="venv_tcn"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo "Removed existing venv"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Detect CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
    echo "✓ CUDA detected: $CUDA_VERSION"
    
    # Determine PyTorch CUDA version
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "Installing PyTorch with CUDA 12.1 support..."
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "Installing PyTorch with CUDA 11.8 support..."
    else
        echo "Warning: Unsupported CUDA version. Installing CPU version."
        TORCH_INDEX=""
    fi
else
    echo "Warning: CUDA not detected. Installing CPU version."
    TORCH_INDEX=""
fi

# Install PyTorch
if [ -n "$TORCH_INDEX" ]; then
    pip install torch torchvision --index-url "$TORCH_INDEX"
else
    pip install torch torchvision
fi

echo "✓ PyTorch installed"

# Install other requirements
echo "Installing requirements from requirementstcn.txt..."
pip install -r requirementstcn.txt

echo "✓ All requirements installed"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run training:"
echo "  # Stocks (RTX 5080 - filtered)"
echo "  python pretrain_tcn_rv.py --profile rtx5080 --stream stocks"
echo ""
echo "  # Options (H100 - full)"
echo "  python pretrain_tcn_rv.py --profile h100 --stream options"
echo ""
echo "  # Index (H100)"
echo "  python pretrain_tcn_rv.py --profile h100 --stream index"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
