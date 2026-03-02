#!/bin/bash
# VPS Setup Script for TCN Pretraining (Flexible Data Handling)
# Run this on a fresh VPS with CUDA-enabled GPU

set -e  # Exit on error

echo "=========================================="
echo "TCN Pretraining VPS Setup"
echo "=========================================="

# 1. Upgrade PyTorch to latest version
echo "[1/5] Upgrading PyTorch..."
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Clone repository
echo "[2/5] Cloning TCN repository..."
cd /
if [ -d "/TCN" ]; then
    echo "TCN directory exists, pulling latest..."
    cd /TCN
    git pull
else
    git clone https://github.com/yacinemassena/TCN.git
    cd /TCN
fi

# 3. Install Python dependencies
echo "[3/5] Installing dependencies..."
pip install pandas pyarrow tqdm boto3 optuna

# 4. Check data availability
echo "[4/5] Checking data availability..."
python3 check_data_availability.py

# 5. Verify setup
echo "[5/5] Verifying setup..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

# Read available streams and provide recommendations
if [ -f "available_streams.json" ]; then
    echo "Data availability check complete."
    
    # Extract recommended stream using Python
    STREAM=$(python3 -c "import json; d=json.load(open('available_streams.json')); print(d.get('recommended_stream', 'none'))" 2>/dev/null || echo "none")
    
    if [ "$STREAM" != "none" ] && [ "$STREAM" != "None" ]; then
        echo ""
        echo "✓ Ready to train on: $STREAM"
        echo ""
        echo "Start training:"
        echo "  python pretrain_tcn_rv.py --stream $STREAM --profile h100"
        echo ""
        echo "For hyperparameter search (4 GPUs):"
        echo "  python hyperparam_search.py --stream $STREAM --gpus 4 --n_trials 20"
    else
        echo ""
        echo "⚠ No complete datasets found."
        echo ""
        echo "Download data from R2:"
        echo "  python check_data_availability.py --download --stream index"
        echo ""
        echo "Or download all available data:"
        echo "  python check_data_availability.py --download"
    fi
else
    echo "⚠ Could not check data availability"
fi

echo ""
echo "Available profiles: rtx5080 (16GB), h100 (80GB), a100 (80GB), amd (192GB)"
echo "Add --no-checkpoint flag to disable gradient checkpointing (faster, more VRAM)"
echo ""
