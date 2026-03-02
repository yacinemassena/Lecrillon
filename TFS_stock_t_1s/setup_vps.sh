#!/bin/bash
set -euo pipefail

# VPS Setup for Stock → Mamba → VIX Pipeline
# Uses root venv from setupenv.sh (custom mamba_blackwell sm_120)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Setting up Mamba VIX Pipeline on VPS"
echo "============================================================"

# Step 1: Run root setupenv.sh if venv doesn't exist
if [[ ! -d "venv" ]]; then
    echo "Running root setupenv.sh to create venv with Mamba..."
    bash setupenv.sh
fi

source venv/bin/activate

# Step 2: Install additional dependencies
echo "Installing pipeline dependencies..."
pip install optuna boto3 pyarrow tqdm scipy scikit-learn

# Step 3: Check data availability
echo ""
echo "Checking data availability..."
cd "$SCRIPT_DIR"
python check_data_availability.py

# Step 4: Generate top 100 stocks if missing
if [[ ! -f "scripts/top_100_stocks.txt" ]]; then
    echo "Generating top 100 stocks list..."
    python -c "
from scripts.compute_top_stocks import compute_top_stocks
from pathlib import Path
import platform
if platform.system() == 'Linux':
    data_path = '/TCN/datasets/2017-2025/Stock_Data_1s'
else:
    data_path = 'D:/Mamba v2/datasets/Stock_Data_1s'
compute_top_stocks(data_path, top_n=100, output_file='scripts/top_100_stocks.txt')
"
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To train Level 1 (next-day VIX):"
echo "  python train_mamba.py --profile rtx5080 --level 1"
echo ""
echo "To train Level 2 (VIX +30d):"
echo "  python train_mamba.py --profile rtx5080 --level 2"
echo ""
echo "Smoke test:"
echo "  python train_mamba.py --profile rtx5080 --level 1 --smoke"
echo ""
echo "Multi-GPU (e.g. 4 GPUs):"
echo "  torchrun --nproc_per_node=4 train_mamba.py --profile a100 --level 1 --gpus 4"
echo ""
echo "Hyperparameter search (8 GPUs):"
echo "  python hyperparam_search_mamba.py --profile a100 --gpus 8 --n_trials 50"
