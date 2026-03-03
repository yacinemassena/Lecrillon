#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Building custom wheels for Mamba and Causal-Conv1d"
echo "======================================================"

# Ensure we're in the WSL environment and have venv activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "⚠️  Virtual environment not activated. Attempting to activate..."
    if [[ -f "$PROJECT_DIR/venv/bin/activate" ]]; then
        source "$PROJECT_DIR/venv/bin/activate"
    else
        echo "❌ Cannot find venv at $PROJECT_DIR/venv. Please run setupenv.sh first."
        exit 1
    fi
fi

# Ensure pip, wheel, and build are installed
pip install --upgrade pip wheel build packaging

WHEELS_DIR="$PROJECT_DIR/custom_wheels"
mkdir -p "$WHEELS_DIR"

echo "🧹 Cleaning old wheels..."
rm -f "$WHEELS_DIR"/*.whl

# Build causal-conv1d
echo "🌊 Building causal-conv1d-sm120..."
cd "$PROJECT_DIR/custom_packages/causal-conv1d-sm120"
python -m build --wheel --outdir "$WHEELS_DIR"

# Build mamba-ssm
echo "🐍 Building mamba_blackwell..."
cd "$PROJECT_DIR/custom_packages/mamba_blackwell"
python -m build --wheel --outdir "$WHEELS_DIR"

echo "✅ Wheels built successfully in $WHEELS_DIR:"
ls -lh "$WHEELS_DIR"/*.whl

echo ""
echo "Now run upload_to_r2.py to upload these wheels."
