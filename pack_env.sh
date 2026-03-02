#!/bin/bash
set -euo pipefail

# Get the directory where the script is located, which is the project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "📦 Packaging Mamba VIX Virtual Environment"
echo "=========================================="

# Check if venv exists
VENV_PATH="$PROJECT_DIR/venv"
if [ -d "/workspace/venv" ]; then
    VENV_PATH="/workspace/venv"
    echo "Found venv in /workspace/venv"
elif [ -d "$PROJECT_DIR/venv" ]; then
    VENV_PATH="$PROJECT_DIR/venv"
    echo "Found venv in $PROJECT_DIR/venv"
else
    echo "❌ No venv found! Run setupvps.sh first to build the environment."
    exit 1
fi

echo "🧹 Cleaning up unnecessary cache files to reduce size..."
find "$VENV_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$VENV_PATH" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$VENV_PATH" -type d -name "pip-cache" -exec rm -rf {} + 2>/dev/null || true

# Determine CUDA compute capability for naming
GPU_COMPUTE_CAP=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'cpu')" 2>/dev/null | tr -d '(),' | tr ' ' '_' || echo "cpu")
if [ "$GPU_COMPUTE_CAP" == "" ]; then
    GPU_COMPUTE_CAP="cpu"
fi

TAR_FILENAME="mamba_venv_linux_${GPU_COMPUTE_CAP}.tar.zst"
TAR_FILEPATH="$PROJECT_DIR/$TAR_FILENAME"

echo "🗜️ Compressing venv into $TAR_FILENAME using zstd..."
echo "This may take a few minutes depending on CPU speed..."

# Make sure zstd is installed
if ! command -v zstd &> /dev/null; then
    echo "Installing zstd..."
    sudo apt-get update && sudo apt-get install -y zstd
fi

# Go to the parent dir of venv to tar it nicely without full absolute paths
cd "$(dirname "$VENV_PATH")"
VENV_BASENAME="$(basename "$VENV_PATH")"

# Tar and compress with zstd (-T0 uses all cores, -10 is good compression ratio vs speed)
tar -I 'zstd -T0 -10' -cf "$TAR_FILEPATH" "$VENV_BASENAME"
cd "$PROJECT_DIR"

SIZE=$(du -h "$TAR_FILEPATH" | cut -f1)
echo "✅ Packaging complete! File size: $SIZE"

echo "☁️ Uploading to Cloudflare R2..."
# Quick python script to upload using boto3
cat > "$PROJECT_DIR/upload_env.py" << EOF
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

filepath = sys.argv[1]
filename = os.path.basename(filepath)
object_name = f"{PREFIX}{filename}"

print(f"Connecting to R2...")
s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL,
                  aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                  config=Config(signature_version='s3v4'), region_name='auto')

size_mb = os.path.getsize(filepath) / (1024 * 1024)
print(f"Uploading {filename} ({size_mb:.1f} MB)... This might take a while depending on bandwidth.")

try:
    s3.upload_file(filepath, BUCKET_NAME, object_name)
    print("✅ Upload successful!")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)
EOF

"$VENV_PATH/bin/python" "$PROJECT_DIR/upload_env.py" "$TAR_FILEPATH"
rm "$PROJECT_DIR/upload_env.py"

echo "🎉 All done! You can now use the pre-packaged environment on new VPS instances."
echo "Keep $TAR_FILEPATH as a local backup if needed."
