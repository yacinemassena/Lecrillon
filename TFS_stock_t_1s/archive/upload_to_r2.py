"""Upload TCN training data to Cloudflare R2 bucket."""
import boto3
from pathlib import Path
from tqdm import tqdm
import os

# R2 Configuration
R2_ENDPOINT = "https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com"
ACCESS_KEY = "fdfa18bf64b18c61bbee64fda98ca20b"
SECRET_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
BUCKET_NAME = "europe"

# Initialize S3 client for R2
s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='auto'
)

def upload_file(local_path, s3_key):
    """Upload a single file to R2."""
    file_size = os.path.getsize(local_path)
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {Path(local_path).name}") as pbar:
        s3.upload_file(
            str(local_path),
            BUCKET_NAME,
            s3_key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
        )
    print(f"✓ Uploaded: {s3_key}")

def upload_directory(local_dir, s3_prefix):
    """Upload directory recursively to R2."""
    local_path = Path(local_dir)
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
            upload_file(file_path, s3_key)

def main():
    print("=" * 60)
    print("Uploading Dataset to Cloudflare R2")
    print("=" * 60)
    
    # 1. Upload minimal index data (Feb-Mar 2022, ~2 months for testing)
    print("\n[1/2] Uploading minimal index data (2 months)...")
    index_data_path = Path("../datasets/2022-2023/index_data/2022")
    if index_data_path.exists():
        files_to_upload = sorted(index_data_path.glob("2022-0[23]-*.parquet"))[:40]
        for file_path in files_to_upload:
            s3_key = f"datasets/index_data/2022/{file_path.name}"
            upload_file(file_path, s3_key)
    else:
        print(f"⚠ Index data not found at {index_data_path}")
    
    # 2. Upload RV targets
    print("\n[2/2] Uploading RV targets...")
    rv_file = Path("../datasets/2022-2023/spy_daily_rv.parquet")
    if rv_file.exists():
        upload_file(rv_file, "datasets/spy_daily_rv.parquet")
    else:
        print(f"⚠ RV file not found at {rv_file}")
    
    print("\n" + "=" * 60)
    print("✓ Upload complete!")
    print("=" * 60)
    print("\nFiles uploaded to R2 bucket 'europe':")
    print("  - datasets/index_data/   (minimal 2-month dataset)")
    print("  - datasets/spy_rv_30d.parquet")
    print("\nTo setup on GPU server:")
    print("  1. Clone code: git clone https://github.com/yacinemassena/TCN.git")
    print("  2. Download data: aws s3 sync s3://europe/datasets/ ./datasets/ --endpoint-url=" + R2_ENDPOINT)
    print("  3. Run setup: bash setup_tcn.sh")
    print("  4. Train: python pretrain_tcn_rv.py --profile h100 --stream index")

if __name__ == "__main__":
    main()
