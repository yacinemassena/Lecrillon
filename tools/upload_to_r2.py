import os
import glob
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

def upload_wheels():
    print("🚀 Uploading wheels to Cloudflare R2...")
    
    # R2 Configuration
    # It's better to use environment variables for secrets in production,
    # but for this script we'll use the provided credentials.
    R2_ACCOUNT_ID = "2a139e9393f803634546ad9d541d37b9"
    R2_ACCESS_KEY_ID = "fdfa18bf64b18c61bbee64fda98ca20b"
    R2_SECRET_ACCESS_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
    R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    BUCKET_NAME = "europe"
    PREFIX = "custom_mamba_wheels/"

    # Initialize S3 client for R2
    s3 = boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto' # R2 requires region to be 'auto' or 'us-east-1' generally
    )

    # Find wheels
    project_dir = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(project_dir, "custom_wheels")
    
    if not os.path.exists(wheels_dir):
        print(f"❌ Wheels directory not found: {wheels_dir}")
        print("Please run build_wheels.sh first.")
        return

    wheel_files = glob.glob(os.path.join(wheels_dir, "*.whl"))
    
    if not wheel_files:
        print("❌ No .whl files found in custom_wheels directory.")
        return

    print(f"📦 Found {len(wheel_files)} wheels to upload.")

    for wheel_path in wheel_files:
        filename = os.path.basename(wheel_path)
        s3_key = f"{PREFIX}{filename}"
        
        print(f"Uploading {filename} to {BUCKET_NAME}/{s3_key}...")
        try:
            s3.upload_file(wheel_path, BUCKET_NAME, s3_key)
            print(f"✅ Successfully uploaded {filename}")
        except ClientError as e:
            print(f"❌ Failed to upload {filename}: {e}")

    print("🎉 All uploads complete!")

if __name__ == "__main__":
    upload_wheels()
