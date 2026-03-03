#!/usr/bin/env python3
"""
Upload VIX data to Cloudflare R2 storage.

Usage:
    python upload_vix.py
    python upload_vix.py --vix-dir D:\Mamba v2\datasets\VIX
"""

import argparse
import boto3
from pathlib import Path


# R2 Configuration
R2_ENDPOINT = 'https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com'
R2_ACCESS_KEY = 'fdfa18bf64b18c61bbee64fda98ca20b'
R2_SECRET_KEY = '394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8'
R2_BUCKET = 'europe'
VIX_PREFIX = 'VIX/'


def upload_vix_data(s3_client, local_dir: Path):
    """Upload VIX CSV files to R2."""
    if not local_dir.exists():
        print(f'❌ Directory not found: {local_dir}')
        return 0
    
    print(f'📈 Uploading VIX data to R2...')
    print(f'   Local: {local_dir}')
    print(f'   Bucket: {R2_BUCKET}')
    print(f'   Prefix: {VIX_PREFIX}')
    print()
    
    # Get all CSV files
    csv_files = sorted(local_dir.glob('*.csv'))
    
    if not csv_files:
        print(f'❌ No CSV files found in {local_dir}')
        return 0
    
    count = 0
    total_size = 0
    
    for local_file in csv_files:
        filename = local_file.name
        r2_key = f'{VIX_PREFIX}{filename}'
        size_mb = local_file.stat().st_size / 1e6
        
        print(f'  [{count+1}/{len(csv_files)}] {filename} ({size_mb:.2f} MB)')
        
        s3_client.upload_file(
            str(local_file),
            R2_BUCKET,
            r2_key
        )
        
        count += 1
        total_size += local_file.stat().st_size
    
    print()
    print(f'✅ Upload complete! {count} files, {total_size/1e9:.2f} GB total')
    return count


def main():
    parser = argparse.ArgumentParser(
        description='Upload VIX data to Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from default location
  python upload_vix.py
  
  # Upload from custom location
  python upload_vix.py --vix-dir D:\\Mamba v2\\datasets\\VIX
        """
    )
    
    parser.add_argument('--vix-dir', type=Path, default=Path('datasets/VIX'),
                       help='Local directory with VIX CSV files (default: datasets/VIX)')
    
    args = parser.parse_args()
    
    # Initialize S3 client
    print('🔌 Connecting to R2...')
    s3 = boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )
    print('✅ Connected to R2\n')
    
    # Upload VIX data
    upload_vix_data(s3, local_dir=args.vix_dir)
    
    print('\n🎉 Upload complete!')


if __name__ == '__main__':
    main()
