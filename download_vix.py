#!/usr/bin/env python3
"""
Download VIX data from Cloudflare R2 storage.

Usage:
    python download_vix.py
    python download_vix.py --vix-dir /workspace/datasets/VIX
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


def download_vix_data(s3_client, local_dir: Path = Path('datasets/VIX')):
    """Download VIX data from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📈 Downloading VIX data from R2...')
    print(f'   Bucket: {R2_BUCKET}')
    print(f'   Prefix: {VIX_PREFIX}')
    print(f'   Local: {local_dir}')
    
    paginator = s3_client.get_paginator('list_objects_v2')
    count = 0
    total_size = 0
    
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=VIX_PREFIX):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            
            # Get filename without prefix
            filename = key.replace(VIX_PREFIX, '')
            if not filename:
                continue
                
            local_file = local_dir / filename
            size_kb = obj['Size'] / 1e3
            
            print(f'  [{count+1}] {filename} ({size_kb:.1f} KB)')
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            count += 1
            total_size += obj['Size']
    
    print(f'✅ VIX data download complete! {count} files, {total_size/1e6:.2f} MB total')
    return count


def main():
    parser = argparse.ArgumentParser(
        description='Download VIX data from Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python download_vix.py
  
  # Download to /workspace
  python download_vix.py --vix-dir /workspace/datasets/VIX
        """
    )
    
    parser.add_argument('--vix-dir', type=Path, default=Path('datasets/VIX'),
                       help='Local directory for VIX data (default: datasets/VIX)')
    
    args = parser.parse_args()
    
    # Initialize S3 client
    print('🔌 Connecting to R2...')
    s3 = boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )
    print('✅ Connected to R2\n')
    
    # Download VIX data
    download_vix_data(s3, local_dir=args.vix_dir)
    
    print('\n🎉 Download complete!')


if __name__ == '__main__':
    main()
