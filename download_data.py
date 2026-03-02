#!/usr/bin/env python3
"""
Download stock and VIX data from Cloudflare R2 storage.

Usage:
    python download_data.py --year 2024 --data-type stock
    python download_data.py --year 2023 --data-type vix
    python download_data.py --year 2024 --data-type both
    python download_data.py --start-year 2023 --end-year 2024 --data-type stock
"""

import argparse
import boto3
from pathlib import Path
from typing import Optional


# R2 Configuration
R2_ENDPOINT = 'https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com'
R2_ACCESS_KEY = 'fdfa18bf64b18c61bbee64fda98ca20b'
R2_SECRET_KEY = '394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8'
R2_BUCKET = 'europe'

# Data paths
STOCK_PREFIX = 'datasets/Stock_Data_1s/'
VIX_PREFIX = 'datasets/VIX/'


def download_stock_data(s3_client, year: Optional[int] = None, 
                       start_year: Optional[int] = None, 
                       end_year: Optional[int] = None,
                       local_dir: Path = Path('datasets/Stock_Data_1s')):
    """Download 1s stock bar data from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📊 Downloading Stock_Data_1s from R2...')
    if year:
        print(f'   Filtering for year: {year}')
    elif start_year and end_year:
        print(f'   Filtering for years: {start_year}-{end_year}')
    
    paginator = s3_client.get_paginator('list_objects_v2')
    count = 0
    total_size = 0
    
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=STOCK_PREFIX):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            
            filename = key.split('/')[-1]
            
            # Filter by year(s)
            if year:
                if not filename.startswith(f'{year}-'):
                    continue
            elif start_year and end_year:
                try:
                    file_year = int(filename.split('-')[0])
                    if file_year < start_year or file_year > end_year:
                        continue
                except (ValueError, IndexError):
                    continue
            
            local_file = local_dir / filename
            size_mb = obj['Size'] / 1e6
            
            print(f'  [{count+1}] {filename} ({size_mb:.1f} MB)')
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            count += 1
            total_size += obj['Size']
    
    print(f'✅ Stock data download complete! {count} files, {total_size/1e9:.2f} GB total')
    return count


def download_vix_data(s3_client, year: Optional[int] = None,
                     start_year: Optional[int] = None,
                     end_year: Optional[int] = None,
                     local_dir: Path = Path('datasets/VIX')):
    """Download VIX daily close data from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📈 Downloading VIX data from R2...')
    
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
            
            filename = key.split('/')[-1]
            
            # VIX files are typically named like 'vix_daily_close.csv' or year-specific
            # Download all VIX files (they're small)
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
        description='Download stock and VIX data from Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2024 stock data only
  python download_data.py --year 2024 --data-type stock
  
  # Download 2023-2024 stock data
  python download_data.py --start-year 2023 --end-year 2024 --data-type stock
  
  # Download VIX data
  python download_data.py --data-type vix
  
  # Download both stock and VIX for 2024
  python download_data.py --year 2024 --data-type both
  
  # Download all available data
  python download_data.py --data-type both
        """
    )
    
    parser.add_argument('--year', type=int, help='Specific year to download (e.g., 2024)')
    parser.add_argument('--start-year', type=int, help='Start year for range download')
    parser.add_argument('--end-year', type=int, help='End year for range download')
    parser.add_argument('--data-type', choices=['stock', 'vix', 'both'], default='both',
                       help='Type of data to download (default: both)')
    parser.add_argument('--stock-dir', type=Path, default=Path('datasets/Stock_Data_1s'),
                       help='Local directory for stock data (default: datasets/Stock_Data_1s)')
    parser.add_argument('--vix-dir', type=Path, default=Path('datasets/VIX'),
                       help='Local directory for VIX data (default: datasets/VIX)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.year and (args.start_year or args.end_year):
        parser.error('Cannot specify both --year and --start-year/--end-year')
    
    if (args.start_year and not args.end_year) or (args.end_year and not args.start_year):
        parser.error('Must specify both --start-year and --end-year for range download')
    
    if args.start_year and args.end_year and args.start_year > args.end_year:
        parser.error('--start-year must be <= --end-year')
    
    # Initialize S3 client
    print('🔌 Connecting to R2...')
    s3 = boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )
    print('✅ Connected to R2\n')
    
    # Download requested data
    if args.data_type in ['stock', 'both']:
        download_stock_data(
            s3, 
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=args.stock_dir
        )
        print()
    
    if args.data_type in ['vix', 'both']:
        download_vix_data(
            s3,
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=args.vix_dir
        )
        print()
    
    print('🎉 All downloads complete!')


if __name__ == '__main__':
    main()
