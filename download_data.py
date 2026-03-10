#!/usr/bin/env python3
"""
Download stock, VIX, options, and news data from Cloudflare R2 storage.

Usage:
    python download_data.py --year 2024 --data-type stock
    python download_data.py --year 2023 --data-type vix
    python download_data.py --year 2024 --data-type all
    python download_data.py --start-year 2023 --end-year 2024 --data-type stock
    python download_data.py --year 2024 --data-type options
    python download_data.py --year 2024 --data-type news
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
OPTIONS_PREFIX = 'datasets/opt_trade_1sec/'
NEWS_PREFIX = 'datasets/benzinga_embeddings/'


def download_stock_data(s3_client, year: Optional[int] = None, 
                       start_year: Optional[int] = None, 
                       end_year: Optional[int] = None,
                       local_dir: Path = Path('datasets/Stock_Data_1s'),
                       force: bool = False):
    """Download 1s stock bar data from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📊 Downloading Stock_Data_1s from R2...')
    if year:
        print(f'   Filtering for year: {year}')
    elif start_year and end_year:
        print(f'   Filtering for years: {start_year}-{end_year}')
    
    paginator = s3_client.get_paginator('list_objects_v2')
    downloaded_count = 0
    skipped_count = 0
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
            remote_size = obj['Size']
            size_mb = remote_size / 1e6
            
            # Skip if file exists with matching size (resume capability)
            if not force and local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == remote_size:
                    skipped_count += 1
                    continue
                else:
                    print(f'  [{downloaded_count+1}] {filename} ({size_mb:.1f} MB) [size mismatch, re-downloading]')
            else:
                print(f'  [{downloaded_count+1}] {filename} ({size_mb:.1f} MB)')
            
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            downloaded_count += 1
            total_size += remote_size
    
    print(f'✅ Stock data: {downloaded_count} downloaded, {skipped_count} skipped (already exist), {total_size/1e9:.2f} GB transferred')
    return downloaded_count


def download_vix_data(s3_client, year: Optional[int] = None,
                     start_year: Optional[int] = None,
                     end_year: Optional[int] = None,
                     local_dir: Path = Path('datasets/VIX'),
                     force: bool = False):
    """Download VIX daily close data from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📈 Downloading VIX data from R2...')
    
    paginator = s3_client.get_paginator('list_objects_v2')
    downloaded_count = 0
    skipped_count = 0
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
            remote_size = obj['Size']
            size_kb = remote_size / 1e3
            
            # Skip if file exists with matching size (resume capability)
            if not force and local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == remote_size:
                    skipped_count += 1
                    continue
                else:
                    print(f'  [{downloaded_count+1}] {filename} ({size_kb:.1f} KB) [size mismatch, re-downloading]')
            else:
                print(f'  [{downloaded_count+1}] {filename} ({size_kb:.1f} KB)')
            
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            downloaded_count += 1
            total_size += remote_size
    
    print(f'✅ VIX data: {downloaded_count} downloaded, {skipped_count} skipped (already exist), {total_size/1e6:.2f} MB transferred')
    return downloaded_count


def download_options_data(s3_client, year: Optional[int] = None,
                          start_year: Optional[int] = None,
                          end_year: Optional[int] = None,
                          local_dir: Path = Path('datasets/opt_trade_1sec'),
                          force: bool = False):
    """Download options trade data (contract and underlying bars) from R2."""
    print(f'📊 Downloading opt_trade_1sec from R2...')
    if year:
        print(f'   Filtering for year: {year}')
    elif start_year and end_year:
        print(f'   Filtering for years: {start_year}-{end_year}')
    
    paginator = s3_client.get_paginator('list_objects_v2')
    downloaded_count = 0
    skipped_count = 0
    total_size = 0
    
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=OPTIONS_PREFIX):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            
            # Key format: datasets/opt_trade_1sec/option_underlying_bars_1s/2024-01-02.parquet
            parts = key.replace(OPTIONS_PREFIX, '').split('/')
            if len(parts) < 2:
                continue
            
            subdir = parts[0]  # e.g., option_underlying_bars_1s
            filename = parts[1]  # e.g., 2024-01-02.parquet
            
            # Filter by year(s) - filename format: YYYY-MM-DD.parquet
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
            
            # Create subdirectory structure
            subdir_path = local_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            
            local_file = subdir_path / filename
            remote_size = obj['Size']
            size_mb = remote_size / 1e6
            
            # Skip if file exists with matching size
            if not force and local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == remote_size:
                    skipped_count += 1
                    continue
                else:
                    print(f'  [{downloaded_count+1}] {subdir}/{filename} ({size_mb:.1f} MB) [re-downloading]')
            else:
                print(f'  [{downloaded_count+1}] {subdir}/{filename} ({size_mb:.1f} MB)')
            
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            downloaded_count += 1
            total_size += remote_size
    
    print(f'✅ Options data: {downloaded_count} downloaded, {skipped_count} skipped (already exist), {total_size/1e9:.2f} GB transferred')
    return downloaded_count


def download_news_data(s3_client, year: Optional[int] = None,
                       start_year: Optional[int] = None,
                       end_year: Optional[int] = None,
                       local_dir: Path = Path('datasets/benzinga_embeddings'),
                       force: bool = False):
    """Download Benzinga news embeddings from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📰 Downloading benzinga_embeddings from R2...')
    if year:
        print(f'   Filtering for year: {year}')
    elif start_year and end_year:
        print(f'   Filtering for years: {start_year}-{end_year}')
    
    paginator = s3_client.get_paginator('list_objects_v2')
    downloaded_count = 0
    skipped_count = 0
    total_size = 0
    
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=NEWS_PREFIX):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            
            filename = key.split('/')[-1]
            
            # Filter by year - filename format: YYYY_embedded.parquet
            if year or (start_year and end_year):
                try:
                    file_year = int(filename.split('_')[0])
                    if year and file_year != year:
                        continue
                    if start_year and end_year:
                        if file_year < start_year or file_year > end_year:
                            continue
                except (ValueError, IndexError):
                    continue
            
            local_file = local_dir / filename
            remote_size = obj['Size']
            size_gb = remote_size / 1e9
            
            # Skip if file exists with matching size
            if not force and local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == remote_size:
                    skipped_count += 1
                    continue
                else:
                    print(f'  [{downloaded_count+1}] {filename} ({size_gb:.2f} GB) [re-downloading]')
            else:
                print(f'  [{downloaded_count+1}] {filename} ({size_gb:.2f} GB)')
            
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            downloaded_count += 1
            total_size += remote_size
    
    print(f'✅ News data: {downloaded_count} downloaded, {skipped_count} skipped (already exist), {total_size/1e9:.2f} GB transferred')
    return downloaded_count


def main():
    parser = argparse.ArgumentParser(
        description='Download stock, VIX, options, and news data from Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2024 stock data only
  python download_data.py --year 2024 --data-type stock
  
  # Download 2023-2024 stock data
  python download_data.py --start-year 2023 --end-year 2024 --data-type stock
  
  # Download VIX data
  python download_data.py --data-type vix
  
  # Download options data for 2024
  python download_data.py --year 2024 --data-type options
  
  # Download news embeddings for 2024
  python download_data.py --year 2024 --data-type news
  
  # Download all data types for 2024 (year filter applies to ALL types)
  python download_data.py --year 2024 --data-type all
  
  # Download all available data
  python download_data.py --data-type all
  
  # Force re-download (ignore existing files)
  python download_data.py --data-type all --force
        """
    )
    
    parser.add_argument('--year', type=int, help='Specific year to download (e.g., 2024) - applies to ALL data types')
    parser.add_argument('--start-year', type=int, help='Start year for range download - applies to ALL data types')
    parser.add_argument('--end-year', type=int, help='End year for range download - applies to ALL data types')
    parser.add_argument('--data-type', choices=['stock', 'vix', 'options', 'news', 'all'], default='all',
                       help='Type of data to download (default: all)')
    parser.add_argument('--stock-dir', type=Path, default=Path('datasets/Stock_Data_1s'),
                       help='Local directory for stock data (default: datasets/Stock_Data_1s)')
    parser.add_argument('--vix-dir', type=Path, default=Path('datasets/VIX'),
                       help='Local directory for VIX data (default: datasets/VIX)')
    parser.add_argument('--options-dir', type=Path, default=Path('datasets/opt_trade_1sec'),
                       help='Local directory for options data (default: datasets/opt_trade_1sec)')
    parser.add_argument('--news-dir', type=Path, default=Path('datasets/benzinga_embeddings'),
                       help='Local directory for news data (default: datasets/benzinga_embeddings)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download all files (ignore existing)')
    
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
    if args.data_type in ['stock', 'all']:
        download_stock_data(
            s3, 
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=args.stock_dir,
            force=args.force
        )
        print()
    
    if args.data_type in ['vix', 'all']:
        download_vix_data(
            s3,
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=args.vix_dir,
            force=args.force
        )
        print()
    
    if args.data_type in ['options', 'all']:
        download_options_data(
            s3,
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=args.options_dir,
            force=args.force
        )
        print()
    
    if args.data_type in ['news', 'all']:
        download_news_data(
            s3,
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=args.news_dir,
            force=args.force
        )
        print()
    
    print('🎉 All downloads complete!')


if __name__ == '__main__':
    main()
