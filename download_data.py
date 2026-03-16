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
DATASETS_ROOT_PREFIX = 'datasets/'
STOCK_PREFIX = f'{DATASETS_ROOT_PREFIX}Stock_Data_2min/'
VIX_PREFIX = f'{DATASETS_ROOT_PREFIX}VIX/'
OPTIONS_PREFIX = f'{DATASETS_ROOT_PREFIX}opt_trade_2min/'
NEWS_PREFIX = f'{DATASETS_ROOT_PREFIX}benzinga_embeddings/news_daily/'


def _matches_year_filter(relative_path: str,
                         year: Optional[int] = None,
                         start_year: Optional[int] = None,
                         end_year: Optional[int] = None) -> bool:
    """Best-effort year filter for mixed dataset trees.

    Supports filenames like YYYY-MM-DD.parquet, YYYY_embedded.parquet,
    and directory layouts containing a 4-digit year component.
    Files without a detectable year are kept.
    """
    if not year and not (start_year and end_year):
        return True

    parts = [p for p in relative_path.replace('\\', '/').split('/') if p]
    filename = parts[-1] if parts else relative_path
    candidate_year = None

    for part in reversed(parts[:-1]):
        if len(part) == 4 and part.isdigit():
            candidate_year = int(part)
            break

    if candidate_year is None:
        prefix = filename.split('-')[0]
        if len(prefix) == 4 and prefix.isdigit():
            candidate_year = int(prefix)
        else:
            prefix = filename.split('_')[0]
            if len(prefix) == 4 and prefix.isdigit():
                candidate_year = int(prefix)

    if candidate_year is None:
        return True

    if year:
        return candidate_year == year
    return start_year <= candidate_year <= end_year


def download_full_dataset_tree(s3_client, year: Optional[int] = None,
                               start_year: Optional[int] = None,
                               end_year: Optional[int] = None,
                               local_dir: Path = Path('datasets'),
                               force: bool = False):
    """Download the full datasets tree from R2, preserving subdirectories."""
    local_dir.mkdir(parents=True, exist_ok=True)

    print('🗂️ Downloading full datasets tree from R2...')
    if year:
        print(f'   Filtering for year: {year} where detectable')
    elif start_year and end_year:
        print(f'   Filtering for years: {start_year}-{end_year} where detectable')

    paginator = s3_client.get_paginator('list_objects_v2')
    downloaded_count = 0
    skipped_count = 0
    total_size = 0

    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=DATASETS_ROOT_PREFIX):
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue

            relative_path = key.replace(DATASETS_ROOT_PREFIX, '', 1)
            if not relative_path:
                continue
            if not _matches_year_filter(relative_path, year, start_year, end_year):
                continue

            local_file = local_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            remote_size = obj['Size']
            size_mb = remote_size / 1e6

            if not force and local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == remote_size:
                    skipped_count += 1
                    continue
                else:
                    print(f'  [{downloaded_count+1}] {relative_path} ({size_mb:.1f} MB) [size mismatch, re-downloading]')
            else:
                print(f'  [{downloaded_count+1}] {relative_path} ({size_mb:.1f} MB)')

            s3_client.download_file(R2_BUCKET, key, str(local_file))

            downloaded_count += 1
            total_size += remote_size

    print(f'✅ Full datasets tree: {downloaded_count} downloaded, {skipped_count} skipped (already exist), {total_size/1e9:.2f} GB transferred')
    return downloaded_count


def download_stock_data(s3_client, year: Optional[int] = None, 
                       start_year: Optional[int] = None, 
                       end_year: Optional[int] = None,
                       local_dir: Path = Path('datasets/Stock_Data_2min'),
                       force: bool = False):
    """Download 2min stock bar data from R2."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'📊 Downloading Stock_Data_2min from R2...')
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
                          local_dir: Path = Path('datasets/opt_trade_2min'),
                          force: bool = False):
    """Download 2min options trade data from R2."""
    print(f'📊 Downloading opt_trade_2min from R2...')
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
            
            filename = key.split('/')[-1]
            
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
            
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / filename
            remote_size = obj['Size']
            size_mb = remote_size / 1e6
            
            # Skip if file exists with matching size
            if not force and local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == remote_size:
                    skipped_count += 1
                    continue
                else:
                    print(f'  [{downloaded_count+1}] {filename} ({size_mb:.1f} MB) [re-downloading]')
            else:
                print(f'  [{downloaded_count+1}] {filename} ({size_mb:.1f} MB)')
            
            s3_client.download_file(R2_BUCKET, key, str(local_file))
            
            downloaded_count += 1
            total_size += remote_size
    
    print(f'✅ Options data: {downloaded_count} downloaded, {skipped_count} skipped (already exist), {total_size/1e9:.2f} GB transferred')
    return downloaded_count


def download_news_data(s3_client, year: Optional[int] = None,
                       start_year: Optional[int] = None,
                       end_year: Optional[int] = None,
                       local_dir: Path = Path('datasets/benzinga_embeddings/news_daily'),
                       force: bool = False):
    """Download Benzinga daily news embeddings from R2."""
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
            
            # Filter by year - filename format: YYYY-MM-DD.parquet
            if year or (start_year and end_year):
                try:
                    file_year = int(filename.split('-')[0])
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
        description='Download stock, VIX, options, news, or the full datasets tree from Cloudflare R2',
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
  
  # Download daily news embeddings for 2024
  python download_data.py --year 2024 --data-type news
  
  # Download all data types for 2024 (year filter applies to ALL types)
  python download_data.py --year 2024 --data-type all
  
  # Download all available data
  python download_data.py --data-type all

  # Download full datasets tree into ./datasets
  python download_data.py --data-type full
  
  # Force re-download (ignore existing files)
  python download_data.py --data-type all --force
        """
    )
    
    parser.add_argument('--year', type=int, help='Specific year to download (e.g., 2024) - applies to ALL data types')
    parser.add_argument('--start-year', type=int, help='Start year for range download - applies to ALL data types')
    parser.add_argument('--end-year', type=int, help='End year for range download - applies to ALL data types')
    parser.add_argument('--data-type', choices=['stock', 'vix', 'options', 'news', 'all', 'full'], default='all',
                       help='Type of data to download (default: all)')
    parser.add_argument('--stock-dir', type=Path, default=Path('datasets/Stock_Data_2min'),
                       help='Local directory for stock data (default: datasets/Stock_Data_2min)')
    parser.add_argument('--vix-dir', type=Path, default=Path('datasets/VIX'),
                       help='Local directory for VIX data (default: datasets/VIX)')
    parser.add_argument('--options-dir', type=Path, default=Path('datasets/opt_trade_2min'),
                       help='Local directory for options data (default: datasets/opt_trade_2min)')
    parser.add_argument('--news-dir', type=Path, default=Path('datasets/benzinga_embeddings/news_daily'),
                       help='Local directory for news data (default: datasets/benzinga_embeddings/news_daily)')
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

    if args.data_type == 'full':
        download_full_dataset_tree(
            s3,
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            local_dir=Path('datasets'),
            force=args.force
        )
        print('🎉 All downloads complete!')
        return
    
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
