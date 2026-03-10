#!/usr/bin/env python3
"""
Upload opt_trade_1sec and benzinga_embeddings datasets to Cloudflare R2.

Usage:
    python tools/upload_datasets.py --data-type options
    python tools/upload_datasets.py --data-type news
    python tools/upload_datasets.py --data-type both
    python tools/upload_datasets.py --data-type both --year 2024
"""

import argparse
import os
import glob
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional


# R2 Configuration
R2_ACCOUNT_ID = "2a139e9393f803634546ad9d541d37b9"
R2_ACCESS_KEY_ID = "fdfa18bf64b18c61bbee64fda98ca20b"
R2_SECRET_ACCESS_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET_NAME = "europe"

# Dataset paths (relative to project root)
LOCAL_OPTIONS_DIR = "datasets/opt_trade_1sec"
LOCAL_NEWS_DIR = "datasets/benzinga_embeddings/news"

# R2 prefixes (matching data loader expectations)
R2_OPTIONS_PREFIX = "datasets/opt_trade_1sec/"
R2_NEWS_PREFIX = "datasets/benzinga_embeddings/"


def get_s3_client():
    """Initialize S3 client for R2."""
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )


def upload_options_data(s3_client, project_dir: Path, year: Optional[int] = None,
                        start_year: Optional[int] = None, end_year: Optional[int] = None):
    """Upload option trade data (both contract and underlying bars)."""
    print("🔄 Uploading opt_trade_1sec data to R2...")
    
    options_dir = project_dir / LOCAL_OPTIONS_DIR
    if not options_dir.exists():
        print(f"❌ Options directory not found: {options_dir}")
        return 0
    
    subdirs = ['option_contract_bars_1s', 'option_underlying_bars_1s']
    total_uploaded = 0
    total_skipped = 0
    
    for subdir in subdirs:
        subdir_path = options_dir / subdir
        if not subdir_path.exists():
            print(f"  ⚠️ Subdirectory not found: {subdir}")
            continue
        
        parquet_files = sorted(subdir_path.glob('*.parquet'))
        print(f"  📁 {subdir}: {len(parquet_files)} files found")
        
        for file_path in parquet_files:
            filename = file_path.name
            
            # Filter by year if specified
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
            
            s3_key = f"{R2_OPTIONS_PREFIX}{subdir}/{filename}"
            
            # Check if file already exists in R2
            try:
                head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
                remote_size = head['ContentLength']
                local_size = file_path.stat().st_size
                if remote_size == local_size:
                    total_skipped += 1
                    continue
            except ClientError:
                pass  # File doesn't exist, proceed with upload
            
            size_mb = file_path.stat().st_size / 1e6
            print(f"    [{total_uploaded+1}] {subdir}/{filename} ({size_mb:.1f} MB)")
            
            try:
                s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
                total_uploaded += 1
            except ClientError as e:
                print(f"    ❌ Failed: {e}")
    
    print(f"✅ Options data: {total_uploaded} uploaded, {total_skipped} skipped (already exist)")
    return total_uploaded


def upload_news_data(s3_client, project_dir: Path, year: Optional[int] = None,
                     start_year: Optional[int] = None, end_year: Optional[int] = None):
    """Upload Benzinga news embeddings."""
    print("🔄 Uploading benzinga_embeddings data to R2...")
    
    news_dir = project_dir / LOCAL_NEWS_DIR
    if not news_dir.exists():
        print(f"❌ News directory not found: {news_dir}")
        return 0
    
    parquet_files = sorted(news_dir.glob('*_embedded.parquet'))
    print(f"  📁 Found {len(parquet_files)} embedding files")
    
    total_uploaded = 0
    total_skipped = 0
    
    for file_path in parquet_files:
        filename = file_path.name
        
        # Extract year from filename (e.g., "2024_embedded.parquet" -> 2024)
        try:
            file_year = int(filename.split('_')[0])
            if year and file_year != year:
                continue
            if start_year and end_year:
                if file_year < start_year or file_year > end_year:
                    continue
        except (ValueError, IndexError):
            continue
        
        # Upload directly to benzinga_embeddings/ (data loader expects {year}_embedded.parquet)
        s3_key = f"{R2_NEWS_PREFIX}{filename}"
        
        # Check if file already exists
        try:
            head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
            remote_size = head['ContentLength']
            local_size = file_path.stat().st_size
            if remote_size == local_size:
                total_skipped += 1
                continue
        except ClientError:
            pass
        
        size_gb = file_path.stat().st_size / 1e9
        print(f"    [{total_uploaded+1}] {filename} ({size_gb:.2f} GB)")
        
        try:
            s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
            total_uploaded += 1
        except ClientError as e:
            print(f"    ❌ Failed: {e}")
    
    print(f"✅ News data: {total_uploaded} uploaded, {total_skipped} skipped (already exist)")
    return total_uploaded


def main():
    parser = argparse.ArgumentParser(
        description='Upload datasets to Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all options data
  python tools/upload_datasets.py --data-type options
  
  # Upload only 2024 options data
  python tools/upload_datasets.py --data-type options --year 2024
  
  # Upload news embeddings for 2023-2024
  python tools/upload_datasets.py --data-type news --start-year 2023 --end-year 2024
  
  # Upload everything
  python tools/upload_datasets.py --data-type both
        """
    )
    
    parser.add_argument('--data-type', choices=['options', 'news', 'both'], default='both',
                       help='Type of data to upload (default: both)')
    parser.add_argument('--year', type=int, help='Specific year to upload')
    parser.add_argument('--start-year', type=int, help='Start year for range upload')
    parser.add_argument('--end-year', type=int, help='End year for range upload')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.year and (args.start_year or args.end_year):
        parser.error('Cannot specify both --year and --start-year/--end-year')
    
    if (args.start_year and not args.end_year) or (args.end_year and not args.start_year):
        parser.error('Must specify both --start-year and --end-year for range upload')
    
    # Get project directory
    project_dir = Path(__file__).parent.parent.resolve()
    
    print(f"📂 Project directory: {project_dir}")
    print("🔌 Connecting to R2...")
    
    s3 = get_s3_client()
    print("✅ Connected to R2\n")
    
    if args.data_type in ['options', 'both']:
        upload_options_data(s3, project_dir, args.year, args.start_year, args.end_year)
        print()
    
    if args.data_type in ['news', 'both']:
        upload_news_data(s3, project_dir, args.year, args.start_year, args.end_year)
        print()
    
    print("🎉 All uploads complete!")


if __name__ == "__main__":
    main()
