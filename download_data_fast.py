#!/usr/bin/env python3
"""
Fast parallel download of datasets from Cloudflare R2 using rclone.

Usage:
    python download_data_fast.py --year 2024 --data-type stock
    python download_data_fast.py --year 2024 --data-type all
    python download_data_fast.py --start-year 2023 --end-year 2024 --data-type options
    python download_data_fast.py --data-type all --transfers 64

Prerequisites:
    1. Install rclone: curl https://rclone.org/install.sh | sudo bash
    2. Script auto-configures rclone for R2 on first run (no manual config needed)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


# R2 Configuration
R2_ACCOUNT_ID = "2a139e9393f803634546ad9d541d37b9"
R2_ACCESS_KEY = "fdfa18bf64b18c61bbee64fda98ca20b"
R2_SECRET_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "europe"
RCLONE_REMOTE = "r2mamba"

# R2 paths
DATASETS_ROOT = 'datasets'
PATHS = {
    'stock': f'{DATASETS_ROOT}/Stock_Data_2min',
    'vix': f'{DATASETS_ROOT}/VIX',
    'options': f'{DATASETS_ROOT}/opt_trade_2min',
    'news': f'{DATASETS_ROOT}/benzinga_embeddings/news_daily',
    'macro': f'{DATASETS_ROOT}/MACRO',
    'gdelt': f'{DATASETS_ROOT}/GDELT',
    'econ': f'{DATASETS_ROOT}/econ_calendar',
    'fundamentals': f'{DATASETS_ROOT}/fundamentals',
    'preprocessed': f'{DATASETS_ROOT}/preprocessed',
    'full': DATASETS_ROOT,
}


def check_rclone_installed() -> bool:
    """Check if rclone is installed."""
    try:
        subprocess.run(['rclone', 'version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_rclone_remote():
    """Configure rclone remote for R2 if not already configured."""
    # Check if remote already exists
    result = subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True)
    if f"{RCLONE_REMOTE}:" in result.stdout:
        print(f"✅ Rclone remote '{RCLONE_REMOTE}' already configured")
        return True
    
    print(f"🔧 Configuring rclone remote '{RCLONE_REMOTE}' for R2...")
    
    # Create rclone config via environment variables approach
    config_content = f"""[{RCLONE_REMOTE}]
type = s3
provider = Other
access_key_id = {R2_ACCESS_KEY}
secret_access_key = {R2_SECRET_KEY}
endpoint = {R2_ENDPOINT}
acl = private
no_check_bucket = true
"""
    
    # Find rclone config path
    result = subprocess.run(['rclone', 'config', 'file'], capture_output=True, text=True)
    config_path = result.stdout.strip().split('\n')[-1]
    
    # Append to config file
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'a') as f:
        f.write(config_content)
    
    print(f"✅ Rclone remote '{RCLONE_REMOTE}' configured successfully")
    return True


def build_include_filters(year: Optional[int] = None,
                          start_year: Optional[int] = None,
                          end_year: Optional[int] = None,
                          data_type: str = 'stock') -> List[str]:
    """Build rclone --include filters for year filtering.
    
    Returns empty list when no year filter is active — rclone copies
    the entire remote folder by default (no --include needed).
    """
    # No year filter → download everything in the folder
    if not year and not (start_year and end_year):
        return []
    
    filters = []
    if data_type in ['stock', 'options', 'news']:
        if year:
            filters.extend(['--include', f'{year}-*'])
        else:
            for y in range(start_year, end_year + 1):
                filters.extend(['--include', f'{y}-*'])
    # Other data types (vix, macro, gdelt, econ, fundamentals, preprocessed)
    # are small or don't have year-based filenames — download everything
    return filters


def run_rclone_sync(remote_path: str, local_path: str,
                    filters: List[str],
                    transfers: int = 32,
                    checkers: int = 16,
                    chunk_size: str = '64M',
                    dry_run: bool = False,
                    force: bool = False) -> bool:
    """Run rclone copy with parallel transfers."""
    
    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'rclone', 'copy',
        f'{RCLONE_REMOTE}:{R2_BUCKET}/{remote_path}',
        local_path,
        '--transfers', str(transfers),
        '--checkers', str(checkers),
        '--fast-list',
        '--progress',
        '--s3-chunk-size', chunk_size,
        '--stats', '5s',
        '--stats-one-line',
    ]
    
    # Add filters
    cmd.extend(filters)
    
    # Skip existing files (default behavior, use --force to override)
    if not force:
        cmd.append('--ignore-existing')
    
    if dry_run:
        cmd.append('--dry-run')
        print(f"🔍 Dry run: {' '.join(cmd)}")
    
    print(f"📥 Syncing {remote_path} -> {local_path}")
    print(f"   Transfers: {transfers}, Checkers: {checkers}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Rclone error: {e}")
        return False


def download_data_type(data_type: str,
                       local_dir: Path,
                       year: Optional[int] = None,
                       start_year: Optional[int] = None,
                       end_year: Optional[int] = None,
                       transfers: int = 32,
                       checkers: int = 16,
                       dry_run: bool = False,
                       force: bool = False) -> bool:
    """Download a specific data type with year filtering."""
    
    remote_path = PATHS[data_type]
    filters = build_include_filters(year, start_year, end_year, data_type)
    
    # Print what we're downloading
    if year:
        print(f"\n📊 Downloading {data_type} data for year {year}...")
    elif start_year and end_year:
        print(f"\n📊 Downloading {data_type} data for years {start_year}-{end_year}...")
    else:
        print(f"\n📊 Downloading all {data_type} data...")
    
    return run_rclone_sync(
        remote_path=remote_path,
        local_path=str(local_dir),
        filters=filters,
        transfers=transfers,
        checkers=checkers,
        dry_run=dry_run,
        force=force,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Fast parallel download of datasets or the full datasets tree from R2 using rclone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2024 stock data with 32 parallel transfers
  python download_data_fast.py --year 2024 --data-type stock
  
  # Download 2023-2024 for all data types with 64 transfers
  python download_data_fast.py --start-year 2023 --end-year 2024 --data-type all --transfers 64
  
  # Download only options data for 2024
  python download_data_fast.py --year 2024 --data-type options
  
  # Download all news embeddings
  python download_data_fast.py --data-type news

  # Mirror the full datasets tree into ./datasets
  python download_data_fast.py --data-type full
  
  # Dry run (see what would be downloaded)
  python download_data_fast.py --year 2024 --data-type all --dry-run
  
  # Force re-download (ignore existing files)
  python download_data_fast.py --year 2024 --data-type stock --force

Performance Tips:
  - Use --transfers 64 or higher for fast connections (10Gbps+)
  - Default --transfers 32 is good for most VPS instances
  - Options data has many small files, benefits from high --transfers
  - News data has few large files, --transfers 16 is sufficient
        """
    )
    
    parser.add_argument('--year', type=int, 
                        help='Specific year to download (e.g., 2024)')
    parser.add_argument('--start-year', type=int, 
                        help='Start year for range download')
    parser.add_argument('--end-year', type=int, 
                        help='End year for range download')
    parser.add_argument('--data-type', choices=['stock', 'vix', 'options', 'news', 'macro', 'gdelt', 'econ', 'fundamentals', 'preprocessed', 'all', 'full'], 
                        default='all', help='Type of data to download (default: all)')
    
    # Directory options
    parser.add_argument('--stock-dir', type=Path, default=Path('datasets/Stock_Data_2min'),
                        help='Local directory for stock data')
    parser.add_argument('--vix-dir', type=Path, default=Path('datasets/VIX'),
                        help='Local directory for VIX data')
    parser.add_argument('--options-dir', type=Path, default=Path('datasets/opt_trade_2min'),
                        help='Local directory for options data')
    parser.add_argument('--news-dir', type=Path, default=Path('datasets/benzinga_embeddings/news_daily'),
                        help='Local directory for news data')
    
    # Performance options
    parser.add_argument('--transfers', type=int, default=32,
                        help='Number of parallel file transfers (default: 32)')
    parser.add_argument('--checkers', type=int, default=16,
                        help='Number of parallel checkers (default: 16)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
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
    
    # Check rclone is installed
    if not check_rclone_installed():
        print("❌ rclone is not installed. Install it with:")
        print("   curl https://rclone.org/install.sh | sudo bash")
        sys.exit(1)
    
    # Setup rclone remote
    setup_rclone_remote()
    
    print(f"\n🚀 Starting parallel download (transfers={args.transfers}, checkers={args.checkers})")
    
    # Directory mapping
    dir_map = {
        'stock': args.stock_dir,
        'vix': args.vix_dir,
        'options': args.options_dir,
        'news': args.news_dir,
        'macro': Path('datasets/MACRO'),
        'gdelt': Path('datasets/GDELT'),
        'econ': Path('datasets/econ_calendar'),
        'fundamentals': Path('datasets/fundamentals'),
        'preprocessed': Path('datasets/preprocessed'),
        'full': Path('datasets'),
    }
    
    # Determine which data types to download
    if args.data_type == 'all':
        data_types = ['stock', 'vix', 'options', 'news', 'macro', 'gdelt', 'econ', 'fundamentals', 'preprocessed']
    else:
        data_types = [args.data_type]
    
    # Download each data type
    success = True
    for dtype in data_types:
        result = download_data_type(
            data_type=dtype,
            local_dir=dir_map[dtype],
            year=args.year,
            start_year=args.start_year,
            end_year=args.end_year,
            transfers=args.transfers,
            checkers=args.checkers,
            dry_run=args.dry_run,
            force=args.force,
        )
        if not result:
            success = False
    
    if success:
        print("\n🎉 All downloads complete!")
    else:
        print("\n⚠️ Some downloads may have failed. Check output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
