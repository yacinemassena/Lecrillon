#!/usr/bin/env python3
"""
Convert news embedding parquet timestamps to nanoseconds since epoch.

This makes timestamp handling consistent with stock/options data and avoids
datetime parsing overhead at training time.
"""

import argparse
from pathlib import Path
import pandas as pd


def convert_file(path: Path, dry_run: bool = False) -> bool:
    """Convert a single parquet file's timestamps to nanoseconds."""
    print(f"Processing {path.name}...")
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  ❌ Failed to read: {e}")
        return False
    
    if 'timestamp' not in df.columns:
        print(f"  ⚠️ No timestamp column, skipping")
        return False
    
    # Check current dtype
    ts_dtype = df['timestamp'].dtype
    print(f"  Current dtype: {ts_dtype}")
    
    # If already int64 (nanoseconds), skip
    if ts_dtype == 'int64':
        print(f"  ✓ Already int64 (nanoseconds), skipping")
        return True
    
    # Convert to datetime if needed, then to nanoseconds
    # Use utc=True to handle mixed timezones
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    
    # Convert to nanoseconds since epoch
    df['timestamp_ns'] = df['timestamp'].astype('int64')
    
    # Replace timestamp column
    df['timestamp'] = df['timestamp_ns']
    df = df.drop(columns=['timestamp_ns'])
    
    print(f"  Converted to int64 nanoseconds")
    print(f"  Sample: {df['timestamp'].iloc[0]} ns")
    
    if dry_run:
        print(f"  [DRY RUN] Would save to {path}")
    else:
        df.to_parquet(path, index=False)
        print(f"  ✓ Saved")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert news timestamps to nanoseconds')
    parser.add_argument('--news-dir', type=Path, default=Path('datasets/benzinga_embeddings/news'),
                        help='Directory containing news parquet files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without saving')
    args = parser.parse_args()
    
    news_dir = args.news_dir
    if not news_dir.exists():
        print(f"❌ News directory not found: {news_dir}")
        return 1
    
    parquet_files = sorted(news_dir.glob('*_embedded.parquet'))
    if not parquet_files:
        print(f"❌ No *_embedded.parquet files found in {news_dir}")
        return 1
    
    print(f"Found {len(parquet_files)} parquet files")
    if args.dry_run:
        print("🔍 DRY RUN MODE - no files will be modified\n")
    else:
        print("⚠️ Files will be modified in place\n")
    
    success = 0
    for path in parquet_files:
        if convert_file(path, args.dry_run):
            success += 1
    
    print(f"\n✓ Converted {success}/{len(parquet_files)} files")
    return 0


if __name__ == '__main__':
    exit(main())
