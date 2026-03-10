#!/usr/bin/env python3
"""
Convert yearly news parquet files to daily parquet files.
Matches the naming convention of stock/options data: YYYY-MM-DD.parquet
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys

def convert_news_to_daily(input_dir: str, output_dir: str, years: list = None):
    """Convert yearly news parquet files to daily files.
    
    Args:
        input_dir: Directory containing YYYY_embedded.parquet files
        output_dir: Directory to write YYYY-MM-DD.parquet files
        years: List of years to process (None = all available)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all yearly files
    yearly_files = sorted(input_path.glob("*_embedded.parquet"))
    
    if not yearly_files:
        print(f"No yearly parquet files found in {input_dir}")
        return
    
    print(f"Found {len(yearly_files)} yearly files")
    
    total_days = 0
    total_articles = 0
    
    for yearly_file in yearly_files:
        year = int(yearly_file.stem.split('_')[0])
        
        if years and year not in years:
            print(f"Skipping {year} (not in requested years)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {yearly_file.name}...")
        
        try:
            df = pd.read_parquet(yearly_file)
            print(f"  Loaded {len(df):,} articles")
        except Exception as e:
            print(f"  ERROR loading {yearly_file}: {e}")
            continue
        
        # Check timestamp column
        if 'timestamp' not in df.columns:
            print(f"  ERROR: No 'timestamp' column found. Columns: {list(df.columns)}")
            continue
        
        # Filter invalid timestamps (negative or zero values)
        valid_mask = df['timestamp'] > 0
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"  Filtering {invalid_count:,} invalid timestamps")
            df = df[valid_mask].copy()
        
        # Auto-detect timestamp unit based on magnitude
        # 19 digits = nanoseconds, 16 digits = microseconds, 13 digits = milliseconds
        sample_ts = df['timestamp'].iloc[0]
        ts_len = len(str(sample_ts))
        if ts_len >= 19:
            unit = 'ns'
        elif ts_len >= 16:
            unit = 'us'
        elif ts_len >= 13:
            unit = 'ms'
        else:
            unit = 's'
        print(f"  Timestamp unit detected: {unit} (sample: {sample_ts})")
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
        df['date'] = df['datetime'].dt.date
        
        # Group by date and save
        grouped = df.groupby('date')
        days_written = 0
        
        for date, group in grouped:
            date_str = date.strftime('%Y-%m-%d')
            output_file = output_path / f"{date_str}.parquet"
            
            # Drop the temporary columns before saving
            group_to_save = group.drop(columns=['datetime', 'date'])
            group_to_save.to_parquet(output_file, index=False)
            
            days_written += 1
            total_articles += len(group)
        
        total_days += days_written
        print(f"  Written {days_written} daily files")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_days} daily files, {total_articles:,} total articles")
    print(f"Output: {output_path}")


def verify_timestamps(input_dir: str, output_dir: str, sample_date: str = None):
    """Verify that news timestamps align with stock/options data.
    
    Args:
        input_dir: Directory containing daily news parquet files
        output_dir: Stock or options directory to compare against
        sample_date: Specific date to check (YYYY-MM-DD), or None for random
    """
    input_path = Path(input_dir)
    compare_path = Path(output_dir)
    
    # Find a common date
    news_files = {f.stem for f in input_path.glob("*.parquet")}
    compare_files = {f.stem for f in compare_path.glob("*.parquet")}
    
    common_dates = news_files & compare_files
    
    if not common_dates:
        print("No common dates found between news and comparison data")
        return
    
    if sample_date and sample_date in common_dates:
        check_date = sample_date
    else:
        check_date = sorted(common_dates)[-1]  # Most recent common date
    
    print(f"Checking date: {check_date}")
    
    # Load news
    news_df = pd.read_parquet(input_path / f"{check_date}.parquet")
    news_df['datetime'] = pd.to_datetime(news_df['timestamp'], unit='ns', utc=True)
    
    print(f"\nNews articles: {len(news_df)}")
    print(f"  First: {news_df['datetime'].min()}")
    print(f"  Last:  {news_df['datetime'].max()}")
    
    # Load comparison (stock or options)
    compare_df = pd.read_parquet(compare_path / f"{check_date}.parquet")
    
    # Try to find timestamp column
    ts_col = None
    for col in ['timestamp', 'bar_timestamp', 'datetime']:
        if col in compare_df.columns:
            ts_col = col
            break
    
    if ts_col:
        if compare_df[ts_col].dtype == 'int64':
            # Nanoseconds
            compare_df['datetime'] = pd.to_datetime(compare_df[ts_col], unit='ns', utc=True)
        else:
            compare_df['datetime'] = pd.to_datetime(compare_df[ts_col], utc=True)
        
        print(f"\nComparison data ({compare_path.name}): {len(compare_df)} rows")
        print(f"  First: {compare_df['datetime'].min()}")
        print(f"  Last:  {compare_df['datetime'].max()}")
    else:
        print(f"\nComparison data columns: {list(compare_df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert yearly news to daily parquet files")
    parser.add_argument("--input", type=str, default="datasets/benzinga_embeddings/news",
                        help="Input directory with yearly files")
    parser.add_argument("--output", type=str, default="datasets/benzinga_embeddings/news_daily",
                        help="Output directory for daily files")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process (default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify timestamps against stock data")
    parser.add_argument("--verify-path", type=str, default="datasets/Stock_Data_1s",
                        help="Path to compare timestamps against")
    parser.add_argument("--verify-date", type=str, default=None,
                        help="Specific date to verify (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_timestamps(args.output, args.verify_path, args.verify_date)
    else:
        convert_news_to_daily(args.input, args.output, args.years)
