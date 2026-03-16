#!/usr/bin/env python3
"""
Build intraday realized volatility forecast features for stock 2-min bars.

Adds rolling RV at multiple horizons:
- rv_5m, rv_15m, rv_30m, rv_1h, rv_2h (backward-looking)
- rv_ratio_5m_30m, rv_ratio_15m_1h (vol regime)
- rv_acceleration (change in RV momentum)
- rv_zscore_1h (normalized RV level)

Output: Adds columns to existing Stock_Data_2min parquet files OR creates
        a separate SPY_intraday_rv directory with just SPY RV features.
"""
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Tuple
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_rolling_rv_pandas(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """Compute rolling RV features for a single ticker DataFrame (pandas)."""
    # Sort by timestamp
    df = df.sort_values('bar_timestamp').copy()
    
    # Log returns
    df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))
    df['log_ret_sq'] = df['log_ret'] ** 2
    
    # Rolling RV at different horizons (in number of 2-min bars)
    # 5m = 2.5 bars, 15m = 7.5 bars, 30m = 15 bars, 1h = 30 bars, 2h = 60 bars
    horizons = {
        'rv_5m': 3,
        'rv_15m': 8,
        'rv_30m': 15,
        'rv_1h': 30,
        'rv_2h': 60,
    }
    
    for name, window in horizons.items():
        # RV = sqrt(sum of squared returns) * sqrt(252 * 195) for annualization
        # But we keep it as raw for model to learn scaling
        df[name] = df['log_ret_sq'].rolling(window, min_periods=max(1, window//2)).sum().apply(np.sqrt)
    
    # RV ratios (short vs long term vol regime)
    df['rv_ratio_5m_30m'] = df['rv_5m'] / (df['rv_30m'] + 1e-10)
    df['rv_ratio_15m_1h'] = df['rv_15m'] / (df['rv_1h'] + 1e-10)
    df['rv_ratio_30m_2h'] = df['rv_30m'] / (df['rv_2h'] + 1e-10)
    
    # RV momentum and acceleration
    df['rv_change_5'] = df['rv_30m'].diff(5)
    df['rv_change_15'] = df['rv_30m'].diff(15)
    df['rv_acceleration'] = df['rv_change_5'].diff(5)
    
    # RV z-score (normalized level over 1h lookback)
    rv_mean = df['rv_30m'].rolling(30).mean()
    rv_std = df['rv_30m'].rolling(30).std()
    df['rv_zscore_1h'] = (df['rv_30m'] - rv_mean) / (rv_std + 1e-10)
    
    # RV rank (fast approximation of percentile using z-score sigmoid)
    # Maps z-score to 0-1 range, approximates percentile without slow rolling apply
    df['rv_rank_2h'] = 1 / (1 + np.exp(-df['rv_zscore_1h'].clip(-3, 3)))
    
    # Drop intermediate columns
    df = df.drop(columns=['log_ret', 'log_ret_sq'], errors='ignore')
    
    # Fill NaN from rolling windows
    rv_cols = [c for c in df.columns if c.startswith('rv_')]
    df[rv_cols] = df[rv_cols].fillna(0)
    
    return df


def compute_rolling_rv_polars(df: pl.DataFrame, price_col: str = 'close') -> pl.DataFrame:
    """Compute rolling RV features using Polars with native group_by (FAST)."""
    # Sort by ticker then timestamp for proper grouping
    df = df.sort(['ticker', 'bar_timestamp'])
    
    # All operations use .over('ticker') for per-ticker rolling
    df = df.with_columns([
        # Log returns (per ticker)
        (pl.col(price_col).log() - pl.col(price_col).shift(1).over('ticker').log()).alias('log_ret'),
    ])
    df = df.with_columns([
        (pl.col('log_ret') ** 2).alias('log_ret_sq'),
    ])
    
    # Rolling RV at different horizons (all vectorized per-ticker)
    df = df.with_columns([
        pl.col('log_ret_sq').rolling_sum(window_size=3, min_periods=1).over('ticker').sqrt().alias('rv_5m'),
        pl.col('log_ret_sq').rolling_sum(window_size=8, min_periods=4).over('ticker').sqrt().alias('rv_15m'),
        pl.col('log_ret_sq').rolling_sum(window_size=15, min_periods=7).over('ticker').sqrt().alias('rv_30m'),
        pl.col('log_ret_sq').rolling_sum(window_size=30, min_periods=15).over('ticker').sqrt().alias('rv_1h'),
        pl.col('log_ret_sq').rolling_sum(window_size=60, min_periods=30).over('ticker').sqrt().alias('rv_2h'),
    ])
    
    # RV ratios (vectorized)
    df = df.with_columns([
        (pl.col('rv_5m') / (pl.col('rv_30m') + 1e-10)).alias('rv_ratio_5m_30m'),
        (pl.col('rv_15m') / (pl.col('rv_1h') + 1e-10)).alias('rv_ratio_15m_1h'),
        (pl.col('rv_30m') / (pl.col('rv_2h') + 1e-10)).alias('rv_ratio_30m_2h'),
    ])
    
    # RV momentum (per ticker)
    df = df.with_columns([
        (pl.col('rv_30m') - pl.col('rv_30m').shift(5).over('ticker')).alias('rv_change_5'),
        (pl.col('rv_30m') - pl.col('rv_30m').shift(15).over('ticker')).alias('rv_change_15'),
    ])
    df = df.with_columns([
        (pl.col('rv_change_5') - pl.col('rv_change_5').shift(5).over('ticker')).alias('rv_acceleration'),
    ])
    
    # RV z-score (per ticker)
    df = df.with_columns([
        ((pl.col('rv_30m') - pl.col('rv_30m').rolling_mean(window_size=30).over('ticker')) /
         (pl.col('rv_30m').rolling_std(window_size=30).over('ticker') + 1e-10)).alias('rv_zscore_1h'),
    ])
    
    # RV rank (sigmoid of z-score)
    df = df.with_columns([
        (1 / (1 + (-pl.col('rv_zscore_1h').clip(-3, 3)).exp())).alias('rv_rank_2h'),
    ])
    
    # Drop intermediate columns and fill nulls
    df = df.drop(['log_ret', 'log_ret_sq'])
    rv_cols = [c for c in df.columns if c.startswith('rv_')]
    df = df.with_columns([pl.col(c).fill_null(0) for c in rv_cols])
    
    return df


def process_stock_day(input_path: Path, output_path: Path, spy_only: bool = False) -> Tuple[bool, str]:
    """Process one day of stock data, adding RV features."""
    try:
        if HAS_POLARS:
            df = pl.read_parquet(input_path)
            
            if spy_only:
                df = df.filter(pl.col('ticker') == 'SPY')
                if len(df) == 0:
                    return False, f"No SPY data: {input_path.name}"
            
            # Process ALL tickers at once using .over('ticker') - no Python loop!
            result = compute_rolling_rv_polars(df)
            result.write_parquet(output_path)
        else:
            df = pd.read_parquet(input_path)
            
            if spy_only:
                df = df[df['ticker'] == 'SPY']
                if len(df) == 0:
                    return False, f"No SPY data: {input_path.name}"
            
            # Process each ticker
            result_dfs = []
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker].copy()
                ticker_df = compute_rolling_rv_pandas(ticker_df)
                result_dfs.append(ticker_df)
            
            result = pd.concat(result_dfs, ignore_index=True)
            result.to_parquet(output_path, index=False)
        
        return True, f"OK: {input_path.name}"
    
    except Exception as e:
        return False, f"ERROR {input_path.name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Add intraday RV features to stock bars')
    parser.add_argument('--input-dir', type=str, default='datasets/Stock_Data_2min',
                        help='Input stock data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: overwrite input)')
    parser.add_argument('--spy-only', action='store_true',
                        help='Only process SPY (for cross-asset RV reference)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--force', action='store_true',
                        help='Re-process even if output exists')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / args.input_dir
    
    if args.output_dir:
        output_dir = script_dir / args.output_dir
    elif args.spy_only:
        output_dir = script_dir / 'datasets' / 'SPY_intraday_rv'
    else:
        output_dir = input_dir  # Overwrite in place
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    input_files = sorted(input_dir.glob('*.parquet'))
    
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        input_files = [f for f in input_files if f.stem >= args.start_date]
    
    # Check which files already have RV columns (resume support)
    to_process = []
    skipped = 0
    for f in input_files:
        out_path = output_dir / f.name
        needs_processing = True
        
        if out_path.exists() and not args.force:
            # Check if RV columns already exist in the file
            try:
                if HAS_POLARS:
                    cols = pl.read_parquet(out_path, n_rows=0).columns
                else:
                    cols = pd.read_parquet(out_path, columns=[]).columns.tolist()
                if 'rv_5m' in cols and 'rv_2h' in cols:
                    needs_processing = False
            except:
                pass
        
        if needs_processing:
            to_process.append(f)
        else:
            skipped += 1
    
    logger.info(f"Total files: {len(input_files)}")
    logger.info(f"Already done (skipping): {skipped}")
    logger.info(f"To process: {len(to_process)}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"SPY only: {args.spy_only}")
    logger.info(f"Using {'Polars' if HAS_POLARS else 'Pandas'}")
    
    if len(to_process) == 0:
        logger.info("Nothing to do!")
        return
    
    # Estimate time (rough: ~0.5s per file with Polars, ~2s with Pandas)
    est_per_file = 0.5 if HAS_POLARS else 2.0
    est_total = len(to_process) * est_per_file / args.workers
    logger.info(f"Estimated time: ~{est_total/60:.1f} minutes ({est_per_file:.1f}s/file, {args.workers} workers)")
    
    # Process in parallel with progress bar
    success = 0
    failed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for f in to_process:
            out_path = output_dir / f.name
            futures[executor.submit(process_stock_day, f, out_path, args.spy_only)] = f
        
        with tqdm(total=len(to_process), desc="Processing", unit="file") as pbar:
            for future in as_completed(futures):
                ok, msg = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    logger.warning(msg)
                pbar.update(1)
    
    elapsed = time.time() - start_time
    logger.info(f"Done in {elapsed/60:.1f} min: {success} succeeded, {failed} failed, {skipped} skipped")


if __name__ == '__main__':
    main()
