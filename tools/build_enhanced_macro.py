#!/usr/bin/env python3
"""
Build enhanced macro conditioning data combining:
1. FED data (yields, rates, spreads, balance sheet)
2. Cross-asset momentum (gold, bonds via FRED)
3. Credit spreads (high-yield, investment grade)
4. Existing sector fundamentals

Output: macro_daily_enhanced.parquet with all features merged on date
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fed_data(fed_dir: Path) -> pd.DataFrame:
    """Load and merge all FED parquet files into a single daily DataFrame."""
    fed_dfs = []
    
    # Treasury yields - critical for vol regime
    yields_path = fed_dir / 'treasury_yields.parquet'
    if yields_path.exists():
        df = pd.read_parquet(yields_path)
        df.index = pd.to_datetime(df.index)
        # Add derived features
        df['yield_2s10s'] = df['DGS10'] - df['DGS2']  # 2s10s spread
        df['yield_3m10y'] = df['DGS10'] - df['DGS3MO']  # 3m10y spread
        df['yield_curve_steepness'] = df['DGS30'] - df['DGS2']  # long-short spread
        fed_dfs.append(df)
        logger.info(f"Loaded treasury yields: {len(df)} rows, {list(df.columns)}")
    
    # Yield curve inversions (direct from FRED)
    yc_path = fed_dir / 'yield_curve.parquet'
    if yc_path.exists():
        df = pd.read_parquet(yc_path)
        df.index = pd.to_datetime(df.index)
        # Add inversion flags
        df['yc_2s10s_inverted'] = (df['T10Y2Y'] < 0).astype(float)
        df['yc_3m10y_inverted'] = (df['T10Y3M'] < 0).astype(float)
        fed_dfs.append(df)
        logger.info(f"Loaded yield curve: {len(df)} rows")
    
    # Fed funds rate and policy
    rates_path = fed_dir / 'rates.parquet'
    if rates_path.exists():
        df = pd.read_parquet(rates_path)
        df.index = pd.to_datetime(df.index)
        # Effective fed funds rate spread vs target
        df['fed_funds_vs_target'] = df['DFF'] - df['DFEDTARU']
        fed_dfs.append(df)
        logger.info(f"Loaded rates: {len(df)} rows")
    
    # Inflation expectations
    infl_path = fed_dir / 'inflation.parquet'
    if infl_path.exists():
        df = pd.read_parquet(infl_path)
        df.index = pd.to_datetime(df.index)
        # YoY changes
        for col in df.columns:
            df[f'{col}_yoy'] = df[col].pct_change(12) * 100  # monthly data, 12-month change
        fed_dfs.append(df)
        logger.info(f"Loaded inflation: {len(df)} rows")
    
    # Employment
    emp_path = fed_dir / 'employment.parquet'
    if emp_path.exists():
        df = pd.read_parquet(emp_path)
        df.index = pd.to_datetime(df.index)
        # ICSA (jobless claims) momentum
        df['ICSA_4wma'] = df['ICSA'].rolling(4).mean()
        df['ICSA_vs_4wma'] = df['ICSA'] / df['ICSA_4wma'] - 1
        fed_dfs.append(df)
        logger.info(f"Loaded employment: {len(df)} rows")
    
    # Balance sheet
    bs_path = fed_dir / 'balance_sheet.parquet'
    if bs_path.exists():
        df = pd.read_parquet(bs_path)
        df.index = pd.to_datetime(df.index)
        # Total assets growth
        df['WALCL_mom'] = df['WALCL'].pct_change(4) * 100  # 4-week change
        fed_dfs.append(df)
        logger.info(f"Loaded balance sheet: {len(df)} rows")
    
    if not fed_dfs:
        logger.warning("No FED data found!")
        return pd.DataFrame()
    
    # Merge all on date index
    merged = fed_dfs[0]
    for df in fed_dfs[1:]:
        merged = merged.join(df, how='outer', rsuffix='_dup')
        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]
    
    # Forward fill missing values (most FED data is weekly/monthly)
    merged = merged.ffill()
    
    logger.info(f"Merged FED data: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def download_cross_asset_data(start_date: str = '2000-01-01') -> pd.DataFrame:
    """Download cross-asset data from FRED API."""
    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()
    
    import os
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        logger.warning("FRED_API_KEY not set. Skipping cross-asset download.")
        return pd.DataFrame()
    
    fred = Fred(api_key=api_key)
    
    series = {
        # Gold
        'GOLDAMGBD228NLBM': 'gold_price',
        # Credit spreads
        'BAMLH0A0HYM2': 'hy_spread',  # High yield spread
        'BAMLC0A0CM': 'ig_spread',    # Investment grade spread
        'BAMLH0A0HYM2EY': 'hy_yield', # High yield effective yield
        # Bond ETF proxies (if available)
        'DFII10': 'tips_10y',         # 10-year TIPS
        'T10YIE': 'breakeven_10y',    # 10-year breakeven inflation
        # Market stress
        'VIXCLS': 'vix_close',
        'STLFSI4': 'stl_fsi',         # St. Louis Fed Financial Stress Index
        # Dollar index
        'DTWEXBGS': 'dollar_index',
    }
    
    dfs = []
    for code, name in series.items():
        try:
            s = fred.get_series(code, observation_start=start_date)
            df = s.to_frame(name=name)
            dfs.append(df)
            logger.info(f"Downloaded {name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to download {code}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer')
    
    # Add derived features
    if 'gold_price' in merged.columns:
        merged['gold_mom_5d'] = merged['gold_price'].pct_change(5) * 100
        merged['gold_mom_20d'] = merged['gold_price'].pct_change(20) * 100
    
    if 'hy_spread' in merged.columns and 'ig_spread' in merged.columns:
        merged['credit_spread_diff'] = merged['hy_spread'] - merged['ig_spread']
        merged['hy_spread_z'] = (merged['hy_spread'] - merged['hy_spread'].rolling(252).mean()) / merged['hy_spread'].rolling(252).std()
    
    if 'dollar_index' in merged.columns:
        merged['dollar_mom_5d'] = merged['dollar_index'].pct_change(5) * 100
        merged['dollar_mom_20d'] = merged['dollar_index'].pct_change(20) * 100
    
    merged = merged.ffill()
    logger.info(f"Cross-asset data: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def load_existing_macro(macro_path: Path) -> pd.DataFrame:
    """Load existing macro_daily.parquet if it exists."""
    if not macro_path.exists():
        logger.info("No existing macro_daily.parquet found")
        return pd.DataFrame()
    
    df = pd.read_parquet(macro_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        df.index = pd.to_datetime(df.index)
    
    logger.info(f"Loaded existing macro: {len(df)} rows, {len(df.columns)} columns")
    return df


def build_enhanced_macro(
    fed_dir: Path,
    existing_macro_path: Path,
    output_path: Path,
    download_cross_asset: bool = False,
) -> pd.DataFrame:
    """Build enhanced macro dataset combining all sources."""
    
    dfs = []
    
    # 1. Load FED data
    fed_df = load_fed_data(fed_dir)
    if len(fed_df) > 0:
        dfs.append(fed_df)
    
    # 2. Load existing macro (sector fundamentals)
    existing_df = load_existing_macro(existing_macro_path)
    if len(existing_df) > 0:
        dfs.append(existing_df)
    
    # 3. Download cross-asset data (optional, needs API key)
    if download_cross_asset:
        cross_df = download_cross_asset_data()
        if len(cross_df) > 0:
            dfs.append(cross_df)
    
    if not dfs:
        logger.error("No data sources available!")
        return pd.DataFrame()
    
    # Merge all DataFrames on date
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer', rsuffix='_dup')
        merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]
    
    # Forward fill and handle NaN
    merged = merged.ffill()
    
    # Drop rows with too many NaN (early dates)
    min_valid = len(merged.columns) * 0.5
    merged = merged.dropna(thresh=int(min_valid))
    
    # Fill remaining NaN with 0
    merged = merged.fillna(0)
    
    # Sort by date
    merged = merged.sort_index()
    
    # Shift by 1 day to avoid lookahead (T-1 data for day T prediction)
    merged = merged.shift(1).dropna(how='all')
    
    # Convert index to date column for compatibility
    merged = merged.reset_index()
    merged = merged.rename(columns={'index': 'date'})
    merged['date'] = pd.to_datetime(merged['date']).dt.date
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    
    logger.info(f"Saved enhanced macro: {output_path}")
    logger.info(f"  Shape: {merged.shape}")
    logger.info(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    logger.info(f"  Columns: {list(merged.columns)}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Build enhanced macro conditioning data')
    parser.add_argument('--fed-dir', type=str, default='datasets/FED',
                        help='Directory containing FED parquet files')
    parser.add_argument('--existing-macro', type=str, default='datasets/MACRO/macro_daily.parquet',
                        help='Path to existing macro_daily.parquet')
    parser.add_argument('--output', type=str, default='datasets/MACRO/macro_daily_enhanced.parquet',
                        help='Output path for enhanced macro data')
    parser.add_argument('--download-cross-asset', action='store_true',
                        help='Download cross-asset data from FRED (requires FRED_API_KEY)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    fed_dir = script_dir / args.fed_dir
    existing_macro = script_dir / args.existing_macro
    output_path = script_dir / args.output
    
    build_enhanced_macro(
        fed_dir=fed_dir,
        existing_macro_path=existing_macro,
        output_path=output_path,
        download_cross_asset=args.download_cross_asset,
    )


if __name__ == '__main__':
    main()
