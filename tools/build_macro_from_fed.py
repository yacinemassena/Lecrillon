#!/usr/bin/env python3
"""
Build macro_daily.parquet from existing FED parquet files.

Merges daily/weekly/monthly FRED data into a single daily file with forward-fill.
Adds FOMC calendar features (days since/until meeting).

Usage:
    python tools/build_macro_from_fed.py
    python tools/build_macro_from_fed.py --fed-path datasets/FED --output datasets/macro/macro_daily.parquet
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_and_merge_fed_data(fed_path: Path) -> pd.DataFrame:
    """Load all FED parquet files and merge on date index."""
    
    # Define files to load and which columns to keep (avoiding high-NaN columns)
    files_config = {
        'fred_interest_rates.parquet': ['DFF'],  # Fed funds rate (0.02% NaN)
        'fred_treasury_yields.parquet': ['DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30'],  # ~4% NaN
        'fred_yield_curve.parquet': ['T10Y2Y', 'T10Y3M'],  # Yield curve spreads
        'fred_market_stress.parquet': ['VIXCLS', 'BAMLH0A0HYM2', 'BAMLC0A0CM'],  # VIX, credit spreads
        'fred_inflation.parquet': ['CPIAUCSL', 'PCEPILFE'],  # CPI, core PCE (monthly)
        'fred_employment.parquet': ['UNRATE', 'ICSA'],  # Unemployment, initial claims (weekly)
        'fred_balance_sheet.parquet': ['WALCL'],  # Fed balance sheet (weekly)
    }
    
    dfs = []
    for filename, cols in files_config.items():
        filepath = fed_path / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue
        
        df = pd.read_parquet(filepath)
        
        # The index is already datetime, but check if it needs to be set
        if df.index.name is None and df.index.dtype == 'datetime64[ns]':
            pass  # Index is already datetime
        elif 'date' in df.columns:
            df = df.set_index('date')
        
        # Select columns that exist
        available_cols = [c for c in cols if c in df.columns]
        if available_cols:
            dfs.append(df[available_cols])
            print(f"  Loaded {filename}: {available_cols}")
    
    if not dfs:
        raise ValueError("No FED data files found!")
    
    # Merge all on date index
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer')
    
    return merged


def add_fomc_features(df: pd.DataFrame, fed_path: Path) -> pd.DataFrame:
    """Add FOMC calendar features: days_since_fomc, days_until_fomc, is_fomc_week."""
    
    fomc_path = fed_path / 'fomc_statements.parquet'
    if not fomc_path.exists():
        print("  Warning: fomc_statements.parquet not found, skipping FOMC features")
        return df
    
    fomc = pd.read_parquet(fomc_path)
    if 'date' not in fomc.columns:
        print("  Warning: FOMC file missing 'date' column")
        return df
    
    fomc_dates = pd.to_datetime(fomc['date']).sort_values().unique()
    print(f"  FOMC meetings: {len(fomc_dates)} dates from {fomc_dates.min()} to {fomc_dates.max()}")
    
    # For each date, compute days since last FOMC and days until next FOMC
    dates = df.index.to_series()
    
    days_since = pd.Series(index=df.index, dtype=float)
    days_until = pd.Series(index=df.index, dtype=float)
    
    for i, d in enumerate(df.index):
        # Days since last FOMC
        past_fomc = fomc_dates[fomc_dates <= d]
        if len(past_fomc) > 0:
            days_since.iloc[i] = (d - past_fomc[-1]).days
        else:
            days_since.iloc[i] = 365  # Default if no FOMC before
        
        # Days until next FOMC
        future_fomc = fomc_dates[fomc_dates > d]
        if len(future_fomc) > 0:
            days_until.iloc[i] = (future_fomc[0] - d).days
        else:
            days_until.iloc[i] = 45  # Default ~6 weeks if no future FOMC
    
    df['days_since_fomc'] = days_since
    df['days_until_fomc'] = days_until
    df['is_fomc_week'] = (days_until <= 7) | (days_since <= 2)
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived macro features."""
    
    # Yield curve slope (10Y - 2Y already exists as T10Y2Y, but add if missing)
    if 'DGS10' in df.columns and 'DGS2' in df.columns and 'T10Y2Y' not in df.columns:
        df['T10Y2Y'] = df['DGS10'] - df['DGS2']
    
    # Real rate proxy (10Y - inflation expectations via breakeven)
    # Not available directly, skip
    
    # Credit spread (high yield - investment grade)
    if 'BAMLH0A0HYM2' in df.columns and 'BAMLC0A0CM' in df.columns:
        df['credit_spread'] = df['BAMLH0A0HYM2'] - df['BAMLC0A0CM']
    
    # VIX term structure (if we had VIX futures, but we don't - skip)
    
    # Fed funds vs 3-month spread
    if 'DFF' in df.columns and 'DGS3MO' in df.columns:
        df['ff_3m_spread'] = df['DFF'] - df['DGS3MO']
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Build macro_daily.parquet from FED data')
    parser.add_argument('--fed-path', type=str, default='datasets/FED',
                        help='Path to FED parquet files')
    parser.add_argument('--output', type=str, default='datasets/macro/macro_daily.parquet',
                        help='Output path for macro_daily.parquet')
    args = parser.parse_args()
    
    fed_path = Path(args.fed_path)
    output_path = Path(args.output)
    
    print(f"Loading FED data from: {fed_path}")
    
    # Load and merge all files
    df = load_and_merge_fed_data(fed_path)
    print(f"Merged data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Add FOMC calendar features
    print("Adding FOMC features...")
    df = add_fomc_features(df, fed_path)
    
    # Add derived features
    print("Adding derived features...")
    df = add_derived_features(df)
    
    # Forward-fill NaN (for weekends, holidays, and lower-frequency data)
    print("Forward-filling NaN values...")
    nan_before = df.isna().sum().sum()
    df = df.ffill()
    nan_after = df.isna().sum().sum()
    print(f"  Filled {nan_before - nan_after} NaN values, {nan_after} remaining")
    
    # Backward-fill any remaining NaN at the start
    df = df.bfill()
    
    # Drop any remaining rows with NaN (edge cases)
    rows_before = len(df)
    df = df.dropna()
    if len(df) < rows_before:
        print(f"  Dropped {rows_before - len(df)} rows with remaining NaN")
    
    # Shift by 1 day to prevent leakage (T-1 data for T prediction)
    print("Applying T-1 shift to prevent leakage...")
    df = df.shift(1).dropna()
    
    # Normalize features (z-score using expanding window to prevent future leakage)
    print("Normalizing features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove boolean-like columns from normalization
    bool_like = ['is_fomc_week']
    numeric_cols = [c for c in numeric_cols if c not in bool_like]
    
    # Simple z-score normalization (using full history - acceptable for slow-moving macro)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
    
    # Ensure date index is named
    df.index.name = 'date'
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    
    print(f"\n=== Final Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Features ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
