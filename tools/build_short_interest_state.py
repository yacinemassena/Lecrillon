#!/usr/bin/env python3
"""
Build short interest sector state from Polygon REST API data.

Creates bi-weekly sector aggregates for cross-attention conditioning:
- Sector-level days-to-cover
- Short interest concentration
- Short interest changes

Usage:
    python tools/build_short_interest_state.py

Output:
    datasets/short_interest/sector_state.parquet
    datasets/short_interest/sector_daily/YYYY-MM-DD.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sector mapping (simplified - in production use proper mapping)
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'META': 'Technology', 'AMD': 'Technology',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
}

def get_sector(ticker: str) -> str:
    """Get sector for ticker (simplified mapping)."""
    return SECTOR_MAP.get(ticker, 'Other')


def main():
    rest_data_path = Path('datasets/MACRO/rest_data')
    output_dir = Path('datasets/short_interest')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Building Short Interest Sector State")
    logger.info("=" * 60)
    
    # Load short interest data
    si_path = rest_data_path / 'short_interest' / 'short_interest.csv'
    df = pd.read_csv(si_path)
    logger.info(f"Loaded short interest: {len(df):,} rows, {df['ticker'].nunique():,} tickers")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['settlement_date'])
    df['sector'] = df['ticker'].apply(get_sector)
    
    # Aggregate by date and sector
    sector_agg = df.groupby(['date', 'sector']).agg({
        'short_interest': ['sum', 'mean', 'count'],
        'avg_daily_volume': 'sum',
        'days_to_cover': 'mean',
    }).reset_index()
    
    # Flatten column names
    sector_agg.columns = ['date', 'sector', 'total_short_interest', 'avg_short_interest', 
                          'n_tickers', 'total_volume', 'avg_days_to_cover']
    
    # Compute derived features
    sector_agg['short_ratio'] = sector_agg['total_short_interest'] / sector_agg['total_volume'].replace(0, np.nan)
    sector_agg['short_ratio'] = sector_agg['short_ratio'].fillna(0)
    
    # Compute changes vs previous period
    sector_agg = sector_agg.sort_values(['sector', 'date'])
    sector_agg['short_interest_pct_change'] = sector_agg.groupby('sector')['total_short_interest'].pct_change()
    sector_agg['days_to_cover_change'] = sector_agg.groupby('sector')['avg_days_to_cover'].diff()
    
    # Fill NaN
    sector_agg = sector_agg.fillna(0)
    
    # Save main file
    sector_agg.to_parquet(output_dir / 'sector_state.parquet', index=False)
    logger.info(f"Saved: {output_dir / 'sector_state.parquet'} ({len(sector_agg):,} records)")
    
    # Create daily files
    daily_dir = output_dir / 'sector_daily'
    daily_dir.mkdir(parents=True, exist_ok=True)
    
    for d, group in sector_agg.groupby(sector_agg['date'].dt.date):
        date_str = d.strftime('%Y-%m-%d')
        group.to_parquet(daily_dir / f'{date_str}.parquet', index=False)
    
    n_daily = len(sector_agg['date'].dt.date.unique())
    logger.info(f"Created {n_daily} daily files in {daily_dir}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info(f"Date range: {sector_agg['date'].min().date()} to {sector_agg['date'].max().date()}")
    logger.info(f"Sectors: {sector_agg['sector'].nunique()}")
    logger.info(f"Features: total_short_interest, avg_days_to_cover, short_ratio, short_interest_pct_change")


if __name__ == '__main__':
    main()
