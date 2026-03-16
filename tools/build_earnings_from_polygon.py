#!/usr/bin/env python3
"""
Build earnings dataset from Polygon REST API data (rest_data/).

Creates two outputs for the Mamba architecture:
1. Daily earnings tokens (top 50 mega-caps) → news Mamba stream
2. Sector aggregates (all companies) → cross-attention state at checkpoints

Much faster than SEC EDGAR parsing - uses pre-cleaned CSVs.

Usage:
    python tools/build_earnings_from_polygon.py
    python tools/build_earnings_from_polygon.py --top-n 50

Output:
    datasets/earnings/earnings_tokens/YYYY-MM-DD.parquet  (daily mega-cap events)
    datasets/earnings/sector_state.parquet                 (sector aggregates)
    datasets/earnings/company_fundamentals.parquet         (all companies, all metrics)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GICS sector mapping (approximate based on first letter of ticker or manual mapping)
# In production, you'd use a proper CIK->sector mapping
MEGA_CAP_SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
    'TSLA': 'Consumer Discretionary', 'BRK.A': 'Financials', 'BRK.B': 'Financials',
    'UNH': 'Health Care', 'JNJ': 'Health Care', 'XOM': 'Energy', 'JPM': 'Financials',
    'V': 'Financials', 'PG': 'Consumer Staples', 'MA': 'Financials', 'HD': 'Consumer Discretionary',
    'CVX': 'Energy', 'MRK': 'Health Care', 'ABBV': 'Health Care', 'LLY': 'Health Care',
    'PEP': 'Consumer Staples', 'KO': 'Consumer Staples', 'COST': 'Consumer Staples',
    'AVGO': 'Technology', 'TMO': 'Health Care', 'WMT': 'Consumer Staples', 'MCD': 'Consumer Discretionary',
    'CSCO': 'Technology', 'ACN': 'Technology', 'ABT': 'Health Care', 'DHR': 'Health Care',
    'NEE': 'Utilities', 'VZ': 'Communication Services', 'ADBE': 'Technology', 'NKE': 'Consumer Discretionary',
    'CMCSA': 'Communication Services', 'TXN': 'Technology', 'PM': 'Consumer Staples', 'WFC': 'Financials',
    'DIS': 'Communication Services', 'BMY': 'Health Care', 'COP': 'Energy', 'RTX': 'Industrials',
    'ORCL': 'Technology', 'QCOM': 'Technology', 'T': 'Communication Services', 'MS': 'Financials',
    'UNP': 'Industrials', 'INTC': 'Technology', 'AMD': 'Technology', 'CRM': 'Technology',
}

DEFAULT_SECTOR = 'Unknown'


def parse_tickers(tickers_str: str) -> List[str]:
    """Parse tickers from string like "['AAPL']" or "['GME', 'GMEw']"."""
    if pd.isna(tickers_str):
        return []
    try:
        # Remove brackets and quotes, split by comma
        cleaned = tickers_str.strip("[]'\"").replace("'", "").replace('"', '')
        tickers = [t.strip() for t in cleaned.split(',') if t.strip()]
        return tickers
    except:
        return []


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return MEGA_CAP_SECTORS.get(ticker, DEFAULT_SECTOR)


def load_financial_statements(rest_data_path: Path) -> pd.DataFrame:
    """Load and merge all financial statement data."""
    logger.info("Loading financial statements from Polygon REST data...")
    
    dfs = []
    
    # Income statements
    income_path = rest_data_path / 'income_statements' / 'income_statements.csv'
    if income_path.exists():
        df = pd.read_csv(income_path, on_bad_lines='skip')
        df['source'] = 'income'
        dfs.append(df)
        logger.info(f"  Loaded income_statements: {len(df):,} rows")
    
    # Balance sheets
    balance_path = rest_data_path / 'balance_sheets' / 'balance_sheets.csv'
    if balance_path.exists():
        df = pd.read_csv(balance_path, on_bad_lines='skip')
        df['source'] = 'balance'
        dfs.append(df)
        logger.info(f"  Loaded balance_sheets: {len(df):,} rows")
    
    # Cash flow statements
    cashflow_path = rest_data_path / 'cash_flow_statements' / 'cash_flow_statements.csv'
    if cashflow_path.exists():
        df = pd.read_csv(cashflow_path, on_bad_lines='skip')
        df['source'] = 'cashflow'
        dfs.append(df)
        logger.info(f"  Loaded cash_flow_statements: {len(df):,} rows")
    
    if not dfs:
        logger.error("No financial statement files found!")
        return pd.DataFrame()
    
    # Merge on common columns
    # Each source has: tickers, cik, period_end, filing_date, fiscal_quarter, fiscal_year, timeframe
    common_cols = ['tickers', 'cik', 'period_end', 'filing_date', 'fiscal_quarter', 'fiscal_year', 'timeframe']
    
    # Process each dataframe
    processed = []
    for df in dfs:
        # Extract primary ticker
        df['ticker'] = df['tickers'].apply(lambda x: parse_tickers(x)[0] if parse_tickers(x) else None)
        df = df.dropna(subset=['ticker', 'filing_date'])
        df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        df = df.dropna(subset=['filing_date'])
        processed.append(df)
    
    # Merge all sources
    merged = processed[0]
    for df in processed[1:]:
        # Merge on common keys, keeping all columns
        merged = pd.merge(
            merged, df,
            on=['ticker', 'cik', 'filing_date', 'fiscal_year', 'fiscal_quarter'],
            how='outer',
            suffixes=('', '_dup')
        )
        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]
    
    logger.info(f"Merged financial data: {len(merged):,} rows, {merged['ticker'].nunique():,} tickers")
    return merged


def build_company_fundamentals(merged_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Build company fundamentals parquet."""
    logger.info("Building company fundamentals...")
    
    # Select key columns
    key_cols = [
        'ticker', 'cik', 'filing_date', 'fiscal_year', 'fiscal_quarter', 'timeframe',
        # Income statement
        'revenue', 'cost_of_revenue', 'gross_profit', 'operating_income', 'net_income_loss',
        'basic_earnings_per_share', 'diluted_earnings_per_share', 'ebitda',
        # Balance sheet  
        'total_assets', 'total_liabilities', 'total_equity', 'cash_and_equivalents',
        'total_current_assets', 'total_current_liabilities',
        # Cash flow
        'net_cash_flow_from_operating_activities', 'net_cash_flow_from_investing_activities',
        'net_cash_flow_from_financing_activities',
    ]
    
    # Keep only columns that exist
    available_cols = [c for c in key_cols if c in merged_df.columns]
    df = merged_df[available_cols].copy()
    
    # Add sector
    df['sector'] = df['ticker'].apply(get_sector)
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'filing_date'])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved company fundamentals: {output_path} ({len(df):,} rows)")
    
    return df


def build_earnings_tokens(
    fundamentals_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 50,
) -> None:
    """Build daily earnings token files for top N companies by filing frequency."""
    logger.info(f"Building daily earnings tokens for top {top_n} companies...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find top N by number of filings (proxy for importance/coverage)
    ticker_counts = fundamentals_df.groupby('ticker').size().sort_values(ascending=False)
    top_tickers = set(ticker_counts.head(top_n).index.tolist())
    logger.info(f"Top {top_n} tickers by coverage: {sorted(top_tickers)[:10]}...")
    
    # Filter to top tickers
    df = fundamentals_df[fundamentals_df['ticker'].isin(top_tickers)].copy()
    
    # Features for embedding
    EMBED_FEATURES = [
        'revenue', 'gross_profit', 'operating_income', 'net_income_loss',
        'basic_earnings_per_share', 'ebitda',
        'total_assets', 'total_equity', 'cash_and_equivalents',
    ]
    
    # Keep only available features
    available_features = [f for f in EMBED_FEATURES if f in df.columns]
    
    # Compute derived features
    if 'gross_profit' in df.columns and 'revenue' in df.columns:
        df['gross_margin'] = df['gross_profit'] / df['revenue'].replace(0, np.nan)
    if 'net_income_loss' in df.columns and 'revenue' in df.columns:
        df['net_margin'] = df['net_income_loss'] / df['revenue'].replace(0, np.nan)
    if 'net_income_loss' in df.columns and 'total_assets' in df.columns:
        df['roa'] = df['net_income_loss'] / df['total_assets'].replace(0, np.nan)
    
    derived_features = ['gross_margin', 'net_margin', 'roa']
    available_derived = [f for f in derived_features if f in df.columns]
    
    # Compute YoY changes
    df = df.sort_values(['ticker', 'filing_date'])
    for feat in ['revenue', 'net_income_loss', 'basic_earnings_per_share']:
        if feat in df.columns:
            df[f'{feat}_yoy'] = df.groupby('ticker')[feat].pct_change(periods=4)  # ~1 year for quarterly
    
    yoy_features = [f'{f}_yoy' for f in ['revenue', 'net_income_loss', 'basic_earnings_per_share'] if f in df.columns]
    
    # All embedding columns
    all_embed_cols = available_features + available_derived + yoy_features
    
    # Fill NaN and normalize
    for col in all_embed_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Z-score normalization
    stats = {}
    for col in all_embed_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}_norm'] = (df[col] - mean) / std
            else:
                df[f'{col}_norm'] = 0.0
            stats[col] = {'mean': mean, 'std': std}
    
    norm_cols = [f'{c}_norm' for c in all_embed_cols if f'{c}_norm' in df.columns]
    
    # Group by filing date and create daily files
    df['date'] = df['filing_date'].dt.date
    
    files_created = 0
    for filing_date, group in df.groupby('date'):
        if len(group) == 0:
            continue
        
        # Create embeddings
        embeddings = group[norm_cols].values.astype(np.float32) if norm_cols else np.zeros((len(group), 1), dtype=np.float32)
        
        # Timestamp at market close (4 PM ET = 21:00 UTC)
        filing_dt = datetime.combine(filing_date, datetime.min.time())
        base_ts = int(filing_dt.timestamp()) + 16 * 3600
        
        daily_df = pd.DataFrame({
            'ticker': group['ticker'].values,
            'timestamp': base_ts,
            'embedding': list(embeddings),
            'revenue': group['revenue'].values if 'revenue' in group.columns else 0,
            'net_income': group['net_income_loss'].values if 'net_income_loss' in group.columns else 0,
            'eps': group['basic_earnings_per_share'].values if 'basic_earnings_per_share' in group.columns else 0,
            'sector': group['sector'].values,
        })
        
        date_str = filing_date.strftime('%Y-%m-%d')
        daily_df.to_parquet(output_dir / f'{date_str}.parquet', index=False)
        files_created += 1
    
    logger.info(f"Created {files_created} daily earnings token files in {output_dir}")
    
    # Save normalization stats
    if stats:
        stats_df = pd.DataFrame(stats).T
        stats_df.to_parquet(output_dir.parent / 'earnings_norm_stats.parquet')


def build_sector_state(
    fundamentals_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Build sector-level aggregate state for cross-attention."""
    logger.info("Building sector aggregate state...")
    
    df = fundamentals_df.copy()
    df['sector'] = df['ticker'].apply(get_sector)
    
    # Group by filing date and sector
    df['date'] = df['filing_date'].dt.date
    
    sector_states = []
    
    for (filing_date, sector), group in df.groupby(['date', 'sector']):
        agg = {
            'date': filing_date,
            'sector': sector,
            'n_reported': len(group),
            'n_tickers': group['ticker'].nunique(),
        }
        
        # Revenue aggregates
        if 'revenue' in group.columns:
            rev = group['revenue'].dropna()
            if len(rev) > 0:
                agg['total_revenue'] = rev.sum()
                agg['avg_revenue'] = rev.mean()
        
        # Net income aggregates
        if 'net_income_loss' in group.columns:
            ni = group['net_income_loss'].dropna()
            if len(ni) > 0:
                agg['total_net_income'] = ni.sum()
                agg['avg_net_income'] = ni.mean()
                agg['pct_profitable'] = (ni > 0).mean() * 100
        
        # EPS aggregates
        if 'basic_earnings_per_share' in group.columns:
            eps = group['basic_earnings_per_share'].dropna()
            if len(eps) > 0:
                agg['avg_eps'] = eps.mean()
        
        sector_states.append(agg)
    
    if not sector_states:
        logger.warning("No sector states computed!")
        return pd.DataFrame()
    
    result = pd.DataFrame(sector_states)
    result['date'] = pd.to_datetime(result['date'])
    
    # Fill NaN
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)
    
    # Save main file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info(f"Saved sector state: {output_path} ({len(result):,} records, {result['sector'].nunique()} sectors)")
    
    # Create daily files
    daily_dir = output_path.parent / 'sector_daily'
    daily_dir.mkdir(parents=True, exist_ok=True)
    
    for d, group in result.groupby(result['date'].dt.date):
        date_str = d.strftime('%Y-%m-%d')
        group.to_parquet(daily_dir / f'{date_str}.parquet', index=False)
    
    logger.info(f"Created {len(result['date'].dt.date.unique())} daily sector state files")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Build earnings dataset from Polygon REST API data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--rest-data', type=str, default='datasets/MACRO/rest_data',
                        help='Path to rest_data directory')
    parser.add_argument('--output-dir', type=str, default='datasets/earnings',
                        help='Output directory for processed data')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top companies for individual earnings tokens')
    
    args = parser.parse_args()
    
    rest_data_path = Path(args.rest_data)
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Building Earnings Dataset from Polygon REST API")
    logger.info("=" * 60)
    logger.info(f"REST data: {rest_data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Top N: {args.top_n}")
    
    # Step 1: Load and merge financial statements
    logger.info("\n[1/4] Loading financial statements...")
    merged_df = load_financial_statements(rest_data_path)
    
    if merged_df.empty:
        logger.error("No data loaded! Exiting.")
        return
    
    # Step 2: Build company fundamentals
    logger.info("\n[2/4] Building company fundamentals...")
    fundamentals_path = output_dir / 'company_fundamentals.parquet'
    fundamentals_df = build_company_fundamentals(merged_df, fundamentals_path)
    
    # Step 3: Build earnings tokens
    logger.info("\n[3/4] Building daily earnings tokens...")
    earnings_tokens_dir = output_dir / 'earnings_tokens'
    build_earnings_tokens(fundamentals_df, earnings_tokens_dir, top_n=args.top_n)
    
    # Step 4: Build sector state
    logger.info("\n[4/4] Building sector aggregate state...")
    sector_state_path = output_dir / 'sector_state.parquet'
    build_sector_state(fundamentals_df, sector_state_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Outputs:")
    logger.info(f"  - {fundamentals_path}")
    logger.info(f"  - {earnings_tokens_dir}/")
    logger.info(f"  - {sector_state_path}")
    logger.info(f"  - {output_dir}/sector_daily/")


if __name__ == '__main__':
    main()
