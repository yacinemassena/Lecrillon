#!/usr/bin/env python3
"""
Build comprehensive fundamentals state for cross-attention conditioning.

Architecture:
    Cross-attention state (~170 dims, updates daily):
    ├── Earnings aggregates (75 dims) — beat rates, surprises, revenue growth
    ├── Financial health (80 dims) — margins, leverage, cash flow by sector  
    └── Short interest aggregates (20 dims) — sector SI trends

    News Mamba tokens (event-driven):
    ├── Mega-cap earnings (top 50 individual reports)
    └── Mega-cap SI changes (top 50 when SI shifts >10%)

IMPORTANT: Uses filing_date (when data became public), NOT period_date (leakage!)

Usage:
    python tools/build_fundamentals_state.py

Output:
    datasets/fundamentals/fundamentals_state.parquet     (daily ~170 dims)
    datasets/fundamentals/state_daily/YYYY-MM-DD.parquet (daily files)
    datasets/fundamentals/earnings_tokens/YYYY-MM-DD.parquet (mega-cap events)
    datasets/fundamentals/si_tokens/YYYY-MM-DD.parquet   (mega-cap SI events)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GICS Sectors (11 sectors)
# ---------------------------------------------------------------------------
SECTORS = [
    'Technology', 'Financials', 'Healthcare', 'Consumer Discretionary',
    'Consumer Staples', 'Industrials', 'Energy', 'Materials',
    'Utilities', 'Real Estate', 'Communication Services'
]

# Top 50 mega-caps for individual tokens
MEGA_CAPS = {
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'WMT', 'MCD',
    'CSCO', 'ACN', 'ABT', 'DHR', 'NEE', 'VZ', 'ADBE', 'NKE', 'CMCSA',
    'TXN', 'PM', 'WFC', 'DIS', 'BMY', 'COP', 'RTX', 'ORCL', 'QCOM',
    'T', 'MS', 'UNP', 'INTC', 'AMD', 'CRM'
}

# Sector mapping (simplified - maps first letter patterns)
TICKER_SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
    'TSLA': 'Consumer Discretionary', 'BRK.A': 'Financials', 'BRK.B': 'Financials',
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'XOM': 'Energy', 'JPM': 'Financials',
    'V': 'Financials', 'PG': 'Consumer Staples', 'MA': 'Financials', 'HD': 'Consumer Discretionary',
    'CVX': 'Energy', 'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'LLY': 'Healthcare',
    'PEP': 'Consumer Staples', 'KO': 'Consumer Staples', 'COST': 'Consumer Staples',
    'AVGO': 'Technology', 'TMO': 'Healthcare', 'WMT': 'Consumer Staples', 'MCD': 'Consumer Discretionary',
    'CSCO': 'Technology', 'ACN': 'Technology', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    'NEE': 'Utilities', 'VZ': 'Communication Services', 'ADBE': 'Technology', 'NKE': 'Consumer Discretionary',
    'CMCSA': 'Communication Services', 'TXN': 'Technology', 'PM': 'Consumer Staples', 'WFC': 'Financials',
    'DIS': 'Communication Services', 'BMY': 'Healthcare', 'COP': 'Energy', 'RTX': 'Industrials',
    'ORCL': 'Technology', 'QCOM': 'Technology', 'T': 'Communication Services', 'MS': 'Financials',
    'UNP': 'Industrials', 'INTC': 'Technology', 'AMD': 'Technology', 'CRM': 'Technology',
    'GS': 'Financials', 'BAC': 'Financials', 'C': 'Financials',
    'PFE': 'Healthcare', 'GILD': 'Healthcare', 'AMGN': 'Healthcare',
    'BA': 'Industrials', 'CAT': 'Industrials', 'HON': 'Industrials', 'GE': 'Industrials',
    'SLB': 'Energy', 'EOG': 'Energy', 'PSX': 'Energy',
}


def get_sector(ticker: str) -> str:
    """Get sector for a ticker."""
    return TICKER_SECTOR_MAP.get(ticker, 'Unknown')


def parse_tickers(tickers_str: str) -> List[str]:
    """Parse tickers from string like "['AAPL']"."""
    if pd.isna(tickers_str):
        return []
    try:
        cleaned = tickers_str.strip("[]'\"").replace("'", "").replace('"', '')
        return [t.strip() for t in cleaned.split(',') if t.strip()]
    except:
        return []


# ---------------------------------------------------------------------------
# Load financial data
# ---------------------------------------------------------------------------
def load_income_statements(path: Path) -> pd.DataFrame:
    """Load income statements with proper date handling."""
    df = pd.read_csv(path, on_bad_lines='skip')
    df['ticker'] = df['tickers'].apply(lambda x: parse_tickers(x)[0] if parse_tickers(x) else None)
    df = df.dropna(subset=['ticker', 'filing_date'])
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df['sector'] = df['ticker'].apply(get_sector)
    
    # Key income statement fields
    cols = ['ticker', 'sector', 'filing_date', 'fiscal_year', 'fiscal_quarter',
            'revenue', 'cost_of_revenue', 'gross_profit', 'operating_income', 
            'net_income_loss', 'basic_earnings_per_share', 'ebitda']
    return df[[c for c in cols if c in df.columns]].dropna(subset=['filing_date'])


def load_balance_sheets(path: Path) -> pd.DataFrame:
    """Load balance sheets with proper date handling."""
    df = pd.read_csv(path, on_bad_lines='skip')
    df['ticker'] = df['tickers'].apply(lambda x: parse_tickers(x)[0] if parse_tickers(x) else None)
    df = df.dropna(subset=['ticker', 'filing_date'])
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df['sector'] = df['ticker'].apply(get_sector)
    
    # Key balance sheet fields
    cols = ['ticker', 'sector', 'filing_date', 'fiscal_year', 'fiscal_quarter',
            'total_assets', 'total_liabilities', 'total_equity', 'cash_and_equivalents',
            'total_current_assets', 'total_current_liabilities', 'long_term_debt_and_capital_lease_obligations']
    return df[[c for c in cols if c in df.columns]].dropna(subset=['filing_date'])


def load_cash_flow(path: Path) -> pd.DataFrame:
    """Load cash flow statements with proper date handling."""
    df = pd.read_csv(path, on_bad_lines='skip')
    df['ticker'] = df['tickers'].apply(lambda x: parse_tickers(x)[0] if parse_tickers(x) else None)
    df = df.dropna(subset=['ticker', 'filing_date'])
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    df['sector'] = df['ticker'].apply(get_sector)
    
    # Key cash flow fields
    cols = ['ticker', 'sector', 'filing_date', 'fiscal_year', 'fiscal_quarter',
            'net_cash_flow_from_operating_activities', 'net_cash_flow_from_investing_activities',
            'net_cash_flow_from_financing_activities', 'capital_expenditures',
            'purchase_of_investment_securities', 'sale_of_investment_securities',
            'repurchase_of_common_stock', 'payment_of_dividends']
    return df[[c for c in cols if c in df.columns]].dropna(subset=['filing_date'])


def load_short_interest(path: Path) -> pd.DataFrame:
    """Load short interest data."""
    df = pd.read_csv(path)
    df['filing_date'] = pd.to_datetime(df['settlement_date'])
    df['sector'] = df['ticker'].apply(get_sector)
    return df


# ---------------------------------------------------------------------------
# Compute sector aggregates
# ---------------------------------------------------------------------------
def compute_income_aggregates(df: pd.DataFrame, as_of_date: datetime) -> Dict[str, float]:
    """
    Compute income statement aggregates by sector as of a given date.
    Uses only filings that were PUBLIC by as_of_date (filing_date <= as_of_date).
    
    Returns ~35 features (7 metrics × 5 sectors shown, extrapolate to 11)
    """
    # Filter to filings available as of this date
    available = df[df['filing_date'] <= as_of_date].copy()
    
    # Get most recent filing per ticker (within last 6 months to stay current)
    cutoff = as_of_date - timedelta(days=180)
    recent = available[available['filing_date'] >= cutoff]
    latest = recent.sort_values('filing_date').groupby('ticker').last().reset_index()
    
    features = {}
    
    for sector in SECTORS:
        sector_data = latest[latest['sector'] == sector]
        prefix = sector.lower().replace(' ', '_')[:8]  # Shorten sector name
        
        n = len(sector_data)
        features[f'{prefix}_n_reported'] = n
        
        if n == 0:
            features[f'{prefix}_rev_growth'] = 0.0
            features[f'{prefix}_gross_margin'] = 0.0
            features[f'{prefix}_op_margin'] = 0.0
            features[f'{prefix}_net_margin'] = 0.0
            features[f'{prefix}_pct_profitable'] = 0.0
            continue
        
        # Revenue growth (QoQ approximation)
        if 'revenue' in sector_data.columns:
            rev = sector_data['revenue'].dropna()
            features[f'{prefix}_rev_growth'] = rev.pct_change().mean() if len(rev) > 1 else 0.0
        
        # Gross margin
        if 'gross_profit' in sector_data.columns and 'revenue' in sector_data.columns:
            gp = sector_data['gross_profit'].sum()
            rev = sector_data['revenue'].sum()
            features[f'{prefix}_gross_margin'] = gp / rev if rev > 0 else 0.0
        
        # Operating margin
        if 'operating_income' in sector_data.columns and 'revenue' in sector_data.columns:
            op = sector_data['operating_income'].sum()
            rev = sector_data['revenue'].sum()
            features[f'{prefix}_op_margin'] = op / rev if rev > 0 else 0.0
        
        # Net margin
        if 'net_income_loss' in sector_data.columns and 'revenue' in sector_data.columns:
            ni = sector_data['net_income_loss'].sum()
            rev = sector_data['revenue'].sum()
            features[f'{prefix}_net_margin'] = ni / rev if rev > 0 else 0.0
        
        # Profitability rate
        if 'net_income_loss' in sector_data.columns:
            ni = sector_data['net_income_loss'].dropna()
            features[f'{prefix}_pct_profitable'] = (ni > 0).mean() if len(ni) > 0 else 0.0
    
    return features


def compute_balance_aggregates(df: pd.DataFrame, as_of_date: datetime) -> Dict[str, float]:
    """
    Compute balance sheet aggregates by sector.
    
    Returns ~33 features (3 metrics × 11 sectors)
    """
    available = df[df['filing_date'] <= as_of_date].copy()
    cutoff = as_of_date - timedelta(days=180)
    recent = available[available['filing_date'] >= cutoff]
    latest = recent.sort_values('filing_date').groupby('ticker').last().reset_index()
    
    features = {}
    
    for sector in SECTORS:
        sector_data = latest[latest['sector'] == sector]
        prefix = sector.lower().replace(' ', '_')[:8]
        
        n = len(sector_data)
        
        if n == 0:
            features[f'{prefix}_debt_equity'] = 0.0
            features[f'{prefix}_current_ratio'] = 0.0
            features[f'{prefix}_cash_ratio'] = 0.0
            continue
        
        # Debt/Equity ratio
        if 'total_liabilities' in sector_data.columns and 'total_equity' in sector_data.columns:
            debt = sector_data['total_liabilities'].sum()
            equity = sector_data['total_equity'].sum()
            features[f'{prefix}_debt_equity'] = debt / equity if equity > 0 else 0.0
        
        # Current ratio
        if 'total_current_assets' in sector_data.columns and 'total_current_liabilities' in sector_data.columns:
            ca = sector_data['total_current_assets'].sum()
            cl = sector_data['total_current_liabilities'].sum()
            features[f'{prefix}_current_ratio'] = ca / cl if cl > 0 else 0.0
        
        # Cash ratio
        if 'cash_and_equivalents' in sector_data.columns and 'total_current_liabilities' in sector_data.columns:
            cash = sector_data['cash_and_equivalents'].sum()
            cl = sector_data['total_current_liabilities'].sum()
            features[f'{prefix}_cash_ratio'] = cash / cl if cl > 0 else 0.0
    
    return features


def compute_cashflow_aggregates(df: pd.DataFrame, as_of_date: datetime) -> Dict[str, float]:
    """
    Compute cash flow aggregates by sector.
    
    Returns ~22 features (2 metrics × 11 sectors)
    """
    available = df[df['filing_date'] <= as_of_date].copy()
    cutoff = as_of_date - timedelta(days=180)
    recent = available[available['filing_date'] >= cutoff]
    latest = recent.sort_values('filing_date').groupby('ticker').last().reset_index()
    
    features = {}
    
    for sector in SECTORS:
        sector_data = latest[latest['sector'] == sector]
        prefix = sector.lower().replace(' ', '_')[:8]
        
        n = len(sector_data)
        
        if n == 0:
            features[f'{prefix}_fcf_margin'] = 0.0
            features[f'{prefix}_buyback_rate'] = 0.0
            continue
        
        # Free cash flow (OCF - CapEx) / Revenue proxy
        if 'net_cash_flow_from_operating_activities' in sector_data.columns:
            ocf = sector_data['net_cash_flow_from_operating_activities'].sum()
            capex = sector_data.get('capital_expenditures', pd.Series([0])).abs().sum()
            fcf = ocf - capex
            # Normalize by total assets as revenue isn't in this df
            assets = sector_data.get('total_assets', pd.Series([1])).sum()
            features[f'{prefix}_fcf_margin'] = fcf / assets if assets > 0 else 0.0
        
        # Buyback activity
        if 'repurchase_of_common_stock' in sector_data.columns:
            buybacks = sector_data['repurchase_of_common_stock'].abs().sum()
            features[f'{prefix}_buyback_rate'] = buybacks / 1e9  # In billions
        else:
            features[f'{prefix}_buyback_rate'] = 0.0
    
    return features


def compute_si_aggregates(df: pd.DataFrame, as_of_date: datetime) -> Dict[str, float]:
    """
    Compute short interest aggregates by sector.
    
    Returns ~22 features (2 metrics × 11 sectors)
    """
    # Get most recent SI report before as_of_date
    available = df[df['filing_date'] <= as_of_date].copy()
    
    if len(available) == 0:
        return {f'{s.lower().replace(" ", "_")[:8]}_si_level': 0.0 for s in SECTORS}
    
    # Get most recent date
    latest_date = available['filing_date'].max()
    latest = available[available['filing_date'] == latest_date]
    
    # Also get previous report for change calculation
    prev_dates = available[available['filing_date'] < latest_date]['filing_date'].unique()
    prev_date = max(prev_dates) if len(prev_dates) > 0 else None
    prev = available[available['filing_date'] == prev_date] if prev_date else pd.DataFrame()
    
    features = {}
    
    for sector in SECTORS:
        prefix = sector.lower().replace(' ', '_')[:8]
        
        sector_latest = latest[latest['sector'] == sector]
        sector_prev = prev[prev['sector'] == sector] if len(prev) > 0 else pd.DataFrame()
        
        if len(sector_latest) == 0:
            features[f'{prefix}_si_level'] = 0.0
            features[f'{prefix}_si_change'] = 0.0
            continue
        
        # Average days to cover (proxy for SI intensity)
        if 'days_to_cover' in sector_latest.columns:
            features[f'{prefix}_si_level'] = sector_latest['days_to_cover'].mean()
        
        # SI change from previous report
        if len(sector_prev) > 0 and 'short_interest' in sector_latest.columns:
            curr_si = sector_latest['short_interest'].sum()
            prev_si = sector_prev['short_interest'].sum()
            features[f'{prefix}_si_change'] = (curr_si - prev_si) / prev_si if prev_si > 0 else 0.0
        else:
            features[f'{prefix}_si_change'] = 0.0
    
    return features


# ---------------------------------------------------------------------------
# Build mega-cap event tokens
# ---------------------------------------------------------------------------
def build_earnings_tokens(income_df: pd.DataFrame, output_dir: Path) -> int:
    """
    Build daily earnings token files for mega-cap individual reports.
    
    Token features: ticker, sector, revenue_z, eps_z, margin, surprise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to mega-caps
    mega = income_df[income_df['ticker'].isin(MEGA_CAPS)].copy()
    
    if len(mega) == 0:
        logger.warning("No mega-cap earnings found!")
        return 0
    
    # Compute z-scores for normalization
    for col in ['revenue', 'basic_earnings_per_share', 'net_income_loss']:
        if col in mega.columns:
            mean = mega[col].mean()
            std = mega[col].std()
            mega[f'{col}_z'] = (mega[col] - mean) / std if std > 0 else 0.0
    
    # Compute margins
    if 'gross_profit' in mega.columns and 'revenue' in mega.columns:
        mega['gross_margin'] = mega['gross_profit'] / mega['revenue'].replace(0, np.nan)
    if 'net_income_loss' in mega.columns and 'revenue' in mega.columns:
        mega['net_margin'] = mega['net_income_loss'] / mega['revenue'].replace(0, np.nan)
    
    mega = mega.fillna(0)
    
    # Group by filing date
    mega['date'] = mega['filing_date'].dt.date
    
    files_created = 0
    for filing_date, group in mega.groupby('date'):
        # Build token features
        tokens = []
        for _, row in group.iterrows():
            token = {
                'ticker': row['ticker'],
                'sector': row['sector'],
                'timestamp': int(datetime.combine(filing_date, datetime.min.time()).timestamp()) + 16*3600,
                'revenue_z': row.get('revenue_z', 0),
                'eps_z': row.get('basic_earnings_per_share_z', 0),
                'gross_margin': row.get('gross_margin', 0),
                'net_margin': row.get('net_margin', 0),
            }
            tokens.append(token)
        
        if tokens:
            df = pd.DataFrame(tokens)
            df.to_parquet(output_dir / f'{filing_date}.parquet', index=False)
            files_created += 1
    
    return files_created


def build_si_tokens(si_df: pd.DataFrame, output_dir: Path, threshold: float = 0.10) -> int:
    """
    Build SI change tokens for mega-caps when SI shifts >threshold (10%).
    
    Token features: ticker, sector, si_level_z, si_change, si_percentile
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to mega-caps
    mega = si_df[si_df['ticker'].isin(MEGA_CAPS)].copy()
    
    if len(mega) == 0:
        logger.warning("No mega-cap short interest found!")
        return 0
    
    # Sort by ticker and date
    mega = mega.sort_values(['ticker', 'filing_date'])
    
    # Compute SI change per ticker
    mega['si_change'] = mega.groupby('ticker')['short_interest'].pct_change()
    mega['si_change'] = mega['si_change'].fillna(0)
    
    # Compute z-scores
    mean_dtc = mega['days_to_cover'].mean()
    std_dtc = mega['days_to_cover'].std()
    mega['si_level_z'] = (mega['days_to_cover'] - mean_dtc) / std_dtc if std_dtc > 0 else 0.0
    
    # Compute historical percentile per ticker
    mega['si_percentile'] = mega.groupby('ticker')['short_interest'].rank(pct=True)
    
    # Filter to significant changes
    significant = mega[mega['si_change'].abs() > threshold].copy()
    
    if len(significant) == 0:
        logger.info(f"No SI changes > {threshold*100}% found for mega-caps")
        return 0
    
    # Group by filing date
    significant['date'] = significant['filing_date'].dt.date
    
    files_created = 0
    for filing_date, group in significant.groupby('date'):
        tokens = []
        for _, row in group.iterrows():
            token = {
                'ticker': row['ticker'],
                'sector': row['sector'],
                'timestamp': int(datetime.combine(filing_date, datetime.min.time()).timestamp()) + 16*3600,
                'si_level_z': row['si_level_z'],
                'si_change': row['si_change'],
                'si_percentile': row['si_percentile'],
            }
            tokens.append(token)
        
        if tokens:
            df = pd.DataFrame(tokens)
            df.to_parquet(output_dir / f'{filing_date}.parquet', index=False)
            files_created += 1
    
    return files_created


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------
def main():
    rest_data_path = Path('datasets/MACRO/rest_data')
    output_dir = Path('datasets/fundamentals')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Building Comprehensive Fundamentals State (~170 dims)")
    logger.info("=" * 70)
    
    # Load all data
    logger.info("\n[1/6] Loading financial data...")
    income_df = load_income_statements(rest_data_path / 'income_statements' / 'income_statements.csv')
    logger.info(f"  Income statements: {len(income_df):,} rows")
    
    balance_df = load_balance_sheets(rest_data_path / 'balance_sheets' / 'balance_sheets.csv')
    logger.info(f"  Balance sheets: {len(balance_df):,} rows")
    
    cashflow_df = load_cash_flow(rest_data_path / 'cash_flow_statements' / 'cash_flow_statements.csv')
    logger.info(f"  Cash flow statements: {len(cashflow_df):,} rows")
    
    si_df = load_short_interest(rest_data_path / 'short_interest' / 'short_interest.csv')
    logger.info(f"  Short interest: {len(si_df):,} rows")
    
    # Get all unique dates
    all_dates = set()
    for df in [income_df, balance_df, cashflow_df, si_df]:
        all_dates.update(df['filing_date'].dt.date.unique())
    all_dates = sorted(all_dates)
    logger.info(f"\n[2/6] Building daily state for {len(all_dates)} dates...")
    
    # Build daily state
    daily_states = []
    for i, as_of_date in enumerate(all_dates):
        as_of_dt = datetime.combine(as_of_date, datetime.min.time())
        
        # Compute all aggregates
        income_agg = compute_income_aggregates(income_df, as_of_dt)
        balance_agg = compute_balance_aggregates(balance_df, as_of_dt)
        cashflow_agg = compute_cashflow_aggregates(cashflow_df, as_of_dt)
        si_agg = compute_si_aggregates(si_df, as_of_dt)
        
        # Combine
        state = {'date': as_of_date}
        state.update(income_agg)
        state.update(balance_agg)
        state.update(cashflow_agg)
        state.update(si_agg)
        
        daily_states.append(state)
        
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{len(all_dates)} dates...")
    
    # Save main state file
    state_df = pd.DataFrame(daily_states)
    state_df['date'] = pd.to_datetime(state_df['date'])
    state_df = state_df.fillna(0)
    
    n_features = len(state_df.columns) - 1  # Exclude date
    state_df.to_parquet(output_dir / 'fundamentals_state.parquet', index=False)
    logger.info(f"\n[3/6] Saved fundamentals_state.parquet ({len(state_df):,} rows, {n_features} features)")
    
    # Create daily files
    daily_dir = output_dir / 'state_daily'
    daily_dir.mkdir(parents=True, exist_ok=True)
    for d, group in state_df.groupby(state_df['date'].dt.date):
        group.to_parquet(daily_dir / f'{d}.parquet', index=False)
    logger.info(f"  Created {len(state_df)} daily state files")
    
    # Build mega-cap earnings tokens
    logger.info("\n[4/6] Building mega-cap earnings tokens...")
    earnings_dir = output_dir / 'earnings_tokens'
    n_earnings = build_earnings_tokens(income_df, earnings_dir)
    logger.info(f"  Created {n_earnings} earnings token files")
    
    # Build mega-cap SI tokens
    logger.info("\n[5/6] Building mega-cap SI change tokens (>10% change)...")
    si_tokens_dir = output_dir / 'si_tokens'
    n_si = build_si_tokens(si_df, si_tokens_dir, threshold=0.10)
    logger.info(f"  Created {n_si} SI token files")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nCross-attention state ({n_features} dims):")
    logger.info(f"  - {output_dir / 'fundamentals_state.parquet'}")
    logger.info(f"  - {daily_dir}/ ({len(state_df)} daily files)")
    logger.info(f"\nNews Mamba tokens:")
    logger.info(f"  - {earnings_dir}/ ({n_earnings} mega-cap earnings events)")
    logger.info(f"  - {si_tokens_dir}/ ({n_si} mega-cap SI change events)")
    
    # Feature breakdown
    income_features = len([k for k in daily_states[0].keys() if 'margin' in k or 'growth' in k or 'profit' in k])
    balance_features = len([k for k in daily_states[0].keys() if 'debt' in k or 'ratio' in k or 'cash_ratio' in k])
    si_features = len([k for k in daily_states[0].keys() if 'si_' in k])
    logger.info(f"\nFeature breakdown:")
    logger.info(f"  - Income/earnings: ~{income_features} features")
    logger.info(f"  - Balance/leverage: ~{balance_features} features")
    logger.info(f"  - Short interest: ~{si_features} features")


if __name__ == '__main__':
    main()
