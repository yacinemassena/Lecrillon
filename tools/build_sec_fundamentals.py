#!/usr/bin/env python3
"""
Build SEC fundamentals dataset from EDGAR companyfacts JSON files.

Creates two outputs for the Mamba architecture:
1. Daily earnings tokens (top 50 mega-caps) → news Mamba stream
2. Sector aggregates (all 19K companies) → cross-attention state at checkpoints

Architecture integration:
    News Mamba token — individual earnings events for mega-caps
    Cross-attention state — sector aggregates that update at checkpoints (300 steps)

Usage:
    python tools/build_sec_fundamentals.py --sec-data datasets/MACRO/sec_data
    python tools/build_sec_fundamentals.py --sec-data datasets/MACRO/sec_data --top-n 50

Output:
    datasets/earnings/earnings_tokens/YYYY-MM-DD.parquet  (daily mega-cap events)
    datasets/earnings/sector_state.parquet                 (sector aggregates)
    datasets/earnings/company_fundamentals.parquet         (all companies, all metrics)
"""

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC EDGAR key metrics to extract
# ---------------------------------------------------------------------------
# These are the most common US-GAAP tags found in companyfacts
KEY_METRICS = {
    # Income statement
    'Revenues': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet', 
                 'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueGoodsNet'],
    'NetIncome': ['NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'],
    'GrossProfit': ['GrossProfit'],
    'OperatingIncome': ['OperatingIncomeLoss', 'IncomeLossFromContinuingOperations'],
    'CostOfRevenue': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold'],
    
    # Per share
    'EPS_Basic': ['EarningsPerShareBasic'],
    'EPS_Diluted': ['EarningsPerShareDiluted'],
    
    # Balance sheet
    'TotalAssets': ['Assets'],
    'TotalLiabilities': ['Liabilities'],
    'TotalEquity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'Cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments'],
    'TotalDebt': ['LongTermDebt', 'DebtCurrent', 'LongTermDebtAndCapitalLeaseObligations'],
    'CurrentAssets': ['AssetsCurrent'],
    'CurrentLiabilities': ['LiabilitiesCurrent'],
    
    # Cash flow
    'OperatingCashFlow': ['NetCashProvidedByUsedInOperatingActivities'],
    'CapEx': ['PaymentsToAcquirePropertyPlantAndEquipment'],
    'FreeCashFlow': ['NetCashProvidedByUsedInOperatingActivities'],  # Will compute FCF = OCF - CapEx
    
    # Shares
    'SharesOutstanding': ['CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingBasic'],
}

# GICS sector mapping (approximate based on common SIC codes)
SIC_TO_SECTOR = {
    range(100, 1000): 'Energy',
    range(1000, 1500): 'Materials', 
    range(1500, 1800): 'Materials',
    range(2000, 4000): 'Industrials',
    range(4000, 5000): 'Utilities',
    range(5000, 5200): 'Consumer Discretionary',
    range(5200, 6000): 'Consumer Staples',
    range(6000, 6800): 'Financials',
    range(7000, 7400): 'Communication Services',
    range(7370, 7380): 'Information Technology',
    range(7300, 7400): 'Information Technology',
    range(8000, 8100): 'Health Care',
    range(8700, 8800): 'Information Technology',
    range(9000, 10000): 'Real Estate',
}

def get_sector_from_sic(sic_code: Optional[int]) -> str:
    """Map SIC code to GICS sector."""
    if sic_code is None:
        return 'Unknown'
    for sic_range, sector in SIC_TO_SECTOR.items():
        if sic_code in sic_range:
            return sector
    return 'Unknown'


# ---------------------------------------------------------------------------
# SEC data extraction
# ---------------------------------------------------------------------------
def load_submissions(submissions_path: Path) -> Dict[str, Dict]:
    """Load CIK → company info mapping from submissions files."""
    logger.info(f"Loading company submissions from {submissions_path}...")
    
    cik_to_info = {}
    submission_files = list(submissions_path.glob('CIK*.json'))
    
    for i, f in enumerate(submission_files):
        if i % 50000 == 0:
            logger.info(f"  Processed {i}/{len(submission_files)} submission files...")
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            
            cik = data.get('cik', '').zfill(10)
            tickers = data.get('tickers', [])
            
            cik_to_info[cik] = {
                'cik': cik,
                'name': data.get('name', ''),
                'tickers': tickers,
                'ticker': tickers[0] if tickers else None,
                'sic': data.get('sic'),
                'sicDescription': data.get('sicDescription', ''),
                'exchanges': data.get('exchanges', []),
            }
        except Exception as e:
            continue
    
    logger.info(f"Loaded {len(cik_to_info)} company submissions")
    return cik_to_info


def extract_metric_value(facts: Dict, metric_tags: List[str]) -> List[Dict]:
    """Extract all reported values for a metric from companyfacts."""
    us_gaap = facts.get('facts', {}).get('us-gaap', {})
    
    results = []
    for tag in metric_tags:
        if tag not in us_gaap:
            continue
        
        concept = us_gaap[tag]
        units = concept.get('units', {})
        
        # Try USD first, then USD/shares, then shares
        for unit_type in ['USD', 'USD/shares', 'shares', 'pure']:
            if unit_type in units:
                for entry in units[unit_type]:
                    # Only use annual (10-K) and quarterly (10-Q) filings
                    form = entry.get('form', '')
                    if form not in ['10-K', '10-Q', '10-K/A', '10-Q/A']:
                        continue
                    
                    results.append({
                        'tag': tag,
                        'value': entry.get('val'),
                        'start': entry.get('start'),
                        'end': entry.get('end'),
                        'filed': entry.get('filed'),
                        'form': form,
                        'fy': entry.get('fy'),
                        'fp': entry.get('fp'),
                        'frame': entry.get('frame'),
                    })
                break  # Use first matching unit type
    
    return results


def process_companyfacts(file_path: Path, cik_to_info: Dict) -> Optional[Dict]:
    """Process a single companyfacts JSON file."""
    try:
        cik = file_path.stem.replace('CIK', '').zfill(10)
        
        with open(file_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        company_info = cik_to_info.get(cik, {})
        ticker = company_info.get('ticker')
        
        if not ticker:
            return None
        
        # Extract all metrics
        metrics_data = {}
        for metric_name, tags in KEY_METRICS.items():
            values = extract_metric_value(data, tags)
            if values:
                metrics_data[metric_name] = values
        
        if not metrics_data:
            return None
        
        return {
            'cik': cik,
            'ticker': ticker,
            'name': company_info.get('name', ''),
            'sic': company_info.get('sic'),
            'sector': get_sector_from_sic(int(company_info['sic']) if company_info.get('sic') else None),
            'metrics': metrics_data,
        }
    except Exception as e:
        return None


def build_company_fundamentals(
    sec_data_path: Path,
    output_path: Path,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Extract fundamentals from all companyfacts files."""
    submissions_path = sec_data_path / 'submissions'
    companyfacts_path = sec_data_path / 'companyfacts'
    
    # Load company info
    cik_to_info = load_submissions(submissions_path)
    
    # Process all companyfacts
    logger.info(f"Processing companyfacts from {companyfacts_path}...")
    files = list(companyfacts_path.glob('CIK*.json'))
    logger.info(f"Found {len(files)} companyfacts files")
    
    all_records = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_companyfacts, f, cik_to_info): f for f in files}
        
        for future in as_completed(futures):
            result = future.result()
            processed += 1
            
            if processed % 1000 == 0:
                logger.info(f"  Processed {processed}/{len(files)} files, found {len(all_records)} with data")
            
            if result is None:
                continue
            
            # Flatten metrics into records by filing date
            for metric_name, values in result['metrics'].items():
                for v in values:
                    if v['value'] is None or v['filed'] is None:
                        continue
                    
                    all_records.append({
                        'cik': result['cik'],
                        'ticker': result['ticker'],
                        'name': result['name'],
                        'sic': result['sic'],
                        'sector': result['sector'],
                        'metric': metric_name,
                        'value': v['value'],
                        'period_start': v['start'],
                        'period_end': v['end'],
                        'filed_date': v['filed'],
                        'form': v['form'],
                        'fiscal_year': v['fy'],
                        'fiscal_period': v['fp'],
                    })
    
    logger.info(f"Extracted {len(all_records)} metric records from {processed} files")
    
    if not all_records:
        logger.warning("No records extracted!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df['filed_date'] = pd.to_datetime(df['filed_date'])
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    # Save full fundamentals
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved company fundamentals to {output_path}")
    
    return df


def compute_surprise(actual: float, expected: float) -> float:
    """Compute earnings surprise as percentage."""
    if expected == 0:
        return 0.0
    return (actual - expected) / abs(expected) * 100


def build_earnings_tokens(
    fundamentals_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 50,
) -> None:
    """
    Build daily earnings token files for top N mega-caps.
    
    Each token represents an earnings event with:
    - Embedding: normalized fundamentals vector
    - Timestamp: filing datetime (Unix seconds)
    - Metadata: ticker, metrics, surprises
    
    Output: daily parquet files like news (YYYY-MM-DD.parquet)
    """
    logger.info(f"Building daily earnings tokens for top {top_n} mega-caps...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pivot to get latest metrics per company
    # Use the most recent filing for market cap approximation
    latest = fundamentals_df.sort_values('filed_date').groupby(['ticker', 'metric']).last().reset_index()
    pivot = latest.pivot(index='ticker', columns='metric', values='value')
    
    # Approximate market cap using shares * (revenue/shares as proxy)
    # In practice, you'd use actual market cap data
    if 'SharesOutstanding' in pivot.columns and 'Revenues' in pivot.columns:
        pivot['market_cap_proxy'] = pivot['Revenues']  # Simplified: use revenue as size proxy
    else:
        pivot['market_cap_proxy'] = 0
    
    # Get top N by market cap proxy
    top_tickers = set(pivot.nlargest(top_n, 'market_cap_proxy').index.tolist())
    logger.info(f"Top {top_n} tickers: {sorted(top_tickers)[:10]}...")
    
    # Filter to top tickers only
    mega_cap_filings = fundamentals_df[fundamentals_df['ticker'].isin(top_tickers)].copy()
    
    # Create earnings feature vector (normalized)
    # Features: Revenue, NetIncome, EPS, GrossProfit, OperatingIncome, etc.
    EARNINGS_FEATURES = [
        'Revenues', 'NetIncome', 'GrossProfit', 'OperatingIncome', 'EPS_Basic', 'EPS_Diluted',
        'TotalAssets', 'TotalLiabilities', 'TotalEquity', 'Cash', 'OperatingCashFlow',
    ]
    
    # Pivot filings to get feature vectors
    filings_pivot = mega_cap_filings.pivot_table(
        index=['ticker', 'filed_date'],
        columns='metric',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Ensure all features exist
    for feat in EARNINGS_FEATURES:
        if feat not in filings_pivot.columns:
            filings_pivot[feat] = np.nan
    
    # Create embedding vector (will be fed to earnings encoder)
    feature_cols = [c for c in EARNINGS_FEATURES if c in filings_pivot.columns]
    
    # Compute derived features
    filings_pivot['GrossMargin'] = filings_pivot['GrossProfit'] / filings_pivot['Revenues'].replace(0, np.nan)
    filings_pivot['NetMargin'] = filings_pivot['NetIncome'] / filings_pivot['Revenues'].replace(0, np.nan)
    filings_pivot['ROA'] = filings_pivot['NetIncome'] / filings_pivot['TotalAssets'].replace(0, np.nan)
    
    # Compute YoY changes (surprise proxies)
    filings_pivot = filings_pivot.sort_values(['ticker', 'filed_date'])
    for feat in ['Revenues', 'NetIncome', 'EPS_Basic']:
        if feat in filings_pivot.columns:
            filings_pivot[f'{feat}_YoY'] = filings_pivot.groupby('ticker')[feat].pct_change(periods=4)  # ~1 year for quarterly
    
    # Fill NaN and normalize
    numeric_cols = filings_pivot.select_dtypes(include=[np.number]).columns
    filings_pivot[numeric_cols] = filings_pivot[numeric_cols].fillna(0)
    
    # Create daily files
    filings_pivot['date'] = filings_pivot['filed_date'].dt.date
    
    # Build embedding columns (fixed dimension for model)
    EMBED_COLS = [
        'Revenues', 'NetIncome', 'GrossProfit', 'OperatingIncome', 'EPS_Basic',
        'TotalAssets', 'TotalEquity', 'Cash', 'OperatingCashFlow',
        'GrossMargin', 'NetMargin', 'ROA',
        'Revenues_YoY', 'NetIncome_YoY', 'EPS_Basic_YoY',
    ]
    
    for col in EMBED_COLS:
        if col not in filings_pivot.columns:
            filings_pivot[col] = 0.0
    
    # Normalize each column (z-score across all data)
    stats = {}
    for col in EMBED_COLS:
        mean = filings_pivot[col].mean()
        std = filings_pivot[col].std()
        if std > 0:
            filings_pivot[f'{col}_norm'] = (filings_pivot[col] - mean) / std
        else:
            filings_pivot[f'{col}_norm'] = 0.0
        stats[col] = {'mean': mean, 'std': std}
    
    norm_cols = [f'{c}_norm' for c in EMBED_COLS]
    
    # Group by date and save daily files
    daily_groups = filings_pivot.groupby('date')
    
    files_created = 0
    for filing_date, group in daily_groups:
        if len(group) == 0:
            continue
        
        # Create embeddings array
        embeddings = group[norm_cols].values.astype(np.float32)
        
        # Create timestamps (Unix seconds at market close, 4 PM ET = 21:00 UTC)
        filing_dt = datetime.combine(filing_date, datetime.min.time())
        base_ts = int(filing_dt.timestamp()) + 16 * 3600  # 4 PM
        
        daily_df = pd.DataFrame({
            'ticker': group['ticker'].values,
            'timestamp': base_ts,  # All filings on same day get same timestamp
            'embedding': list(embeddings),
            'revenues': group['Revenues'].values,
            'net_income': group['NetIncome'].values,
            'eps': group['EPS_Basic'].values,
            'revenues_yoy': group['Revenues_YoY'].values,
            'net_income_yoy': group['NetIncome_YoY'].values,
        })
        
        date_str = filing_date.strftime('%Y-%m-%d')
        daily_df.to_parquet(output_dir / f'{date_str}.parquet', index=False)
        files_created += 1
    
    logger.info(f"Created {files_created} daily earnings token files in {output_dir}")
    
    # Save normalization stats
    stats_df = pd.DataFrame(stats).T
    stats_df.to_parquet(output_dir.parent / 'earnings_norm_stats.parquet')


def build_sector_state(
    fundamentals_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Build sector-level aggregate state for cross-attention.
    
    Updates daily with:
    - Sector earnings beat rate
    - Sector revenue growth (aggregate)
    - % of sector reported
    - Aggregate P/E, margins, etc.
    
    This becomes the conditioning state for cross-attention at checkpoints.
    """
    logger.info("Building sector aggregate state...")
    
    # Get unique filing dates
    fundamentals_df['filed_date'] = pd.to_datetime(fundamentals_df['filed_date'])
    
    # Pivot to wide format per filing
    filings = fundamentals_df.pivot_table(
        index=['ticker', 'filed_date', 'sector', 'fiscal_year', 'fiscal_period'],
        columns='metric',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Sort by date
    filings = filings.sort_values('filed_date')
    
    # Build cumulative sector state for each date
    all_dates = sorted(filings['filed_date'].dt.date.unique())
    logger.info(f"Processing {len(all_dates)} unique filing dates...")
    
    sector_states = []
    
    # Track cumulative state per fiscal period
    reported_by_period = defaultdict(set)  # (fiscal_year, fiscal_period, sector) -> set of tickers
    metrics_by_period = defaultdict(list)  # (fiscal_year, fiscal_period, sector) -> list of metric dicts
    
    for i, current_date in enumerate(all_dates):
        if i % 500 == 0:
            logger.info(f"  Processing date {i}/{len(all_dates)}: {current_date}")
        
        # Get filings up to this date
        mask = filings['filed_date'].dt.date <= current_date
        filings_to_date = filings[mask]
        
        # Get current fiscal period (approximate: Q4 if Oct-Dec, etc.)
        month = current_date.month
        if month <= 3:
            current_fy = current_date.year - 1
            current_fp = 'Q4'
        elif month <= 6:
            current_fy = current_date.year
            current_fp = 'Q1'
        elif month <= 9:
            current_fy = current_date.year
            current_fp = 'Q2'
        else:
            current_fy = current_date.year
            current_fp = 'Q3'
        
        # Compute sector aggregates for current period
        current_period = filings_to_date[
            (filings_to_date['fiscal_year'] == current_fy) |
            (filings_to_date['fiscal_year'] == current_fy - 1)  # Include recent FY too
        ]
        
        sectors = current_period['sector'].unique()
        
        for sector in sectors:
            sector_data = current_period[current_period['sector'] == sector]
            
            if len(sector_data) == 0:
                continue
            
            n_reported = sector_data['ticker'].nunique()
            
            # Aggregate metrics
            agg = {
                'date': current_date,
                'sector': sector,
                'n_reported': n_reported,
            }
            
            # Revenue aggregates
            if 'Revenues' in sector_data.columns:
                rev = sector_data['Revenues'].dropna()
                if len(rev) > 0:
                    agg['total_revenue'] = rev.sum()
                    agg['avg_revenue'] = rev.mean()
                    agg['median_revenue'] = rev.median()
            
            # Net income aggregates
            if 'NetIncome' in sector_data.columns:
                ni = sector_data['NetIncome'].dropna()
                if len(ni) > 0:
                    agg['total_net_income'] = ni.sum()
                    agg['avg_net_income'] = ni.mean()
                    agg['pct_profitable'] = (ni > 0).mean() * 100
            
            # EPS aggregates
            if 'EPS_Basic' in sector_data.columns:
                eps = sector_data['EPS_Basic'].dropna()
                if len(eps) > 0:
                    agg['avg_eps'] = eps.mean()
                    agg['median_eps'] = eps.median()
            
            # Margin aggregates
            if 'GrossProfit' in sector_data.columns and 'Revenues' in sector_data.columns:
                gp = sector_data['GrossProfit'].dropna()
                rev = sector_data['Revenues'].dropna()
                if len(gp) > 0 and len(rev) > 0 and rev.sum() > 0:
                    agg['avg_gross_margin'] = (gp.sum() / rev.sum()) * 100
            
            if 'NetIncome' in sector_data.columns and 'Revenues' in sector_data.columns:
                ni = sector_data['NetIncome'].dropna()
                rev = sector_data['Revenues'].dropna()
                if len(ni) > 0 and len(rev) > 0 and rev.sum() > 0:
                    agg['avg_net_margin'] = (ni.sum() / rev.sum()) * 100
            
            sector_states.append(agg)
    
    if not sector_states:
        logger.warning("No sector states computed!")
        return pd.DataFrame()
    
    df = pd.DataFrame(sector_states)
    df['date'] = pd.to_datetime(df['date'])
    
    # Fill NaN with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved sector state to {output_path}: {len(df)} records, {df['sector'].nunique()} sectors")
    
    # Also create daily files for efficient loading
    daily_dir = output_path.parent / 'sector_daily'
    daily_dir.mkdir(parents=True, exist_ok=True)
    
    for d, group in df.groupby(df['date'].dt.date):
        date_str = d.strftime('%Y-%m-%d')
        group.to_parquet(daily_dir / f'{date_str}.parquet', index=False)
    
    logger.info(f"Created {len(df['date'].dt.date.unique())} daily sector state files")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Build SEC fundamentals dataset for Mamba architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--sec-data', type=str, default='datasets/MACRO/sec_data',
                        help='Path to SEC data directory (with submissions/ and companyfacts/)')
    parser.add_argument('--output-dir', type=str, default='datasets/earnings',
                        help='Output directory for processed data')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top mega-caps for individual earnings tokens')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--skip-fundamentals', action='store_true',
                        help='Skip building fundamentals (use existing file)')
    
    args = parser.parse_args()
    
    sec_data_path = Path(args.sec_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Building SEC Fundamentals Dataset")
    logger.info("=" * 60)
    logger.info(f"SEC data: {sec_data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Top N mega-caps: {args.top_n}")
    
    # Step 1: Extract fundamentals from all companies
    fundamentals_path = output_dir / 'company_fundamentals.parquet'
    
    if args.skip_fundamentals and fundamentals_path.exists():
        logger.info(f"\n[1/3] Loading existing fundamentals from {fundamentals_path}")
        fundamentals_df = pd.read_parquet(fundamentals_path)
    else:
        logger.info("\n[1/3] Extracting company fundamentals...")
        fundamentals_df = build_company_fundamentals(
            sec_data_path,
            fundamentals_path,
            max_workers=args.workers,
        )
    
    if fundamentals_df.empty:
        logger.error("No fundamentals extracted! Exiting.")
        return
    
    logger.info(f"Fundamentals: {len(fundamentals_df)} records, {fundamentals_df['ticker'].nunique()} companies")
    
    # Step 2: Build daily earnings tokens for mega-caps
    logger.info("\n[2/3] Building daily earnings tokens for mega-caps...")
    earnings_tokens_dir = output_dir / 'earnings_tokens'
    build_earnings_tokens(fundamentals_df, earnings_tokens_dir, top_n=args.top_n)
    
    # Step 3: Build sector aggregate state
    logger.info("\n[3/3] Building sector aggregate state...")
    sector_state_path = output_dir / 'sector_state.parquet'
    build_sector_state(fundamentals_df, sector_state_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Outputs:")
    logger.info(f"  - {fundamentals_path} (all companies, all metrics)")
    logger.info(f"  - {earnings_tokens_dir}/ (daily mega-cap earnings tokens)")
    logger.info(f"  - {sector_state_path} (sector aggregates)")
    logger.info(f"  - {output_dir}/sector_daily/ (daily sector state files)")


if __name__ == '__main__':
    main()
