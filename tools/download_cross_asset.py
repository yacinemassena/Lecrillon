#!/usr/bin/env python3
"""
Download cross-asset data for macro conditioning:
- Gold (GLD proxy via FRED)
- Bond ETFs / Treasury futures
- Credit spreads (HY, IG)
- Dollar index
- Market stress indicators

Uses FRED API (free, requires API key from https://fred.stlouisfed.org/docs/api/api_key.html)
"""
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FRED_SERIES = {
    # Gold & commodities
    'GOLDAMGBD228NLBM': 'gold_price_london',      # Gold fixing price London
    'DCOILWTICO': 'oil_wti',                       # WTI crude oil
    
    # Credit spreads - critical for vol regime
    'BAMLH0A0HYM2': 'hy_oas',                      # ICE BofA US High Yield OAS
    'BAMLC0A0CM': 'ig_oas',                        # ICE BofA US Corp Master OAS
    'BAMLH0A0HYM2EY': 'hy_effective_yield',       # HY effective yield
    'BAMLC0A4CBBB': 'bbb_oas',                    # BBB corporate OAS
    'BAMLC0A1CAAA': 'aaa_oas',                    # AAA corporate OAS
    
    # Treasury & TIPS
    'DFII5': 'tips_5y_real',                       # 5-year TIPS real yield
    'DFII10': 'tips_10y_real',                     # 10-year TIPS real yield
    'T5YIE': 'breakeven_5y',                       # 5-year breakeven inflation
    'T10YIE': 'breakeven_10y',                     # 10-year breakeven inflation
    
    # Market stress indices
    'VIXCLS': 'vix_close',                         # CBOE VIX close
    'STLFSI4': 'stl_stress_index',                 # St. Louis Fed Financial Stress
    'NFCI': 'chicago_fci',                         # Chicago Fed National Financial Conditions
    'ANFCI': 'adjusted_nfci',                      # Adjusted NFCI
    
    # Dollar & FX
    'DTWEXBGS': 'dollar_broad',                    # Trade weighted dollar (broad)
    'DTWEXAFEGS': 'dollar_afe',                    # Trade weighted dollar (AFE)
    'DEXJPUS': 'usdjpy',                           # USD/JPY
    'DEXUSEU': 'eurusd',                           # EUR/USD
    
    # Term premia & risk
    'THREEFYTP10': 'term_premium_10y',             # 10-year term premium
    
    # Equity risk
    'TEDRATE': 'ted_spread',                       # TED spread (LIBOR - T-bill)
    
    # Economic surprise
    'STLFSI4': 'fin_stress',                       # Financial stress index
}


def download_fred_series(api_key: str, start_date: str = '2000-01-01') -> pd.DataFrame:
    """Download all FRED series and merge into single DataFrame."""
    try:
        from fredapi import Fred
    except ImportError:
        logger.error("fredapi not installed. Run: pip install fredapi")
        raise
    
    fred = Fred(api_key=api_key)
    dfs = []
    
    for series_id, col_name in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            if s is not None and len(s) > 0:
                df = s.to_frame(name=col_name)
                df.index = pd.to_datetime(df.index)
                dfs.append(df)
                logger.info(f"✓ {col_name} ({series_id}): {len(df)} observations")
            else:
                logger.warning(f"✗ {col_name} ({series_id}): No data")
        except Exception as e:
            logger.warning(f"✗ {col_name} ({series_id}): {e}")
    
    if not dfs:
        raise ValueError("No FRED series downloaded successfully")
    
    # Merge all
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer')
    
    return merged


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum, spreads, and z-scores from raw data."""
    out = df.copy()
    
    # Gold momentum
    if 'gold_price_london' in out.columns:
        out['gold_ret_1d'] = out['gold_price_london'].pct_change(1)
        out['gold_ret_5d'] = out['gold_price_london'].pct_change(5)
        out['gold_ret_20d'] = out['gold_price_london'].pct_change(20)
        out['gold_z_20d'] = (out['gold_price_london'] - out['gold_price_london'].rolling(20).mean()) / out['gold_price_london'].rolling(20).std()
    
    # Oil momentum
    if 'oil_wti' in out.columns:
        out['oil_ret_1d'] = out['oil_wti'].pct_change(1)
        out['oil_ret_5d'] = out['oil_wti'].pct_change(5)
        out['oil_ret_20d'] = out['oil_wti'].pct_change(20)
    
    # Credit spread dynamics
    if 'hy_oas' in out.columns:
        out['hy_oas_chg_5d'] = out['hy_oas'].diff(5)
        out['hy_oas_z_60d'] = (out['hy_oas'] - out['hy_oas'].rolling(60).mean()) / out['hy_oas'].rolling(60).std()
    
    if 'ig_oas' in out.columns:
        out['ig_oas_chg_5d'] = out['ig_oas'].diff(5)
    
    if 'hy_oas' in out.columns and 'ig_oas' in out.columns:
        out['hy_ig_diff'] = out['hy_oas'] - out['ig_oas']
        out['hy_ig_ratio'] = out['hy_oas'] / out['ig_oas'].replace(0, 1)
    
    # Credit quality spread
    if 'bbb_oas' in out.columns and 'aaa_oas' in out.columns:
        out['bbb_aaa_spread'] = out['bbb_oas'] - out['aaa_oas']
    
    # Real yields & breakevens
    if 'tips_10y_real' in out.columns:
        out['real_yield_chg_5d'] = out['tips_10y_real'].diff(5)
    
    if 'breakeven_10y' in out.columns:
        out['infl_exp_chg_5d'] = out['breakeven_10y'].diff(5)
    
    # Dollar momentum
    if 'dollar_broad' in out.columns:
        out['dollar_ret_5d'] = out['dollar_broad'].pct_change(5)
        out['dollar_ret_20d'] = out['dollar_broad'].pct_change(20)
    
    # VIX dynamics
    if 'vix_close' in out.columns:
        out['vix_chg_1d'] = out['vix_close'].diff(1)
        out['vix_chg_5d'] = out['vix_close'].diff(5)
        out['vix_z_20d'] = (out['vix_close'] - out['vix_close'].rolling(20).mean()) / out['vix_close'].rolling(20).std()
        out['vix_term_struct'] = out['vix_close'].diff(1)  # Placeholder for VIX futures term structure
    
    # Stress regime flags
    if 'hy_oas' in out.columns:
        out['credit_stress'] = (out['hy_oas'] > out['hy_oas'].rolling(252).quantile(0.9)).astype(float)
    
    if 'stl_stress_index' in out.columns:
        out['fin_stress_elevated'] = (out['stl_stress_index'] > 0).astype(float)
    
    return out


def main():
    parser = argparse.ArgumentParser(description='Download cross-asset data from FRED')
    parser.add_argument('--api-key', type=str, default=None,
                        help='FRED API key (or set FRED_API_KEY env var)')
    parser.add_argument('--start-date', type=str, default='2000-01-01',
                        help='Start date for data download')
    parser.add_argument('--output', type=str, default='datasets/cross_asset/cross_asset_daily.parquet',
                        help='Output path')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('FRED_API_KEY')
    if not api_key:
        logger.error("FRED API key required. Set FRED_API_KEY env var or use --api-key")
        logger.info("Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    # Download raw data
    logger.info("Downloading FRED series...")
    raw_df = download_fred_series(api_key, args.start_date)
    
    # Compute derived features
    logger.info("Computing derived features...")
    enhanced_df = compute_derived_features(raw_df)
    
    # Forward fill (many series are weekly/monthly)
    enhanced_df = enhanced_df.ffill()
    
    # Shift by 1 day (T-1 for prediction at T)
    enhanced_df = enhanced_df.shift(1)
    
    # Drop early rows with too many NaN
    min_valid = len(enhanced_df.columns) * 0.3
    enhanced_df = enhanced_df.dropna(thresh=int(min_valid))
    
    # Fill remaining NaN
    enhanced_df = enhanced_df.fillna(0)
    
    # Save
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    enhanced_df.to_parquet(output_path)
    
    logger.info(f"Saved: {output_path}")
    logger.info(f"  Shape: {enhanced_df.shape}")
    logger.info(f"  Date range: {enhanced_df.index.min()} to {enhanced_df.index.max()}")
    logger.info(f"  Features: {list(enhanced_df.columns)}")


if __name__ == '__main__':
    main()
