#!/usr/bin/env python3
"""
Build VIX features at 2-min resolution for extended hours (04:00-22:00 ET).

VIX Mamba runs ~540 bars/day (18h) vs Stock Mamba's 195 bars (6.5h market hours).
Timestamps stored as nanoseconds since epoch (int64) for consistent alignment.

Features computed (21 total):
- Base OHLC: open, high, low, close (4) - volume dropped (unreliable overnight)
- VVIX: vvix, previousclose (2)
- Moving averages: 5dMA, 10dMA, 20dMA (3)
- Realized volatility: rv_5m, rv_30m, rv_2h, rv_acceleration, rv_change_5, rv_change_30, rv_ratio (7)
- VIX technicals: vix_vvix_ratio, vix_zscore_20d, vix_percentile_252d, distance_from_20dMA (4)
- Derivatives: vix_velocity_15, vix_velocity_75, vix_acceleration_15, vix_acceleration_75 (4)
- Variance risk premium: rv_ratio_to_vix (1)

Usage:
    python tools/build_vix_features.py
    python tools/build_vix_features.py --start-date 2020-01-01
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Constants
# VIX extended hours: ~04:00-22:00 ET = 18 hours = 540 bars/day at 2-min
VIX_BARS_PER_DAY = 540
# Stock market hours for MA/rolling calculations still use trading day convention
BARS_PER_DAY = 195  # ~6.5 hours * 60 / 2 = 195 bars per trading day
MA_WINDOWS = {
    '5dMA': 5 * BARS_PER_DAY,      # 975 bars
    '10dMA': 10 * BARS_PER_DAY,    # 1950 bars
    '20dMA': 20 * BARS_PER_DAY,    # 3900 bars
}
ZSCORE_WINDOW = 20 * BARS_PER_DAY   # 3900 bars for 20-day z-score
PERCENTILE_WINDOW = 252 * BARS_PER_DAY  # ~49140 bars for 252-day percentile

# RV windows (in 2-min bars)
RV_5M_BARS = 3      # 5 min / 2 = 2.5 -> 3 bars
RV_30M_BARS = 15    # 30 min / 2 = 15 bars
RV_2H_BARS = 60     # 2 hours / 2 = 60 bars

# Velocity/acceleration windows
VELOCITY_SHORT = 15   # 30 min
VELOCITY_LONG = 75    # 2.5 hours


def load_vix_1min(vix_dir: Path) -> pd.DataFrame:
    """Load all VIX 1-min data from yearly CSV files."""
    all_data = []
    
    csv_files = sorted(vix_dir.glob('VIX_*.csv'))
    logger.info(f"Loading {len(csv_files)} VIX yearly files...")
    
    for csv_file in tqdm(csv_files, desc="Loading VIX"):
        try:
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'], utc=True)
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No VIX data loaded")
    
    vix_df = pd.concat(all_data, ignore_index=True)
    vix_df = vix_df.sort_values('date').reset_index(drop=True)
    logger.info(f"Loaded VIX: {len(vix_df)} rows, {vix_df['date'].min()} to {vix_df['date'].max()}")
    
    return vix_df


def load_vvix_1min(vix_dir: Path) -> pd.DataFrame:
    """Load VVIX 1-min data."""
    vvix_file = vix_dir / 'vvix_1min_historical.csv'
    if not vvix_file.exists():
        logger.warning("VVIX file not found, will use NaN")
        return pd.DataFrame()
    
    vvix_df = pd.read_csv(vvix_file)
    vvix_df['timestamp'] = pd.to_datetime(vvix_df['timestamp'], utc=True)
    vvix_df = vvix_df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded VVIX: {len(vvix_df)} rows")
    
    return vvix_df


def aggregate_to_2min(df: pd.DataFrame, ts_col: str = 'date') -> pd.DataFrame:
    """Aggregate 1-min bars to 2-min bars. Timestamps converted to nanoseconds."""
    df = df.copy()
    
    # Create 2-min bucket (floor to even minutes)
    df['bucket'] = df[ts_col].dt.floor('2min')
    
    # Aggregate OHLC (drop volume - unreliable overnight)
    agg = df.groupby('bucket').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).reset_index()
    
    agg.rename(columns={'bucket': 'bar_timestamp'}, inplace=True)
    return agg


def compute_rv(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling realized volatility as sqrt(sum(returns^2))."""
    if len(returns) < window:
        return np.full(len(returns), np.nan)
    
    # Use rolling sum of squared returns
    returns_sq = returns ** 2
    rv = np.full(len(returns), np.nan)
    
    for i in range(window - 1, len(returns)):
        rv[i] = np.sqrt(np.sum(returns_sq[i - window + 1:i + 1]))
    
    return rv


def compute_velocity(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rate of change (first derivative) over window."""
    velocity = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        velocity[i] = series[i] - series[i - window]
    return velocity


def compute_acceleration(series: np.ndarray, window: int) -> np.ndarray:
    """Compute change of velocity (second derivative) over window."""
    velocity = compute_velocity(series, window)
    accel = np.full(len(series), np.nan)
    for i in range(window, len(velocity)):
        if not np.isnan(velocity[i]) and not np.isnan(velocity[i - window]):
            accel[i] = velocity[i] - velocity[i - window]
    return accel


def compute_rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score: (x - rolling_mean) / rolling_std."""
    zscore = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1:i + 1]
        mu = np.mean(window_data)
        sigma = np.std(window_data)
        if sigma > 1e-8:
            zscore[i] = (series[i] - mu) / sigma
    return zscore


def compute_rolling_percentile(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling percentile rank (0-1)."""
    pct = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1:i + 1]
        rank = np.sum(window_data <= series[i])
        pct[i] = rank / window
    return pct


def compute_rolling_ma(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling simple moving average."""
    ma = np.full(len(series), np.nan)
    cumsum = np.nancumsum(series)
    ma[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    return ma


def load_spy_rv(stock_dir: Path, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """Load SPY 2-min data and compute 5-day rolling RV for each trading day.
    
    Returns dict: trading_day -> 5d RV (computed from last 975 bars of SPY returns)
    """
    logger.info("Loading SPY data for RV computation...")
    
    spy_returns = []
    spy_dates = []
    
    # Get unique trading dates from VIX timestamps
    unique_dates = sorted(set(d.date() for d in dates))
    
    for d in tqdm(unique_dates, desc="Loading SPY"):
        date_str = d.strftime('%Y-%m-%d')
        parquet_file = stock_dir / f"{date_str}.parquet"
        
        if not parquet_file.exists():
            continue
        
        try:
            # Load only SPY
            df = pl.read_parquet(parquet_file)
            spy_df = df.filter(pl.col('ticker') == 'SPY')
            
            if len(spy_df) == 0:
                continue
            
            # Get close prices and compute returns
            closes = spy_df.sort('bar_timestamp')['close'].to_numpy()
            if len(closes) > 1:
                returns = np.log(closes[1:] / closes[:-1])
                returns = returns[np.isfinite(returns)]
                spy_returns.extend(returns.tolist())
                spy_dates.extend([d] * len(returns))
                
        except Exception as e:
            logger.warning(f"Error loading SPY for {date_str}: {e}")
            continue
    
    if not spy_returns:
        logger.warning("No SPY data loaded")
        return {}
    
    # Compute 5-day rolling RV
    spy_returns = np.array(spy_returns)
    spy_dates_arr = np.array(spy_dates)
    
    # Window: 5 days * ~195 bars = 975 bars
    rv_window = 5 * BARS_PER_DAY
    rv_values = compute_rv(spy_returns, rv_window)
    
    # Map: date -> last RV value for that day (annualized)
    # Annualize: sqrt(252 * 195) * rv (since we have 2-min bars)
    annualize_factor = np.sqrt(252 * BARS_PER_DAY)
    
    spy_rv_by_date = {}
    for i, (d, rv) in enumerate(zip(spy_dates_arr, rv_values)):
        if not np.isnan(rv):
            # Annualize and convert to percentage (VIX is in %)
            spy_rv_by_date[d] = rv * annualize_factor * 100
    
    logger.info(f"Computed SPY 5d RV for {len(spy_rv_by_date)} days")
    return spy_rv_by_date


def build_vix_features(
    vix_dir: Path,
    stock_dir: Path,
    output_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Build all VIX features and save as daily parquet files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    vix_1min = load_vix_1min(vix_dir)
    vvix_1min = load_vvix_1min(vix_dir)
    
    # Aggregate VIX to 2-min
    logger.info("Aggregating VIX to 2-min bars...")
    vix_2min = aggregate_to_2min(vix_1min, ts_col='date')
    
    # Aggregate VVIX to 2-min and merge
    if len(vvix_1min) > 0:
        vvix_2min = aggregate_to_2min(vvix_1min, ts_col='timestamp')
        vvix_2min = vvix_2min[['bar_timestamp', 'close']].rename(columns={'close': 'vvix'})
        vix_2min = vix_2min.merge(vvix_2min, on='bar_timestamp', how='left')
    else:
        vix_2min['vvix'] = np.nan
    
    # Sort by timestamp
    vix_2min = vix_2min.sort_values('bar_timestamp').reset_index(drop=True)
    logger.info(f"VIX 2-min: {len(vix_2min)} bars")
    
    # Filter date range if specified
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        vix_2min = vix_2min[vix_2min['bar_timestamp'] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        vix_2min = vix_2min[vix_2min['bar_timestamp'] <= end_ts]
    
    vix_2min = vix_2min.reset_index(drop=True)
    
    # Load SPY RV
    spy_rv_by_date = load_spy_rv(stock_dir, vix_2min['bar_timestamp'].tolist())
    
    # Convert to numpy for fast computation
    close = vix_2min['close'].values
    vvix = vix_2min['vvix'].values
    
    # Compute log returns for RV
    log_returns = np.zeros(len(close))
    log_returns[1:] = np.log(close[1:] / close[:-1])
    log_returns[0] = np.nan
    log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info("Computing features...")
    
    # Previous close (lag 1)
    previousclose = np.roll(close, 1)
    previousclose[0] = np.nan
    
    # Moving averages
    ma_5d = compute_rolling_ma(close, MA_WINDOWS['5dMA'])
    ma_10d = compute_rolling_ma(close, MA_WINDOWS['10dMA'])
    ma_20d = compute_rolling_ma(close, MA_WINDOWS['20dMA'])
    
    # Realized volatility (VIX price-based, not SPY)
    rv_5m = compute_rv(log_returns, RV_5M_BARS)
    rv_30m = compute_rv(log_returns, RV_30M_BARS)
    rv_2h = compute_rv(log_returns, RV_2H_BARS)
    
    # RV ratios and changes
    rv_acceleration = rv_30m / (rv_2h + 1e-8)
    rv_ratio = rv_5m / (rv_30m + 1e-8)
    
    # RV changes (percent change)
    rv_change_5 = np.full(len(rv_5m), np.nan)
    rv_change_30 = np.full(len(rv_30m), np.nan)
    for i in range(RV_5M_BARS, len(rv_5m)):
        if rv_5m[i - RV_5M_BARS] > 1e-8:
            rv_change_5[i] = (rv_5m[i] - rv_5m[i - RV_5M_BARS]) / rv_5m[i - RV_5M_BARS]
    for i in range(RV_30M_BARS, len(rv_30m)):
        if rv_30m[i - RV_30M_BARS] > 1e-8:
            rv_change_30[i] = (rv_30m[i] - rv_30m[i - RV_30M_BARS]) / rv_30m[i - RV_30M_BARS]
    
    # VIX/VVIX ratio
    vix_vvix_ratio = close / (vvix + 1e-8)
    
    # Z-score (20-day)
    vix_zscore_20d = compute_rolling_zscore(close, ZSCORE_WINDOW)
    
    # Percentile (252-day) - use smaller window to avoid excessive NaN
    # For practical purposes, use 60 days initially then expand
    pct_window = min(PERCENTILE_WINDOW, len(close))
    vix_percentile_252d = compute_rolling_percentile(close, min(252 * BARS_PER_DAY, len(close)))
    
    # Distance from 20dMA
    distance_from_20dMA = close - ma_20d
    
    # Velocity at two timescales
    vix_velocity_15 = compute_velocity(close, VELOCITY_SHORT)
    vix_velocity_75 = compute_velocity(close, VELOCITY_LONG)
    
    # Acceleration at two timescales
    vix_acceleration_15 = compute_acceleration(close, VELOCITY_SHORT)
    vix_acceleration_75 = compute_acceleration(close, VELOCITY_LONG)
    
    # Variance risk premium: SPY RV / VIX
    # Map SPY RV to each bar based on trading date
    rv_ratio_to_vix = np.full(len(close), np.nan)
    for i, ts in enumerate(vix_2min['bar_timestamp']):
        d = ts.date()
        if d in spy_rv_by_date:
            # SPY RV (annualized %) / VIX (%)
            rv_ratio_to_vix[i] = spy_rv_by_date[d] / (close[i] + 1e-8)
    
    # Convert timestamps to nanoseconds since epoch (int64)
    # .values.view('int64') gives microseconds for tz-aware, multiply by 1000 for ns
    # This ensures consistent alignment across all data streams
    bar_ts_ns = vix_2min['bar_timestamp'].values.view('int64') * 1000
    
    # Build output DataFrame (21 features + timestamp)
    features_df = pd.DataFrame({
        'bar_timestamp': bar_ts_ns,  # nanoseconds since epoch
        'open': vix_2min['open'],
        'high': vix_2min['high'],
        'low': vix_2min['low'],
        'close': close,
        'vvix': vvix,
        'previousclose': previousclose,
        '5dMA': ma_5d,
        '10dMA': ma_10d,
        '20dMA': ma_20d,
        'rv_5m': rv_5m,
        'rv_30m': rv_30m,
        'rv_2h': rv_2h,
        'rv_acceleration': rv_acceleration,
        'rv_change_5': rv_change_5,
        'rv_change_30': rv_change_30,
        'rv_ratio': rv_ratio,
        'vix_vvix_ratio': vix_vvix_ratio,
        'vix_zscore_20d': vix_zscore_20d,
        'vix_percentile_252d': vix_percentile_252d,
        'distance_from_20dMA': distance_from_20dMA,
        'vix_velocity_15': vix_velocity_15,
        'vix_velocity_75': vix_velocity_75,
        'vix_acceleration_15': vix_acceleration_15,
        'vix_acceleration_75': vix_acceleration_75,
        'rv_ratio_to_vix': rv_ratio_to_vix,
    })
    
    # Replace inf with nan
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Extract trading date from nanoseconds timestamp
    features_df['trading_date'] = pd.to_datetime(features_df['bar_timestamp'], unit='ns').dt.date
    
    # Save as daily parquet files
    logger.info("Saving daily parquet files...")
    dates = features_df['trading_date'].unique()
    
    for d in tqdm(dates, desc="Saving"):
        day_df = features_df[features_df['trading_date'] == d].drop(columns=['trading_date'])
        output_file = output_dir / f"{d}.parquet"
        day_df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved {len(dates)} daily files to {output_dir}")
    
    # Print feature stats
    print("\n" + "=" * 60)
    print("VIX Feature Statistics")
    print("=" * 60)
    print(f"Total bars: {len(features_df)}")
    print(f"Date range: {features_df['bar_timestamp'].min()} to {features_df['bar_timestamp'].max()}")
    print(f"Trading days: {len(dates)}")
    print(f"\nFeature coverage (non-NaN %):")
    for col in features_df.columns:
        if col not in ['bar_timestamp', 'trading_date']:
            pct = features_df[col].notna().mean() * 100
            print(f"  {col}: {pct:.1f}%")


def validate_features(output_dir: Path) -> None:
    """Validate VIX features structure and coverage."""
    logger.info("Validating VIX features...")
    
    vix_files = sorted(output_dir.glob('*.parquet'))[:10]
    
    for vix_file in vix_files:
        date_str = vix_file.stem
        vix_df = pd.read_parquet(vix_file)
        
        # Check timestamp is int64 nanoseconds
        assert vix_df['bar_timestamp'].dtype == 'int64', f"{date_str}: timestamp not int64"
        
        # Check bar count (extended hours: ~540 bars/day)
        n_bars = len(vix_df)
        logger.info(f"{date_str}: {n_bars} bars")
        
        # Check feature coverage
        for col in ['open', 'high', 'low', 'close']:
            assert col in vix_df.columns, f"{date_str}: missing {col}"
            coverage = vix_df[col].notna().mean() * 100
            if coverage < 90:
                logger.warning(f"{date_str}: {col} coverage only {coverage:.1f}%")
    
    logger.info("Validation passed")


def main():
    parser = argparse.ArgumentParser(
        description='Build VIX features at 2-min resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--vix-dir', type=str,
                        default='datasets/VIX',
                        help='VIX data directory')
    parser.add_argument('--stock-dir', type=str,
                        default='datasets/Stock_Data_2min',
                        help='Stock 2-min data directory (for SPY RV)')
    parser.add_argument('--output-dir', type=str,
                        default='datasets/VIX/Vix_features',
                        help='Output directory for VIX features')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after building')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    vix_dir = (script_dir / args.vix_dir).resolve()
    stock_dir = (script_dir / args.stock_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    logger.info(f"VIX dir: {vix_dir}")
    logger.info(f"Stock dir: {stock_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    build_vix_features(
        vix_dir=vix_dir,
        stock_dir=stock_dir,
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    if args.validate:
        validate_features(output_dir)


if __name__ == '__main__':
    main()
