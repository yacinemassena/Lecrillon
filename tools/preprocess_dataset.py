"""
Preprocess all parquet data into numpy memmaps for instant loading.

Each Mamba feed gets its own memmap + index file:
  stock_features.npy      + stock_index.json
  stock_timestamps.npy
  options_features.npy    + options_index.json
  vix_features.npy        + vix_index.json
  vix_timestamps.npy
  news_embeddings.npy     + news_index.json
  news_timestamps.npy
  gdelt_features.npy      + gdelt_index.json
  gdelt_timestamps.npy
  macro_features.npy      + macro_index.json
  fundamentals_features.npy + fundamentals_index.json
  econ_numeric.npy        + econ_index.json
  econ_event_ids.npy
  econ_currency_ids.npy
  econ_timestamps.npy

Usage:
    python tools/preprocess_dataset.py --data-root datasets --workers 8
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from loader.bar_mamba_dataset import (
    DEFAULT_FEATURES,
    OPTION_FEATURES,
    OPTION_DERIVED_FEATURES,
    VIX_FEATURES,
    NUM_STOCK_FEATURES,
    NUM_OPTION_FEATURES,
    NUM_VIX_FEATURES,
    compute_ticker_dispersion,
    compute_liquidity_stress,
    compute_ofi_derived,
    compute_intraday_vol_skew,
    compute_option_derived,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_BARS_PER_DAY = 195  # stock cap


# ---------------------------------------------------------------------------
# Per-day processing functions (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def process_stock_day(file_path: str) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Process one day of stock data. Returns (date_str, features, timestamps)."""
    fp = Path(file_path)
    date_str = fp.stem  # YYYY-MM-DD

    try:
        df = pl.read_parquet(fp)
    except Exception as e:
        logger.warning(f"Failed to read {fp.name}: {e}")
        return None

    if len(df) == 0:
        return None

    ts_col = 'bar_timestamp' if 'bar_timestamp' in df.columns else 'timestamp'
    if ts_col not in df.columns:
        return None

    df = df.sort(ts_col)

    derived_features = {'liquidity_stress', 'ofi_acceleration', 'abs_ofi',
                        'intraday_vol_skew', 'ticker_dispersion'}

    # Compute ticker_dispersion before feature extraction
    ticker_disp = None
    if 'ticker_dispersion' in DEFAULT_FEATURES:
        ticker_disp = compute_ticker_dispersion(df, list(df.columns), ts_col)

    # Extract raw features (excluding derived)
    avail = [f for f in DEFAULT_FEATURES if f in df.columns and f not in derived_features]
    if not avail:
        return None

    features = df.select(avail).to_numpy().astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute derived features
    if 'liquidity_stress' in DEFAULT_FEATURES:
        ls = compute_liquidity_stress(features, avail)
        features = np.column_stack([features, ls])
        avail = avail + ['liquidity_stress']

    if 'ofi_acceleration' in DEFAULT_FEATURES or 'abs_ofi' in DEFAULT_FEATURES:
        ofi_accel, abs_ofi = compute_ofi_derived(features, avail)
        if 'ofi_acceleration' in DEFAULT_FEATURES:
            features = np.column_stack([features, ofi_accel])
            avail = avail + ['ofi_acceleration']
        if 'abs_ofi' in DEFAULT_FEATURES:
            features = np.column_stack([features, abs_ofi])
            avail = avail + ['abs_ofi']

    if 'intraday_vol_skew' in DEFAULT_FEATURES:
        vol_skew = compute_intraday_vol_skew(features, avail)
        features = np.column_stack([features, vol_skew])
        avail = avail + ['intraday_vol_skew']

    if 'ticker_dispersion' in DEFAULT_FEATURES and ticker_disp is not None:
        features = np.column_stack([features, ticker_disp])
        avail = avail + ['ticker_dispersion']

    # Timestamps
    timestamps = df[ts_col].to_numpy()
    if hasattr(timestamps, 'astype'):
        timestamps = timestamps.astype('datetime64[s]').astype(np.int64)

    # Pad missing features to NUM_STOCK_FEATURES
    if len(avail) < NUM_STOCK_FEATURES:
        padded = np.zeros((len(features), NUM_STOCK_FEATURES), dtype=np.float32)
        padded[:, :len(avail)] = features
        features = padded

    # Cap bars per day
    if len(features) > MAX_BARS_PER_DAY:
        features = features[:MAX_BARS_PER_DAY]
        timestamps = timestamps[:MAX_BARS_PER_DAY]

    return (date_str, features, timestamps)


def process_options_day(file_path: str) -> Optional[Tuple[str, np.ndarray]]:
    """Process one day of options data. Returns (date_str, features)."""
    fp = Path(file_path)
    date_str = fp.stem

    try:
        df = pl.read_parquet(fp)
    except Exception as e:
        logger.warning(f"Failed to read options {fp.name}: {e}")
        return None

    if len(df) == 0:
        return None

    ts_col = 'bar_timestamp' if 'bar_timestamp' in df.columns else 'timestamp'
    if ts_col not in df.columns:
        return None
    df = df.sort(ts_col)

    # Aggregate across underlyings per timestamp
    agg_exprs = []
    for feat in OPTION_FEATURES:
        if feat in OPTION_DERIVED_FEATURES:
            continue
        if feat not in df.columns:
            continue
        if 'ratio' in feat or 'skew' in feat or 'vs_20d' in feat:
            agg_exprs.append(pl.col(feat).mean().alias(feat))
        else:
            agg_exprs.append(pl.col(feat).sum().alias(feat))

    if not agg_exprs:
        return None

    df_agg = df.group_by(ts_col).agg(agg_exprs).sort(ts_col)

    avail = [f for f in OPTION_FEATURES if f in df_agg.columns and f not in OPTION_DERIVED_FEATURES]
    if not avail:
        return None

    features = df_agg.select(avail).to_numpy().astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute derived
    derived = compute_option_derived(features, avail)
    for feat_name in ['skew_change']:
        if feat_name in OPTION_FEATURES:
            features = np.column_stack([features, derived[feat_name]])
            avail = avail + [feat_name]

    # Pad missing features
    if len(avail) < NUM_OPTION_FEATURES:
        padded = np.zeros((len(features), NUM_OPTION_FEATURES), dtype=np.float32)
        padded[:, :len(avail)] = features
        features = padded

    return (date_str, features)


def process_vix_day(file_path: str) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Process one day of VIX feature data. Returns (date_str, features, timestamps)."""
    fp = Path(file_path)
    date_str = fp.stem

    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        logger.warning(f"Failed to read VIX {fp.name}: {e}")
        return None

    if len(df) == 0:
        return None

    df = df.sort_values('bar_timestamp')
    timestamps = pd.to_datetime(df['bar_timestamp']).values.astype('datetime64[s]').astype(np.int64)

    avail = [f for f in VIX_FEATURES if f in df.columns]
    if not avail:
        return None

    features = df[avail].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if len(avail) < NUM_VIX_FEATURES:
        padded = np.zeros((len(features), NUM_VIX_FEATURES), dtype=np.float32)
        padded[:, :len(avail)] = features
        features = padded

    return (date_str, features, timestamps)


def process_gdelt_day(file_path: str) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Process one day of GDELT data. Returns (date_str, features, timestamps)."""
    fp = Path(file_path)
    # Date from path: .../YYYY/MM/DD.parquet
    date_str = f"{fp.parent.parent.name}-{fp.parent.name}-{fp.stem}"

    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        logger.warning(f"Failed to read GDELT {fp}: {e}")
        return None

    if len(df) == 0:
        return None

    df = df.sort_values('bucket_end')

    # Embeddings [N, 384]
    embeddings = np.stack(df['embedding'].values).astype(np.float32)

    # Stats [N, 7]
    GDELT_STATS_DIM = 7
    stats_cols = [
        ('article_count', np.log1p),
        ('goldstein_scale_mean', None),
        ('goldstein_scale_min', None),
        ('tone_mean', None),
        ('tone_negative_max', None),
        ('tone_polarity_mean', None),
        ('num_sources_mean', None),
    ]
    stats = np.zeros((len(df), GDELT_STATS_DIM), dtype=np.float32)
    for i, (col_name, transform) in enumerate(stats_cols):
        if col_name in df.columns:
            vals = df[col_name].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            if transform is not None:
                vals = transform(vals)
            stats[:, i] = vals

    features = np.concatenate([embeddings, stats], axis=1)  # [N, 391]

    # Timestamps
    bucket_end = pd.to_datetime(df['bucket_end'])
    if bucket_end.dt.tz is None:
        bucket_end = bucket_end.dt.tz_localize('UTC')
    else:
        bucket_end = bucket_end.dt.tz_convert('UTC')
    timestamps = (bucket_end.astype(np.int64) // 1_000_000_000).values

    return (date_str, features, timestamps)


def process_news_day(file_path: str) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Process one day of news data. Returns (date_str, embeddings, timestamps)."""
    fp = Path(file_path)
    date_str = fp.stem  # YYYY-MM-DD

    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        logger.warning(f"Failed to read news {fp.name}: {e}")
        return None

    if len(df) == 0:
        return None

    embeddings = np.stack(df['title_embedding'].values).astype(np.float32)

    # Detect timestamp unit
    sample_ts = df['timestamp'].iloc[0]
    ts_len = len(str(abs(sample_ts)))
    if ts_len >= 19:
        divisor = 1_000_000_000
    elif ts_len >= 16:
        divisor = 1_000_000
    else:
        divisor = 1
    timestamps = (df['timestamp'].values // divisor).astype(np.int64)

    return (date_str, embeddings, timestamps)


# ---------------------------------------------------------------------------
# Build memmap from processed results
# ---------------------------------------------------------------------------

def build_memmap(results: List, out_dir: Path, name: str, has_timestamps: bool = True):
    """Build memmap + index from list of (date_str, features[, timestamps]) tuples."""
    # Sort by date
    results.sort(key=lambda x: x[0])

    all_features = [r[1] for r in results]
    feat_dim = all_features[0].shape[1] if all_features[0].ndim == 2 else 1
    total_rows = sum(f.shape[0] for f in all_features)

    logger.info(f"  {name}: {len(results)} days, {total_rows:,} rows, dim={feat_dim}")

    # Concatenate and save features
    features = np.concatenate(all_features, axis=0)
    np.save(out_dir / f"{name}_features.npy", features)

    # Build index
    index = {}
    offset = 0
    for r in results:
        date_str = r[0]
        length = r[1].shape[0]
        index[date_str] = {"offset": offset, "length": length}
        offset += length

    with open(out_dir / f"{name}_index.json", 'w') as f:
        json.dump(index, f)

    # Timestamps
    if has_timestamps:
        all_ts = [r[2] for r in results]
        timestamps = np.concatenate(all_ts, axis=0)
        np.save(out_dir / f"{name}_timestamps.npy", timestamps)

    return total_rows


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_feed(name: str, files: List[Path], process_fn, workers: int,
                 out_dir: Path, has_timestamps: bool = True) -> int:
    """Process a feed type using multiprocessing. Returns row count."""
    t0 = time.time()
    results = []
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_fn, str(f)): f for f in files}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == len(files):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                logger.info(f"  {name}: {done}/{len(files)} ({rate:.0f}/s, ETA {eta:.0f}s)")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.warning(f"  {name} worker error: {e}")

    if not results:
        logger.warning(f"  {name}: no results!")
        return 0

    total = build_memmap(results, out_dir, name, has_timestamps)
    elapsed = time.time() - t0
    logger.info(f"  {name}: done in {elapsed:.1f}s ({len(results)} days, {failed} failed, "
                f"{total:,} rows)")
    return total


def process_econ(data_root: Path, out_dir: Path):
    """Process econ calendar into per-day arrays."""
    econ_dir = data_root / "econ_calendar"
    parquet_file = econ_dir / "econ_events.parquet"
    if not parquet_file.exists():
        logger.info("  econ: no data found, skipping")
        return

    df = pd.read_parquet(parquet_file)
    if len(df) == 0:
        return

    # Group by date
    if 'date' in df.columns:
        df['date_key'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    else:
        logger.warning("  econ: no date column")
        return

    # Collect per-day arrays
    all_event_ids = []
    all_currency_ids = []
    all_numeric = []
    all_timestamps = []
    index = {}
    offset = 0

    for date_str, group in sorted(df.groupby('date_key')):
        n = len(group)
        event_ids = group['event_id'].values.astype(np.int16) if 'event_id' in group.columns else np.zeros(n, dtype=np.int16)
        currency_ids = group['currency_id'].values.astype(np.int8) if 'currency_id' in group.columns else np.zeros(n, dtype=np.int8)

        # Raw numeric fields (anchor-independent)
        numeric_cols = ['impact_ord', 'is_usd', 'time_of_day', 'actual_z', 'forecast_z',
                        'previous_z', 'has_actual', 'has_forecast',
                        'event_rank_today_norm', 'days_since_last_same_norm']
        numeric = np.zeros((n, len(numeric_cols)), dtype=np.float32)
        for i, col in enumerate(numeric_cols):
            if col in group.columns:
                numeric[:, i] = group[col].values.astype(np.float32)
        numeric = np.nan_to_num(numeric, nan=0.0)

        ts = group['timestamp'].values.astype(np.int64) if 'timestamp' in group.columns else np.zeros(n, dtype=np.int64)

        all_event_ids.append(event_ids)
        all_currency_ids.append(currency_ids)
        all_numeric.append(numeric)
        all_timestamps.append(ts)
        index[date_str] = {"offset": offset, "length": n}
        offset += n

    if not all_event_ids:
        return

    np.save(out_dir / "econ_event_ids.npy", np.concatenate(all_event_ids))
    np.save(out_dir / "econ_currency_ids.npy", np.concatenate(all_currency_ids))
    np.save(out_dir / "econ_numeric.npy", np.concatenate(all_numeric))
    np.save(out_dir / "econ_timestamps.npy", np.concatenate(all_timestamps))
    with open(out_dir / "econ_index.json", 'w') as f:
        json.dump(index, f)
    logger.info(f"  econ: {len(index)} days, {offset:,} events")


def process_macro(data_root: Path, out_dir: Path):
    """Process macro data into memmap."""
    macro_path = data_root / "MACRO" / "macro_daily_enhanced.parquet"
    if not macro_path.exists():
        logger.info("  macro: no data found, skipping")
        return

    df = pd.read_parquet(macro_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.set_index('date').sort_index()
    elif df.index.name == 'date' or df.index.dtype == 'datetime64[ns]':
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df = df.sort_index()

    # Compute derived features (must match _compute_macro_derived_features)
    if 'BAMLH0A0HYM2' in df.columns and 'BAMLC0A0CM' in df.columns:
        df['credit_spread'] = df['BAMLH0A0HYM2'] - df['BAMLC0A0CM']
    if 'T10Y2Y' in df.columns:
        df['yield_curve_velocity'] = df['T10Y2Y'] - df['T10Y2Y'].shift(5)
        df['yield_curve_velocity'] = df['yield_curve_velocity'].fillna(0.0)
    if 'STLFSI4' in df.columns:
        df['stlfsi4_change'] = df['STLFSI4'] - df['STLFSI4'].shift(1)
        df['stlfsi4_change'] = df['stlfsi4_change'].fillna(0.0)
    if 'days_until_fomc' in df.columns:
        df['fomc_proximity'] = np.exp(-df['days_until_fomc'] / 5.0)

    features = df.values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(out_dir / "macro_features.npy", features)

    index = {date_str: {"offset": i, "length": 1} for i, date_str in enumerate(df.index)}
    with open(out_dir / "macro_index.json", 'w') as f:
        json.dump(index, f)

    # Save column names for reference
    with open(out_dir / "macro_columns.json", 'w') as f:
        json.dump(list(df.columns), f)

    logger.info(f"  macro: {len(df)} days, {len(df.columns)} features")


def process_fundamentals(data_root: Path, out_dir: Path):
    """Process fundamentals data into memmap."""
    fund_path = data_root / "fundamentals" / "fundamentals_state.parquet"
    if not fund_path.exists():
        logger.info("  fundamentals: no data found, skipping")
        return

    df = pd.read_parquet(fund_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.set_index('date').sort_index()
    elif df.index.dtype == 'datetime64[ns]':
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df = df.sort_index()

    features = df.values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(out_dir / "fundamentals_features.npy", features)

    index = {date_str: {"offset": i, "length": 1} for i, date_str in enumerate(df.index)}
    with open(out_dir / "fundamentals_index.json", 'w') as f:
        json.dump(index, f)

    with open(out_dir / "fundamentals_columns.json", 'w') as f:
        json.dump(list(df.columns), f)

    logger.info(f"  fundamentals: {len(df)} days, {len(df.columns)} features")


def main():
    parser = argparse.ArgumentParser(description="Preprocess parquet data into numpy memmaps")
    parser.add_argument('--data-root', type=str, default='datasets',
                        help='Root directory containing all data subdirectories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data-root/preprocessed)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes (0 = cpu_count-1)')
    parser.add_argument('--feeds', type=str, nargs='*',
                        default=['stock', 'options', 'vix', 'gdelt', 'news',
                                 'econ', 'macro', 'fundamentals'],
                        help='Which feeds to process')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output) if args.output else data_root / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 1)

    logger.info(f"Data root: {data_root}")
    logger.info(f"Output:    {out_dir}")
    logger.info(f"Workers:   {workers}")
    logger.info(f"Feeds:     {args.feeds}")
    t_total = time.time()

    # --- Stock ---
    if 'stock' in args.feeds:
        logger.info("Processing STOCK...")
        stock_dir = data_root / "Stock_Data_2min"
        stock_files = sorted(stock_dir.glob("*.parquet"))
        if stock_files:
            process_feed("stock", stock_files, process_stock_day, workers, out_dir,
                         has_timestamps=True)
        else:
            logger.warning(f"No stock files in {stock_dir}")

    # --- Options ---
    if 'options' in args.feeds:
        logger.info("Processing OPTIONS...")
        opt_dir = data_root / "opt_trade_2min"
        opt_files = sorted(opt_dir.glob("*.parquet"))
        if opt_files:
            process_feed("options", opt_files, process_options_day, workers, out_dir,
                         has_timestamps=False)
        else:
            logger.warning(f"No options files in {opt_dir}")

    # --- VIX features ---
    if 'vix' in args.feeds:
        logger.info("Processing VIX FEATURES...")
        vix_dir = data_root / "VIX" / "Vix_features"
        vix_files = sorted(vix_dir.glob("*.parquet"))
        if vix_files:
            process_feed("vix", vix_files, process_vix_day, workers, out_dir,
                         has_timestamps=True)
        else:
            logger.warning(f"No VIX feature files in {vix_dir}")

    # --- GDELT ---
    if 'gdelt' in args.feeds:
        logger.info("Processing GDELT...")
        gdelt_dir = data_root / "GDELT"
        gdelt_files = sorted(gdelt_dir.rglob("*.parquet"))
        if gdelt_files:
            process_feed("gdelt", gdelt_files, process_gdelt_day, workers, out_dir,
                         has_timestamps=True)
        else:
            logger.warning(f"No GDELT files in {gdelt_dir}")

    # --- News ---
    if 'news' in args.feeds:
        logger.info("Processing NEWS...")
        news_dir = data_root / "benzinga_embeddings" / "news_daily"
        if not news_dir.exists():
            news_dir = data_root / "benzinga_embeddings"
        news_files = sorted(news_dir.glob("*.parquet"))
        if news_files:
            process_feed("news", news_files, process_news_day, workers, out_dir,
                         has_timestamps=True)
            # Rename to news_embeddings.npy for loader compatibility
            nf = out_dir / "news_features.npy"
            ne = out_dir / "news_embeddings.npy"
            if nf.exists() and not ne.exists():
                nf.rename(ne)
        else:
            logger.warning(f"No news files in {news_dir}")

    # --- Econ (single file, no multiprocessing needed) ---
    if 'econ' in args.feeds:
        logger.info("Processing ECON...")
        process_econ(data_root, out_dir)

    # --- Macro (single file) ---
    if 'macro' in args.feeds:
        logger.info("Processing MACRO...")
        process_macro(data_root, out_dir)

    # --- Fundamentals (single file) ---
    if 'fundamentals' in args.feeds:
        logger.info("Processing FUNDAMENTALS...")
        process_fundamentals(data_root, out_dir)

    # --- Summary ---
    elapsed = time.time() - t_total
    logger.info(f"\nAll done in {elapsed/60:.1f} minutes")

    # Print output sizes
    total_bytes = 0
    for f in sorted(out_dir.glob("*")):
        size = f.stat().st_size
        total_bytes += size
        logger.info(f"  {f.name:40s} {size/1e6:8.1f} MB")
    logger.info(f"  {'TOTAL':40s} {total_bytes/1e9:8.2f} GB")


if __name__ == '__main__':
    main()
