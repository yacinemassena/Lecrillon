"""
Build preprocessed economic calendar features for Mamba training.

Reads raw econcalendar.csv → filters to US+EU → parses values → computes
per-event z-score stats → outputs parquet + vocab JSON.

Usage:
    python tools/build_econ_features.py
    python tools/build_econ_features.py --input datasets/econcalendar.csv --output datasets/econ_calendar
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Value parser — handles %, K, M, B, T suffixes, bond auctions (pipe), plain
# ---------------------------------------------------------------------------
SUFFIX_MULT = {
    'K': 1_000,
    'M': 1_000_000,
    'B': 1_000_000_000,
    'T': 1_000_000_000_000,
}

def parse_value(raw) -> float:
    """Parse mixed-format economic value to float.
    
    Handles: '0.3%', '120K', '16.5M', '-785M', '3.94|1.8', '54.8', '', NaN
    """
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    if s == '' or s.lower() == 'nan':
        return np.nan
    
    # Bond auction format: "3.94|1.8" → take first value (yield)
    if '|' in s:
        s = s.split('|')[0].strip()
    
    # Percentage: "0.3%" → 0.003
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return np.nan
    
    # Suffixed numbers: "120K", "16.5M", "-785M", "10.9B"
    upper = s.upper()
    for suffix, mult in SUFFIX_MULT.items():
        if upper.endswith(suffix):
            try:
                return float(upper[:-1]) * mult
            except ValueError:
                return np.nan
    
    # Plain number
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------
def build_econ_features(input_csv: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read raw CSV (no header)
    df = pd.read_csv(
        input_csv, header=None,
        names=['date', 'time', 'currency', 'impact', 'event',
               'col5', 'col6', 'actual', 'forecast', 'previous'],
    )
    print(f"Raw events: {len(df):,}")
    
    # Filter to US + Europe
    df = df[df['currency'].isin(['USD', 'EUR', 'GBP', 'CHF'])].copy()
    print(f"After US+EU filter: {len(df):,}")
    
    # Drop unused columns
    df.drop(columns=['col5', 'col6'], inplace=True)
    
    # Parse date + time into datetime and Unix timestamp
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
    df['timestamp'] = (df['datetime'].astype(np.int64) // 10**9).astype(np.int64)
    df['date_parsed'] = df['datetime'].dt.date
    
    # Sort by datetime
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --- Build vocabs ---
    # Event vocab (sorted alphabetically for determinism)
    events_sorted = sorted(df['event'].unique())
    event_vocab = {name: idx + 1 for idx, name in enumerate(events_sorted)}  # 0 reserved for padding
    
    # Currency vocab
    currencies_sorted = sorted(df['currency'].unique())
    currency_vocab = {name: idx + 1 for idx, name in enumerate(currencies_sorted)}  # 0 reserved for padding
    
    # Impact mapping
    impact_map = {'N': 0, 'L': 1, 'M': 2, 'H': 3}
    
    # --- Map categorical features ---
    df['event_id'] = df['event'].map(event_vocab).astype(np.int16)
    df['currency_id'] = df['currency'].map(currency_vocab).astype(np.int8)
    df['impact_ord'] = df['impact'].map(impact_map).fillna(0).astype(np.int8)
    df['is_usd'] = (df['currency'] == 'USD').astype(np.int8)
    
    # Time of day normalized [0, 1]
    df['time_of_day'] = (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) / 1440.0
    
    # --- Parse numeric values ---
    print("Parsing values...")
    df['actual_raw'] = df['actual'].apply(parse_value)
    df['forecast_raw'] = df['forecast'].apply(parse_value)
    df['previous_raw'] = df['previous'].apply(parse_value)
    
    # --- Per-event-type z-score stats ---
    print("Computing per-event z-score stats...")
    # Combine all values per event type for robust stats
    event_stats = {}
    for eid_name, eid in event_vocab.items():
        mask = df['event_id'] == eid
        vals = pd.concat([
            df.loc[mask, 'actual_raw'],
            df.loc[mask, 'forecast_raw'],
            df.loc[mask, 'previous_raw'],
        ]).dropna()
        
        if len(vals) >= 3:
            mean = float(vals.mean())
            std = float(vals.std())
            if std < 1e-10:
                std = 1.0  # Avoid division by zero
        else:
            mean = 0.0
            std = 1.0
        
        event_stats[str(eid)] = {'mean': mean, 'std': std, 'name': eid_name, 'count': int(len(vals))}
    
    # Apply z-score normalization
    means = df['event_id'].map(lambda x: event_stats[str(x)]['mean'])
    stds = df['event_id'].map(lambda x: event_stats[str(x)]['std'])
    
    df['actual_z'] = ((df['actual_raw'] - means) / stds).fillna(0.0).astype(np.float32)
    df['forecast_z'] = ((df['forecast_raw'] - means) / stds).fillna(0.0).astype(np.float32)
    df['previous_z'] = ((df['previous_raw'] - means) / stds).fillna(0.0).astype(np.float32)
    
    # Has-value flags
    df['has_actual'] = df['actual_raw'].notna().astype(np.int8)
    df['has_forecast'] = df['forecast_raw'].notna().astype(np.int8)
    
    # --- Event rank today (position within each day's events) ---
    print("Computing event_rank_today...")
    df['event_rank_today'] = df.groupby('date').cumcount().astype(np.int16)
    # Normalize by max events in a day
    max_events_per_day = df.groupby('date').size().max()
    df['event_rank_today_norm'] = (df['event_rank_today'] / max(max_events_per_day, 1)).astype(np.float32)
    
    # --- Days since last same event ---
    print("Computing days_since_last_same...")
    df['days_since_last_same'] = np.float32(90.0)  # Default: capped max
    for eid in df['event_id'].unique():
        mask = df['event_id'] == eid
        dates = df.loc[mask, 'date_parsed'].values
        diffs = np.zeros(len(dates), dtype=np.float32)
        diffs[0] = 90.0  # First occurrence
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i-1]).days if hasattr(dates[i] - dates[i-1], 'days') else 90
            diffs[i] = min(float(delta), 90.0)
        df.loc[mask, 'days_since_last_same'] = diffs
    
    # Normalize: /90 so range is [0, 1]
    df['days_since_last_same_norm'] = (df['days_since_last_same'] / 90.0).astype(np.float32)
    
    # --- Select output columns ---
    out_cols = [
        'date_parsed', 'timestamp', 'event_id', 'currency_id', 'impact_ord',
        'is_usd', 'time_of_day', 'actual_z', 'forecast_z', 'previous_z',
        'has_actual', 'has_forecast', 'event_rank_today_norm',
        'days_since_last_same_norm',
    ]
    out = df[out_cols].copy()
    out.rename(columns={'date_parsed': 'date'}, inplace=True)
    
    # Convert date to string for parquet compatibility
    out['date'] = out['date'].astype(str)
    
    # Save parquet
    parquet_path = output_path / 'econ_events.parquet'
    out.to_parquet(parquet_path, index=False)
    print(f"Saved {len(out):,} events to {parquet_path}")
    
    # Save vocab
    vocab = {
        'event_vocab': event_vocab,
        'currency_vocab': currency_vocab,
        'impact_map': impact_map,
        'event_stats': event_stats,
        'num_event_types': len(event_vocab),
        'num_currencies': len(currency_vocab),
        'max_events_per_day': int(max_events_per_day),
    }
    vocab_path = output_path / 'vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to {vocab_path}")
    
    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Event types: {len(event_vocab)}")
    print(f"Currencies: {len(currency_vocab)} ({currencies_sorted})")
    print(f"Date range: {out['date'].min()} to {out['date'].max()}")
    print(f"Max events/day: {max_events_per_day}")
    print(f"Events with actual: {df['has_actual'].sum():,} / {len(df):,}")
    
    # Show sample z-score stats for key events
    key_events = ['Non-Farm Employment Change', 'CPI m/m', 'FOMC Statement',
                  'Unemployment Rate', 'GDP q/q', 'ISM Manufacturing PMI']
    print(f"\nPer-event z-score stats (sample):")
    for name in key_events:
        eid = event_vocab.get(name)
        if eid and str(eid) in event_stats:
            s = event_stats[str(eid)]
            print(f"  {name}: mean={s['mean']:.4f}, std={s['std']:.4f}, n={s['count']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build econ calendar features')
    parser.add_argument('--input', default='datasets/econcalendar.csv',
                        help='Path to raw econ calendar CSV')
    parser.add_argument('--output', default='datasets/econ_calendar',
                        help='Output directory for parquet + vocab')
    args = parser.parse_args()
    
    build_econ_features(args.input, args.output)
