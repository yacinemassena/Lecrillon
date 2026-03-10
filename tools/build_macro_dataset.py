#!/usr/bin/env python3
"""
Build macro conditioning dataset for FiLM layers.

Downloads FRED economic data + FOMC calendar, computes derived features,
applies T-1 shift for leakage prevention, and outputs a single parquet file.

Usage:
    pip install fredapi pandas pyarrow
    python tools/build_macro_dataset.py --api-key YOUR_FRED_KEY
    python tools/build_macro_dataset.py --api-key YOUR_FRED_KEY --start 2004-01-01 --end 2025-12-31

Output:
    datasets/macro/macro_daily.parquet  (~5K rows x 15 cols)
"""

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FOMC meeting dates (announcement day, 2005-2025)
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# ---------------------------------------------------------------------------
FOMC_DATES = [
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-18", "2008-04-30",
    "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-08",
    "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
]

FOMC_DATES_PARSED = [pd.Timestamp(d).date() for d in FOMC_DATES]


# ---------------------------------------------------------------------------
# FRED series to download
# ---------------------------------------------------------------------------
FRED_SERIES = {
    'fed_funds_rate': 'DFF',         # Effective Federal Funds Rate (daily)
    'treasury_2y': 'DGS2',           # 2-Year Treasury Yield (daily)
    'treasury_10y': 'DGS10',         # 10-Year Treasury Yield (daily)
    'treasury_3m': 'DTB3',           # 3-Month Treasury Bill (daily)
    'cpi': 'CPIAUCSL',               # CPI All Urban (monthly, not seasonally adj)
    'unemployment': 'UNRATE',        # Unemployment Rate (monthly)
    'initial_claims': 'ICSA',        # Initial Jobless Claims (weekly)
    'vix3m': 'VXVCLS',               # CBOE VIX3M (daily) - for term ratio
}


def download_fred_data(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Download all FRED series and merge into a single DataFrame."""
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    
    frames = {}
    for name, series_id in FRED_SERIES.items():
        print(f"  Downloading {name} ({series_id})...")
        try:
            data = fred.get_series(series_id, observation_start=start, observation_end=end)
            data = data.dropna()
            frames[name] = data
            print(f"    Got {len(data)} observations")
        except Exception as e:
            print(f"    WARNING: Failed to download {series_id}: {e}")
            frames[name] = pd.Series(dtype=float)
    
    # Merge all series on date index
    df = pd.DataFrame(index=pd.date_range(start, end, freq='D'))
    for name, series in frames.items():
        df[name] = series
    
    return df


def load_vix_daily(vix_dir: str) -> pd.Series:
    """Load VIX daily close from VIX CSV files (same as used by training)."""
    vix_path = Path(vix_dir)
    daily = {}
    
    for csv_file in sorted(vix_path.glob('VIX_*.csv')):
        try:
            raw = pd.read_csv(csv_file, usecols=['date', 'close'])
            raw['date'] = pd.to_datetime(raw['date'], utc=True)
            raw['trading_date'] = raw['date'].dt.date
            raw = raw.sort_values('date')
            last_per_day = raw.groupby('trading_date')['close'].last()
            for tdate, close_val in last_per_day.items():
                daily[tdate] = float(close_val)
        except Exception as e:
            print(f"    WARNING: Error loading {csv_file}: {e}")
    
    print(f"  Loaded VIX daily close for {len(daily)} trading days")
    series = pd.Series(daily, name='vix_close')
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def load_spy_daily(stock_dir: str) -> pd.Series:
    """Load SPY daily close from stock data files."""
    stock_path = Path(stock_dir)
    daily = {}
    
    for f in sorted(stock_path.glob('*.parquet')):
        try:
            # Read just close and ticker columns
            df = pd.read_parquet(f, columns=['ticker', 'close'])
            if 'ticker' in df.columns:
                spy = df[df['ticker'] == 'SPY']
            else:
                spy = df
            if len(spy) > 0:
                # Last close of the day
                d = pd.to_datetime(f.stem).date()
                daily[d] = spy['close'].iloc[-1]
        except Exception:
            continue
    
    print(f"  Loaded SPY daily close for {len(daily)} trading days")
    series = pd.Series(daily, name='spy_close')
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def compute_fomc_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute FOMC calendar features for each date."""
    fomc_sorted = sorted(FOMC_DATES_PARSED)
    
    days_to_fomc = []
    days_since_fomc = []
    fomc_week = []
    
    for d in dates:
        d_date = d.date() if hasattr(d, 'date') else d
        
        # Days to next FOMC
        future = [f for f in fomc_sorted if f > d_date]
        if future:
            delta = (future[0] - d_date).days
            days_to_fomc.append(delta)
        else:
            days_to_fomc.append(90)  # fallback
        
        # Days since last FOMC
        past = [f for f in fomc_sorted if f <= d_date]
        if past:
            delta = (d_date - past[-1]).days
            days_since_fomc.append(delta)
        else:
            days_since_fomc.append(90)  # fallback
        
        # FOMC this week (within 3 calendar days of an FOMC date)
        nearby = any(abs((d_date - f).days) <= 3 for f in fomc_sorted)
        fomc_week.append(1.0 if nearby else 0.0)
    
    return pd.DataFrame({
        'days_to_fomc': days_to_fomc,
        'days_since_fomc': days_since_fomc,
        'fomc_week': fomc_week,
    }, index=dates)


def build_macro_dataset(
    api_key: str,
    output_path: str,
    vix_dir: str = 'datasets/VIX',
    stock_dir: str = 'datasets/Stock_Data_1s',
    start: str = '2004-01-01',
    end: str = '2025-12-31',
):
    """Build the complete macro conditioning dataset."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Building macro conditioning dataset")
    print("=" * 60)
    
    # 1. Download FRED data
    print("\n[1/5] Downloading FRED data...")
    df = download_fred_data(api_key, start, end)
    
    # 2. Load VIX and SPY for derived features
    print("\n[2/5] Loading VIX and SPY daily data...")
    vix_daily = load_vix_daily(vix_dir)
    spy_daily = load_spy_daily(stock_dir)
    
    df['vix_close'] = vix_daily
    df['spy_close'] = spy_daily
    
    # 3. Forward-fill all series (weekends, holidays)
    print("\n[3/5] Computing derived features...")
    df = df.ffill()
    
    # Derived features
    df['yield_spread_2s10s'] = df['treasury_10y'] - df['treasury_2y']
    df['term_premium'] = df['treasury_10y'] - df['treasury_3m']
    
    # CPI YoY (monthly data, forward-filled)
    if 'cpi' in df.columns and df['cpi'].notna().sum() > 12:
        df['cpi_yoy'] = df['cpi'].pct_change(periods=12) * 100  # monthly freq
        # For daily rows, the monthly cpi_yoy will be NaN on non-report days
        # Forward fill to carry the last known value
        df['cpi_yoy'] = df['cpi_yoy'].ffill()
    else:
        df['cpi_yoy'] = 0.0
    
    # VIX term structure ratio
    if 'vix3m' in df.columns and 'vix_close' in df.columns:
        df['vix_term_ratio'] = df['vix_close'] / df['vix3m'].replace(0, np.nan)
        df['vix_term_ratio'] = df['vix_term_ratio'].ffill().fillna(1.0)
    else:
        df['vix_term_ratio'] = 1.0
    
    # SPY vs moving averages
    if 'spy_close' in df.columns and df['spy_close'].notna().sum() > 200:
        spy = df['spy_close']
        df['spy_vs_200d_ma'] = spy / spy.rolling(200, min_periods=50).mean() - 1.0
        df['spy_vs_50d_ma'] = spy / spy.rolling(50, min_periods=20).mean() - 1.0
    else:
        df['spy_vs_200d_ma'] = 0.0
        df['spy_vs_50d_ma'] = 0.0
    
    # 4. FOMC calendar features
    print("\n[4/5] Computing FOMC calendar features...")
    fomc_df = compute_fomc_features(df.index)
    df = df.join(fomc_df)
    
    # 5. Select final columns and apply T-1 shift
    print("\n[5/5] Applying T-1 shift and saving...")
    
    MACRO_FEATURES = [
        'fed_funds_rate',
        'treasury_2y',
        'treasury_10y',
        'yield_spread_2s10s',
        'treasury_3m',
        'term_premium',
        'cpi_yoy',
        'unemployment',
        'initial_claims',
        'vix_term_ratio',
        'spy_vs_200d_ma',
        'spy_vs_50d_ma',
        'days_to_fomc',
        'days_since_fomc',
        'fomc_week',
    ]
    
    # Select and forward-fill any remaining NaNs
    macro = df[MACRO_FEATURES].copy()
    macro = macro.ffill().bfill()
    
    # T-1 SHIFT: use previous trading day's values
    # This prevents any leakage from same-day data
    macro = macro.shift(1)
    macro = macro.ffill().bfill()  # Fill the first row after shift
    
    # Add date column
    macro['date'] = macro.index.date
    macro = macro.reset_index(drop=True)
    
    # Drop rows with all NaN features
    macro = macro.dropna(subset=MACRO_FEATURES, how='all')
    
    # Save
    macro.to_parquet(output, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(macro)} rows, {len(MACRO_FEATURES)} features")
    print(f"Date range: {macro['date'].min()} to {macro['date'].max()}")
    print(f"Output: {output}")
    print(f"Features: {MACRO_FEATURES}")
    print(f"\nSample (first 5 rows):")
    print(macro.head().to_string())
    print(f"\nSample (last 5 rows):")
    print(macro.tail().to_string())
    
    return macro


def main():
    parser = argparse.ArgumentParser(
        description='Build macro conditioning dataset from FRED + FOMC calendar',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--api-key', type=str, required=True,
                        help='FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)')
    parser.add_argument('--output', type=str, default='datasets/macro/macro_daily.parquet',
                        help='Output parquet file path')
    parser.add_argument('--vix-dir', type=str, default='datasets/VIX',
                        help='VIX data directory')
    parser.add_argument('--stock-dir', type=str, default='datasets/Stock_Data_1s',
                        help='Stock data directory (for SPY)')
    parser.add_argument('--start', type=str, default='2004-01-01',
                        help='Start date')
    parser.add_argument('--end', type=str, default='2025-12-31',
                        help='End date')
    
    args = parser.parse_args()
    
    build_macro_dataset(
        api_key=args.api_key,
        output_path=args.output,
        vix_dir=args.vix_dir,
        stock_dir=args.stock_dir,
        start=args.start,
        end=args.end,
    )


if __name__ == '__main__':
    main()
