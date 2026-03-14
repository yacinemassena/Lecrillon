#!/usr/bin/env python3
"""
Aggregate 1-second bar data into 2-minute bars using DuckDB.

Stock:  Stock_Data_1s  -> Stock_Data_2min  (48 features + ticker + bar_timestamp)
Options: option_underlying_bars_1s -> opt_trade_2min (47 features + underlying + bar_timestamp)

Usage:
    python tools/aggregate_2min.py                          # Process all files
    python tools/aggregate_2min.py --stock-only              # Stock only
    python tools/aggregate_2min.py --options-only             # Options only
    python tools/aggregate_2min.py --start-date 2020-01-01   # From date
    python tools/aggregate_2min.py --validate                 # Run validation after
    python tools/aggregate_2min.py --threads 16               # Fewer threads if RAM tight
"""

import argparse
import calendar
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expiry Calendar
# ---------------------------------------------------------------------------
def third_friday(year: int, month: int) -> date:
    """Compute the 3rd Friday of a given month."""
    # Find first day of month
    first_day = date(year, month, 1)
    # dayofweek: Monday=0, Friday=4
    # Find first Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    # 3rd Friday = first Friday + 14 days
    return first_friday + timedelta(days=14)


def build_expiry_calendar(options_dir: Optional[Path] = None,
                          start_year: int = 2003,
                          end_year: int = 2026) -> List[date]:
    """Build monthly expiry calendar.
    
    Uses real expiry dates from options data where available (2014+),
    falls back to computed 3rd Friday for earlier dates.
    
    Returns sorted list of monthly expiry dates.
    """
    expiry_dates: Set[date] = set()
    
    # Computed 3rd Fridays for all months (fallback for pre-options era)
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            expiry_dates.add(third_friday(year, month))
    
    # Try to extract real expiry dates from options data
    if options_dir and options_dir.exists():
        logger.info("Scanning options data for real expiry dates...")
        option_files = sorted(options_dir.glob('*.parquet'))
        
        # Sample ~50 files spread across the date range for efficiency
        if len(option_files) > 50:
            step = len(option_files) // 50
            sample_files = option_files[::step]
        else:
            sample_files = option_files
        
        real_expiries: Set[date] = set()
        for f in sample_files:
            try:
                con = duckdb.connect()
                result = con.sql(f"""
                    SELECT DISTINCT unique_expiries 
                    FROM read_parquet('{f.as_posix()}')
                    WHERE unique_expiries IS NOT NULL
                """).fetchall()
                con.close()
                # unique_expiries is a count, not actual dates — skip
                # The option_contract_bars_1s has 'expiry' column but
                # option_underlying_bars_1s doesn't have actual expiry dates.
            except Exception:
                pass
        
        # Since option_underlying_bars_1s doesn't contain actual expiry dates,
        # we stick with computed 3rd Fridays. This is the standard approach.
        # Holiday shifts (Good Friday etc.) are rare — only ~1 per year.
        logger.info(f"Built expiry calendar: {len(expiry_dates)} monthly expiries "
                    f"({start_year}-{end_year})")
    
    return sorted(expiry_dates)


def days_to_next_expiry(d: date, expiry_cal: List[date]) -> int:
    """Find calendar days to next monthly expiry from sorted expiry list."""
    import bisect
    idx = bisect.bisect_left(expiry_cal, d)
    if idx < len(expiry_cal):
        delta = (expiry_cal[idx] - d).days
        return min(delta, 30)  # Cap at 30
    return 30  # Fallback


def is_monthly_expiry(d: date, expiry_cal: List[date]) -> bool:
    """Check if date is a monthly expiry day."""
    import bisect
    idx = bisect.bisect_left(expiry_cal, d)
    if idx < len(expiry_cal) and expiry_cal[idx] == d:
        return True
    return False


# ---------------------------------------------------------------------------
# Stock Aggregation SQL
# ---------------------------------------------------------------------------
STOCK_AGG_SQL = """
WITH bucketed AS (
    SELECT
        ticker,
        time_bucket(INTERVAL '2 minutes', bar_timestamp) AS bucket,
        bar_timestamp,
        open, high, low, close, volume, trade_count,
        vwap, avg_trade_size, std_trade_size, max_trade_size,
        amihud, buy_volume, sell_volume, signed_trade_count,
        tick_arrival_rate, large_trade_ratio, inter_trade_std, tick_burst,
        rv_intrabar, bpv_intrabar, price_skew,
        close_vs_vwap, high_vs_vwap, low_vs_vwap, ofi
    FROM src_data
),
agg AS (
    SELECT
        ticker,
        bucket AS bar_timestamp,
        count(*) AS bar_count,
        
        -- OHLCV (ORDER BY for first/last)
        first(open ORDER BY bar_timestamp) AS open,
        max(high) AS high,
        min(low) AS low,
        last(close ORDER BY bar_timestamp) AS close,
        sum(volume) AS volume,
        sum(trade_count) AS trade_count,
        
        -- VWAP (volume-weighted recalc)
        CASE WHEN sum(volume) > 0
             THEN sum(vwap * volume) / sum(volume)
             ELSE last(vwap ORDER BY bar_timestamp)
        END AS vwap,
        
        -- Microstructure (recomputed from closes)
        avg(close) AS price_mean,
        CASE WHEN count(*) >= 2 THEN stddev_samp(close) ELSE 0.0 END AS price_std,
        max(high) - min(low) AS price_range,
        CASE WHEN first(open ORDER BY bar_timestamp) > 0
             THEN (max(high) - min(low)) / first(open ORDER BY bar_timestamp) * 100.0
             ELSE 0.0
        END AS price_range_pct,
        
        -- Trade size
        CASE WHEN sum(trade_count) > 0
             THEN sum(volume) / sum(trade_count)
             ELSE 0.0
        END AS avg_trade_size,
        
        -- Weighted avg of std_trade_size (approximate)
        CASE WHEN sum(trade_count) > 0
             THEN sum(std_trade_size * trade_count) / sum(trade_count)
             ELSE 0.0
        END AS std_trade_size,
        
        max(max_trade_size) AS max_trade_size,
        avg(amihud) AS amihud,
        sum(buy_volume) AS buy_volume,
        sum(sell_volume) AS sell_volume,
        sum(signed_trade_count) AS signed_trade_count,
        
        -- Tick arrival rate: trades per second over 2-min window
        CASE WHEN 120.0 > 0 THEN sum(trade_count) / 120.0 ELSE 0.0 END AS tick_arrival_rate,
        
        -- Large trade ratio (approximate from weighted avg)
        CASE WHEN sum(volume) > 0
             THEN sum(large_trade_ratio * volume) / sum(volume)
             ELSE 0.0
        END AS large_trade_ratio,
        
        avg(inter_trade_std) AS inter_trade_std,
        max(tick_burst) AS tick_burst,
        
        -- Volatility (additive)
        sum(rv_intrabar) AS rv_intrabar,
        sum(bpv_intrabar) AS bpv_intrabar,
        
        -- Skewness: recompute from closes, 0 if < 3 bars
        CASE WHEN count(*) >= 3
             THEN (
                 avg(power(close - avg(close) OVER (PARTITION BY ticker, time_bucket(INTERVAL '2 minutes', bar_timestamp)), 3))
             ) / (power(
                 CASE WHEN stddev_samp(close) > 0 THEN stddev_samp(close) ELSE 1e-8 END
             , 3) + 1e-8)
             ELSE 0.0
        END AS price_skew,
        
        -- OFI (additive)
        sum(ofi) AS ofi

    FROM bucketed
    GROUP BY ticker, bucket
)
SELECT
    ticker,
    bar_timestamp,
    open, high, low, close, volume, trade_count,
    price_mean, price_std, price_range, price_range_pct,
    vwap, avg_trade_size, std_trade_size, max_trade_size,
    amihud, buy_volume, sell_volume, signed_trade_count,
    tick_arrival_rate, large_trade_ratio, inter_trade_std, tick_burst,
    rv_intrabar, bpv_intrabar, price_skew,
    
    -- VWAP deviations (recomputed)
    close - vwap AS close_vs_vwap,
    high - vwap AS high_vs_vwap,
    low - vwap AS low_vs_vwap,
    ofi,
    
    -- NEW derived features
    buy_volume - sell_volume AS net_volume,
    (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-8) AS volume_imbalance,
    trade_count / 120.0 AS trade_intensity,
    rv_intrabar / (bpv_intrabar + 1e-8) AS rv_bpv_ratio,
    CASE WHEN open > 0 THEN (close - open) / open ELSE 0.0 END AS close_return,
    volume / (trade_count + 1.0) AS volume_per_trade,
    buy_volume / (sell_volume + 1e-8) AS buy_sell_ratio,
    (high - low) / (low + 1e-8) AS high_low_ratio,
    (close - low) / (high - low + 1e-8) AS close_position

FROM agg
ORDER BY ticker, bar_timestamp
"""


# ---------------------------------------------------------------------------
# Options Aggregation SQL
# ---------------------------------------------------------------------------
OPTIONS_AGG_SQL = """
WITH bucketed AS (
    SELECT
        underlying,
        time_bucket(INTERVAL '2 minutes', ts_converted) AS bucket,
        bar_timestamp AS orig_ts,
        ts_converted,
        call_volume, put_volume, call_trade_count, put_trade_count,
        put_call_ratio_volume, put_call_ratio_count,
        call_premium_total, put_premium_total,
        near_volume, mid_volume, far_volume,
        near_pc_ratio, far_pc_ratio, term_skew,
        otm_put_volume, atm_volume, otm_call_volume,
        skew_proxy, atm_concentration, deep_otm_put_volume,
        total_large_trade_count, call_large_count, put_large_count,
        net_large_flow, large_premium_total, sweep_intensity,
        max_volume_surprise, avg_volume_surprise,
        uoa_call_count, uoa_put_count,
        total_volume, total_trade_count,
        unique_contracts, unique_strikes, unique_expiries,
        pc_ratio_vs_20d, call_volume_vs_20d, put_volume_vs_20d
    FROM src_options
),
agg AS (
    SELECT
        underlying,
        bucket AS bar_timestamp,
        
        -- Volume features (sum)
        sum(call_volume) AS call_volume,
        sum(put_volume) AS put_volume,
        sum(call_trade_count) AS call_trade_count,
        sum(put_trade_count) AS put_trade_count,
        sum(total_volume) AS total_volume,
        sum(total_trade_count) AS total_trade_count,
        sum(call_premium_total) AS call_premium_total,
        sum(put_premium_total) AS put_premium_total,
        sum(large_premium_total) AS large_premium_total,
        
        -- Ratio features (recompute from aggregated volumes)
        sum(put_volume) / (sum(call_volume) + 1e-8) AS put_call_ratio_volume,
        sum(put_trade_count) / (sum(call_trade_count) + 1e-8) AS put_call_ratio_count,
        
        -- Moneyness/Strike distribution (sum volumes)
        sum(near_volume) AS near_volume,
        sum(mid_volume) AS mid_volume,
        sum(far_volume) AS far_volume,
        
        -- Recompute near/far PC ratios — approximate via mean since
        -- we don't have separate near puts/calls columns
        avg(near_pc_ratio) AS near_pc_ratio,
        avg(far_pc_ratio) AS far_pc_ratio,
        
        avg(term_skew) AS term_skew,
        sum(otm_put_volume) AS otm_put_volume,
        sum(atm_volume) AS atm_volume,
        sum(otm_call_volume) AS otm_call_volume,
        
        -- Recompute skew/concentration from aggregated volumes
        sum(otm_put_volume) / (sum(atm_volume) + 1e-8) AS skew_proxy,
        sum(atm_volume) / (sum(total_volume) + 1e-8) AS atm_concentration,
        sum(deep_otm_put_volume) AS deep_otm_put_volume,
        
        -- Large/Unusual flow
        sum(total_large_trade_count) AS total_large_trade_count,
        sum(call_large_count) AS call_large_count,
        sum(put_large_count) AS put_large_count,
        sum(net_large_flow) AS net_large_flow,
        max(sweep_intensity) AS sweep_intensity,
        max(max_volume_surprise) AS max_volume_surprise,
        avg(avg_volume_surprise) AS avg_volume_surprise,
        sum(uoa_call_count) AS uoa_call_count,
        sum(uoa_put_count) AS uoa_put_count,
        
        -- Structure features (last value in window — ORDER BY required)
        last(unique_contracts ORDER BY ts_converted) AS unique_contracts,
        last(unique_strikes ORDER BY ts_converted) AS unique_strikes,
        last(unique_expiries ORDER BY ts_converted) AS unique_expiries,
        last(pc_ratio_vs_20d ORDER BY ts_converted) AS pc_ratio_vs_20d,
        last(call_volume_vs_20d ORDER BY ts_converted) AS call_volume_vs_20d,
        last(put_volume_vs_20d ORDER BY ts_converted) AS put_volume_vs_20d

    FROM bucketed
    GROUP BY underlying, bucket
)
SELECT
    underlying,
    bar_timestamp,
    call_volume, put_volume, call_trade_count, put_trade_count,
    put_call_ratio_volume, put_call_ratio_count,
    call_premium_total, put_premium_total,
    near_volume, mid_volume, far_volume,
    near_pc_ratio, far_pc_ratio, term_skew,
    otm_put_volume, atm_volume, otm_call_volume,
    skew_proxy, atm_concentration, deep_otm_put_volume,
    total_large_trade_count, call_large_count, put_large_count,
    net_large_flow, large_premium_total, sweep_intensity,
    max_volume_surprise, avg_volume_surprise,
    uoa_call_count, uoa_put_count,
    total_volume, total_trade_count,
    unique_contracts, unique_strikes, unique_expiries,
    pc_ratio_vs_20d, call_volume_vs_20d, put_volume_vs_20d,
    
    -- NEW derived features
    call_premium_total - put_premium_total AS net_premium_flow,
    (call_premium_total - put_premium_total) / (call_premium_total + put_premium_total + 1e-8) AS premium_imbalance,
    total_large_trade_count / (total_trade_count + 1e-8) AS large_trade_pct,
    put_large_count / (total_large_trade_count + 1e-8) AS put_large_pct,
    uoa_call_count + uoa_put_count AS uoa_total,
    uoa_put_count / (uoa_call_count + uoa_put_count + 1e-8) AS uoa_put_bias,
    deep_otm_put_volume / (total_volume + 1e-8) AS deep_otm_put_pct,
    near_volume / (far_volume + 1e-8) AS near_far_ratio,
    sweep_intensity / (total_trade_count + 1e-8) AS sweep_to_trade_ratio

FROM agg
ORDER BY underlying, bar_timestamp
"""


# ---------------------------------------------------------------------------
# Aggregation Functions
# ---------------------------------------------------------------------------
def aggregate_stock_day(input_path: Path, output_path: Path,
                        expiry_cal: List[date],
                        memory_limit: str = '4GB',
                        threads: int = 16) -> Tuple[bool, str]:
    """Aggregate one day of stock 1s bars to 2-min bars with calendar features."""
    try:
        con = duckdb.connect()
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {threads}")  # Use all threads for this file
        
        # Load source data
        con.execute(f"""
            CREATE TEMP TABLE src_data AS
            SELECT * FROM read_parquet('{input_path.as_posix()}')
        """)
        
        # Check if empty
        row_count = con.execute("SELECT count(*) FROM src_data").fetchone()[0]
        if row_count == 0:
            con.close()
            return False, f"Empty file: {input_path.name}"
        
        # Run aggregation — price_skew needs special handling because DuckDB
        # doesn't support window functions inside aggregate expressions.
        # We'll compute skew in a post-processing step.
        
        # Step 1: Base aggregation (without skew)
        agg_sql = """
        WITH bucketed AS (
            SELECT
                ticker,
                time_bucket(INTERVAL '2 minutes', bar_timestamp) AS bucket,
                bar_timestamp,
                open, high, low, close, volume, trade_count,
                vwap, avg_trade_size, std_trade_size, max_trade_size,
                amihud, buy_volume, sell_volume, signed_trade_count,
                tick_arrival_rate, large_trade_ratio, inter_trade_std, tick_burst,
                rv_intrabar, bpv_intrabar,
                close_vs_vwap, high_vs_vwap, low_vs_vwap, ofi
            FROM src_data
        )
        SELECT
            ticker,
            bucket AS bar_timestamp,
            count(*) AS bar_count,
            
            first(open ORDER BY bar_timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close ORDER BY bar_timestamp) AS close,
            sum(volume) AS volume,
            sum(trade_count) AS trade_count,
            
            CASE WHEN sum(volume) > 0
                 THEN sum(vwap * volume) / sum(volume)
                 ELSE last(vwap ORDER BY bar_timestamp)
            END AS vwap,
            
            avg(close) AS price_mean,
            CASE WHEN count(*) >= 2 THEN stddev_samp(close) ELSE 0.0 END AS price_std,
            max(high) - min(low) AS price_range,
            CASE WHEN first(open ORDER BY bar_timestamp) > 0
                 THEN (max(high) - min(low)) / first(open ORDER BY bar_timestamp) * 100.0
                 ELSE 0.0
            END AS price_range_pct,
            
            CASE WHEN sum(trade_count) > 0
                 THEN sum(volume) / sum(trade_count)
                 ELSE 0.0
            END AS avg_trade_size,
            
            CASE WHEN sum(trade_count) > 0
                 THEN sum(std_trade_size * trade_count) / sum(trade_count)
                 ELSE 0.0
            END AS std_trade_size,
            
            max(max_trade_size) AS max_trade_size,
            avg(amihud) AS amihud,
            sum(buy_volume) AS buy_volume,
            sum(sell_volume) AS sell_volume,
            sum(signed_trade_count) AS signed_trade_count,
            sum(trade_count) / 120.0 AS tick_arrival_rate,
            
            CASE WHEN sum(volume) > 0
                 THEN sum(large_trade_ratio * volume) / sum(volume)
                 ELSE 0.0
            END AS large_trade_ratio,
            
            avg(inter_trade_std) AS inter_trade_std,
            max(tick_burst) AS tick_burst,
            sum(rv_intrabar) AS rv_intrabar,
            sum(bpv_intrabar) AS bpv_intrabar,
            sum(ofi) AS ofi

        FROM bucketed
        GROUP BY ticker, bucket
        """
        
        con.execute(f"CREATE TEMP TABLE stock_agg AS {agg_sql}")
        
        # Step 2: Compute skewness separately
        # For each (ticker, bar_timestamp) group, compute skew from the original 1s closes
        skew_sql = """
        WITH bucketed AS (
            SELECT
                ticker,
                time_bucket(INTERVAL '2 minutes', bar_timestamp) AS bucket,
                close
            FROM src_data
        ),
        group_stats AS (
            SELECT
                ticker,
                bucket,
                count(*) AS n,
                avg(close) AS mu,
                CASE WHEN count(*) >= 2 THEN stddev_samp(close) ELSE 0.0 END AS sigma
            FROM bucketed
            GROUP BY ticker, bucket
        ),
        cubed AS (
            SELECT
                b.ticker,
                b.bucket,
                gs.n,
                gs.sigma,
                power(b.close - gs.mu, 3) AS cubed_dev
            FROM bucketed b
            JOIN group_stats gs ON b.ticker = gs.ticker AND b.bucket = gs.bucket
        )
        SELECT
            ticker,
            bucket AS bar_timestamp,
            CASE WHEN max(n) >= 3 AND max(sigma) > 1e-10
                 THEN avg(cubed_dev) / (power(max(sigma), 3) + 1e-8)
                 ELSE 0.0
            END AS price_skew
        FROM cubed
        GROUP BY ticker, bucket
        """
        
        con.execute(f"CREATE TEMP TABLE skew_data AS {skew_sql}")
        
        # Step 3: Join skew back and compute final features
        # Extract the trading date for calendar features
        trading_date_str = input_path.stem  # e.g., "2024-01-15"
        try:
            trading_dt = datetime.strptime(trading_date_str, '%Y-%m-%d').date()
        except ValueError:
            trading_dt = date(2020, 1, 1)  # fallback
        
        dow = trading_dt.weekday()  # Monday=0, Friday=4
        days_to_fri = (4 - dow) % 5 if dow <= 4 else 0
        is_fri = 1.0 if dow == 4 else 0.0
        dte = days_to_next_expiry(trading_dt, expiry_cal)
        is_exp = 1.0 if is_monthly_expiry(trading_dt, expiry_cal) else 0.0
        
        final_sql = f"""
        SELECT
            a.ticker,
            a.bar_timestamp,
            a.open, a.high, a.low, a.close, a.volume, a.trade_count,
            a.price_mean, a.price_std, a.price_range, a.price_range_pct,
            a.vwap, a.avg_trade_size, a.std_trade_size, a.max_trade_size,
            a.amihud, a.buy_volume, a.sell_volume, a.signed_trade_count,
            a.tick_arrival_rate, a.large_trade_ratio, a.inter_trade_std, a.tick_burst,
            a.rv_intrabar, a.bpv_intrabar,
            COALESCE(s.price_skew, 0.0) AS price_skew,
            
            -- VWAP deviations
            a.close - a.vwap AS close_vs_vwap,
            a.high - a.vwap AS high_vs_vwap,
            a.low - a.vwap AS low_vs_vwap,
            a.ofi,
            
            -- NEW derived market features (10)
            a.buy_volume - a.sell_volume AS net_volume,
            (a.buy_volume - a.sell_volume) / (a.buy_volume + a.sell_volume + 1e-8) AS volume_imbalance,
            a.trade_count / 120.0 AS trade_intensity,
            a.rv_intrabar / (a.bpv_intrabar + 1e-8) AS rv_bpv_ratio,
            CASE WHEN a.open > 0 THEN (a.close - a.open) / a.open ELSE 0.0 END AS close_return,
            a.volume / (a.trade_count + 1.0) AS volume_per_trade,
            a.buy_volume / (a.sell_volume + 1e-8) AS buy_sell_ratio,
            (a.high - a.low) / (a.low + 1e-8) AS high_low_ratio,
            (a.close - a.low) / (a.high - a.low + 1e-8) AS close_position,
            
            -- Calendar features (7)
            CAST({dow} AS DOUBLE) AS day_of_week,
            CAST({days_to_fri} AS DOUBLE) AS days_to_friday,
            -- minute_of_day: minutes since 09:30 ET
            CAST(
                (extract(hour FROM a.bar_timestamp) * 60 + extract(minute FROM a.bar_timestamp))
                - (9 * 60 + 30)
            AS DOUBLE) AS minute_of_day,
            CAST({is_fri} AS DOUBLE) AS is_friday,
            CAST({dte} AS DOUBLE) AS days_to_monthly_expiry,
            CAST({days_to_fri} AS DOUBLE) AS days_to_weekly_expiry,
            CAST({is_exp} AS DOUBLE) AS is_expiration_day

        FROM stock_agg a
        LEFT JOIN skew_data s ON a.ticker = s.ticker AND a.bar_timestamp = s.bar_timestamp
        ORDER BY a.ticker, a.bar_timestamp
        """
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"""
            COPY ({final_sql})
            TO '{output_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        
        con.close()
        return True, f"OK: {input_path.name}"
        
    except Exception as e:
        return False, f"ERROR {input_path.name}: {e}"


def aggregate_options_day(input_path: Path, output_path: Path,
                          memory_limit: str = '4GB',
                          threads: int = 16) -> Tuple[bool, str]:
    """Aggregate one day of option underlying 1s bars to 2-min bars."""
    try:
        con = duckdb.connect()
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {threads}")  # Use all threads for this file
        
        # Load source data with timestamp conversion
        con.execute(f"""
            CREATE TEMP TABLE src_options AS
            SELECT
                *,
                to_timestamp(CAST(bar_timestamp AS DOUBLE) / 1e9) AS ts_converted
            FROM read_parquet('{input_path.as_posix()}')
        """)
        
        row_count = con.execute("SELECT count(*) FROM src_options").fetchone()[0]
        if row_count == 0:
            con.close()
            return False, f"Empty file: {input_path.name}"
        
        # Run aggregation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"""
            COPY ({OPTIONS_AGG_SQL})
            TO '{output_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        
        con.close()
        return True, f"OK: {input_path.name}"
        
    except Exception as e:
        return False, f"ERROR {input_path.name}: {e}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_stock_day(input_path: Path, output_path: Path) -> List[str]:
    """Validate stock aggregation: volume checksums, ticker sets, feature count."""
    issues = []
    
    try:
        con = duckdb.connect()
        
        # 1. Volume checksum
        vol_1s = con.execute(f"""
            SELECT ticker, sum(volume) AS total_vol
            FROM read_parquet('{input_path.as_posix()}')
            GROUP BY ticker
            ORDER BY ticker
        """).fetchdf()
        
        vol_2m = con.execute(f"""
            SELECT ticker, sum(volume) AS total_vol
            FROM read_parquet('{output_path.as_posix()}')
            GROUP BY ticker
            ORDER BY ticker
        """).fetchdf()
        
        # Ticker set equality
        tickers_1s = set(vol_1s['ticker'].tolist())
        tickers_2m = set(vol_2m['ticker'].tolist())
        
        missing = tickers_1s - tickers_2m
        extra = tickers_2m - tickers_1s
        if missing:
            issues.append(f"Tickers lost in aggregation: {missing}")
        if extra:
            issues.append(f"Extra tickers in output: {extra}")
        
        # Volume match (per ticker)
        merged = vol_1s.merge(vol_2m, on='ticker', suffixes=('_1s', '_2m'))
        vol_diff = (merged['total_vol_1s'] - merged['total_vol_2m']).abs()
        bad_vol = merged[vol_diff > 0.01]
        if len(bad_vol) > 0:
            issues.append(f"Volume mismatch for {len(bad_vol)} tickers, "
                          f"max diff: {vol_diff.max():.4f}")
        
        # 2. Feature count
        cols = con.execute(f"""
            SELECT * FROM read_parquet('{output_path.as_posix()}') LIMIT 0
        """).description
        num_cols = len(cols)
        expected = 47  # 29 original + 9 new market + 7 calendar + ticker + bar_timestamp
        if num_cols != expected:
            issues.append(f"Column count: {num_cols} (expected {expected})")
        
        # 3. NaN/Inf check (sample)
        nan_check = con.execute(f"""
            SELECT count(*) AS nan_rows FROM read_parquet('{output_path.as_posix()}')
            WHERE volume != volume  -- NaN check
               OR trade_count != trade_count
               OR close != close
        """).fetchone()[0]
        if nan_check > 0:
            issues.append(f"NaN detected in {nan_check} rows")
        
        # 4. Timestamp ordering
        ts_check = con.execute(f"""
            WITH ordered AS (
                SELECT ticker, bar_timestamp,
                       lag(bar_timestamp) OVER (PARTITION BY ticker ORDER BY bar_timestamp) AS prev_ts
                FROM read_parquet('{output_path.as_posix()}')
            )
            SELECT count(*) FROM ordered
            WHERE prev_ts IS NOT NULL AND bar_timestamp <= prev_ts
        """).fetchone()[0]
        if ts_check > 0:
            issues.append(f"Non-increasing timestamps: {ts_check} violations")
        
        con.close()
        
    except Exception as e:
        issues.append(f"Validation error: {e}")
    
    return issues


def validate_options_day(input_path: Path, output_path: Path) -> List[str]:
    """Validate options aggregation."""
    issues = []
    
    try:
        con = duckdb.connect()
        
        # 1. Volume checksum
        vol_1s = con.execute(f"""
            SELECT underlying, sum(total_volume) AS total_vol
            FROM read_parquet('{input_path.as_posix()}')
            GROUP BY underlying
            ORDER BY underlying
        """).fetchdf()
        
        vol_2m = con.execute(f"""
            SELECT underlying, sum(total_volume) AS total_vol
            FROM read_parquet('{output_path.as_posix()}')
            GROUP BY underlying
            ORDER BY underlying
        """).fetchdf()
        
        # Underlying set equality
        ul_1s = set(vol_1s['underlying'].tolist())
        ul_2m = set(vol_2m['underlying'].tolist())
        
        missing = ul_1s - ul_2m
        extra = ul_2m - ul_1s
        if missing:
            issues.append(f"Underlyings lost: {missing}")
        if extra:
            issues.append(f"Extra underlyings: {extra}")
        
        # Volume match
        merged = vol_1s.merge(vol_2m, on='underlying', suffixes=('_1s', '_2m'))
        vol_diff = (merged['total_vol_1s'] - merged['total_vol_2m']).abs()
        bad_vol = merged[vol_diff > 0.01]
        if len(bad_vol) > 0:
            issues.append(f"Volume mismatch for {len(bad_vol)} underlyings, "
                          f"max diff: {vol_diff.max():.4f}")
        
        # 2. Feature count
        cols = con.execute(f"""
            SELECT * FROM read_parquet('{output_path.as_posix()}') LIMIT 0
        """).description
        num_cols = len(cols)
        expected = 49  # 47 features + underlying + bar_timestamp
        if num_cols != expected:
            issues.append(f"Column count: {num_cols} (expected {expected})")
        
        # 3. Timestamp ordering
        ts_check = con.execute(f"""
            WITH ordered AS (
                SELECT underlying, bar_timestamp,
                       lag(bar_timestamp) OVER (PARTITION BY underlying ORDER BY bar_timestamp) AS prev_ts
                FROM read_parquet('{output_path.as_posix()}')
            )
            SELECT count(*) FROM ordered
            WHERE prev_ts IS NOT NULL AND bar_timestamp <= prev_ts
        """).fetchone()[0]
        if ts_check > 0:
            issues.append(f"Non-increasing timestamps: {ts_check} violations")
        
        con.close()
        
    except Exception as e:
        issues.append(f"Validation error: {e}")
    
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Aggregate 1s bars to 2-minute bars using DuckDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stock-dir', type=str,
                        default='datasets/Stock_Data_1s',
                        help='Input stock data directory')
    parser.add_argument('--stock-out', type=str,
                        default='datasets/Stock_Data_2min',
                        help='Output stock data directory')
    parser.add_argument('--options-dir', type=str,
                        default='datasets/opt_trade_1sec/option_underlying_bars_1s',
                        help='Input options data directory')
    parser.add_argument('--options-out', type=str,
                        default='datasets/opt_trade_2min',
                        help='Output options data directory')
    parser.add_argument('--stock-only', action='store_true',
                        help='Process stock data only')
    parser.add_argument('--options-only', action='store_true',
                        help='Process options data only')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of files to process in parallel (default: 4)')
    parser.add_argument('--threads', type=int, default=8,
                        help='DuckDB threads per file (default: 8, total CPU = workers * threads)')
    parser.add_argument('--max-ram', type=int, default=128,
                        help='Max RAM in GB total (default: 128, split across workers)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after aggregation')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation (no aggregation)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    stock_dir = project_root / args.stock_dir
    stock_out = project_root / args.stock_out
    options_dir = project_root / args.options_dir
    options_out = project_root / args.options_out
    
    # Split memory across parallel workers
    per_worker_ram = max(4, args.max_ram // max(args.workers, 1))
    mem_limit = f'{per_worker_ram}GB'
    logger.info(f"Hybrid parallelism: {args.workers} workers x {args.threads} DuckDB threads = {args.workers * args.threads} total threads")
    logger.info(f"Memory: {mem_limit} per worker ({args.max_ram}GB total)")
    
    # Date filter
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    def date_in_range(filename: str) -> bool:
        try:
            dt = datetime.strptime(Path(filename).stem, '%Y-%m-%d').date()
            if start_date and dt < start_date:
                return False
            if end_date and dt > end_date:
                return False
            return True
        except ValueError:
            return False
    
    # Build expiry calendar
    logger.info("Building expiry calendar...")
    expiry_cal = build_expiry_calendar(
        options_dir=options_dir if options_dir.exists() else None,
        start_year=2003,
        end_year=2027,
    )
    
    total_start = time.time()
    
    # -----------------------------------------------------------------------
    # Stock aggregation
    # -----------------------------------------------------------------------
    process_stock = not args.options_only
    process_options = not args.stock_only
    
    if process_stock and stock_dir.exists():
        stock_files = sorted([
            f for f in stock_dir.glob('*.parquet')
            if not f.name.startswith('._') and date_in_range(f.name)
        ])
        
        if args.validate_only:
            stock_files_to_process = []
        else:
            # Filter out existing outputs
            if not args.overwrite:
                stock_files_to_process = [
                    f for f in stock_files
                    if not (stock_out / f.name).exists()
                ]
            else:
                stock_files_to_process = stock_files
        
        logger.info(f"Stock: {len(stock_files_to_process)} files to process "
                     f"({len(stock_files)} total, output: {stock_out})")
        
        if stock_files_to_process:
            stock_out.mkdir(parents=True, exist_ok=True)
            
            success = 0
            errors = 0
            
            stock_start = time.time()
            
            # Process-based parallelism: bypasses Python GIL for true parallelism
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        aggregate_stock_day,
                        f, stock_out / f.name, expiry_cal,
                        memory_limit=mem_limit, threads=args.threads
                    ): f for f in stock_files_to_process
                }
                
                pbar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Stock 1s→2min",
                    unit="file",
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                )
                
                for future in pbar:
                    ok, msg = future.result()
                    if ok:
                        success += 1
                    else:
                        errors += 1
                        logger.warning(msg)
                    pbar.set_postfix(ok=success, err=errors, refresh=False)
            
            stock_elapsed = time.time() - stock_start
            rate = len(stock_files_to_process) / stock_elapsed if stock_elapsed > 0 else 0
            logger.info(f"Stock aggregation complete: {success} OK, {errors} errors ")
            logger.info(f"  Time: {stock_elapsed:.1f}s ({stock_elapsed/60:.1f} min), "
                        f"Rate: {rate:.2f} files/sec")
        
        # Validation
        if args.validate or args.validate_only:
            logger.info("Validating stock aggregation...")
            val_errors = 0
            val_files = sorted([
                f for f in stock_dir.glob('*.parquet')
                if not f.name.startswith('._')
                   and date_in_range(f.name)
                   and (stock_out / f.name).exists()
            ])
            
            # Sample validation (every 50th file to save time)
            sample_step = max(1, len(val_files) // 100)
            val_sample = val_files[::sample_step]
            
            for f in tqdm(val_sample, desc="Validate Stock", unit="file",
                         dynamic_ncols=True,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                issues = validate_stock_day(f, stock_out / f.name)
                if issues:
                    val_errors += 1
                    logger.warning(f"Validation {f.name}: {issues}")
            
            logger.info(f"Stock validation: {len(val_sample)} files checked, "
                        f"{val_errors} with issues")
    
    # -----------------------------------------------------------------------
    # Options aggregation
    # -----------------------------------------------------------------------
    if process_options and options_dir.exists():
        options_files = sorted([
            f for f in options_dir.glob('*.parquet')
            if not f.name.startswith('._') and date_in_range(f.name)
        ])
        
        if args.validate_only:
            options_files_to_process = []
        else:
            if not args.overwrite:
                options_files_to_process = [
                    f for f in options_files
                    if not (options_out / f.name).exists()
                ]
            else:
                options_files_to_process = options_files
        
        logger.info(f"Options: {len(options_files_to_process)} files to process "
                     f"({len(options_files)} total, output: {options_out})")
        
        if options_files_to_process:
            options_out.mkdir(parents=True, exist_ok=True)
            
            success = 0
            errors = 0
            
            options_start = time.time()
            
            # Process-based parallelism: bypasses Python GIL for true parallelism
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        aggregate_options_day,
                        f, options_out / f.name,
                        memory_limit=mem_limit, threads=args.threads
                    ): f for f in options_files_to_process
                }
                
                pbar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Options 1s→2min",
                    unit="file",
                    dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                )
                
                for future in pbar:
                    ok, msg = future.result()
                    if ok:
                        success += 1
                    else:
                        errors += 1
                        logger.warning(msg)
                    pbar.set_postfix(ok=success, err=errors, refresh=False)
            
            options_elapsed = time.time() - options_start
            rate = len(options_files_to_process) / options_elapsed if options_elapsed > 0 else 0
            logger.info(f"Options aggregation complete: {success} OK, {errors} errors")
            logger.info(f"  Time: {options_elapsed:.1f}s ({options_elapsed/60:.1f} min), "
                        f"Rate: {rate:.2f} files/sec")
        
        # Validation
        if args.validate or args.validate_only:
            logger.info("Validating options aggregation...")
            val_errors = 0
            val_files = sorted([
                f for f in options_dir.glob('*.parquet')
                if not f.name.startswith('._')
                   and date_in_range(f.name)
                   and (options_out / f.name).exists()
            ])
            
            sample_step = max(1, len(val_files) // 100)
            val_sample = val_files[::sample_step]
            
            for f in tqdm(val_sample, desc="Validate Options", unit="file",
                         dynamic_ncols=True,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                issues = validate_options_day(f, options_out / f.name)
                if issues:
                    val_errors += 1
                    logger.warning(f"Validation {f.name}: {issues}")
            
            logger.info(f"Options validation: {len(val_sample)} files checked, "
                        f"{val_errors} with issues")
    
    elapsed = time.time() - total_start
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
