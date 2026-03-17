"""
Bar-to-Mamba Dataset: Direct 2-min bar sequences for Mamba-only VIX prediction.

Multi-horizon targets: +1d, +7d, +15d, +30d VIX change prediction.
Spike-weighted loss support via per-sample VIX change magnitudes.

No caching - just load files directly, train, discard.
Simple and clean.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threaded I/O pool for parallel file loading
# ---------------------------------------------------------------------------
_IO_THREADS = max(1, (os.cpu_count() or 4) - 1)
_IO_POOL: Optional[ThreadPoolExecutor] = None

def set_io_threads(n: int):
    """Set the number of I/O threads and (re)create the pool."""
    global _IO_THREADS, _IO_POOL
    _IO_THREADS = max(1, n)
    if _IO_POOL is not None:
        _IO_POOL.shutdown(wait=False)
    _IO_POOL = ThreadPoolExecutor(max_workers=_IO_THREADS)
    logger.info(f"I/O thread pool: {_IO_THREADS} threads")

def _get_io_pool() -> ThreadPoolExecutor:
    """Lazy-init the thread pool."""
    global _IO_POOL
    if _IO_POOL is None:
        _IO_POOL = ThreadPoolExecutor(max_workers=_IO_THREADS)
        logger.info(f"I/O thread pool: {_IO_THREADS} threads (auto)")
    return _IO_POOL


# ---------------------------------------------------------------------------
# Default bar features (45 parquet columns + 5 derived = 50 total)
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'trade_count',
    'price_mean', 'price_std', 'price_range', 'price_range_pct', 'vwap',
    'avg_trade_size', 'std_trade_size', 'max_trade_size', 'amihud',
    'buy_volume', 'sell_volume', 'signed_trade_count',
    'tick_arrival_rate', 'large_trade_ratio', 'inter_trade_std', 'tick_burst',
    'rv_intrabar', 'bpv_intrabar', 'price_skew',
    'close_vs_vwap', 'high_vs_vwap', 'low_vs_vwap', 'ofi',
    # New 2-min features
    'net_volume', 'volume_imbalance', 'trade_intensity', 'rv_bpv_ratio',
    'close_return', 'volume_per_trade', 'buy_sell_ratio', 'high_low_ratio',
    'close_position', 'day_of_week', 'days_to_friday', 'minute_of_day',
    'is_friday', 'days_to_monthly_expiry', 'days_to_weekly_expiry', 'is_expiration_day',
    # Derived features (computed on-the-fly)
    'liquidity_stress',
    'ofi_acceleration',  # OFI(t) - OFI(t-10)
    'abs_ofi',           # |OFI|
    'intraday_vol_skew', # rv_last_hour / rv_first_hour
    'ticker_dispersion', # std(close_return) across tickers
]

NUM_STOCK_FEATURES = len(DEFAULT_FEATURES)


def compute_cumulative_zscore(arr: np.ndarray, min_samples: int = 2) -> np.ndarray:
    """Compute cumulative z-score using only past values (no lookahead).
    
    For each position i, z-score is computed using mean/std of arr[0:i+1].
    First min_samples-1 values are set to 0 (not enough history).
    
    Args:
        arr: 1D array of values
        min_samples: Minimum samples needed before computing z-score
    
    Returns:
        1D array of cumulative z-scores
    """
    n = len(arr)
    result = np.zeros(n, dtype=np.float32)
    
    if n < min_samples:
        return result
    
    # Cumulative sum and sum of squares for online computation
    cumsum = np.cumsum(arr)
    cumsum_sq = np.cumsum(arr ** 2)
    
    for i in range(min_samples - 1, n):
        count = i + 1
        mean = cumsum[i] / count
        var = (cumsum_sq[i] / count) - (mean ** 2)
        std = np.sqrt(max(var, 1e-8))  # Avoid division by zero
        result[i] = (arr[i] - mean) / std
    
    return result


def compute_intraday_vol_skew(features: np.ndarray, feature_names: list, 
                               bars_per_hour: int = 30) -> np.ndarray:
    """Compute intraday_vol_skew = rv_last_hour / rv_first_hour.
    
    Computes ratio of realized volatility in last hour vs first hour of the day.
    Uses rv_intrabar summed over first/last hour windows.
    
    Args:
        features: [T, num_features] array (single day)
        feature_names: List of feature names
        bars_per_hour: Number of 2-min bars per hour (default 30)
    
    Returns:
        [T] array with same skew value broadcast to all bars
    """
    n = len(features)
    result = np.ones(n, dtype=np.float32)  # Default to 1.0 (neutral)
    
    try:
        rv_idx = feature_names.index('rv_intrabar')
    except ValueError:
        return result
    
    rv = features[:, rv_idx]
    
    # Need at least 2 hours of data (first + last hour)
    if n < bars_per_hour * 2:
        return result
    
    # Sum RV for first hour and last hour
    rv_first_hour = np.sum(rv[:bars_per_hour])
    rv_last_hour = np.sum(rv[-bars_per_hour:])
    
    # Compute ratio (avoid division by zero)
    if rv_first_hour > 1e-10:
        skew = rv_last_hour / rv_first_hour
    else:
        skew = 1.0
    
    # Broadcast to all bars in the day
    result[:] = skew
    return result


def compute_ofi_derived(features: np.ndarray, feature_names: list, lag: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute OFI-derived features: ofi_acceleration and abs_ofi.
    
    Args:
        features: [T, num_features] array
        feature_names: List of feature names corresponding to columns
        lag: Lookback for acceleration (default 10 bars = 20 min)
    
    Returns:
        (ofi_acceleration, abs_ofi) - each [T] array
    """
    try:
        ofi_idx = feature_names.index('ofi')
    except ValueError:
        n = len(features)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    
    ofi = features[:, ofi_idx]
    n = len(ofi)
    
    # abs_ofi = |OFI|
    abs_ofi = np.abs(ofi).astype(np.float32)
    
    # ofi_acceleration = OFI(t) - OFI(t-lag)
    # First `lag` values are 0 (no history)
    ofi_acceleration = np.zeros(n, dtype=np.float32)
    if n > lag:
        ofi_acceleration[lag:] = ofi[lag:] - ofi[:-lag]
    
    return ofi_acceleration, abs_ofi


def compute_ticker_dispersion(df, feature_names: list, ts_col: str = 'bar_timestamp') -> np.ndarray:
    """Compute ticker_dispersion = std(close_return) across tickers per timestamp.
    
    Args:
        df: Polars DataFrame with ticker and timestamp columns
        feature_names: List of feature names (to check if close_return exists)
        ts_col: Timestamp column name
    
    Returns:
        [T] array of dispersion values (one per unique timestamp, sorted)
    """
    if 'close_return' not in df.columns or 'ticker' not in df.columns:
        # No ticker info or close_return - return zeros
        return np.zeros(len(df), dtype=np.float32)
    
    # Group by timestamp and compute std of close_return across tickers
    dispersion = (
        df.group_by(ts_col)
        .agg(pl.col('close_return').std().alias('dispersion'))
        .sort(ts_col)
    )
    
    # Map back to original row order
    result = df.sort(ts_col).join(
        dispersion, on=ts_col, how='left'
    )['dispersion'].fill_null(0.0).to_numpy().astype(np.float32)
    
    return result


def compute_liquidity_stress(features: np.ndarray, feature_names: list) -> np.ndarray:
    """Compute liquidity_stress = zscore(amihud) + zscore(inter_trade_std) - zscore(tick_arrival_rate).
    
    Uses cumulative z-score (past bars only, no lookahead).
    
    Args:
        features: [T, num_features] array
        feature_names: List of feature names corresponding to columns
    
    Returns:
        [T] array of liquidity_stress values
    """
    # Find column indices for input features
    try:
        amihud_idx = feature_names.index('amihud')
        inter_trade_std_idx = feature_names.index('inter_trade_std')
        tick_arrival_rate_idx = feature_names.index('tick_arrival_rate')
    except ValueError:
        # Missing required features, return zeros
        return np.zeros(len(features), dtype=np.float32)
    
    # Extract columns
    amihud = features[:, amihud_idx]
    inter_trade_std = features[:, inter_trade_std_idx]
    tick_arrival_rate = features[:, tick_arrival_rate_idx]
    
    # Compute cumulative z-scores (no lookahead)
    z_amihud = compute_cumulative_zscore(amihud)
    z_inter_trade_std = compute_cumulative_zscore(inter_trade_std)
    z_tick_arrival_rate = compute_cumulative_zscore(tick_arrival_rate)
    
    # liquidity_stress = zscore(amihud) + zscore(inter_trade_std) - zscore(tick_arrival_rate)
    liquidity_stress = z_amihud + z_inter_trade_std - z_tick_arrival_rate
    
    return liquidity_stress


# Option features (all 49 features from opt_trade_2min)
OPTION_FEATURES = [
    'call_volume', 'put_volume', 'call_trade_count', 'put_trade_count',
    'put_call_ratio_volume', 'put_call_ratio_count',
    'call_premium_total', 'put_premium_total',
    'near_volume', 'mid_volume', 'far_volume',
    'near_pc_ratio', 'far_pc_ratio', 'term_skew',
    'otm_put_volume', 'atm_volume', 'otm_call_volume',
    'skew_proxy', 'atm_concentration', 'deep_otm_put_volume',
    'total_large_trade_count', 'call_large_count', 'put_large_count',
    'net_large_flow', 'large_premium_total', 'sweep_intensity',
    'max_volume_surprise', 'avg_volume_surprise',
    'uoa_call_count', 'uoa_put_count',
    'total_volume', 'total_trade_count',
    'unique_contracts', 'unique_strikes', 'unique_expiries',
    'pc_ratio_vs_20d', 'call_volume_vs_20d', 'put_volume_vs_20d',
    # New 2-min features
    'net_premium_flow', 'premium_imbalance', 'large_trade_pct', 'put_large_pct',
    'uoa_total', 'uoa_put_bias', 'deep_otm_put_pct',
    'near_far_ratio', 'sweep_to_trade_ratio',
    # Derived features (computed on-the-fly)
    'skew_change',  # skew_proxy(t) - skew_proxy(t-30)
]
NUM_OPTION_FEATURES = len(OPTION_FEATURES)

# Option derived features
OPTION_DERIVED_FEATURES = {'skew_change'}


def compute_option_derived(features: np.ndarray, feature_names: list, lag: int = 30) -> dict:
    """Compute derived option features.
    
    Args:
        features: [T, num_features] array
        feature_names: List of feature names
        lag: Lookback for skew_change (default 30 bars = 1 hour)
    
    Returns:
        Dict of feature_name -> [T] array
    """
    n = len(features)
    result = {}
    
    # skew_change = skew_proxy(t) - skew_proxy(t-30)
    try:
        skew_idx = feature_names.index('skew_proxy')
        skew = features[:, skew_idx]
        skew_change = np.zeros(n, dtype=np.float32)
        if n > lag:
            skew_change[lag:] = skew[lag:] - skew[:-lag]
        result['skew_change'] = skew_change
    except ValueError:
        result['skew_change'] = np.zeros(n, dtype=np.float32)
    
    return result

# VIX features (from Vix_features parquet files) - 25 features
# Extended hours: ~540 bars/day (04:00-22:00 ET) vs stock's 195 bars
VIX_FEATURES = [
    'open', 'high', 'low', 'close',  # OHLC (volume dropped - unreliable overnight)
    'vvix', 'previousclose',
    '5dMA', '10dMA', '20dMA',
    'rv_5m', 'rv_30m', 'rv_2h', 'rv_acceleration', 'rv_change_5', 'rv_change_30', 'rv_ratio',
    'vix_vvix_ratio', 'vix_zscore_20d', 'vix_percentile_252d', 'distance_from_20dMA',
    'vix_velocity_15', 'vix_velocity_75', 'vix_acceleration_15', 'vix_acceleration_75',
    'rv_ratio_to_vix',
]
NUM_VIX_FEATURES = len(VIX_FEATURES)  # 25

# Multi-horizon prediction targets (days ahead)
HORIZONS = [1, 7, 15, 30]  # +1d, +7d, +15d, +30d
NUM_HORIZONS = len(HORIZONS)

# Class-level caches (shared across all dataset instances/workers)
# Loaded once in main process, shared via copy-on-write in workers
_NEWS_CACHE: Dict[int, pd.DataFrame] = {}
_BARS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}   # file_path -> (features, timestamps)
_GDELT_CACHE: Dict[str, pd.DataFrame] = {}                   # date_str -> DataFrame
_VIX_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}    # date_str -> (features, timestamps)
_RAW_PARQUET_CACHE: Dict[str, object] = {}                    # file_path -> raw DataFrame (polars/pandas)

def _read_parquet_to_cache(path_str: str, use_polars: bool = True):
    """Read a parquet file into raw cache. Only I/O, no processing. GIL-free."""
    if path_str in _RAW_PARQUET_CACHE:
        return
    try:
        if use_polars:
            _RAW_PARQUET_CACHE[path_str] = pl.read_parquet(path_str)
        else:
            _RAW_PARQUET_CACHE[path_str] = pd.read_parquet(path_str)
    except Exception:
        _RAW_PARQUET_CACHE[path_str] = None


# ---------------------------------------------------------------------------
# VIX daily close loader
# ---------------------------------------------------------------------------
def load_vix_daily_close(vix_dir: str) -> Dict:
    """Load VIX 1-min CSVs and extract daily close (last bar per trading day)."""
    vix_path = Path(vix_dir)
    daily_close = {}

    csv_files = sorted(vix_path.glob('VIX_*.csv'))
    if not csv_files:
        logger.warning(f"No VIX CSV files found in {vix_dir}")
        return daily_close

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, usecols=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df['trading_date'] = df['date'].dt.date
            df = df.sort_values('date')
            last_per_day = df.groupby('trading_date')['close'].last()
            for tdate, close_val in last_per_day.items():
                daily_close[tdate] = float(close_val)
        except Exception as e:
            logger.warning(f"Error loading {csv_file}: {e}")
            continue

    logger.info(f"Loaded VIX daily close for {len(daily_close)} trading days")
    return daily_close


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BarMambaDataset(Dataset):
    """
    Map-style dataset for Mamba VIX prediction.
    
    Data is cached in RAM after first load - __getitem__ is just a slice.
    Supports native DataLoader shuffling.

    Each sample:
    - Input: lookback_days of 1s bars as a flat sequence [T, num_features]
    - Target: next-day VIX change in raw VIX points (not normalized)
    """

    def __init__(
        self,
        stock_data_path: str,
        vix_data_path: str,
        split: str = 'train',
        features: Optional[List[str]] = None,
        max_bars_per_day: int = 195,  # ~195 bars per day at 2-min resolution
        max_total_bars: int = 1000,   # ~5 days at 2-min resolution
        train_start: str = '2005-01-01',
        train_end: str = '2023-11-30',
        val_end: str = '2024-12-31',
        allowed_tickers_file: Optional[str] = None,
        news_data_path: Optional[str] = None,
        use_news: bool = False,
        options_data_path: Optional[str] = None,
        use_options: bool = False,
        macro_data_path: Optional[str] = None,
        use_macro: bool = False,
        gdelt_data_path: Optional[str] = None,
        use_gdelt: bool = False,
        econ_calendar_path: Optional[str] = None,
        use_econ: bool = False,
        fundamentals_data_path: Optional[str] = None,
        use_fundamentals: bool = False,
        vix_features_path: Optional[str] = None,
        use_vix_features: bool = False,
        max_vix_bars: int = 0,  # Cap on VIX sequence length (0 = auto: 2.7× max_total_bars)
        shared_state: Optional[Dict] = None,  # Pre-built indexes from train dataset
        preprocessed_path: Optional[str] = None,  # Path to preprocessed memmaps
    ):
        self.split = split
        self.features = features or DEFAULT_FEATURES
        self.num_features = len(self.features)
        self.max_bars_per_day = max_bars_per_day
        self.max_total_bars = max_total_bars
        self.use_news = use_news
        self.use_options = use_options
        self.use_macro = use_macro
        self.use_gdelt = use_gdelt
        self.use_econ = use_econ
        self.use_fundamentals = use_fundamentals
        self.use_vix_features = use_vix_features
        # VIX runs ~2.77× more bars than stock (540 vs 195 per day)
        # Default cap: proportional to stock cap to keep memory bounded
        self.max_vix_bars = max_vix_bars if max_vix_bars > 0 else int(max_total_bars * 2.77)
        
        # Store date boundaries early (needed for z-score computation)
        self.train_end = pd.to_datetime(train_end).date()
        self.val_end = pd.to_datetime(val_end).date()
        
        # VIX features data
        self.vix_features_path = Path(vix_features_path) if vix_features_path else None
        self.vix_features = VIX_FEATURES
        self.num_vix_features = NUM_VIX_FEATURES
        self.vix_feature_files: Dict = {}  # date -> file path
        
        # News data (Benzinga embeddings)
        self.news_path = Path(news_data_path) if news_data_path else None
        self.news_cache: Dict[int, pd.DataFrame] = {}  # year -> DataFrame
        self.news_dim = 3072  # OpenAI embedding dimension
        
        # GDELT data (world state embeddings every 15 min)
        self.gdelt_path = Path(gdelt_data_path) if gdelt_data_path else None
        self.gdelt_embed_dim = 384  # MiniLM embedding dimension
        self.gdelt_stats_dim = 7    # Summary stats (article_count, goldstein, tone, etc.)
        self.gdelt_dim = self.gdelt_embed_dim + self.gdelt_stats_dim  # 391 total
        
        # Calculate days needed based on seq_len (~195 bars per trading day at 2-min)
        bars_per_day = 195
        self.lookback_days = max(1, (max_total_bars // bars_per_day) + 1)
        # Anchor dates
        self.anchor_start = pd.to_datetime(train_start).date()
        # train_end and val_end already set earlier for z-score computation

        # --- Memmap fast path: preprocessed binary data ---
        self._use_memmaps = False
        self._mm = {}  # memmap handles
        self._mm_idx = {}  # index dicts
        pp = Path(preprocessed_path) if preprocessed_path else None
        if pp is not None and (pp / "stock_index.json").exists():
            self._init_memmaps(pp)

        # --- Fast path: reuse indexes/static data from train dataset ---
        if shared_state is not None:
            self.stock_path = shared_state['stock_path']
            self.stock_files = shared_state['stock_files']
            self.option_files = shared_state.get('option_files', {})
            self.options_path = shared_state.get('options_path')
            self.option_features = OPTION_FEATURES
            self.num_option_features = NUM_OPTION_FEATURES
            self.vix_feature_files = shared_state.get('vix_feature_files', {})
            self.vix_daily = shared_state['vix_daily']
            self.macro_data = shared_state.get('macro_data')
            self.macro_dim = shared_state.get('macro_dim', 0)
            self.macro_features = shared_state.get('macro_features', [])
            self.macro_mean = shared_state.get('macro_mean')
            self.macro_std = shared_state.get('macro_std')
            self.fundamentals_data = shared_state.get('fundamentals_data')
            self.fundamentals_dim = shared_state.get('fundamentals_dim', 0)
            self.fundamentals_features = shared_state.get('fundamentals_features', [])
            self.fundamentals_mean = shared_state.get('fundamentals_mean')
            self.fundamentals_std = shared_state.get('fundamentals_std')
            self.econ_data = shared_state.get('econ_data')
            self.econ_by_date = shared_state.get('econ_by_date', {})
            self.econ_num_features = 13
            self.econ_num_event_types = shared_state.get('econ_num_event_types', 0)
            self.econ_num_currencies = shared_state.get('econ_num_currencies', 0)
            self.allowed_tickers = shared_state.get('allowed_tickers', set())
            self.news_path = shared_state.get('news_path')
            self.news_cache = shared_state.get('news_cache', {})
            self.news_dim = 3072
            self.gdelt_path = shared_state.get('gdelt_path')
            self.gdelt_embed_dim = 384
            self.gdelt_stats_dim = 7
            self.gdelt_dim = 391
            # Only rebuild anchor dates for this split's date range
            self.anchor_dates = self._build_anchor_dates()
            sources = ['stock']
            if self.use_options:
                sources.append(f'options({len(self.option_files)} days)')
            if self.use_news:
                sources.append('news')
            if self.use_vix_features:
                sources.append(f'vix_features({len(self.vix_feature_files)} days)')
            logger.info(
                f"BarMambaDataset [{split}]: {len(self.anchor_dates)} samples, "
                f"seq={max_total_bars} (~{self.lookback_days}d), sources={sources} (shared indexes)"
            )
            return
        # --- End fast path ---

        # Macro conditioning data (FiLM)
        self.macro_data: Optional[pd.DataFrame] = None
        self.macro_dim = 0
        self.macro_features: List[str] = []
        self.macro_mean: Optional[np.ndarray] = None  # Global z-score stats
        self.macro_std: Optional[np.ndarray] = None
        if use_macro and macro_data_path:
            macro_path = Path(macro_data_path)
            if macro_path.exists():
                self.macro_data = pd.read_parquet(macro_path)
                # Handle both index-based and column-based date
                if 'date' in self.macro_data.columns:
                    self.macro_data['date'] = pd.to_datetime(self.macro_data['date']).dt.date
                    self.macro_data = self.macro_data.set_index('date').sort_index()
                elif self.macro_data.index.name == 'date' or self.macro_data.index.dtype == 'datetime64[ns]':
                    self.macro_data.index = pd.to_datetime(self.macro_data.index).date
                    self.macro_data.index.name = 'date'
                
                # Compute derived macro features
                self._compute_macro_derived_features()
                
                self.macro_features = [c for c in self.macro_data.columns]
                self.macro_dim = len(self.macro_features)
                
                # Compute global z-score stats from TRAINING period only
                train_mask = self.macro_data.index <= self.train_end
                train_data = self.macro_data.loc[train_mask].values.astype(np.float32)
                train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
                self.macro_mean = train_data.mean(axis=0)
                self.macro_std = train_data.std(axis=0) + 1e-8
                
                logger.info(f"Loaded macro data: {len(self.macro_data)} days, {self.macro_dim} features (global z-score from {train_mask.sum()} train days)")
            else:
                logger.warning(f"Macro data not found: {macro_path}")
        
        # Fundamentals data (cross-attention state)
        self.fundamentals_data: Optional[pd.DataFrame] = None
        self.fundamentals_dim = 0
        self.fundamentals_features: List[str] = []
        self.fundamentals_mean: Optional[np.ndarray] = None  # Global z-score stats
        self.fundamentals_std: Optional[np.ndarray] = None
        if use_fundamentals and fundamentals_data_path:
            fund_path = Path(fundamentals_data_path)
            if fund_path.exists():
                self.fundamentals_data = pd.read_parquet(fund_path)
                # Handle both index-based and column-based date
                if 'date' in self.fundamentals_data.columns:
                    self.fundamentals_data['date'] = pd.to_datetime(self.fundamentals_data['date']).dt.date
                    self.fundamentals_data = self.fundamentals_data.set_index('date').sort_index()
                elif self.fundamentals_data.index.name == 'date' or self.fundamentals_data.index.dtype == 'datetime64[ns]':
                    self.fundamentals_data.index = pd.to_datetime(self.fundamentals_data.index).date
                    self.fundamentals_data.index.name = 'date'
                self.fundamentals_features = [c for c in self.fundamentals_data.columns]
                self.fundamentals_dim = len(self.fundamentals_features)
                
                # Compute global z-score stats from TRAINING period only
                train_mask = self.fundamentals_data.index <= self.train_end
                train_data = self.fundamentals_data.loc[train_mask].values.astype(np.float32)
                train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
                self.fundamentals_mean = train_data.mean(axis=0)
                self.fundamentals_std = train_data.std(axis=0) + 1e-8
                
                logger.info(f"Loaded fundamentals data: {len(self.fundamentals_data)} days, {self.fundamentals_dim} features (global z-score from {train_mask.sum()} train days)")
            else:
                logger.warning(f"Fundamentals data not found: {fund_path}")
        
        # Options data
        self.options_path = Path(options_data_path) if options_data_path else None
        self.option_features = OPTION_FEATURES
        self.num_option_features = NUM_OPTION_FEATURES
        self.option_files: Dict = {}  # date -> file path
        
        # Economic calendar data (preprocessed parquet + vocab)
        self.econ_data: Optional[pd.DataFrame] = None
        self.econ_by_date: Dict = {}  # date_str -> DataFrame slice
        self.econ_num_features = 13  # numeric features per econ token
        self.econ_num_event_types = 0
        self.econ_num_currencies = 0
        if use_econ and econ_calendar_path:
            econ_path = Path(econ_calendar_path)
            parquet_file = econ_path / 'econ_events.parquet'
            if parquet_file.exists():
                self.econ_data = pd.read_parquet(parquet_file)
                # Index by date for fast lookup
                for date_str, group in self.econ_data.groupby('date'):
                    self.econ_by_date[date_str] = group
                # Load vocab for metadata
                vocab_file = econ_path / 'vocab.json'
                if vocab_file.exists():
                    import json
                    with open(vocab_file) as f:
                        econ_vocab = json.load(f)
                    self.econ_num_event_types = econ_vocab['num_event_types']
                    self.econ_num_currencies = econ_vocab['num_currencies']
                logger.info(f"Loaded econ calendar: {len(self.econ_data)} events, "
                           f"{len(self.econ_by_date)} days, "
                           f"{self.econ_num_event_types} event types")
            else:
                logger.warning(f"Econ calendar not found: {parquet_file}")
        
        # Load ticker filter
        self.allowed_tickers: Set[str] = set()
        if allowed_tickers_file and Path(allowed_tickers_file).exists():
            with open(allowed_tickers_file, 'r') as f:
                self.allowed_tickers = set(line.strip() for line in f if line.strip())
            logger.info(f"Ticker filter: {len(self.allowed_tickers)} tickers")

        # Index stock files by date
        self.stock_path = Path(stock_data_path).resolve()
        self.stock_files: Dict = {}
        self._index_stock_files()
        
        # Index option files by date (if enabled)
        if self.use_options and self.options_path:
            self._index_option_files()
        
        # Index VIX feature files by date (if enabled)
        if self.use_vix_features and self.vix_features_path:
            self._index_vix_feature_files()

        # Load VIX daily close
        self.vix_daily = load_vix_daily_close(vix_data_path)

        # Build valid anchor dates
        self.anchor_dates = self._build_anchor_dates()
        
        sources = ['stock']
        if self.use_options:
            sources.append(f'options({len(self.option_files)} days)')
        if self.use_news:
            sources.append('news')
        if self.use_vix_features:
            sources.append(f'vix_features({len(self.vix_feature_files)} days)')
        logger.info(
            f"BarMambaDataset [{split}]: {len(self.anchor_dates)} samples, "
            f"seq={max_total_bars} (~{self.lookback_days}d), sources={sources}"
        )

    def get_shared_state(self) -> Dict:
        """Return indexes and static data so a val dataset can skip re-loading."""
        return {
            'stock_path': self.stock_path,
            'stock_files': self.stock_files,
            'option_files': getattr(self, 'option_files', {}),
            'options_path': getattr(self, 'options_path', None),
            'vix_feature_files': getattr(self, 'vix_feature_files', {}),
            'vix_daily': self.vix_daily,
            'macro_data': self.macro_data,
            'macro_dim': self.macro_dim,
            'macro_features': self.macro_features,
            'macro_mean': self.macro_mean,
            'macro_std': self.macro_std,
            'fundamentals_data': self.fundamentals_data,
            'fundamentals_dim': self.fundamentals_dim,
            'fundamentals_features': self.fundamentals_features,
            'fundamentals_mean': getattr(self, 'fundamentals_mean', None),
            'fundamentals_std': getattr(self, 'fundamentals_std', None),
            'econ_data': self.econ_data,
            'econ_by_date': self.econ_by_date,
            'econ_num_event_types': self.econ_num_event_types,
            'econ_num_currencies': self.econ_num_currencies,
            'allowed_tickers': self.allowed_tickers,
            'news_path': self.news_path,
            'news_cache': self.news_cache,
            'gdelt_path': self.gdelt_path,
        }

    def _compute_macro_derived_features(self):
        """Compute derived macro features from raw FED/FRED data.
        
        Adds:
            - credit_spread: BAMLH0A0HYM2 - BAMLC0A0CM (high yield - investment grade)
            - yield_curve_velocity: T10Y2Y(t) - T10Y2Y(t-5) (5-day change)
            - stlfsi4_change: STLFSI4(t) - STLFSI4(t-1) (daily change in stress index)
            - fomc_proximity: exp(-days_until_fomc / 5) (exponential decay)
        """
        if self.macro_data is None:
            return
        
        df = self.macro_data
        
        # credit_spread = BAMLH0A0HYM2 - BAMLC0A0CM
        if 'BAMLH0A0HYM2' in df.columns and 'BAMLC0A0CM' in df.columns:
            df['credit_spread'] = df['BAMLH0A0HYM2'] - df['BAMLC0A0CM']
        
        # yield_curve_velocity = T10Y2Y(t) - T10Y2Y(t-5)
        if 'T10Y2Y' in df.columns:
            df['yield_curve_velocity'] = df['T10Y2Y'] - df['T10Y2Y'].shift(5)
            df['yield_curve_velocity'] = df['yield_curve_velocity'].fillna(0.0)
        
        # stlfsi4_change = STLFSI4(t) - STLFSI4(t-1)
        if 'STLFSI4' in df.columns:
            df['stlfsi4_change'] = df['STLFSI4'] - df['STLFSI4'].shift(1)
            df['stlfsi4_change'] = df['stlfsi4_change'].fillna(0.0)
        
        # fomc_proximity = exp(-days_until_fomc / 5)
        if 'days_until_fomc' in df.columns:
            df['fomc_proximity'] = np.exp(-df['days_until_fomc'] / 5.0)
        
        self.macro_data = df

    def _index_stock_files(self):
        """Build date → file path mapping from all available stock files."""
        for f in self.stock_path.glob('*.parquet'):
            if f.name.startswith('._'):
                continue
            date_str = f.stem.split('.')[0]
            try:
                dt = pd.to_datetime(date_str).date()
                self.stock_files[dt] = f.resolve()
            except Exception:
                continue
        logger.info(f"Indexed {len(self.stock_files)} stock data files")

    def _index_option_files(self):
        """Build date → file path mapping for option 2-min bars."""
        # 2-min data: files directly in options_path (not in subdirectory)
        option_dir = self.options_path
        if not option_dir.exists():
            logger.warning(f"Option directory not found: {option_dir}")
            return
        
        for f in option_dir.glob('*.parquet'):
            if f.name.startswith('._'):
                continue
            date_str = f.stem.split('.')[0]
            try:
                dt = pd.to_datetime(date_str).date()
                self.option_files[dt] = f.resolve()
            except Exception:
                continue
        logger.info(f"Indexed {len(self.option_files)} option data files")

    def _index_vix_feature_files(self):
        """Build date → file path mapping for VIX feature 2-min bars."""
        vix_dir = self.vix_features_path
        if not vix_dir.exists():
            logger.warning(f"VIX features directory not found: {vix_dir}")
            return
        
        for f in vix_dir.glob('*.parquet'):
            if f.name.startswith('._'):
                continue
            date_str = f.stem.split('.')[0]
            try:
                dt = pd.to_datetime(date_str).date()
                self.vix_feature_files[dt] = f.resolve()
            except Exception:
                continue
        logger.info(f"Indexed {len(self.vix_feature_files)} VIX feature files")

    def _find_vix_at_horizon(self, anchor_date: date, horizon_days: int, 
                               vix_dates: set, all_dates: List[date], date_to_idx: Dict[date, int]) -> Optional[float]:
        """Find VIX value at approximately horizon_days ahead.
        
        Searches for the nearest trading day with VIX data around the target date.
        """
        target_date = anchor_date + timedelta(days=horizon_days)
        
        # Search window: +/- 5 trading days from target
        for offset in range(0, 10):
            for sign in [0, 1, -1]:
                if offset == 0 and sign != 0:
                    continue
                candidate = target_date + timedelta(days=offset * sign if sign else 0)
                if candidate in vix_dates:
                    return self.vix_daily[candidate]
        return None

    def _build_anchor_dates(self) -> List[Dict]:
        """Build valid anchor dates with multi-horizon targets."""
        all_dates = sorted(self.stock_files.keys())
        vix_dates = set(self.vix_daily.keys())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # Determine split range for anchor dates
        if self.split == 'train':
            split_start = self.anchor_start
            split_end = self.train_end
        elif self.split == 'val':
            split_start = self.train_end + pd.Timedelta(days=1)
            split_end = self.val_end
        else:  # test
            split_start = self.val_end + pd.Timedelta(days=1)
            split_end = None

        # Handle Timestamp vs date
        if hasattr(split_start, 'date'):
            split_start = split_start.date()
        if split_end is not None and hasattr(split_end, 'date'):
            split_end = split_end.date()

        valid = []
        for i, d in enumerate(all_dates):
            if d < split_start:
                continue
            if split_end and d > split_end:
                continue

            # Need anchor-day VIX to compute change
            if d not in vix_dates:
                continue
            anchor_vix = self.vix_daily[d]

            # Compute multi-horizon targets: +1d, +7d, +15d, +30d
            vix_targets = []  # [num_horizons] VIX changes in points
            horizon_mask = []  # [num_horizons] 1.0 if target exists, 0.0 otherwise
            
            for h in HORIZONS:
                target_vix = self._find_vix_at_horizon(d, h, vix_dates, all_dates, date_to_idx)
                if target_vix is not None:
                    vix_targets.append(target_vix - anchor_vix)
                    horizon_mask.append(1.0)
                else:
                    vix_targets.append(0.0)  # Placeholder, will be masked
                    horizon_mask.append(0.0)
            
            # Must have at least +1d target (first horizon)
            if horizon_mask[0] == 0.0:
                continue

            # Lookback window
            lookback_start_idx = i - self.lookback_days + 1
            if lookback_start_idx < 0:
                continue

            window_dates = all_dates[lookback_start_idx:i + 1]
            if len(window_dates) < self.lookback_days:
                continue

            valid.append({
                'anchor_date': d,
                'window_dates': window_dates,
                'vix_targets': vix_targets,      # [4] VIX changes for each horizon
                'horizon_mask': horizon_mask,    # [4] validity mask
                'anchor_vix': anchor_vix,        # For reference/debugging
            })

        return valid

    def _load_day_bars(self, file_path: Path) -> Optional[np.ndarray]:
        """Load one day of bars as feature matrix. Uses RAM cache via _load_day_bars_with_timestamps."""
        result = self._load_day_bars_with_timestamps(file_path)
        if result is None:
            return None
        return result[0]  # Return only features, not timestamps

    def _load_day_bars_with_timestamps(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load one day of bars with timestamps. Cached in RAM after first load."""
        global _BARS_CACHE
        cache_key = str(file_path)
        
        if cache_key in _BARS_CACHE:
            return _BARS_CACHE[cache_key]
        
        # Check raw parquet cache (populated by parallel prefetch threads)
        if cache_key in _RAW_PARQUET_CACHE:
            df = _RAW_PARQUET_CACHE[cache_key]
            if df is None:
                return None
        else:
            try:
                df = pl.read_parquet(file_path)
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
                return None

        # Filter tickers
        if self.allowed_tickers and 'ticker' in df.columns:
            df = df.filter(pl.col('ticker').is_in(self.allowed_tickers))
        if len(df) == 0:
            return None

        # Determine timestamp column
        ts_col = 'bar_timestamp' if 'bar_timestamp' in df.columns else 'timestamp'
        if ts_col not in df.columns:
            return None

        # Sort by timestamp
        df = df.sort(ts_col)

        # Derived features computed on-the-fly (not in parquet)
        derived_features = {'liquidity_stress', 'ofi_acceleration', 'abs_ofi', 
                           'intraday_vol_skew', 'ticker_dispersion'}
        
        # Compute ticker_dispersion before aggregation (needs raw df with ticker column)
        ticker_disp = None
        if 'ticker_dispersion' in self.features:
            ticker_disp = compute_ticker_dispersion(df, list(df.columns), ts_col)
        
        # Extract features (excluding derived features)
        avail = [f for f in self.features if f in df.columns and f not in derived_features]
        if not avail:
            return None

        features = df.select(avail).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute derived features
        if 'liquidity_stress' in self.features:
            liquidity_stress = compute_liquidity_stress(features, avail)
            features = np.column_stack([features, liquidity_stress])
            avail = avail + ['liquidity_stress']
        
        if 'ofi_acceleration' in self.features or 'abs_ofi' in self.features:
            ofi_accel, abs_ofi = compute_ofi_derived(features, avail)
            if 'ofi_acceleration' in self.features:
                features = np.column_stack([features, ofi_accel])
                avail = avail + ['ofi_acceleration']
            if 'abs_ofi' in self.features:
                features = np.column_stack([features, abs_ofi])
                avail = avail + ['abs_ofi']
        
        if 'intraday_vol_skew' in self.features:
            vol_skew = compute_intraday_vol_skew(features, avail)
            features = np.column_stack([features, vol_skew])
            avail = avail + ['intraday_vol_skew']
        
        if 'ticker_dispersion' in self.features and ticker_disp is not None:
            features = np.column_stack([features, ticker_disp])
            avail = avail + ['ticker_dispersion']

        # Extract timestamps as Unix nanoseconds, convert to seconds
        timestamps = df[ts_col].to_numpy()
        if hasattr(timestamps, 'astype'):
            # Convert datetime64 to int64 (nanoseconds), then to seconds
            timestamps = timestamps.astype('datetime64[s]').astype(np.int64)

        # Pad missing features
        if len(avail) < self.num_features:
            padded = np.zeros((len(features), self.num_features), dtype=np.float32)
            padded[:, :len(avail)] = features
            features = padded

        # Cap bars per day
        if len(features) > self.max_bars_per_day:
            features = features[:self.max_bars_per_day]
            timestamps = timestamps[:self.max_bars_per_day]

        _BARS_CACHE[cache_key] = (features, timestamps)
        return features, timestamps

    def _load_news(self, window_dates: List[date]) -> Tuple[np.ndarray, np.ndarray]:
        """Load news embeddings for window dates from daily parquet files.
        
        Uses daily files (YYYY-MM-DD.parquet) from news_daily directory for efficient
        lazy loading that matches stock/options data granularity.
        
        Returns (embeddings [N, 3072], timestamps [N]).
        """
        global _NEWS_CACHE
        
        if not self.use_news or self.news_path is None:
            return np.zeros((0, self.news_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

        # Check for daily files directory (news_daily)
        daily_path = self.news_path.parent / "news_daily"
        use_daily = daily_path.exists()
        
        all_news = []

        if use_daily:
            # Load daily files directly (efficient, no filtering needed)
            for d in window_dates:
                date_str = d.strftime('%Y-%m-%d')
                cache_key = f"daily_{date_str}"
                
                if cache_key not in _NEWS_CACHE:
                    path = daily_path / f"{date_str}.parquet"
                    path_str = str(path)
                    # Check raw parquet cache (populated by parallel prefetch)
                    if path_str in _RAW_PARQUET_CACHE:
                        raw = _RAW_PARQUET_CACHE[path_str]
                        if raw is not None:
                            _NEWS_CACHE[cache_key] = raw
                    elif path.exists():
                        try:
                            _NEWS_CACHE[cache_key] = pd.read_parquet(path)
                        except Exception as e:
                            logger.warning(f"Failed to load news for {date_str}: {e}")
                            continue
                
                if cache_key in _NEWS_CACHE:
                    all_news.append(_NEWS_CACHE[cache_key])
        else:
            # Fallback to yearly files (legacy)
            years = set(d.year for d in window_dates)
            for year in years:
                if year not in _NEWS_CACHE:
                    path = self.news_path / f"{year}_embedded.parquet"
                    if path.exists():
                        try:
                            _NEWS_CACHE[year] = pd.read_parquet(path)
                            logger.info(f"Loaded news cache for {year}: {len(_NEWS_CACHE[year])} articles")
                        except Exception as e:
                            logger.warning(f"Failed to load news for {year}: {e}")
                            continue

                if year in _NEWS_CACHE:
                    df = _NEWS_CACHE[year]
                    start_ns = int(pd.Timestamp(min(window_dates)).value)
                    end_ns = int(pd.Timestamp(max(window_dates) + pd.Timedelta(days=1)).value)
                    mask = (df['timestamp'] >= start_ns) & (df['timestamp'] < end_ns)
                    all_news.append(df[mask])

        if not all_news:
            return np.zeros((0, self.news_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

        df = pd.concat(all_news, ignore_index=True)
        
        if len(df) == 0:
            return np.zeros((0, self.news_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

        # Extract embeddings (use title_embedding)
        embeddings = np.stack(df['title_embedding'].values).astype(np.float32)
        
        # Detect timestamp unit and convert to seconds
        # Daily files may have mixed units (us for older, ns for newer)
        sample_ts = df['timestamp'].iloc[0]
        ts_len = len(str(abs(sample_ts)))
        if ts_len >= 19:
            divisor = 1_000_000_000  # nanoseconds
        elif ts_len >= 16:
            divisor = 1_000_000  # microseconds
        else:
            divisor = 1  # already seconds
        
        timestamps = (df['timestamp'].values // divisor).astype(np.int64)

        return embeddings, timestamps

    def _load_gdelt(self, window_dates: List[date]) -> Tuple[np.ndarray, np.ndarray]:
        """Load GDELT world-state embeddings for window dates.
        
        Returns (features [N, 391], timestamps [N] as Unix seconds).
        Features = concat(embedding[384], stats[7]).
        Uses bucket_end as timestamp (when bucket becomes visible - no data leakage).
        
        Stats extracted (7 dims):
            - log1p(article_count)
            - goldstein_scale_mean
            - goldstein_scale_min  
            - tone_mean
            - tone_negative_max
            - tone_polarity_mean
            - num_sources_mean
        """
        if not self.use_gdelt or self.gdelt_path is None:
            return np.zeros((0, self.gdelt_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
        
        global _GDELT_CACHE
        all_gdelt = []
        
        for d in window_dates:
            cache_key = f"gdelt_{d}"
            
            if cache_key in _GDELT_CACHE:
                if _GDELT_CACHE[cache_key] is not None:
                    all_gdelt.append(_GDELT_CACHE[cache_key])
                continue
            
            year = d.year
            month = d.month
            day = d.day
            path = self.gdelt_path / str(year) / f"{month:02d}" / f"{day:02d}.parquet"
            raw_key = str(path)
            
            # Check raw parquet cache (populated by parallel prefetch threads)
            if raw_key in _RAW_PARQUET_CACHE:
                df = _RAW_PARQUET_CACHE[raw_key]
                if df is None or len(df) == 0:
                    _GDELT_CACHE[cache_key] = None
                    continue
                _GDELT_CACHE[cache_key] = df
                all_gdelt.append(df)
                continue
            
            if not path.exists():
                _GDELT_CACHE[cache_key] = None
                continue
            
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Failed to load GDELT for {d}: {e}")
                _GDELT_CACHE[cache_key] = None
                continue
            
            if len(df) == 0:
                _GDELT_CACHE[cache_key] = None
                continue
            
            _GDELT_CACHE[cache_key] = df
            all_gdelt.append(df)
        
        if not all_gdelt:
            return np.zeros((0, self.gdelt_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
        
        df = pd.concat(all_gdelt, ignore_index=True)
        
        if len(df) == 0:
            return np.zeros((0, self.gdelt_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
        
        # Sort by bucket_end (when each bucket becomes visible)
        df = df.sort_values('bucket_end')
        
        # Extract embeddings [N, 384]
        embeddings = np.stack(df['embedding'].values).astype(np.float32)
        
        # Extract stats [N, 7] - select available columns with fallbacks
        stats_cols = [
            ('article_count', 'article_count', np.log1p),  # log-transform
            ('goldstein_scale_mean', 'goldstein_scale_mean', None),
            ('goldstein_scale_min', 'goldstein_scale_min', None),
            ('tone_mean', 'tone_mean', None),
            ('tone_negative_max', 'tone_negative_max', None),
            ('tone_polarity_mean', 'tone_polarity_mean', None),
            ('num_sources_mean', 'num_sources_mean', None),
        ]
        
        stats = np.zeros((len(df), self.gdelt_stats_dim), dtype=np.float32)
        for i, (col_name, _, transform) in enumerate(stats_cols):
            if col_name in df.columns:
                vals = df[col_name].values.astype(np.float32)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                if transform is not None:
                    vals = transform(vals)
                stats[:, i] = vals
        
        # Combine: [N, 384 + 7] = [N, 391]
        features = np.concatenate([embeddings, stats], axis=1)
        
        # Convert bucket_end to Unix seconds
        bucket_end = pd.to_datetime(df['bucket_end'])
        # Handle timezone - convert to UTC if needed
        if bucket_end.dt.tz is None:
            bucket_end = bucket_end.dt.tz_localize('UTC')
        else:
            bucket_end = bucket_end.dt.tz_convert('UTC')
        timestamps = (bucket_end.astype(np.int64) // 1_000_000_000).values
        
        return features, timestamps

    def _load_day_options(self, d: date, num_bars: int) -> Optional[np.ndarray]:
        """Load option features for a day, aggregated to match stock bar count.
        
        Since option data is per-underlying per second, we aggregate across all
        underlyings to get market-wide option flow signals.
        
        Args:
            d: Date to load
            num_bars: Number of stock bars for this day (to match length)
        
        Returns:
            Option features [num_bars, num_option_features] or None if unavailable
        """
        if not self.use_options or d not in self.option_files:
            return None
        
        # Check raw parquet cache (populated by parallel prefetch threads)
        cache_key = str(self.option_files[d])
        if cache_key in _RAW_PARQUET_CACHE:
            df = _RAW_PARQUET_CACHE[cache_key]
            if df is None:
                return None
        else:
            try:
                df = pl.read_parquet(self.option_files[d])
                _RAW_PARQUET_CACHE[cache_key] = df
            except Exception as e:
                logger.warning(f"Failed to load options for {d}: {e}")
                _RAW_PARQUET_CACHE[cache_key] = None
                return None
        
        if len(df) == 0:
            return None
        
        # Sort by timestamp
        ts_col = 'bar_timestamp' if 'bar_timestamp' in df.columns else 'timestamp'
        if ts_col not in df.columns:
            return None
        df = df.sort(ts_col)
        
        # Aggregate across all underlyings per second (sum volumes, mean ratios)
        # Group by timestamp and aggregate (excluding derived features)
        agg_exprs = []
        for feat in self.option_features:
            if feat in OPTION_DERIVED_FEATURES:
                continue  # Skip derived features
            if feat not in df.columns:
                continue
            if 'ratio' in feat or 'skew' in feat or 'vs_20d' in feat:
                # Mean for ratios
                agg_exprs.append(pl.col(feat).mean().alias(feat))
            else:
                # Sum for volumes/counts
                agg_exprs.append(pl.col(feat).sum().alias(feat))
        
        if not agg_exprs:
            return None
        
        df_agg = df.group_by(ts_col).agg(agg_exprs).sort(ts_col)
        
        # Extract features (excluding derived)
        avail = [f for f in self.option_features if f in df_agg.columns and f not in OPTION_DERIVED_FEATURES]
        if not avail:
            return None
        
        features = df_agg.select(avail).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute derived option features
        derived = compute_option_derived(features, avail)
        for feat_name in ['skew_change']:
            if feat_name in self.option_features:
                features = np.column_stack([features, derived[feat_name]])
                avail = avail + [feat_name]
        
        # Pad missing features
        if len(avail) < self.num_option_features:
            padded = np.zeros((len(features), self.num_option_features), dtype=np.float32)
            padded[:, :len(avail)] = features
            features = padded
        
        # Match length to stock bars (truncate or pad)
        if len(features) > num_bars:
            features = features[:num_bars]
        elif len(features) < num_bars:
            padded = np.zeros((num_bars, self.num_option_features), dtype=np.float32)
            padded[:len(features)] = features
            features = padded
        
        return features

    def _load_vix_features_with_timestamps(self, d: date) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load VIX features for a day with timestamps for checkpoint alignment.
        
        VIX runs extended hours (~540 bars/day vs stock's 195). Returns full
        sequence with timestamps so model can align at checkpoints.
        Cached in RAM after first load.
        
        Args:
            d: Date to load
        
        Returns:
            (features [N, num_vix_features], timestamps [N]) or None if unavailable
            Timestamps are nanoseconds since epoch (int64).
        """
        global _VIX_CACHE
        
        if not self.use_vix_features or d not in self.vix_feature_files:
            return None
        
        cache_key = f"vix_{d}"
        if cache_key in _VIX_CACHE:
            return _VIX_CACHE[cache_key]
        
        # Check raw parquet cache (populated by parallel prefetch threads)
        raw_key = str(self.vix_feature_files[d])
        if raw_key in _RAW_PARQUET_CACHE:
            df = _RAW_PARQUET_CACHE[raw_key]
            if df is None:
                _VIX_CACHE[cache_key] = None
                return None
        else:
            try:
                df = pd.read_parquet(self.vix_feature_files[d])
            except Exception as e:
                logger.warning(f"Failed to load VIX features for {d}: {e}")
                _VIX_CACHE[cache_key] = None
                return None
        
        if len(df) == 0:
            _VIX_CACHE[cache_key] = None
            return None
        
        # Sort by timestamp
        df = df.sort_values('bar_timestamp')
        
        # Convert timestamps to Unix seconds (matching stock bar_timestamps)
        timestamps = pd.to_datetime(df['bar_timestamp']).values.astype('datetime64[s]').astype(np.int64)
        
        # Extract features
        avail = [f for f in self.vix_features if f in df.columns]
        if not avail:
            return None
        
        features = df[avail].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad missing features
        if len(avail) < self.num_vix_features:
            padded = np.zeros((len(features), self.num_vix_features), dtype=np.float32)
            padded[:, :len(avail)] = features
            features = padded
        
        _VIX_CACHE[cache_key] = (features, timestamps)
        return features, timestamps

    def _load_econ_events(self, window_dates: List[date], anchor_date) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load economic calendar events for lookback + D+15 forward window.
        
        Lookback events (in window_dates): full features including actual_z.
        Forward events (anchor+1 to anchor+15): actual_z masked to 0, is_future=1.
        
        Returns:
            event_ids: [N] int array of event type IDs
            currency_ids: [N] int array of currency IDs  
            numeric_features: [N, 13] float array of numeric features
            timestamps: [N] int64 array of Unix timestamps
        """
        if not self.use_econ or self.econ_data is None:
            return (np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int8),
                    np.zeros((0, self.econ_num_features), dtype=np.float32),
                    np.zeros(0, dtype=np.int64))
        
        from datetime import timedelta
        
        all_events = []
        
        # Lookback events — full features
        for d in window_dates:
            date_str = str(d)
            if date_str in self.econ_by_date:
                all_events.append((self.econ_by_date[date_str], False))
        
        # Forward events — D+1 to D+15 (actual masked)
        for offset in range(1, 16):
            fwd_date = anchor_date + timedelta(days=offset)
            date_str = str(fwd_date)
            if date_str in self.econ_by_date:
                all_events.append((self.econ_by_date[date_str], True))
        
        if not all_events:
            return (np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int8),
                    np.zeros((0, self.econ_num_features), dtype=np.float32),
                    np.zeros(0, dtype=np.int64))
        
        event_ids_list = []
        currency_ids_list = []
        numeric_list = []
        timestamps_list = []
        
        for df_slice, is_future in all_events:
            for _, row in df_slice.iterrows():
                event_ids_list.append(row['event_id'])
                currency_ids_list.append(row['currency_id'])
                timestamps_list.append(row['timestamp'])
                
                # Compute days_until relative to anchor date
                event_date = pd.to_datetime(row['date']).date() if isinstance(row['date'], str) else row['date']
                days_until = (event_date - anchor_date).days
                
                # Build numeric feature vector [13 features]
                # All fields scaled to roughly [-2, 2] range
                actual_z = 0.0 if is_future else float(row['actual_z'])
                has_actual = 0 if is_future else int(row['has_actual'])
                
                # Normalize days_until: divide by 15 to get ~[-3, 1] range
                days_until_norm = float(days_until) / 15.0
                # Normalize time_of_day: assume 0-24 hours, scale to [0, 1]
                time_of_day = float(row['time_of_day'])
                time_of_day_norm = time_of_day / 24.0 if time_of_day > 1.0 else time_of_day
                # Normalize impact: 0-3 -> 0-1
                impact_norm = float(row['impact_ord']) / 3.0
                
                numeric = np.array([
                    impact_norm,                         # 0: impact [0, 1]
                    float(row['is_usd']),               # 1: is_usd [0, 1]
                    days_until_norm,                     # 2: days_until normalized [~-3, 1]
                    time_of_day_norm,                    # 3: time_of_day [0, 1]
                    float(is_future),                    # 4: is_future [0, 1]
                    actual_z,                            # 5: actual_z (z-scored, masked if future)
                    float(row['forecast_z']),            # 6: forecast_z (z-scored)
                    float(row['previous_z']),            # 7: previous_z (z-scored)
                    float(has_actual),                   # 8: has_actual [0, 1]
                    float(row['has_forecast']),          # 9: has_forecast [0, 1]
                    float(row['event_rank_today_norm']), # 10: event_rank_today [0, 1]
                    float(row['days_since_last_same_norm']),  # 11: days_since_last_same [0, 1]
                    days_until_norm,                     # 12: days_until normalized (duplicate for backwards compat)
                ], dtype=np.float32)
                numeric_list.append(numeric)
        
        event_ids = np.array(event_ids_list, dtype=np.int16)
        currency_ids = np.array(currency_ids_list, dtype=np.int8)
        numeric_features = np.stack(numeric_list, axis=0)
        raw_ts = np.array(timestamps_list, dtype=np.int64)
        # Normalize to Unix seconds (matching stock bar_timestamps)
        # Auto-detect unit: nanoseconds (>=1e18), microseconds (>=1e15), or seconds
        if len(raw_ts) > 0 and abs(raw_ts[0]) > 1e15:
            timestamps = (raw_ts // 1_000_000_000).astype(np.int64)
        elif len(raw_ts) > 0 and abs(raw_ts[0]) > 1e12:
            timestamps = (raw_ts // 1_000_000).astype(np.int64)
        else:
            timestamps = raw_ts
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        event_ids = event_ids[sort_idx]
        currency_ids = currency_ids[sort_idx]
        numeric_features = numeric_features[sort_idx]
        timestamps = timestamps[sort_idx]
        
        return event_ids, currency_ids, numeric_features, timestamps

    def _normalize_bars(self, bars: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Z-score normalize bars across the sequence.
        
        Args:
            bars: [T, F] array of features
            mask: [T] optional mask where 1=valid, 0=padded. If provided,
                  stats are computed only over valid positions.
        """
        if mask is not None and mask.sum() > 0:
            # Compute stats only over non-masked (valid) positions
            valid_mask = mask.astype(bool)
            valid_bars = bars[valid_mask]
            mu = valid_bars.mean(axis=0, keepdims=True)
            std = valid_bars.std(axis=0, keepdims=True) + 1e-8
            # Normalize all positions (masked will stay ~0 after normalization)
            normalized = (bars - mu) / std
            # Zero out masked positions to ensure they're exactly 0
            normalized[~valid_mask] = 0.0
            return normalized
        else:
            mu = bars.mean(axis=0, keepdims=True)
            std = bars.std(axis=0, keepdims=True) + 1e-8
            return (bars - mu) / std

    # ------------------------------------------------------------------
    # Memmap fast path
    # ------------------------------------------------------------------

    def _init_memmaps(self, pp: Path):
        """Open preprocessed numpy memmaps. Called once at init."""
        import json as _json

        def _open(name, suffix="_features"):
            npy = pp / f"{name}{suffix}.npy"
            idx = pp / f"{name}_index.json"
            if npy.exists() and idx.exists():
                self._mm[f"{name}{suffix}"] = np.load(str(npy), mmap_mode='r')
                with open(idx) as f:
                    self._mm_idx[name] = _json.load(f)
                return True
            return False

        # Stock (required)
        if not _open("stock"):
            logger.warning("Preprocessed stock not found — falling back to parquet")
            return
        _open("stock", "_timestamps")

        # Optional feeds
        if self.use_options:
            _open("options")
        if self.use_vix_features:
            _open("vix")
            _open("vix", "_timestamps")
        if self.use_news:
            _open("news", "_embeddings")  # news uses _embeddings suffix
            if not f"news_embeddings" in self._mm:
                _open("news", "_features")  # fallback name
            _open("news", "_timestamps")
        if self.use_gdelt:
            _open("gdelt")
            _open("gdelt", "_timestamps")
        if self.use_macro:
            _open("macro")
        if self.use_fundamentals:
            _open("fundamentals")
        if self.use_econ:
            for suffix in ["_numeric", "_event_ids", "_currency_ids", "_timestamps"]:
                npy = pp / f"econ{suffix}.npy"
                if npy.exists():
                    self._mm[f"econ{suffix}"] = np.load(str(npy), mmap_mode='r')
            idx = pp / "econ_index.json"
            if idx.exists():
                with open(idx) as f:
                    self._mm_idx["econ"] = _json.load(f)

        self._use_memmaps = True
        total_mb = sum(m.nbytes / 1e6 for m in self._mm.values())
        logger.info(f"Memmap mode: {len(self._mm)} arrays opened ({total_mb:.0f} MB mapped), "
                    f"{len(self._mm_idx)} indexes loaded")

    def _mm_slice(self, feed: str, date_str: str, suffix: str = "_features"):
        """Slice a memmap for a given date. Returns numpy array or None."""
        idx = self._mm_idx.get(feed)
        key = f"{feed}{suffix}"
        mm = self._mm.get(key)
        if idx is None or mm is None or date_str not in idx:
            return None
        entry = idx[date_str]
        o, l = entry["offset"], entry["length"]
        return np.array(mm[o:o + l])  # copy from mmap into RAM (tiny slice)

    def _getitem_memmap(self, idx: int) -> Dict:
        """Fast __getitem__ using preprocessed memmaps. Microsecond access."""
        sample = self.anchor_dates[idx]
        window_dates = sample['window_dates']
        vix_targets = sample['vix_targets']
        horizon_mask = sample['horizon_mask']

        # ── Stock bars ────────────────────────────────────────────────
        all_bars, all_ts, loaded_days = [], [], []
        for d in window_dates:
            ds = str(d)
            feat = self._mm_slice("stock", ds)
            if feat is not None and len(feat) > 0:
                all_bars.append(feat)
                ts = self._mm_slice("stock", ds, "_timestamps")
                if ts is not None:
                    all_ts.append(ts)
                loaded_days.append((d, len(feat)))

        if not all_bars:
            return self._empty_result(sample)

        bars = np.concatenate(all_bars, axis=0)
        bar_timestamps = np.concatenate(all_ts, axis=0) if all_ts else None
        if len(bars) > self.max_total_bars:
            bars = bars[-self.max_total_bars:]
            if bar_timestamps is not None:
                bar_timestamps = bar_timestamps[-self.max_total_bars:]
        bars = self._normalize_bars(bars)

        result = {
            'bars': torch.from_numpy(bars),
            'vix_targets': torch.tensor(vix_targets, dtype=torch.float32),
            'horizon_mask': torch.tensor(horizon_mask, dtype=torch.float32),
            'num_bars': len(bars),
            'anchor_date': str(sample['anchor_date']),
        }
        if bar_timestamps is not None:
            result['bar_timestamps'] = torch.from_numpy(bar_timestamps)

        # ── News ──────────────────────────────────────────────────────
        if self.use_news:
            emb_key = "news_embeddings" if "news_embeddings" in self._mm else "news_features"
            all_embs, all_nts = [], []
            for d in window_dates:
                ds = str(d)
                e = self._mm_slice("news", ds, "_embeddings")
                if e is None:
                    e = self._mm_slice("news", ds, "_features")
                if e is not None and len(e) > 0:
                    all_embs.append(e)
                    t = self._mm_slice("news", ds, "_timestamps")
                    if t is not None:
                        all_nts.append(t)
            if all_embs:
                news_embs = np.concatenate(all_embs, axis=0)
                news_ts = np.concatenate(all_nts, axis=0) if all_nts else np.zeros(len(news_embs), dtype=np.int64)
            else:
                news_embs = np.zeros((0, self.news_dim), dtype=np.float32)
                news_ts = np.zeros(0, dtype=np.int64)
            result['news_embs'] = torch.from_numpy(news_embs)
            result['news_timestamps'] = torch.from_numpy(news_ts)
            result['num_news'] = len(news_embs)

        # ── GDELT ─────────────────────────────────────────────────────
        if self.use_gdelt:
            all_gf, all_gt = [], []
            for d in window_dates:
                ds = str(d)
                f = self._mm_slice("gdelt", ds)
                if f is not None and len(f) > 0:
                    all_gf.append(f)
                    t = self._mm_slice("gdelt", ds, "_timestamps")
                    if t is not None:
                        all_gt.append(t)
            if all_gf:
                gdelt_embs = np.concatenate(all_gf, axis=0)
                gdelt_ts = np.concatenate(all_gt, axis=0) if all_gt else np.zeros(len(gdelt_embs), dtype=np.int64)
            else:
                gdelt_embs = np.zeros((0, self.gdelt_dim), dtype=np.float32)
                gdelt_ts = np.zeros(0, dtype=np.int64)
            result['gdelt_embs'] = torch.from_numpy(gdelt_embs)
            result['gdelt_timestamps'] = torch.from_numpy(gdelt_ts)
            result['num_gdelt'] = len(gdelt_embs)

        # ── Macro (instant lookup) ────────────────────────────────────
        if self.use_macro and "macro" in self._mm_idx:
            anchor_str = str(sample['anchor_date'])
            mv = self._mm_slice("macro", anchor_str)
            if mv is None:
                # Find nearest earlier date
                dates = sorted(self._mm_idx["macro"].keys())
                earlier = [d for d in dates if d <= anchor_str]
                if earlier:
                    mv = self._mm_slice("macro", earlier[-1])
            if mv is not None:
                macro_vec = mv.flatten().astype(np.float32)
            else:
                macro_vec = np.zeros(max(self.macro_dim, 1), dtype=np.float32)
            macro_vec = np.nan_to_num(macro_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if self.macro_mean is not None and self.macro_std is not None:
                macro_vec = (macro_vec - self.macro_mean) / self.macro_std
            result['macro_context'] = torch.from_numpy(macro_vec)
        elif self.use_macro:
            # Fallback to DataFrame lookup
            anchor = sample['anchor_date']
            if self.macro_data is not None and anchor in self.macro_data.index:
                macro_vec = self.macro_data.loc[anchor].values.astype(np.float32)
            else:
                macro_vec = np.zeros(self.macro_dim, dtype=np.float32)
            macro_vec = np.nan_to_num(macro_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if self.macro_mean is not None and self.macro_std is not None:
                macro_vec = (macro_vec - self.macro_mean) / self.macro_std
            result['macro_context'] = torch.from_numpy(macro_vec)

        # ── Fundamentals (instant lookup) ─────────────────────────────
        if self.use_fundamentals and "fundamentals" in self._mm_idx:
            anchor_str = str(sample['anchor_date'])
            fv = self._mm_slice("fundamentals", anchor_str)
            if fv is None:
                dates = sorted(self._mm_idx["fundamentals"].keys())
                earlier = [d for d in dates if d <= anchor_str]
                if earlier:
                    fv = self._mm_slice("fundamentals", earlier[-1])
            if fv is not None:
                fund_vec = fv.flatten().astype(np.float32)
            else:
                fund_vec = np.zeros(max(self.fundamentals_dim, 1), dtype=np.float32)
            fund_vec = np.nan_to_num(fund_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if self.fundamentals_mean is not None and self.fundamentals_std is not None:
                fund_vec = (fund_vec - self.fundamentals_mean) / self.fundamentals_std
            result['fundamentals_context'] = torch.from_numpy(fund_vec)
        elif self.use_fundamentals:
            anchor = sample['anchor_date']
            if self.fundamentals_data is not None and anchor in self.fundamentals_data.index:
                fund_vec = self.fundamentals_data.loc[anchor].values.astype(np.float32)
            else:
                fund_vec = np.zeros(self.fundamentals_dim, dtype=np.float32)
            fund_vec = np.nan_to_num(fund_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if self.fundamentals_mean is not None and self.fundamentals_std is not None:
                fund_vec = (fund_vec - self.fundamentals_mean) / self.fundamentals_std
            result['fundamentals_context'] = torch.from_numpy(fund_vec)

        # ── Econ ──────────────────────────────────────────────────────
        if self.use_econ and "econ" in self._mm_idx:
            from datetime import timedelta
            anchor = sample['anchor_date']
            econ_idx = self._mm_idx["econ"]
            all_eids, all_cids, all_enum, all_ets = [], [], [], []

            def _econ_batch(ds, is_future):
                """Load one day's econ events and build 13-feature vectors."""
                if ds not in econ_idx:
                    return
                e = econ_idx[ds]
                o, l = e["offset"], e["length"]
                all_eids.append(np.array(self._mm["econ_event_ids"][o:o+l]))
                all_cids.append(np.array(self._mm["econ_currency_ids"][o:o+l]))
                all_ets.append(np.array(self._mm["econ_timestamps"][o:o+l]))
                # Raw stored cols: [impact_ord, is_usd, time_of_day, actual_z,
                #   forecast_z, previous_z, has_actual, has_forecast,
                #   event_rank_today_norm, days_since_last_same_norm]
                raw = np.array(self._mm["econ_numeric"][o:o+l])  # [N, 10]
                n = len(raw)
                event_date = date.fromisoformat(ds)
                days_until = (event_date - anchor).days
                days_until_norm = float(days_until) / 15.0
                # Build 13-feature vectors matching _load_econ_events
                out = np.zeros((n, 13), dtype=np.float32)
                out[:, 0] = raw[:, 0] / 3.0                       # impact_norm
                out[:, 1] = raw[:, 1]                              # is_usd
                out[:, 2] = days_until_norm                        # days_until_norm
                tod = raw[:, 2]
                out[:, 3] = np.where(tod > 1.0, tod / 24.0, tod)  # time_of_day_norm
                out[:, 4] = float(is_future)                       # is_future
                if is_future:
                    out[:, 5] = 0.0                                # actual_z masked
                    out[:, 8] = 0.0                                # has_actual masked
                else:
                    out[:, 5] = raw[:, 3]                          # actual_z
                    out[:, 8] = raw[:, 6]                          # has_actual
                out[:, 6] = raw[:, 4]                              # forecast_z
                out[:, 7] = raw[:, 5]                              # previous_z
                out[:, 9] = raw[:, 7]                              # has_forecast
                out[:, 10] = raw[:, 8]                             # event_rank_today_norm
                out[:, 11] = raw[:, 9]                             # days_since_last_same_norm
                out[:, 12] = days_until_norm                       # days_until (dup)
                all_enum.append(out)

            # Lookback events
            for d in window_dates:
                _econ_batch(str(d), False)
            # Forward events (D+1 to D+15)
            for fwd_offset in range(1, 16):
                _econ_batch(str(anchor + timedelta(days=fwd_offset)), True)

            if all_eids:
                eids = np.concatenate(all_eids)
                cids = np.concatenate(all_cids)
                enum = np.concatenate(all_enum)
                ets = np.concatenate(all_ets)
                # Sort by timestamp
                sort_idx = np.argsort(ets)
                result['econ_event_ids'] = torch.from_numpy(eids[sort_idx].astype(np.int64))
                result['econ_currency_ids'] = torch.from_numpy(cids[sort_idx].astype(np.int64))
                result['econ_numeric'] = torch.from_numpy(enum[sort_idx])
                result['econ_timestamps'] = torch.from_numpy(ets[sort_idx])
                result['num_econ'] = len(eids)
            else:
                result['econ_event_ids'] = torch.zeros(0, dtype=torch.long)
                result['econ_currency_ids'] = torch.zeros(0, dtype=torch.long)
                result['econ_numeric'] = torch.zeros((0, 13), dtype=torch.float32)
                result['econ_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['num_econ'] = 0
        elif self.use_econ:
            anchor = sample['anchor_date']
            econ_ids, econ_cur, econ_num, econ_ts = self._load_econ_events(window_dates, anchor)
            result['econ_event_ids'] = torch.from_numpy(econ_ids.astype(np.int64))
            result['econ_currency_ids'] = torch.from_numpy(econ_cur.astype(np.int64))
            result['econ_numeric'] = torch.from_numpy(econ_num)
            result['econ_timestamps'] = torch.from_numpy(econ_ts)
            result['num_econ'] = len(econ_ids)

        # ── Options (depends on stock loaded_days) ────────────────────
        if self.use_options:
            all_options = []
            for d, day_bar_count in loaded_days:
                ds = str(d)
                opt = self._mm_slice("options", ds)
                if opt is not None and len(opt) > 0:
                    if len(opt) > day_bar_count:
                        opt = opt[:day_bar_count]
                    elif len(opt) < day_bar_count:
                        padded = np.zeros((day_bar_count, opt.shape[1]), dtype=np.float32)
                        padded[:len(opt)] = opt
                        opt = padded
                    all_options.append(opt)
                else:
                    all_options.append(np.zeros((day_bar_count, self.num_option_features), dtype=np.float32))
            if all_options:
                options = np.concatenate(all_options, axis=0)
                if len(options) > len(bars):
                    options = options[-len(bars):]
                elif len(options) < len(bars):
                    padded = np.zeros((len(bars), self.num_option_features), dtype=np.float32)
                    padded[-len(options):] = options
                    options = padded
                options_mask = (np.abs(options).sum(axis=1) > 1e-8).astype(np.float32)
                if options_mask.sum() > 0:
                    options = self._normalize_bars(options, mask=options_mask)
                result['options'] = torch.from_numpy(options)
                result['options_mask'] = torch.from_numpy(options_mask)
            else:
                result['options'] = torch.zeros(len(bars), self.num_option_features)
                result['options_mask'] = torch.zeros(len(bars))

        # ── VIX features ──────────────────────────────────────────────
        if self.use_vix_features:
            all_vf, all_vt = [], []
            for d in window_dates:
                ds = str(d)
                vf = self._mm_slice("vix", ds)
                if vf is not None and len(vf) > 0:
                    all_vf.append(vf)
                    vt = self._mm_slice("vix", ds, "_timestamps")
                    if vt is not None:
                        all_vt.append(vt)
            if all_vf:
                vix_feats = np.concatenate(all_vf, axis=0)
                vix_ts = np.concatenate(all_vt, axis=0) if all_vt else np.zeros(len(vix_feats), dtype=np.int64)
                if len(vix_feats) > self.max_vix_bars:
                    vix_feats = vix_feats[-self.max_vix_bars:]
                    vix_ts = vix_ts[-self.max_vix_bars:]
                vix_mask = (np.abs(vix_feats).sum(axis=1) > 1e-8).astype(np.float32)
                if vix_mask.sum() > 0:
                    vix_feats = self._normalize_bars(vix_feats, mask=vix_mask)
                result['vix_features'] = torch.from_numpy(vix_feats)
                result['vix_timestamps'] = torch.from_numpy(vix_ts)
                result['vix_mask'] = torch.from_numpy(vix_mask)
                result['num_vix'] = len(vix_feats)
            else:
                result['vix_features'] = torch.zeros(0, self.num_vix_features)
                result['vix_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['vix_mask'] = torch.zeros(0)
                result['num_vix'] = 0

        return result

    def _empty_result(self, sample) -> Dict:
        """Return empty sample with all required keys."""
        result = {
            'bars': torch.zeros(1, self.num_features),
            'vix_targets': torch.zeros(NUM_HORIZONS, dtype=torch.float32),
            'horizon_mask': torch.zeros(NUM_HORIZONS, dtype=torch.float32),
            'num_bars': 0,
            'anchor_date': str(sample['anchor_date']),
        }
        if self.use_news:
            result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
            result['news_embs'] = torch.zeros(0, self.news_dim)
            result['news_timestamps'] = torch.zeros(0, dtype=torch.long)
            result['num_news'] = 0
        if self.use_gdelt:
            if 'bar_timestamps' not in result:
                result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
            result['gdelt_embs'] = torch.zeros(0, self.gdelt_dim)
            result['gdelt_timestamps'] = torch.zeros(0, dtype=torch.long)
            result['num_gdelt'] = 0
        if self.use_macro:
            if 'bar_timestamps' not in result:
                result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
            result['macro_context'] = torch.zeros(max(self.macro_dim, 1), dtype=torch.float32)
        if self.use_econ:
            if 'bar_timestamps' not in result:
                result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
            result['econ_event_ids'] = torch.zeros(0, dtype=torch.long)
            result['econ_currency_ids'] = torch.zeros(0, dtype=torch.long)
            result['econ_numeric'] = torch.zeros(0, self.econ_num_features)
            result['econ_timestamps'] = torch.zeros(0, dtype=torch.long)
            result['num_econ'] = 0
        if self.use_options:
            result['options'] = torch.zeros(1, self.num_option_features)
            result['options_mask'] = torch.zeros(1)
        if self.use_vix_features:
            result['vix_features'] = torch.zeros(0, self.num_vix_features)
            result['vix_timestamps'] = torch.zeros(0, dtype=torch.long)
            result['vix_mask'] = torch.zeros(0)
            result['num_vix'] = 0
        if self.use_fundamentals:
            result['fundamentals_context'] = torch.zeros(max(self.fundamentals_dim, 1), dtype=torch.float32)
        return result

    def __len__(self) -> int:
        return len(self.anchor_dates)

    _samples_loaded = 0  # Class-level counter for first-batch progress

    # ------------------------------------------------------------------
    # Per-feed worker methods (each does its own I/O + processing)
    # ------------------------------------------------------------------

    def _feed_stock(self, window_dates: List[date]) -> Dict:
        """Worker: load + process all stock bar data for the window."""
        need_timestamps = (self.use_news or self.use_macro or self.use_gdelt
                           or self.use_econ or self.use_vix_features)
        all_bars, all_bar_ts, loaded_days = [], [], []
        for d in window_dates:
            fp = self.stock_files.get(d)
            if fp is None:
                continue
            if need_timestamps:
                res = self._load_day_bars_with_timestamps(fp)
                if res is not None:
                    all_bars.append(res[0])
                    all_bar_ts.append(res[1])
                    loaded_days.append((d, len(res[0])))
            else:
                day_bars = self._load_day_bars(fp)
                if day_bars is not None:
                    all_bars.append(day_bars)
                    loaded_days.append((d, len(day_bars)))
        return {'all_bars': all_bars, 'all_bar_ts': all_bar_ts,
                'loaded_days': loaded_days, 'need_timestamps': need_timestamps}

    def _feed_news(self, window_dates: List[date]) -> Dict:
        """Worker: load + process all news data for the window."""
        news_embs, news_ts = self._load_news(window_dates)
        return {'news_embs': news_embs, 'news_ts': news_ts}

    def _feed_gdelt(self, window_dates: List[date]) -> Dict:
        """Worker: load + process all GDELT data for the window."""
        gdelt_embs, gdelt_ts = self._load_gdelt(window_dates)
        return {'gdelt_embs': gdelt_embs, 'gdelt_ts': gdelt_ts}

    def _feed_vix(self, window_dates: List[date]) -> Dict:
        """Worker: load + process all VIX feature data for the window."""
        all_vix_feats, all_vix_ts = [], []
        for d in window_dates:
            res = self._load_vix_features_with_timestamps(d)
            if res is not None:
                all_vix_feats.append(res[0])
                all_vix_ts.append(res[1])
        if not all_vix_feats:
            return {'vix_feats': None}
        vix_feats = np.concatenate(all_vix_feats, axis=0)
        vix_ts = np.concatenate(all_vix_ts, axis=0)
        if len(vix_feats) > self.max_vix_bars:
            vix_feats = vix_feats[-self.max_vix_bars:]
            vix_ts = vix_ts[-self.max_vix_bars:]
        vix_mask = (np.abs(vix_feats).sum(axis=1) > 1e-8).astype(np.float32)
        if vix_mask.sum() > 0:
            vix_feats = self._normalize_bars(vix_feats, mask=vix_mask)
        return {'vix_feats': vix_feats, 'vix_ts': vix_ts, 'vix_mask': vix_mask}

    def _feed_econ(self, window_dates: List[date], anchor_date) -> Dict:
        """Worker: load + process econ calendar data for the window."""
        econ_ids, econ_cur, econ_num, econ_ts = self._load_econ_events(
            window_dates, anchor_date)
        return {'econ_ids': econ_ids, 'econ_cur': econ_cur,
                'econ_num': econ_num, 'econ_ts': econ_ts}

    def _feed_options(self, loaded_days: List, total_bars: int) -> Dict:
        """Worker: load + process options data (needs stock loaded_days)."""
        all_options = []
        for d, day_bar_count in loaded_days:
            if day_bar_count > 0:
                day_options = self._load_day_options(d, day_bar_count)
                if day_options is not None:
                    all_options.append(day_options)
                else:
                    all_options.append(np.zeros((day_bar_count, self.num_option_features),
                                                dtype=np.float32))
        if not all_options:
            return {'options': None}
        options = np.concatenate(all_options, axis=0)
        if len(options) > total_bars:
            options = options[-total_bars:]
        elif len(options) < total_bars:
            padded = np.zeros((total_bars, self.num_option_features), dtype=np.float32)
            padded[-len(options):] = options
            options = padded
        options_mask = (np.abs(options).sum(axis=1) > 1e-8).astype(np.float32)
        if options_mask.sum() > 0:
            options = self._normalize_bars(options, mask=options_mask)
        return {'options': options, 'options_mask': options_mask}

    # ------------------------------------------------------------------
    # __getitem__  — parallel feed loading
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict:
        """Get single sample by index. Uses memmaps if available, else threads."""
        # ── Memmap fast path (microseconds) ───────────────────────────
        if self._use_memmaps:
            return self._getitem_memmap(idx)

        import time as _time
        _t0 = _time.time()
        sample = self.anchor_dates[idx]
        window_dates = sample['window_dates']
        vix_targets = sample['vix_targets']      # [4] multi-horizon targets
        horizon_mask = sample['horizon_mask']    # [4] validity mask
        pool = _get_io_pool()

        # ── Launch independent feeds in parallel ──────────────────────
        stock_fut = pool.submit(self._feed_stock, window_dates)
        news_fut  = pool.submit(self._feed_news, window_dates)  if self.use_news   else None
        gdelt_fut = pool.submit(self._feed_gdelt, window_dates) if self.use_gdelt  else None
        vix_fut   = pool.submit(self._feed_vix, window_dates)   if self.use_vix_features else None
        econ_fut  = pool.submit(self._feed_econ, window_dates, sample['anchor_date']) if self.use_econ else None

        # ── Wait for stock (others keep running) ──────────────────────
        stock = stock_fut.result()
        all_bars = stock['all_bars']
        all_bar_ts = stock['all_bar_ts']
        loaded_days = stock['loaded_days']
        need_timestamps = stock['need_timestamps']

        if not all_bars:
            # Empty sample — cancel pending futures and return zeros
            result = {
                'bars': torch.zeros(1, self.num_features),
                'vix_targets': torch.zeros(NUM_HORIZONS, dtype=torch.float32),
                'horizon_mask': torch.zeros(NUM_HORIZONS, dtype=torch.float32),
                'num_bars': 0,
                'anchor_date': str(sample['anchor_date']),
            }
            if self.use_news:
                result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
                result['news_embs'] = torch.zeros(0, self.news_dim)
                result['news_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['num_news'] = 0
            if self.use_gdelt:
                if 'bar_timestamps' not in result:
                    result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
                result['gdelt_embs'] = torch.zeros(0, self.gdelt_dim)
                result['gdelt_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['num_gdelt'] = 0
            if self.use_macro:
                if 'bar_timestamps' not in result:
                    result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
                result['macro_context'] = torch.zeros(max(self.macro_dim, 1), dtype=torch.float32)
            if self.use_econ:
                if 'bar_timestamps' not in result:
                    result['bar_timestamps'] = torch.zeros(1, dtype=torch.long)
                result['econ_event_ids'] = torch.zeros(0, dtype=torch.long)
                result['econ_currency_ids'] = torch.zeros(0, dtype=torch.long)
                result['econ_numeric'] = torch.zeros(0, self.econ_num_features)
                result['econ_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['num_econ'] = 0
            if self.use_options:
                result['options'] = torch.zeros(1, self.num_option_features)
                result['options_mask'] = torch.zeros(1)
            if self.use_vix_features:
                result['vix_features'] = torch.zeros(0, self.num_vix_features)
                result['vix_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['vix_mask'] = torch.zeros(0)
                result['num_vix'] = 0
            if self.use_fundamentals:
                result['fundamentals_context'] = torch.zeros(max(self.fundamentals_dim, 1), dtype=torch.float32)
            return result

        # ── Assemble stock bars ───────────────────────────────────────
        bars = np.concatenate(all_bars, axis=0)
        if need_timestamps:
            bar_timestamps = np.concatenate(all_bar_ts, axis=0)
        if len(bars) > self.max_total_bars:
            bars = bars[-self.max_total_bars:]
            if need_timestamps:
                bar_timestamps = bar_timestamps[-self.max_total_bars:]
        bars = self._normalize_bars(bars)

        # ── Launch options now that we have loaded_days ────────────────
        options_fut = (pool.submit(self._feed_options, loaded_days, len(bars))
                       if self.use_options else None)

        result = {
            'bars': torch.from_numpy(bars),
            'vix_targets': torch.tensor(vix_targets, dtype=torch.float32),
            'horizon_mask': torch.tensor(horizon_mask, dtype=torch.float32),
            'num_bars': len(bars),
            'anchor_date': str(sample['anchor_date']),
        }
        if need_timestamps:
            result['bar_timestamps'] = torch.from_numpy(bar_timestamps)

        # ── Gather news (already running in thread) ───────────────────
        if news_fut is not None:
            news = news_fut.result()
            result['news_embs'] = torch.from_numpy(news['news_embs'])
            result['news_timestamps'] = torch.from_numpy(news['news_ts'])
            result['num_news'] = len(news['news_embs'])

        # ── Gather GDELT (already running in thread) ──────────────────
        if gdelt_fut is not None:
            gdelt = gdelt_fut.result()
            result['gdelt_embs'] = torch.from_numpy(gdelt['gdelt_embs'].copy())
            result['gdelt_timestamps'] = torch.from_numpy(gdelt['gdelt_ts'].copy())
            result['num_gdelt'] = len(gdelt['gdelt_embs'])

        # ── Macro context (instant dict lookup, not worth threading) ──
        if self.use_macro and self.macro_data is not None:
            anchor = sample['anchor_date']
            if anchor in self.macro_data.index:
                macro_vec = self.macro_data.loc[anchor].values.astype(np.float32)
            else:
                earlier = self.macro_data.index[self.macro_data.index <= anchor]
                if len(earlier) > 0:
                    macro_vec = self.macro_data.loc[earlier[-1]].values.astype(np.float32)
                else:
                    macro_vec = np.zeros(self.macro_dim, dtype=np.float32)
            macro_vec = np.nan_to_num(macro_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if self.macro_mean is not None and self.macro_std is not None:
                macro_vec = (macro_vec - self.macro_mean) / self.macro_std
            result['macro_context'] = torch.from_numpy(macro_vec)

        # ── Fundamentals (instant dict lookup) ────────────────────────
        if self.use_fundamentals and self.fundamentals_data is not None:
            anchor = sample['anchor_date']
            if anchor in self.fundamentals_data.index:
                fund_vec = self.fundamentals_data.loc[anchor].values.astype(np.float32)
            else:
                earlier = self.fundamentals_data.index[self.fundamentals_data.index <= anchor]
                if len(earlier) > 0:
                    fund_vec = self.fundamentals_data.loc[earlier[-1]].values.astype(np.float32)
                else:
                    fund_vec = np.zeros(self.fundamentals_dim, dtype=np.float32)
            fund_vec = np.nan_to_num(fund_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if self.fundamentals_mean is not None and self.fundamentals_std is not None:
                fund_vec = (fund_vec - self.fundamentals_mean) / self.fundamentals_std
            result['fundamentals_context'] = torch.from_numpy(fund_vec)

        # ── Gather econ (already running in thread) ───────────────────
        if econ_fut is not None:
            econ = econ_fut.result()
            result['econ_event_ids'] = torch.from_numpy(econ['econ_ids'].astype(np.int64))
            result['econ_currency_ids'] = torch.from_numpy(econ['econ_cur'].astype(np.int64))
            result['econ_numeric'] = torch.from_numpy(econ['econ_num'])
            result['econ_timestamps'] = torch.from_numpy(econ['econ_ts'])
            result['num_econ'] = len(econ['econ_ids'])

        # ── Gather options (launched after stock) ─────────────────────
        if options_fut is not None:
            opt = options_fut.result()
            if opt['options'] is not None:
                result['options'] = torch.from_numpy(opt['options'])
                result['options_mask'] = torch.from_numpy(opt['options_mask'])
            else:
                result['options'] = torch.zeros(len(bars), self.num_option_features)
                result['options_mask'] = torch.zeros(len(bars))

        # ── Gather VIX features (already running in thread) ───────────
        if vix_fut is not None:
            vix = vix_fut.result()
            if vix['vix_feats'] is not None:
                result['vix_features'] = torch.from_numpy(vix['vix_feats'])
                result['vix_timestamps'] = torch.from_numpy(vix['vix_ts'])
                result['vix_mask'] = torch.from_numpy(vix['vix_mask'])
                result['num_vix'] = len(vix['vix_feats'])
            else:
                result['vix_features'] = torch.zeros(0, self.num_vix_features)
                result['vix_timestamps'] = torch.zeros(0, dtype=torch.long)
                result['vix_mask'] = torch.zeros(0)
                result['num_vix'] = 0

        # Log progress for first batch (cold cache)
        BarMambaDataset._samples_loaded += 1
        n = BarMambaDataset._samples_loaded
        elapsed = _time.time() - _t0
        if n <= 64 and (n <= 3 or n % 8 == 0):
            logger.info(f"Sample {n} loaded in {elapsed:.1f}s (idx={idx}, {len(window_dates)}d, {result['num_bars']} bars)")

        return result

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate variable-length bar sequences with padding."""
        if len(batch) == 0:
            return {}

        max_len = max(b['num_bars'] for b in batch)
        num_features = batch[0]['bars'].shape[1]
        B = len(batch)

        bars_padded = torch.zeros(B, max_len, num_features)
        bar_mask = torch.zeros(B, max_len)
        
        # Multi-horizon targets: [B, 4] and horizon validity mask
        vix_targets = torch.stack([b['vix_targets'] for b in batch])  # [B, 4]
        horizon_mask = torch.stack([b['horizon_mask'] for b in batch])  # [B, 4]

        for i, b in enumerate(batch):
            T = b['num_bars']
            bars_padded[i, :T, :] = b['bars']
            bar_mask[i, :T] = 1.0

        result = {
            'bars': bars_padded,
            'bar_mask': bar_mask,
            'vix_targets': vix_targets,      # [B, 4] multi-horizon targets
            'horizon_mask': horizon_mask,    # [B, 4] validity mask
            'num_bars': [b['num_bars'] for b in batch],
        }

        # Handle news data if present
        if 'news_embs' in batch[0]:
            news_dim = batch[0]['news_embs'].shape[-1] if batch[0]['news_embs'].numel() > 0 else 3072
            max_news = max(b['num_news'] for b in batch) if any(b['num_news'] > 0 for b in batch) else 1
            
            news_padded = torch.zeros(B, max_news, news_dim)
            news_mask = torch.zeros(B, max_news)
            news_ts_padded = torch.zeros(B, max_news, dtype=torch.long)
            bar_ts_padded = torch.zeros(B, max_len, dtype=torch.long)
            
            for i, b in enumerate(batch):
                # Bar timestamps
                T = b['num_bars']
                if 'bar_timestamps' in b and b['bar_timestamps'].numel() > 0:
                    bar_ts_padded[i, :T] = b['bar_timestamps'][:T]
                
                # News
                N = b['num_news']
                if N > 0:
                    news_padded[i, :N] = b['news_embs']
                    news_mask[i, :N] = 1.0
                    news_ts_padded[i, :N] = b['news_timestamps']

            result['bar_timestamps'] = bar_ts_padded
            result['news_embs'] = news_padded
            result['news_mask'] = news_mask
            result['news_timestamps'] = news_ts_padded
            result['num_news'] = [b['num_news'] for b in batch]
        
        # Handle GDELT data if present
        if 'gdelt_embs' in batch[0]:
            gdelt_dim = batch[0]['gdelt_embs'].shape[-1] if batch[0]['gdelt_embs'].numel() > 0 else 391
            max_gdelt = max(b.get('num_gdelt', 0) for b in batch) if any(b.get('num_gdelt', 0) > 0 for b in batch) else 1
            
            gdelt_padded = torch.zeros(B, max_gdelt, gdelt_dim)
            gdelt_mask = torch.zeros(B, max_gdelt)
            gdelt_ts_padded = torch.zeros(B, max_gdelt, dtype=torch.long)
            
            # Ensure bar_timestamps exists
            if 'bar_timestamps' not in result:
                bar_ts_padded = torch.zeros(B, max_len, dtype=torch.long)
                for i, b in enumerate(batch):
                    T = b['num_bars']
                    if 'bar_timestamps' in b and b['bar_timestamps'].numel() > 0:
                        bar_ts_padded[i, :T] = b['bar_timestamps'][:T]
                result['bar_timestamps'] = bar_ts_padded
            
            for i, b in enumerate(batch):
                G = b.get('num_gdelt', 0)
                if G > 0:
                    gdelt_padded[i, :G] = b['gdelt_embs']
                    gdelt_mask[i, :G] = 1.0
                    gdelt_ts_padded[i, :G] = b['gdelt_timestamps']
            
            result['gdelt_embs'] = gdelt_padded
            result['gdelt_mask'] = gdelt_mask
            result['gdelt_timestamps'] = gdelt_ts_padded
            result['num_gdelt'] = [b.get('num_gdelt', 0) for b in batch]
        
        # Handle bar_timestamps for macro (when news isn't present but macro is)
        if 'bar_timestamps' in batch[0] and 'bar_timestamps' not in result:
            bar_ts_padded = torch.zeros(B, max_len, dtype=torch.long)
            for i, b in enumerate(batch):
                T = b['num_bars']
                if b['bar_timestamps'].numel() > 0:
                    bar_ts_padded[i, :T] = b['bar_timestamps'][:T]
            result['bar_timestamps'] = bar_ts_padded
        
        # Handle macro context if present
        if 'macro_context' in batch[0]:
            result['macro_context'] = torch.stack([b['macro_context'] for b in batch])
        
        # Handle fundamentals context if present
        if 'fundamentals_context' in batch[0]:
            result['fundamentals_context'] = torch.stack([b['fundamentals_context'] for b in batch])
        
        # Handle econ calendar data if present
        if 'econ_event_ids' in batch[0]:
            max_econ = max(b.get('num_econ', 0) for b in batch) if any(b.get('num_econ', 0) > 0 for b in batch) else 1
            econ_num_features = batch[0]['econ_numeric'].shape[-1] if batch[0]['econ_numeric'].numel() > 0 else 13
            
            econ_event_ids_padded = torch.zeros(B, max_econ, dtype=torch.long)
            econ_currency_ids_padded = torch.zeros(B, max_econ, dtype=torch.long)
            econ_numeric_padded = torch.zeros(B, max_econ, econ_num_features)
            econ_mask = torch.zeros(B, max_econ)
            econ_ts_padded = torch.zeros(B, max_econ, dtype=torch.long)
            
            # Ensure bar_timestamps exists
            if 'bar_timestamps' not in result:
                bar_ts_padded = torch.zeros(B, max_len, dtype=torch.long)
                for i, b in enumerate(batch):
                    T = b['num_bars']
                    if 'bar_timestamps' in b and b['bar_timestamps'].numel() > 0:
                        bar_ts_padded[i, :T] = b['bar_timestamps'][:T]
                result['bar_timestamps'] = bar_ts_padded
            
            for i, b in enumerate(batch):
                E = b.get('num_econ', 0)
                if E > 0:
                    econ_event_ids_padded[i, :E] = b['econ_event_ids']
                    econ_currency_ids_padded[i, :E] = b['econ_currency_ids']
                    econ_numeric_padded[i, :E] = b['econ_numeric']
                    econ_mask[i, :E] = 1.0
                    econ_ts_padded[i, :E] = b['econ_timestamps']
            
            result['econ_event_ids'] = econ_event_ids_padded
            result['econ_currency_ids'] = econ_currency_ids_padded
            result['econ_numeric'] = econ_numeric_padded
            result['econ_mask'] = econ_mask
            result['econ_timestamps'] = econ_ts_padded
            result['num_econ'] = [b.get('num_econ', 0) for b in batch]
        
        # Handle options data if present
        if 'options' in batch[0]:
            num_option_features = batch[0]['options'].shape[-1]
            options_padded = torch.zeros(B, max_len, num_option_features)
            options_mask = torch.zeros(B, max_len)
            
            for i, b in enumerate(batch):
                T = b['num_bars']
                if b['options'].numel() > 0:
                    options_padded[i, :T, :] = b['options'][:T]
                    if 'options_mask' in b and b['options_mask'].numel() > 0:
                        options_mask[i, :T] = b['options_mask'][:T]
            
            result['options'] = options_padded
            result['options_mask'] = options_mask
        
        # Handle VIX features if present (separate sequence with own length)
        if 'vix_features' in batch[0]:
            num_vix_features = batch[0]['vix_features'].shape[-1] if batch[0]['vix_features'].numel() > 0 else NUM_VIX_FEATURES
            max_vix = max(b.get('num_vix', 0) for b in batch) if any(b.get('num_vix', 0) > 0 for b in batch) else 1
            
            vix_padded = torch.zeros(B, max_vix, num_vix_features)
            vix_mask = torch.zeros(B, max_vix)
            vix_ts_padded = torch.zeros(B, max_vix, dtype=torch.long)
            
            for i, b in enumerate(batch):
                V = b.get('num_vix', 0)
                if V > 0 and b['vix_features'].numel() > 0:
                    vix_padded[i, :V, :] = b['vix_features'][:V]
                    vix_mask[i, :V] = b['vix_mask'][:V] if 'vix_mask' in b else 1.0
                    if 'vix_timestamps' in b and b['vix_timestamps'].numel() > 0:
                        vix_ts_padded[i, :V] = b['vix_timestamps'][:V]
            
            result['vix_features'] = vix_padded
            result['vix_mask'] = vix_mask
            result['vix_timestamps'] = vix_ts_padded
            result['num_vix'] = [b.get('num_vix', 0) for b in batch]

        return result
