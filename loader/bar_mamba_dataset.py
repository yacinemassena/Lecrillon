"""
Bar-to-Mamba Dataset: Direct 2-min bar sequences for Mamba-only VIX prediction.

Multi-horizon targets: +1d, +7d, +15d, +30d VIX change prediction.
Spike-weighted loss support via per-sample VIX change magnitudes.

No caching - just load files directly, train, discard.
Simple and clean.
"""

import logging
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
# Default bar features (all 47 features from Stock_Data_2min parquet columns)
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
]

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
]
NUM_OPTION_FEATURES = len(OPTION_FEATURES)

# Multi-horizon prediction targets (days ahead)
HORIZONS = [1, 7, 15, 30]  # +1d, +7d, +15d, +30d
NUM_HORIZONS = len(HORIZONS)

# Class-level news cache (shared across all dataset instances/workers)
# Loaded once in main process, shared via copy-on-write in workers
_NEWS_CACHE: Dict[int, pd.DataFrame] = {}
_NEWS_CACHE_PATH: Optional[Path] = None


def preload_news_cache(news_path: str, start_date: str = None, end_date: str = None):
    """Preload news cache in main process before spawning workers.
    
    Call this before creating DataLoader to avoid redundant loading in workers.
    Only loads years within the specified date range.
    
    Args:
        news_path: Path to news embeddings directory
        start_date: Start date string (YYYY-MM-DD) - determines first year to load
        end_date: End date string (YYYY-MM-DD) - determines last year to load
    """
    global _NEWS_CACHE, _NEWS_CACHE_PATH
    news_path = Path(news_path)
    _NEWS_CACHE_PATH = news_path
    
    # Determine years to load from date range
    if start_date and end_date:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        years = list(range(start_year, end_year + 1))
    else:
        # Fallback: load nothing, let lazy loading handle it
        return
    
    for year in years:
        if year not in _NEWS_CACHE:
            path = news_path / f"{year}_embedded.parquet"
            if path.exists():
                try:
                    _NEWS_CACHE[year] = pd.read_parquet(path)
                    logger.info(f"Preloaded news cache for {year}: {len(_NEWS_CACHE[year])} articles")
                except Exception as e:
                    logger.warning(f"Failed to preload news for {year}: {e}")


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
        
        # News data (Benzinga embeddings)
        self.news_path = Path(news_data_path) if news_data_path else None
        self.news_cache: Dict[int, pd.DataFrame] = {}  # year -> DataFrame
        self.news_dim = 3072  # OpenAI embedding dimension
        
        # GDELT data (world state embeddings every 15 min)
        self.gdelt_path = Path(gdelt_data_path) if gdelt_data_path else None
        self.gdelt_embed_dim = 384  # MiniLM embedding dimension
        self.gdelt_stats_dim = 7    # Summary stats (article_count, goldstein, tone, etc.)
        self.gdelt_dim = self.gdelt_embed_dim + self.gdelt_stats_dim  # 391 total
        
        # Macro conditioning data (FiLM)
        self.macro_data: Optional[pd.DataFrame] = None
        self.macro_dim = 0
        self.macro_features: List[str] = []
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
                self.macro_features = [c for c in self.macro_data.columns]
                self.macro_dim = len(self.macro_features)
                logger.info(f"Loaded macro data: {len(self.macro_data)} days, {self.macro_dim} features")
            else:
                logger.warning(f"Macro data not found: {macro_path}")
        
        # Options data
        self.options_path = Path(options_data_path) if options_data_path else None
        self.option_features = OPTION_FEATURES
        self.num_option_features = NUM_OPTION_FEATURES
        self.option_files: Dict = {}  # date -> file path
        
        # Calculate days needed based on seq_len (~195 bars per trading day at 2-min)
        bars_per_day = 195
        self.lookback_days = max(1, (max_total_bars // bars_per_day) + 1)
        # Anchor dates
        self.anchor_start = pd.to_datetime(train_start).date()
        self.train_end = pd.to_datetime(train_end).date()
        self.val_end = pd.to_datetime(val_end).date()

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

        # Load VIX daily close
        self.vix_daily = load_vix_daily_close(vix_data_path)

        # Build valid anchor dates
        self.anchor_dates = self._build_anchor_dates()
        
        sources = ['stock']
        if self.use_options:
            sources.append(f'options({len(self.option_files)} days)')
        if self.use_news:
            sources.append('news')
        logger.info(
            f"BarMambaDataset [{split}]: {len(self.anchor_dates)} samples, "
            f"seq={max_total_bars} (~{self.lookback_days}d), sources={sources}"
        )

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
        """Load one day of bars as feature matrix. No caching."""
        result = self._load_day_bars_with_timestamps(file_path)
        if result is None:
            return None
        return result[0]  # Return only features, not timestamps

    def _load_day_bars_with_timestamps(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load one day of bars with timestamps. Returns (features, timestamps)."""
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

        # Extract features
        avail = [f for f in self.features if f in df.columns]
        if not avail:
            return None

        features = df.select(avail).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

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
                    if path.exists():
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
        
        all_gdelt = []
        
        for d in window_dates:
            year = d.year
            month = d.month
            day = d.day
            path = self.gdelt_path / str(year) / f"{month:02d}" / f"{day:02d}.parquet"
            
            if not path.exists():
                continue
            
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Failed to load GDELT for {d}: {e}")
                continue
            
            if len(df) == 0:
                continue
            
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
        
        try:
            df = pl.read_parquet(self.option_files[d])
        except Exception as e:
            logger.warning(f"Failed to load options for {d}: {e}")
            return None
        
        if len(df) == 0:
            return None
        
        # Sort by timestamp
        ts_col = 'bar_timestamp' if 'bar_timestamp' in df.columns else 'timestamp'
        if ts_col not in df.columns:
            return None
        df = df.sort(ts_col)
        
        # Aggregate across all underlyings per second (sum volumes, mean ratios)
        # Group by timestamp and aggregate
        agg_exprs = []
        for feat in self.option_features:
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
        
        # Extract features
        avail = [f for f in self.option_features if f in df_agg.columns]
        if not avail:
            return None
        
        features = df_agg.select(avail).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
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

    def __len__(self) -> int:
        return len(self.anchor_dates)

    def __getitem__(self, idx: int) -> Dict:
        """Get single sample by index. Data cached in RAM after first load."""
        sample = self.anchor_dates[idx]
        window_dates = sample['window_dates']
        vix_targets = sample['vix_targets']      # [4] multi-horizon targets
        horizon_mask = sample['horizon_mask']    # [4] validity mask

        # Load bars for all lookback days (with timestamps if using news, macro, or gdelt)
        need_timestamps = self.use_news or self.use_macro or self.use_gdelt
        all_bars = []
        all_bar_ts = []
        for d in window_dates:
            file_path = self.stock_files.get(d)
            if file_path is None:
                continue
            if need_timestamps:
                result = self._load_day_bars_with_timestamps(file_path)
                if result is not None:
                    all_bars.append(result[0])
                    all_bar_ts.append(result[1])
            else:
                day_bars = self._load_day_bars(file_path)
                if day_bars is not None:
                    all_bars.append(day_bars)

        if not all_bars:
            # Return empty sample (will be filtered by collate)
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
            return result

        # Concatenate all days into one sequence
        bars = np.concatenate(all_bars, axis=0)
        if need_timestamps:
            bar_timestamps = np.concatenate(all_bar_ts, axis=0)

        # Cap total sequence length
        if len(bars) > self.max_total_bars:
            bars = bars[-self.max_total_bars:]
            if need_timestamps:
                bar_timestamps = bar_timestamps[-self.max_total_bars:]

        # Normalize bars (not targets — VIX change stays in raw points)
        bars = self._normalize_bars(bars)

        result = {
            'bars': torch.from_numpy(bars),
            'vix_targets': torch.tensor(vix_targets, dtype=torch.float32),      # [4] horizons
            'horizon_mask': torch.tensor(horizon_mask, dtype=torch.float32),    # [4] validity
            'num_bars': len(bars),
            'anchor_date': str(sample['anchor_date']),
        }

        # Add bar timestamps if needed by news or macro
        if need_timestamps:
            result['bar_timestamps'] = torch.from_numpy(bar_timestamps)
        
        # Add news data if enabled
        if self.use_news:
            news_embs, news_ts = self._load_news(window_dates)
            result['news_embs'] = torch.from_numpy(news_embs)
            result['news_timestamps'] = torch.from_numpy(news_ts)
            result['num_news'] = len(news_embs)
            
            # Compute news_indices: map each news timestamp to nearest bar index
            if len(news_ts) > 0 and len(bar_timestamps) > 0:
                # Use searchsorted to find insertion point for each news timestamp
                # This gives the index of the bar that comes after (or at) the news time
                news_indices = np.searchsorted(bar_timestamps, news_ts, side='right') - 1
                news_indices = np.clip(news_indices, 0, len(bar_timestamps) - 1)
                result['news_indices'] = torch.from_numpy(news_indices.astype(np.int64))
            else:
                result['news_indices'] = torch.zeros(len(news_embs), dtype=torch.long)
        
        # Add GDELT data if enabled
        if self.use_gdelt:
            gdelt_embs, gdelt_ts = self._load_gdelt(window_dates)
            result['gdelt_embs'] = torch.from_numpy(gdelt_embs.copy())
            result['gdelt_timestamps'] = torch.from_numpy(gdelt_ts.copy())
            result['num_gdelt'] = len(gdelt_embs)
        
        # Add macro context if enabled (T-1 lookup)
        if self.use_macro and self.macro_data is not None:
            anchor = sample['anchor_date']
            # Find T-1: the macro_data is already shifted by 1 day in build script,
            # so we just look up the anchor date directly
            if anchor in self.macro_data.index:
                macro_vec = self.macro_data.loc[anchor].values.astype(np.float32)
            else:
                # Find nearest earlier date
                earlier = self.macro_data.index[self.macro_data.index <= anchor]
                if len(earlier) > 0:
                    macro_vec = self.macro_data.loc[earlier[-1]].values.astype(np.float32)
                else:
                    macro_vec = np.zeros(self.macro_dim, dtype=np.float32)
            result['macro_context'] = torch.from_numpy(macro_vec)
        
        # Add option data if enabled
        if self.use_options:
            all_options = []
            for i, d in enumerate(window_dates):
                # Get stock bar count for this day to match option length
                day_bar_count = len(all_bars[i]) if i < len(all_bars) else 0
                if day_bar_count > 0:
                    day_options = self._load_day_options(d, day_bar_count)
                    if day_options is not None:
                        all_options.append(day_options)
                    else:
                        # No options for this day - zero fill
                        all_options.append(np.zeros((day_bar_count, self.num_option_features), dtype=np.float32))
            
            if all_options:
                options = np.concatenate(all_options, axis=0)
                # Cap to match bars length
                if len(options) > len(bars):
                    options = options[-len(bars):]
                elif len(options) < len(bars):
                    padded = np.zeros((len(bars), self.num_option_features), dtype=np.float32)
                    padded[-len(options):] = options
                    options = padded
                # Create mask BEFORE normalization: 1 where we have actual option data
                options_mask = (np.abs(options).sum(axis=1) > 1e-8).astype(np.float32)
                # Normalize options using only non-masked values for stats
                if options_mask.sum() > 0:
                    options = self._normalize_bars(options, mask=options_mask)
                result['options'] = torch.from_numpy(options)
                result['options_mask'] = torch.from_numpy(options_mask)
            else:
                result['options'] = torch.zeros(len(bars), self.num_option_features)
                result['options_mask'] = torch.zeros(len(bars))

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

            news_indices_padded = torch.zeros(B, max_news, dtype=torch.long)
            
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
                    if 'news_indices' in b and b['news_indices'].numel() > 0:
                        news_indices_padded[i, :N] = b['news_indices']

            result['bar_timestamps'] = bar_ts_padded
            result['news_embs'] = news_padded
            result['news_mask'] = news_mask
            result['news_timestamps'] = news_ts_padded
            result['news_indices'] = news_indices_padded
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

        return result
