"""
Bar-to-Mamba Dataset: Direct 1s bar sequences for Mamba-only VIX prediction.

No Transformer, no frame grouping, no chunking.
Each sample = lookback_days of 1s bars as a flat sequence → predict next-day VIX close.

Training starts at 2005 but lookback windows reach into pre-2005 stock data.

Data flow:
    Stock parquets (1s bars) → filter features → z-score normalize
    VIX CSVs → extract daily close
    Pair: bars[D-lookback..D] → VIX[D+1]

Features async RAM caching with smart prefetch/eviction.
"""

import logging
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Set, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async File Cache (shared with vix_tick_dataset.py pattern)
# ---------------------------------------------------------------------------
class AsyncFileCache:
    """
    Async file cache with background prefetching and smart eviction.
    
    - Prefetches upcoming files in background threads
    - Evicts old files when moving forward through dataset
    - Never blocks training (async I/O)
    """
    
    def __init__(self, max_gb: float = 80.0, prefetch_count: int = 10, 
                 evict_behind: int = 5, num_threads: int = 4):
        self.max_bytes = int(max_gb * 1024**3)
        self.prefetch_count = prefetch_count
        self.evict_behind = evict_behind
        
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_sizes: Dict[str, int] = {}
        self._current_bytes = 0
        self._lock = threading.RLock()
        
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        self._pending_fetches: Dict[str, bool] = {}
        self._file_list: List[Path] = []
        self._current_index = 0
        
        self._stats = {'hits': 0, 'misses': 0, 'prefetches': 0, 'evictions': 0}
    
    def set_file_list(self, files: List[Path]):
        """Set the ordered list of files for prefetching."""
        with self._lock:
            self._file_list = list(files)
            self._current_index = 0
    
    def _estimate_df_size(self, df: pl.DataFrame) -> int:
        """Estimate DataFrame memory usage in bytes."""
        return df.estimated_size()
    
    def _load_file(self, file_path: Path) -> Optional[pl.DataFrame]:
        """Load a parquet file from disk (called in background thread)."""
        try:
            return pl.read_parquet(file_path)
        except Exception as e:
            logger.warning(f"Cache: Failed to load {file_path.name}: {e}")
            return None
    
    def _prefetch_file(self, file_path: Path):
        """Background prefetch a file into cache."""
        key = str(file_path)
        
        with self._lock:
            if key in self._cache or key in self._pending_fetches:
                return
            self._pending_fetches[key] = True
        
        df = self._load_file(file_path)
        
        with self._lock:
            self._pending_fetches.pop(key, None)
            if df is not None and key not in self._cache:
                size = self._estimate_df_size(df)
                
                # Evict if needed to make room
                while self._current_bytes + size > self.max_bytes and self._cache:
                    oldest_key, _ = self._cache.popitem(last=False)
                    evicted_size = self._cache_sizes.pop(oldest_key, 0)
                    self._current_bytes -= evicted_size
                    self._stats['evictions'] += 1
                
                if self._current_bytes + size <= self.max_bytes:
                    self._cache[key] = df
                    self._cache_sizes[key] = size
                    self._current_bytes += size
                    self._stats['prefetches'] += 1
    
    def _trigger_prefetch(self, current_idx: int):
        """Trigger async prefetch of upcoming files."""
        if not self._file_list:
            return
            
        for i in range(1, self.prefetch_count + 1):
            next_idx = current_idx + i
            if next_idx < len(self._file_list):
                self._executor.submit(self._prefetch_file, self._file_list[next_idx])
    
    def _evict_old_files(self, current_idx: int):
        """Evict files that are far behind current position."""
        if not self._file_list or current_idx <= self.evict_behind:
            return
            
        keep_start = max(0, current_idx - self.evict_behind)
        keep_keys = {str(self._file_list[i]) for i in range(keep_start, len(self._file_list)) 
                     if i < len(self._file_list)}
        
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k not in keep_keys]
            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                    evicted_size = self._cache_sizes.pop(key, 0)
                    self._current_bytes -= evicted_size
                    self._stats['evictions'] += 1
    
    def get(self, file_path: Path, file_index: int = -1) -> Optional[pd.DataFrame]:
        """Get a file from cache or load it."""
        key = str(file_path)
        
        if file_index >= 0:
            self._current_index = file_index
            self._trigger_prefetch(file_index)
            self._evict_old_files(file_index)
        
        with self._lock:
            if key in self._cache:
                self._stats['hits'] += 1
                self._cache.move_to_end(key)
                return self._cache[key]
        
        self._stats['misses'] += 1
        df = self._load_file(file_path)
        
        if df is not None:
            with self._lock:
                size = self._estimate_df_size(df)
                
                while self._current_bytes + size > self.max_bytes and self._cache:
                    oldest_key, _ = self._cache.popitem(last=False)
                    evicted_size = self._cache_sizes.pop(oldest_key, 0)
                    self._current_bytes -= evicted_size
                    self._stats['evictions'] += 1
                
                if self._current_bytes + size <= self.max_bytes:
                    self._cache[key] = df
                    self._cache_sizes[key] = size
                    self._current_bytes += size
        
        return df
    
    def get_stats(self) -> Dict:
        """Return cache statistics."""
        with self._lock:
            return {
                **self._stats,
                'cached_files': len(self._cache),
                'cache_mb': self._current_bytes / (1024**2),
                'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
            }
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._cache_sizes.clear()
            self._current_bytes = 0
    
    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)


# Global cache instance
_global_bar_cache: Optional[AsyncFileCache] = None
_cache_lock = threading.Lock()


def get_bar_cache(max_gb: float = 80.0, prefetch_count: int = 10) -> AsyncFileCache:
    """Get or create the global bar file cache.
    
    Note: In multiprocessing (DataLoader with num_workers>0), each worker gets its own cache.
    To avoid OOM, cache size is divided by number of workers.
    """
    global _global_bar_cache
    
    with _cache_lock:
        if _global_bar_cache is None:
            # Detect if we're in a DataLoader worker process
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            
            if worker_info is not None:
                # Divide cache among workers to avoid exceeding total RAM
                num_workers = worker_info.num_workers
                worker_cache_gb = max_gb / num_workers
                logger.info(f"Worker {worker_info.id}/{num_workers}: cache={worker_cache_gb:.1f}GB")
                max_gb = worker_cache_gb
            
            _global_bar_cache = AsyncFileCache(
                max_gb=max_gb,
                prefetch_count=prefetch_count,
                evict_behind=5,
                num_threads=8  # More threads for faster loading
            )
            if worker_info is None:
                logger.info(f"Initialized async bar cache: {max_gb}GB, prefetch={prefetch_count}")
        return _global_bar_cache


def warm_cache(stock_data_path: str, max_gb: float = 80.0, 
               train_start: str = '2005-01-01', train_end: str = '2023-11-30',
               lookback_days: int = 15) -> None:
    """
    Pre-warm the file cache by loading parquet files in the training date range.
    Call this BEFORE creating DataLoaders to eliminate first-epoch slowdown.
    
    Args:
        stock_data_path: Path to stock data directory
        max_gb: Cache size limit
        train_start: Training start date (anchors start here)
        train_end: Training end date
        lookback_days: Days of lookback (need files before train_start)
    """
    import time
    from tqdm import tqdm
    import pandas as pd
    
    stock_path = Path(stock_data_path).resolve()
    all_files = sorted([f.resolve() for f in stock_path.glob('*.parquet')])
    
    if not all_files:
        logger.warning(f"No parquet files found in {stock_data_path}")
        return
    
    # Parse dates from filenames and filter to training range
    # Need files from (train_start - lookback_days) to train_end
    start_date = (pd.to_datetime(train_start) - pd.Timedelta(days=lookback_days * 2)).date()
    end_date = pd.to_datetime(train_end).date()
    
    files_to_cache = []
    for f in all_files:
        try:
            date_str = f.stem.split('.')[0]
            file_date = pd.to_datetime(date_str).date()
            if start_date <= file_date <= end_date:
                files_to_cache.append(f)
        except:
            continue
    
    if not files_to_cache:
        logger.warning(f"No files found in date range {start_date} to {end_date}")
        return
    
    cache = get_bar_cache(max_gb=max_gb, prefetch_count=20)
    cache.set_file_list(files_to_cache)
    
    logger.info(f"🔥 Warming cache: {len(files_to_cache)} files ({start_date} to {end_date})")
    start = time.time()
    
    loaded = 0
    total_mb = 0
    
    for i, f in enumerate(tqdm(files_to_cache, desc="Loading to RAM")):
        df = cache.get(f, i)
        if df is not None:
            loaded += 1
            total_mb = cache._current_bytes / (1024**2)
        
        # Stop if cache is full
        if cache._current_bytes >= cache.max_bytes * 0.95:
            logger.info(f"Cache full at {total_mb:.0f}MB after {loaded} files")
            break
    
    elapsed = time.time() - start
    logger.info(
        f"✅ Cached: {loaded}/{len(files_to_cache)} files, {total_mb:.0f}MB in {elapsed:.1f}s "
        f"({total_mb/elapsed:.1f} MB/s)"
    )

# ---------------------------------------------------------------------------
# Default bar features (matches Stock_Data_1s parquet columns)
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = [
    'close', 'volume', 'trade_count',
    'price_std', 'price_range_pct', 'vwap',
    'avg_trade_size', 'amihud', 'buy_volume', 'sell_volume',
    'tick_arrival_rate', 'large_trade_ratio', 'tick_burst',
    'rv_intrabar', 'ofi',
]


# ---------------------------------------------------------------------------
# VIX daily close loader
# ---------------------------------------------------------------------------
def load_vix_daily_close(vix_dir: str) -> Dict:
    """Load VIX 1-min CSVs and extract daily close (last bar per trading day).

    Returns:
        dict mapping datetime.date → float (VIX close)
    """
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
class BarMambaDataset(IterableDataset):
    """
    Direct bar-to-Mamba dataset.

    Each sample:
    - Input: lookback_days of 1s bars as a flat sequence [T, num_features]
    - Target: next-day VIX close (z-score normalized)

    Split logic:
    - Anchor dates (prediction dates) start at 2005-01-01
    - Lookback windows reach into pre-2005 data (kept for context)
    - train: anchors in [2005-01-01, train_end]
    - val: anchors in (train_end, val_end]
    - test: anchors after val_end
    """

    def __init__(
        self,
        stock_data_path: str,
        vix_data_path: str,
        split: str = 'train',
        features: Optional[List[str]] = None,
        max_bars_per_day: int = 23400,
        max_total_bars: int = 50000,
        train_end: str = '2023-11-30',
        val_end: str = '2024-12-31',
        vix_normalize: str = 'zscore',
        vix_mean: float = 19.14,
        vix_std: float = 8.24,
        allowed_tickers_file: Optional[str] = None,
        prefetch_files: int = 8,
        cache_gb: float = 80.0,
        use_chunked_cache: bool = False,
    ):
        self.split = split
        self.features = features or DEFAULT_FEATURES
        self.num_features = len(self.features)
        self.max_bars_per_day = max_bars_per_day
        self.max_total_bars = max_total_bars
        
        # Calculate days needed based on seq_len (~23,400 bars per trading day)
        bars_per_day = 23400
        self.lookback_days = max(1, (max_total_bars // bars_per_day) + 1)
        self.vix_normalize = vix_normalize
        self.vix_mean = vix_mean
        self.vix_std = vix_std
        self.prefetch_files = prefetch_files
        self.cache_gb = cache_gb
        self.use_chunked_cache = use_chunked_cache
        
        # Initialize async file cache
        self._file_cache = get_bar_cache(max_gb=cache_gb, prefetch_count=prefetch_files)

        # Anchor dates always start at 2005
        self.anchor_start = pd.to_datetime('2005-01-01').date()
        self.train_end = pd.to_datetime(train_end).date()
        self.val_end = pd.to_datetime(val_end).date()

        # Load ticker filter
        self.allowed_tickers: Set[str] = set()
        if allowed_tickers_file and Path(allowed_tickers_file).exists():
            with open(allowed_tickers_file, 'r') as f:
                self.allowed_tickers = set(line.strip() for line in f if line.strip())
            logger.info(f"Ticker filter: {len(self.allowed_tickers)} tickers")

        # Index stock files by date (use resolved path for cache key consistency)
        self.stock_path = Path(stock_data_path).resolve()
        self.stock_files: Dict = {}
        self._index_stock_files()

        # Load VIX daily close
        self.vix_daily = load_vix_daily_close(vix_data_path)

        # Build valid anchor dates
        self.anchor_dates = self._build_anchor_dates()
        logger.info(
            f"BarMambaDataset [{split}]: {len(self.anchor_dates)} samples, "
            f"seq={max_total_bars} (~{self.lookback_days}d), features={self.num_features}"
        )
    
    def warm_dataset_cache(self):
        """Pre-load all files this dataset will use into the cache."""
        import time
        from tqdm import tqdm
        
        files_to_load = list(self.stock_files.values())
        logger.info(f"🔥 Warming cache with {len(files_to_load)} dataset files...")
        
        start = time.time()
        loaded = 0
        
        for i, f in enumerate(tqdm(files_to_load, desc="Loading to RAM")):
            df = self._file_cache.get(f, i)
            if df is not None:
                loaded += 1
            
            # Stop if cache is full
            if self._file_cache._current_bytes >= self._file_cache.max_bytes * 0.95:
                total_mb = self._file_cache._current_bytes / (1024**2)
                logger.info(f"Cache full at {total_mb:.0f}MB after {loaded} files")
                break
        
        elapsed = time.time() - start
        total_mb = self._file_cache._current_bytes / (1024**2)
        logger.info(
            f"✅ Cached: {loaded}/{len(files_to_load)} files, {total_mb:.0f}MB in {elapsed:.1f}s "
            f"({total_mb/elapsed:.1f} MB/s)"
        )

    def _index_stock_files(self):
        """Build date → file path mapping from all available stock files."""
        for f in self.stock_path.glob('*.parquet'):
            if f.name.startswith('._'):
                continue
            date_str = f.stem.split('.')[0]
            try:
                dt = pd.to_datetime(date_str).date()
                self.stock_files[dt] = f.resolve()  # Use absolute path for cache key match
            except Exception:
                continue
        logger.info(f"Indexed {len(self.stock_files)} stock data files")

    def _build_anchor_dates(self) -> List[Dict]:
        """Build valid anchor dates for this split.

        Anchor date D is the last day of stock data in the lookback window.
        Target = VIX close on D+1 (next trading day with VIX data).
        Anchors always start at 2005, but lookback reaches into pre-2005 data.
        """
        all_dates = sorted(self.stock_files.keys())
        vix_dates = set(self.vix_daily.keys())

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
            # Anchor must be >= 2005 and in split range
            if d < split_start:
                continue
            if split_end and d > split_end:
                continue

            # Find target date: next trading day with VIX data
            target_date = None
            for j in range(i + 1, min(i + 10, len(all_dates))):
                candidate = all_dates[j]
                if candidate in vix_dates:
                    target_date = candidate
                    break

            if target_date is None:
                # Also check dates not in stock files but in VIX
                from datetime import timedelta
                for offset in range(1, 10):
                    candidate = d + timedelta(days=offset)
                    if candidate in vix_dates:
                        target_date = candidate
                        break

            if target_date is None:
                continue

            # Lookback window: need lookback_days trading days before and including D
            lookback_start_idx = i - self.lookback_days + 1
            if lookback_start_idx < 0:
                continue

            window_dates = all_dates[lookback_start_idx:i + 1]
            if len(window_dates) < self.lookback_days:
                continue

            valid.append({
                'anchor_date': d,
                'window_dates': window_dates,
                'target_date': target_date,
                'target_vix': self.vix_daily[target_date],
            })

        return valid

    def _load_day_bars(self, file_path: Path, file_index: int = -1) -> Optional[np.ndarray]:
        """Load one day of bars as feature matrix.

        Args:
            file_path: Path to parquet file
            file_index: Position in file list (for cache prefetch/evict)

        Returns:
            [N_bars, num_features] or None if loading fails
        """
        key = str(file_path)
        
        # Get DataFrame from file cache (Polars)
        df = self._file_cache.get(file_path, file_index)
        if df is None:
            return None

        # Filter tickers (Polars)
        if self.allowed_tickers and 'ticker' in df.columns:
            df = df.filter(pl.col('ticker').is_in(self.allowed_tickers))
        if len(df) == 0:
            return None

        # Sort by timestamp (Polars)
        if 'bar_timestamp' in df.columns:
            df = df.sort('bar_timestamp')
        elif 'timestamp' in df.columns:
            df = df.sort('timestamp')

        # Extract available features (Polars → numpy)
        avail = [f for f in self.features if f in df.columns]
        if not avail:
            return None

        features = df.select(avail).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad missing features with zeros if some columns don't exist
        if len(avail) < self.num_features:
            padded = np.zeros((len(features), self.num_features), dtype=np.float32)
            padded[:, :len(avail)] = features
            features = padded

        # Cap bars per day
        if len(features) > self.max_bars_per_day:
            features = features[:self.max_bars_per_day]
        
        # Don't cache numpy arrays - they duplicate memory from DataFrame cache
        # The DataFrame cache already handles size limits
        return features

    def _normalize_bars(self, bars: np.ndarray) -> np.ndarray:
        """Z-score normalize bars across the full sequence."""
        mu = bars.mean(axis=0, keepdims=True)
        std = bars.std(axis=0, keepdims=True) + 1e-8
        return (bars - mu) / std

    def _normalize_vix(self, vix: float) -> float:
        """Normalize VIX target."""
        if self.vix_normalize == 'zscore':
            return (vix - self.vix_mean) / self.vix_std
        elif self.vix_normalize == 'log':
            return np.log(max(vix, 0.01))
        return vix

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over samples with async file caching."""
        worker_info = torch.utils.data.get_worker_info()
        all_indices = list(range(len(self.anchor_dates)))
        
        if worker_info is not None:
            per_worker = len(all_indices) // worker_info.num_workers
            wid = worker_info.id
            start = wid * per_worker
            end = start + per_worker if wid < worker_info.num_workers - 1 else len(all_indices)
            all_indices = all_indices[start:end]

        # Build ordered file list
        all_files = sorted(self.stock_files.values())
        file_to_idx = {str(f): i for i, f in enumerate(all_files)}
        
        # Use chunked cache or full shuffle
        if self.use_chunked_cache:
            # Chunked cache mode: load files in chunks, shuffle within chunks
            # Each file is ~600MB, so 60GB cache fits ~100 files
            # Each sample needs lookback_days files (e.g. 15)
            # With sorted samples, adjacent samples share most files
            # So samples_per_chunk = files_per_cache (files overlap heavily)
            files_per_cache = int(self.cache_gb * 1024 / 600)
            samples_per_cache = max(10, files_per_cache)
            logger.info(f"Chunked cache: {samples_per_cache} samples per chunk (~{files_per_cache} files max)")
            chunk_iter = range(0, len(all_indices), samples_per_cache)
        else:
            # Full shuffle mode: shuffle all samples, use async prefetch
            if self.split == 'train':
                np.random.shuffle(all_indices)
            self._file_cache.set_file_list(all_files)
            chunk_iter = [0]  # Single chunk = all samples
            samples_per_cache = len(all_indices)
        
        # Process in chunks
        for chunk_start in chunk_iter:
            chunk_end = min(chunk_start + samples_per_cache, len(all_indices))
            chunk_indices = all_indices[chunk_start:chunk_end]
            
            # Get files needed for this chunk
            chunk_files = set()
            for idx in chunk_indices:
                sample = self.anchor_dates[idx]
                for d in sample['window_dates']:
                    f = self.stock_files.get(d)
                    if f:
                        chunk_files.add(f)
            
            chunk_files = sorted(chunk_files)
            self._file_cache.set_file_list(chunk_files)
            
            # Warm cache for this chunk (only if using chunked cache)
            if self.use_chunked_cache:
                from tqdm import tqdm
                logger.info(f"Loading chunk {chunk_start//samples_per_cache + 1}: {len(chunk_files)} files")
                
                # Load files sequentially with strict size check
                loaded = 0
                for f in tqdm(chunk_files, desc="Caching files", leave=False):
                    # Check size BEFORE loading
                    if self._file_cache._current_bytes >= self._file_cache.max_bytes * 0.95:
                        logger.info(f"Cache full at {self._file_cache._current_bytes / 1e9:.1f}GB after {loaded} files")
                        break
                    
                    # Use get() which has built-in size enforcement
                    df = self._file_cache.get(f)
                    if df is not None:
                        loaded += 1
                
                logger.info(f"Loaded {loaded}/{len(chunk_files)} files into cache ({self._file_cache._current_bytes / 1e9:.1f}GB)")
            
            # Shuffle samples within this chunk (only if using chunked cache)
            if self.split == 'train' and self.use_chunked_cache:
                np.random.shuffle(chunk_indices)
            
            samples_yielded = 0
            for idx in chunk_indices:
                sample = self.anchor_dates[idx]
                window_dates = sample['window_dates']
                target_vix = sample['target_vix']

                # Load bars for all lookback days
                all_bars = []
                for d in window_dates:
                    file_path = self.stock_files.get(d)
                    if file_path is None:
                        continue
                    day_bars = self._load_day_bars(file_path, -1)
                    if day_bars is not None:
                        all_bars.append(day_bars)

                if not all_bars:
                    continue

                # Concatenate all days into one sequence
                bars = np.concatenate(all_bars, axis=0)

                # Cap total sequence length
                if len(bars) > self.max_total_bars:
                    bars = bars[-self.max_total_bars:]

                # Normalize
                bars = self._normalize_bars(bars)

                # Normalize VIX target
                target_norm = self._normalize_vix(target_vix)

                samples_yielded += 1
                yield {
                    'bars': torch.from_numpy(bars),
                    'vix_target': torch.tensor(target_norm, dtype=torch.float32),
                    'num_bars': len(bars),
                    'anchor_date': str(sample['anchor_date']),
                }
            
            # Clear cache after chunk (only if using chunked cache)
            if self.use_chunked_cache:
                self._file_cache.clear()
        
        # Log final stats
        stats = self._file_cache.get_stats()
        logger.info(
            f"Cache [{self.split}]: hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1%}"
        )

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
        targets = torch.zeros(B)

        for i, b in enumerate(batch):
            T = b['num_bars']
            bars_padded[i, :T, :] = b['bars']
            bar_mask[i, :T] = 1.0
            targets[i] = b['vix_target']

        return {
            'bars': bars_padded,
            'bar_mask': bar_mask,
            'vix_target': targets,
            'num_bars': [b['num_bars'] for b in batch],
        }
