"""
VIX Tick-based Dataset with Chunking and Scalars.
Yields packed tensors for efficient GPU processing.

Features async RAM caching with smart prefetch/eviction.
"""

import os
import sys
import gzip
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple
from torch.utils.data import IterableDataset, get_worker_info
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import time


class AsyncFileCache:
    """
    Async file cache with background prefetching and smart eviction.
    
    - Prefetches upcoming files in background threads
    - Evicts old files when moving forward through dataset
    - Never blocks training (async I/O)
    """
    
    def __init__(self, max_gb: float = 100.0, prefetch_count: int = 10, 
                 evict_behind: int = 2, num_threads: int = 4):
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
        
        self.logger = logging.getLogger(__name__)
        self._stats = {'hits': 0, 'misses': 0, 'prefetches': 0, 'evictions': 0}
    
    def set_file_list(self, files: List[Path]):
        """Set the ordered list of files for prefetching."""
        with self._lock:
            self._file_list = files
            self._current_index = 0
    
    def _estimate_df_size(self, df: pd.DataFrame) -> int:
        """Estimate DataFrame memory usage in bytes."""
        return df.memory_usage(deep=True).sum()
    
    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a file from disk (called in background thread)."""
        try:
            if file_path.suffix.endswith('gz'):
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            return df
        except Exception as e:
            self.logger.warning(f"Cache: Failed to load {file_path.name}: {e}")
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
            
        # Prefetch next N files
        for i in range(1, self.prefetch_count + 1):
            next_idx = current_idx + i
            if next_idx < len(self._file_list):
                self._executor.submit(self._prefetch_file, self._file_list[next_idx])
    
    def _evict_old_files(self, current_idx: int):
        """Evict files that are far behind current position."""
        if not self._file_list or current_idx <= self.evict_behind:
            return
            
        # Files to keep: from (current - evict_behind) to end
        keep_start = max(0, current_idx - self.evict_behind)
        keep_keys = {str(self._file_list[i]) for i in range(keep_start, len(self._file_list))}
        
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k not in keep_keys]
            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                    evicted_size = self._cache_sizes.pop(key, 0)
                    self._current_bytes -= evicted_size
                    self._stats['evictions'] += 1
    
    def get(self, file_path: Path, file_index: int = -1) -> Optional[pd.DataFrame]:
        """
        Get a file from cache or load it.
        
        Args:
            file_path: Path to the file
            file_index: Current position in file list (for prefetch/evict)
            
        Returns:
            DataFrame or None if load failed
        """
        key = str(file_path)
        
        # Update current index for prefetch logic
        if file_index >= 0:
            self._current_index = file_index
            # Trigger async prefetch of upcoming files
            self._trigger_prefetch(file_index)
            # Evict old files we're done with
            self._evict_old_files(file_index)
        
        with self._lock:
            if key in self._cache:
                self._stats['hits'] += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
        
        # Cache miss - load synchronously
        self._stats['misses'] += 1
        df = self._load_file(file_path)
        
        if df is not None:
            with self._lock:
                size = self._estimate_df_size(df)
                
                # Evict if needed
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


# Global cache instance (shared across dataset instances in same process)
_global_cache: Optional[AsyncFileCache] = None
_cache_lock = threading.Lock()


def get_file_cache(config) -> Optional[AsyncFileCache]:
    """Get or create the global file cache."""
    global _global_cache
    
    if not config.dataset.enable_ram_cache:
        return None
        
    with _cache_lock:
        if _global_cache is None:
            _global_cache = AsyncFileCache(
                max_gb=config.dataset.ram_cache_gb,
                prefetch_count=config.dataset.prefetch_files,
                evict_behind=config.dataset.cache_evict_behind,
                num_threads=max(2, config.dataset.num_workers)
            )
            logging.getLogger(__name__).info(
                f"Initialized async file cache: {config.dataset.ram_cache_gb}GB, "
                f"prefetch={config.dataset.prefetch_files}, evict_behind={config.dataset.cache_evict_behind}"
            )
        return _global_cache


class VIXTickDataset(IterableDataset):
    """
    Streaming dataset for VIX tick data.
    
    Features:
    - 15s fixed-interval frames
    - No truncation: ticks split into chunks
    - Per-frame scalars: n_ticks, notional, vol
    - Async RAM caching with prefetch/eviction
    - Packed tensor output
    """
    
    # Class-level cache to avoid re-scanning files for train/val/test splits
    _source_cache = {}
    
    def __init__(self, config, split: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.split = split
        
        # Configuration from dataclass
        ds = config.dataset
        self.resample_interval = ds.resample_interval
        self.chunk_len = ds.chunk_len
        self.num_frames = ds.num_frames
        self.weight_mode = ds.weight_mode
        self.horizons = ds.prediction_horizons
        self.dim_in = config.model.dim_in
        
        # Data sources from config paths
        self.data_sources = self._parse_data_sources()
        self.targets = self._load_targets()

        # Normalization setup
        self.norm_enable = ds.norm_enable
        if self.norm_enable:
            self.norm_median = np.array(ds.norm_median, dtype=np.float32)
            self.norm_iqr = np.array(ds.norm_iqr, dtype=np.float32)
            # Avoid division by zero
            self.norm_iqr[self.norm_iqr == 0] = 1.0

        # Async RAM cache
        self._file_cache = get_file_cache(config)
        self._split_files: Optional[List[Path]] = None
        self._file_index_map: Dict[str, int] = {}
        
    def _parse_data_sources(self) -> Dict[str, List[Path]]:
        """Parse data source paths from config."""
        ds = self.config.dataset
        
        # Check cache first
        cache_key = (str(ds.indices_path), str(ds.vix_path))
        if cache_key in VIXTickDataset._source_cache:
            return VIXTickDataset._source_cache[cache_key]
            
        sources = {'indices': [], 'vix': []}
        extensions = ['*.csv', '*.csv.gz', '*.parquet']
        
        try:
            # Parse indices from config path
            indices_path = Path(ds.indices_path)
            self.logger.info(f"Looking for indices in: {indices_path}")
            if indices_path.exists():
                for ext in extensions:
                    sources['indices'].extend(sorted(indices_path.rglob(ext)))
                # Sort and deduplicate
                sources['indices'] = sorted(list(set(sources['indices'])))
                self.logger.info(f"Found {len(sources['indices'])} index files.")
            else:
                self.logger.warning(f"Indices path does not exist: {indices_path}")
            
            # Parse VIX from config path
            vix_path = Path(ds.vix_path)
            if vix_path.exists():
                for ext in extensions:
                    sources['vix'].extend(sorted(vix_path.rglob(ext)))
                sources['vix'] = sorted(list(set(sources['vix'])))
                self.logger.info(f"Found {len(sources['vix'])} VIX files.")
            else:
                self.logger.warning(f"VIX path does not exist: {vix_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to parse data sources: {e}")
            
        # Update cache
        VIXTickDataset._source_cache[cache_key] = sources
        return sources

    def _load_targets(self) -> Optional[pd.DataFrame]:
        """Load and prepare VIX target data."""
        vix_files = self.data_sources.get('vix', [])
        if not vix_files:
            return None
            
        dfs = []
        for fp in vix_files:
            try:
                if fp.suffix == '.parquet':
                    df = pd.read_parquet(fp)
                elif fp.suffix == '.csv' or fp.name.endswith('.csv.gz'):
                    df = pd.read_csv(fp)
                else:
                    continue
                
                # Normalize columns
                df.columns = df.columns.str.lower()
                
                # Prioritize sip_timestamp for VIX data
                if 'sip_timestamp' in df.columns:
                    # Use sip_timestamp as the main timestamp
                    df['timestamp'] = pd.to_datetime(df['sip_timestamp'])
                    # Remove sip_timestamp and other potentially conflicting timestamp columns
                    cols_to_drop = ['sip_timestamp']
                    for col in ['time', 'date', 'datetime']:
                        if col in df.columns:
                            cols_to_drop.append(col)
                    df.drop(columns=cols_to_drop, inplace=True)

                # Handle separate Date and Time columns if present
                if 'timestamp' not in df.columns and 'date' in df.columns and 'time' in df.columns:
                    try:
                        # Combine date and time
                        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
                        # Drop original columns to avoid confusion
                        df.drop(columns=['date', 'time'], inplace=True)
                    except Exception:
                        pass

                if 'time' in df.columns and 'timestamp' not in df.columns: 
                    df.rename(columns={'time': 'timestamp'}, inplace=True)
                if 'date' in df.columns and 'timestamp' not in df.columns: 
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                if 'datetime' in df.columns and 'timestamp' not in df.columns: 
                    df.rename(columns={'datetime': 'timestamp'}, inplace=True)
                
                # Remove duplicate columns if any (keep first)
                df = df.loc[:, ~df.columns.duplicated()]
                
                if 'timestamp' in df.columns:
                    dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to load VIX file {fp}: {e}")
                
        if not dfs: return None
        
        try:
            full_df = pd.concat(dfs).sort_values('timestamp')
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
            full_df.set_index('timestamp', inplace=True)
            
            # Keep only relevant columns
            target_col = next((c for c in ['close', 'vix', 'last', 'price'] if c in full_df.columns), None)
            if target_col:
                return full_df[[target_col]].rename(columns={target_col: 'target'})
        except Exception as e:
            self.logger.warning(f"Error processing VIX targets: {e}")
            
        return None

    def _get_chronological_files(self) -> List[Path]:
        """Get files sorted chronologically."""
        files_with_dates: List[Tuple[pd.Timestamp, Path]] = []
        for file_path in self.data_sources.get('indices', []):
            try:
                base = file_path.stem.split('.')[0]
                date = pd.to_datetime(base, errors='coerce')
                if pd.isna(date):
                    date = pd.to_datetime(file_path.parent.name, errors='coerce')
                if not pd.isna(date):
                    files_with_dates.append((date, file_path))
            except Exception:
                continue
        files_with_dates.sort(key=lambda x: x[0])
        return [fp for _, fp in files_with_dates]

    def _files_for_split(self) -> List[Path]:
        """Chronological split with date filtering from config."""
        files = self._get_chronological_files()
        if not files or self.split is None: 
            return files

        ds = self.config.dataset
        
        # Helper to check if file date is within range
        def in_range(f_path, start_str, end_str):
            if not start_str and not end_str: 
                return True
            try:
                base = f_path.stem.split('.')[0]
                f_date = pd.to_datetime(base, errors='coerce')
                if pd.isna(f_date):
                    f_date = pd.to_datetime(f_path.parent.name, errors='coerce')
                
                if pd.isna(f_date): 
                    return False
                
                start = pd.to_datetime(start_str) if start_str else pd.Timestamp.min
                end = pd.to_datetime(end_str) if end_str else pd.Timestamp.max
                return start <= f_date <= end
            except:
                return False

        try:
            if self.split == 'train':
                return [f for f in files if in_range(f, ds.train_start, ds.train_end)]
            elif self.split == 'val':
                return [f for f in files if in_range(f, ds.val_start, ds.val_end)]
            elif self.split == 'test':
                return [f for f in files if in_range(f, ds.test_start, ds.test_end)]
        except Exception as e:
            self.logger.warning(f"Date split failed ({e}), falling back to ratio split.")

        # Fallback to ratio split
        n = len(files)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        if self.split == 'train': 
            return files[:n_train]
        elif self.split == 'val': 
            return files[n_train:n_train + n_val]
        elif self.split == 'test': 
            return files[n_train + n_val:]
        return files

    def _shard_files(self, files: List[Path]) -> List[Path]:
        """Shard files across workers."""
        worker_info = get_worker_info()
        if worker_info is None: return files
        return files[worker_info.id::worker_info.num_workers]

    def _load_tick_data(self, file_path: Path, file_index: int = -1) -> pd.DataFrame:
        """
        Load tick data, using async cache if enabled.
        
        Args:
            file_path: Path to the file
            file_index: Position in file list (for cache prefetch/evict)
        """
        # Use cache if available
        if self._file_cache is not None:
            df = self._file_cache.get(file_path, file_index)
            if df is not None:
                return df.copy()  # Return copy to avoid mutation
            return pd.DataFrame()
        
        # Fallback: direct load
        try:
            if file_path.suffix.endswith('gz'):
                with gzip.open(file_path, 'rt') as f: df = pd.read_csv(f)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            return df
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path.name}: {e}")
            return pd.DataFrame()

    def _process_frame_ticks(self, group: pd.DataFrame, ticker_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Process ticks for a single frame.
        Returns:
            chunks: [K, chunk_len, F]
            weights: [K]
            scalars: [3] (n_ticks, notional, vol)
            rv: float (target RV for this frame)
        """
        # 1. Extract features
        # Price
        price_col = next((c for c in ['price', 'close', 'last'] if c in group.columns), None)
        if price_col:
            prices = group[price_col].values.astype(np.float32)
        else:
            prices = np.zeros(len(group), dtype=np.float32)

        # Size/Volume
        if 'size' in group.columns: 
            sizes = group['size'].values.astype(np.float32)
        elif 'volume' in group.columns: 
            sizes = group['volume'].values.astype(np.float32)
        else:
            sizes = np.zeros(len(group), dtype=np.float32)
            
        # Time Delta (dt)
        # Assuming group is sorted by time
        if 'timestamp' in group.columns:
            # diff in seconds
            ts = group['timestamp']
            dt = ts.diff().dt.total_seconds().fillna(0.0).values.astype(np.float32)
        else:
            dt = np.zeros(len(group), dtype=np.float32)
            
        # Combine features: [price, size, dt]
        # Ensure config.model.tcn.dim_in matches (should be 3)
        ticks = np.stack([prices, sizes, dt], axis=1) # [N, 3]

        n_ticks, n_feats = ticks.shape
        
        # Enforce dim_in features (pad with zeros if needed)
        target_dim = self.config.model.tcn.dim_in
        if n_feats < target_dim:
            padding = np.zeros((n_ticks, target_dim - n_feats), dtype=np.float32)
            ticks = np.hstack([ticks, padding])
            n_feats = target_dim
        elif n_feats > target_dim:
             ticks = ticks[:, :target_dim]
             n_feats = target_dim
        
        # 2. Compute Scalars
        n_ticks_scalar = float(n_ticks)
        
        # Notional
        notional = np.sum(prices * sizes)
            
        # Vol proxy = std dev of price (intra-frame)
        if n_ticks > 1:
            vol = np.std(prices)
        else:
            vol = 0.0
            
        scalars = np.array([n_ticks_scalar, notional, vol], dtype=np.float32)

        if getattr(self, 'norm_enable', False):
            # 1. Log Transform: x' = log(1 + x)
            # Apply to price and size (cols 0, 1), but maybe not dt (col 2) if it's small?
            # User request: "log(tick_count)" -> handled in scalars
            # For features: log1p is good for price/size. 
            # dt can be 0.001s or 10s. log1p is fine.
            ticks = np.log1p(np.maximum(ticks, 0))
            
            # 2. Robust Scaling
            # We have norms for first 2 dims in config?
            # Config has norm_median, norm_iqr.
            # We should probably extend them or just apply to first 2.
            n_norm = min(ticks.shape[1], len(self.norm_median))
            if n_norm > 0:
                ticks[:, :n_norm] = (ticks[:, :n_norm] - self.norm_median[:n_norm]) / self.norm_iqr[:n_norm]
        
        # 3. Chunking
        # Calculate number of chunks
        K = int(np.ceil(n_ticks / self.chunk_len))
        if K == 0: K = 1  # Handle empty frame
        
        # Pad to multiple of chunk_len
        pad_len = K * self.chunk_len - n_ticks
        if pad_len > 0:
            padding = np.zeros((pad_len, n_feats), dtype=np.float32)
            ticks_padded = np.vstack([ticks, padding])
        else:
            ticks_padded = ticks
            
        # Reshape to [K, chunk_len, F]
        chunks = ticks_padded.reshape(K, self.chunk_len, n_feats)
        
        # 4. Compute Weights
        if self.weight_mode == 'tick_count':
            chunk_weights = np.full(K, self.chunk_len, dtype=np.float32)
            if pad_len > 0:
                chunk_weights[-1] -= pad_len
        elif self.weight_mode == 'mean':
            chunk_weights = np.ones(K, dtype=np.float32)
        elif self.weight_mode == 'notional':
            # Re-extract prices/sizes from padded (un-log/un-norm if needed? No, weights are approximate)
            # Just using log-prices for weights is weird but okay for attention masks.
            # Ideally use simple count.
            chunk_weights = np.full(K, self.chunk_len, dtype=np.float32)
            if pad_len > 0: chunk_weights[-1] -= pad_len
        else:
            chunk_weights = np.full(K, self.chunk_len, dtype=np.float32)
        
        # Retrieve computed RV for this frame (passed in group metadata or computed?)
        # Since group is a slice of the main df, we might have added 'rv_target' col to it.
        rv_target = 0.0
        if 'rv_target' in group.columns:
            rv_target = group['rv_target'].iloc[0]
            
        return chunks, chunk_weights, scalars, rv_target

    def _compute_forward_rv(self, df: pd.DataFrame, horizon_min: int = 15) -> pd.DataFrame:
        """
        Compute Realized Volatility target for the next X minutes.
        RV = sqrt(sum(log_returns^2))
        """
        # Ensure sorted
        if not df.index.is_monotonic_increasing:
             df = df.sort_index()

        # Check for price column
        price_col = next((c for c in ['price', 'close', 'last'] if c in df.columns), None)
        if not price_col:
            df['rv_target'] = 0.0
            return df
            
        # Resample to 1-second bars to compute returns (reduce noise of micro-ticks)
        # or compute on tick-by-tick? Tick-by-tick is noisy.
        # Let's resample to 1s for RV calculation.
        try:
            # Forward looking window
            # We want for each timestamp t, the RV from t to t+15m.
            # 1. Compute 1s log returns
            resampled = df[price_col].resample('1s').last().ffill()
            log_ret = np.log(resampled / resampled.shift(1)).fillna(0)
            squared_ret = log_ret ** 2
            
            # 2. Rolling sum over horizon (looking forward)
            # rolling() is backward. So we use rolling(window) then shift(-window).
            seconds = horizon_min * 60
            rv_sq = squared_ret.rolling(window=seconds, min_periods=1).sum()
            rv_forward = np.sqrt(rv_sq).shift(-seconds) # Shift back to align t with start of window
            
            # 3. Map back to original dataframe
            # We need to broadcast the 1s RV to the ticks in that second.
            # reindex with method='nearest' or 'ffill'
            rv_series = rv_forward.reindex(df.index, method='ffill').fillna(0.0)
            
            df['rv_target'] = rv_series.values.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"RV computation failed: {e}")
            df['rv_target'] = 0.0
            
        return df

    def _generate_frames_from_file(self, file_path: Path, file_index: int = -1) -> Iterator[Tuple]:
        """
        Yields frames one by one from a file.
        Returns: (chunks, weights, scalars, bin_time, ticker_id, rv_target)
        
        Args:
            file_path: Path to the file
            file_index: Position in file list (for cache prefetch/evict)
        """
        df = self._load_tick_data(file_path, file_index)
        if df.empty: 
            return

        # Determine Ticker ID
        # Heuristic: Check filename against config tickers
        ticker_id = 0
        fname = file_path.stem.upper()
        # Look for known tickers in filename
        known_tickers = self.config.model.tickers.tickers
        current_ticker_name = None
        
        for i, t in enumerate(known_tickers):
            if t in fname or (t == 'VIX' and 'VIX' in str(file_path).upper()):
                 ticker_id = i
                 current_ticker_name = t
                 break
        
        # ... (Timestamp processing same as before) ...
        df.columns = df.columns.str.lower()
        if 'ts_ns' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_ns'], unit='ns', utc=True)
        elif 'sip_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['sip_timestamp'], errors='coerce')
        
        # Cleanup
        if 'timestamp' not in df.columns:
            for col in ['time', 't', 'datetime']:
                if col in df.columns: df['timestamp'] = df[col]; break
        
        if 'timestamp' not in df.columns: return

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df.set_index('timestamp', drop=False, inplace=True)
        
        # Compute RV Targets
        should_compute_rv = self.config.model.rv.predict_rv
        if should_compute_rv:
            # Filter by allowed tickers
            # If current_ticker_name is None (unknown ticker), we default to NOT computing RV
            if current_ticker_name is None or current_ticker_name not in self.config.model.tickers.predict_rv_tickers:
                should_compute_rv = False

        if should_compute_rv:
            df = self._compute_forward_rv(df, horizon_min=self.config.model.rv.horizon_min)
        else:
            df['rv_target'] = 0.0

        # Bin into 15s frames
        df['bin'] = df['timestamp'].dt.floor(self.resample_interval)
        
        frames_dict = {t: g for t, g in df.groupby('bin')}
        sorted_bins = sorted(frames_dict.keys())
        
        for bin_time in sorted_bins:
            group = frames_dict[bin_time]
            chunks, weights, scalars, rv = self._process_frame_ticks(group, ticker_id)
            yield (chunks, weights, scalars, bin_time, ticker_id, rv)

    def _pack_sample(self, buffer_list: List[Tuple]) -> Dict:
        """
        Packs a list of frames into a batch sample.
        """
        all_chunks = [x[0] for x in buffer_list]
        all_weights = [x[1] for x in buffer_list]
        all_scalars = [x[2] for x in buffer_list]
        all_rvs = [x[5] for x in buffer_list] # New
        # ticker_id is usually same for a file, but buffer might cross files if we supported that. 
        # Here we assume one stream per sample. 
        # But wait, Mamba input might need mixed streams? 
        # For now, let's assume one stream.
        ticker_ids = [x[4] for x in buffer_list]
        
        # Build frame_ptr
        frame_ptr = [0]
        current_idx = 0
        for chunks in all_chunks:
            current_idx += len(chunks)
            frame_ptr.append(current_idx)
            
        # Pack tensors
        packed_chunks = np.vstack(all_chunks)  # [SumK, chunk_len, F]
        packed_weights = np.concatenate(all_weights) # [SumK]
        packed_scalars = np.vstack(all_scalars) # [T, 3]
        packed_ptr = np.array(frame_ptr, dtype=np.int64) # [T+1]
        packed_rvs = np.array(all_rvs, dtype=np.float32) # [T]
        packed_ticker_ids = np.array(ticker_ids, dtype=np.int64) # [T]
        
        # Global Target (Sequence Level)
        target = np.full(len(self.horizons), np.nan, dtype=np.float32) 
        if self.targets is not None and not self.targets.empty:
             # (Existing target logic)
             try:
                current_time = buffer_list[-1][3]
                horizons = self.horizons
                for i, h in enumerate(horizons):
                    target_time = current_time + pd.Timedelta(days=h)
                    idx = self.targets.index.get_indexer([target_time], method='nearest')[0]
                    nearest_time = self.targets.index[idx]
                    if abs(nearest_time - target_time) < pd.Timedelta(days=1):
                        target[i] = float(self.targets.iloc[idx]['target'])
             except: pass
        
        target = np.nan_to_num(target, nan=0.0)

        return {
            'chunks': packed_chunks,
            'frame_ptr': packed_ptr,
            'weights': packed_weights,
            'frame_scalars': packed_scalars,
            'ticker_ids': packed_ticker_ids,
            'rv_targets': packed_rvs,
            'target': target
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collate multiple packed samples into a batch.
        Offsets frame_ptr and frame_id for batch-wide pooling.
        """
        batch_chunks = []
        batch_weights = []
        batch_scalars = []
        batch_targets = []
        batch_rv_targets = []
        batch_ticker_ids = []
        
        batch_frame_id = []
        total_frames = 0
        
        for sample in batch:
            chunks = torch.from_numpy(sample['chunks'])
            weights = torch.from_numpy(sample['weights'])
            scalars = torch.from_numpy(sample['frame_scalars'])
            target = torch.from_numpy(sample['target'])
            rv_targets = torch.from_numpy(sample['rv_targets'])
            ticker_ids = torch.from_numpy(sample['ticker_ids'])
            frame_ptr = sample['frame_ptr']
            
            batch_chunks.append(chunks)
            batch_weights.append(weights)
            batch_scalars.append(scalars)
            batch_targets.append(target)
            batch_rv_targets.append(rv_targets)
            batch_ticker_ids.append(ticker_ids)
            
            # frame_ptr -> frame_id
            n_chunks = len(chunks)
            frame_ids = torch.zeros(n_chunks, dtype=torch.long)
            for i in range(len(frame_ptr) - 1):
                start = frame_ptr[i]
                end = frame_ptr[i+1]
                frame_ids[start:end] = i + total_frames
            
            batch_frame_id.append(frame_ids)
            total_frames += (len(frame_ptr) - 1)
            
        return {
            'chunks': torch.cat(batch_chunks, dim=0),
            'frame_id': torch.cat(batch_frame_id, dim=0),
            'weights': torch.cat(batch_weights, dim=0),
            'frame_scalars': torch.cat(batch_scalars, dim=0),
            'ticker_ids': torch.cat(batch_ticker_ids, dim=0),
            'rv_targets': torch.cat(batch_rv_targets, dim=0), # [TotalFrames]
            'target': torch.stack(batch_targets),
            'num_frames': total_frames
        }

    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate over the dataset, yielding packed samples.
        
        Uses async RAM cache with prefetch/eviction for fast I/O.
        """
        # Get files for this split
        files = self._files_for_split()
        if not files:
            self.logger.warning(f"No files found for split '{self.split}'")
            return
        
        # Shard across workers
        files = self._shard_files(files)
        if not files:
            return
            
        # Initialize cache with file list for prefetching
        if self._file_cache is not None:
            self._file_cache.set_file_list(files)
            # Log cache stats periodically
            self.logger.info(f"Cache initialized with {len(files)} files for split '{self.split}'")
        
        # Build file index map for cache prefetch
        self._file_index_map = {str(f): i for i, f in enumerate(files)}
        
        # Stream through files
        buffer = []
        
        for file_idx, file_path in enumerate(files):
            # Generate frames from this file (with file index for cache)
            for frame_data in self._generate_frames_from_file(file_path, file_idx):
                buffer.append(frame_data)
                
                # When buffer reaches num_frames, yield a sample
                if len(buffer) >= self.num_frames:
                    yield self._pack_sample(buffer[:self.num_frames])
                    # Sliding window: keep last half for overlap
                    buffer = buffer[self.num_frames // 2:]
        
        # Yield remaining if any
        if len(buffer) >= self.num_frames // 2:
            yield self._pack_sample(buffer)
        
        # Log cache stats at end
        if self._file_cache is not None:
            stats = self._file_cache.get_stats()
            self.logger.info(
                f"Cache stats: hits={stats['hits']}, misses={stats['misses']}, "
                f"hit_rate={stats['hit_rate']:.2%}, cached={stats['cached_files']} files, "
                f"size={stats['cache_mb']:.1f}MB"
            )
