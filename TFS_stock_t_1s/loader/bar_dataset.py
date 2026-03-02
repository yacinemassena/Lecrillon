"""
Bar-Based Dataset for 1-Second Stock Data.

Unlike tick data which requires chunking, bar data is already aggregated.
Each bar becomes a single input unit with its pre-computed features.

Architecture:
- Each 1s bar has 31 features (OHLCV + microstructure metrics)
- Bars are grouped into 5-minute frames
- Each frame aggregates ~300 bars (5 min × 60 sec)
- Model learns patterns from bar sequences, not tick sequences
"""

import threading
import queue
import logging
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


# Features to use from 1s bar data (selecting most informative)
BAR_FEATURES = [
    'close', 'volume', 'trade_count',
    'price_std', 'price_range_pct', 'vwap',
    'avg_trade_size', 'amihud', 'buy_volume', 'sell_volume',
    'tick_arrival_rate', 'large_trade_ratio', 'tick_burst',
    'rv_intrabar', 'ofi'
]


@dataclass
class BarBatch:
    """
    A batch of bar frames with sample boundary tracking.
    
    Attributes:
        bars: [N_frames, bars_per_frame, num_features] - Bar features per frame
        ticker_ids: [N_frames, bars_per_frame] - Ticker IDs (optional)
        frame_mask: [N_frames, bars_per_frame] - Valid bar mask
        sample_boundaries: List of (start_frame, end_frame, rv_target) tuples
        num_frames: Total frames in batch
        stream: Name of the stream
    """
    __slots__ = ['bars', 'ticker_ids', 'frame_mask', 'sample_boundaries', 
                 'num_frames', 'stream']
    
    def __init__(self, stream: str):
        self.bars = None
        self.ticker_ids = None
        self.frame_mask = None
        self.sample_boundaries = []
        self.num_frames = 0
        self.stream = stream
    
    def to_device(self, device: torch.device) -> 'BarBatch':
        """Move tensors to device."""
        self.bars = self.bars.to(device)
        if self.ticker_ids is not None:
            self.ticker_ids = self.ticker_ids.to(device)
        self.frame_mask = self.frame_mask.to(device)
        return self


class BarDataset(IterableDataset):
    """
    Dataset for 1-second bar data with pre-computed features.
    
    Unlike tick data:
    - No chunking needed (bars are already 1s units)
    - Use pre-computed features directly
    - Group bars into 5-minute frames
    """
    
    def __init__(
        self,
        data_path: str,
        rv_file: str,
        split: str = 'train',
        frame_interval: str = '5min',
        max_bars_per_frame: int = 500,  # Max bars in one 5-min frame
        max_frames_per_batch: int = 100,
        prefetch_files: int = 8,
        rv_horizon_days: int = 1,
        train_end: str = '2023-12-31',
        val_end: str = '2024-12-31',
        filter_tickers: bool = True,
        allowed_tickers_file: Optional[str] = None,
        features: List[str] = None,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.frame_interval = frame_interval
        self.max_bars_per_frame = max_bars_per_frame
        self.max_frames_per_batch = max_frames_per_batch
        self.prefetch_files = prefetch_files
        self.rv_horizon_days = rv_horizon_days
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.filter_tickers = filter_tickers
        self.features = features or BAR_FEATURES
        self.num_features = len(self.features)
        
        # Load ticker filter and create ticker-to-ID mapping
        self.allowed_tickers: Set[str] = set()
        self.ticker_to_id: Dict[str, int] = {}
        if filter_tickers and allowed_tickers_file and Path(allowed_tickers_file).exists():
            self._load_ticker_filter(allowed_tickers_file)
        
        self.num_tickers = len(self.ticker_to_id) if self.ticker_to_id else 0
        
        # Find data files
        self.files = self._find_files()
        logger.info(f"BarDataset [{split}]: {len(self.files)} files, {self.num_features} features")
        
        # Load precomputed RV
        self._forward_rv_lookup = {}
        if rv_file and Path(rv_file).exists():
            self._load_rv(rv_file)
        
        # Prefetch state
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()
    
    def _load_ticker_filter(self, file_path: str):
        """Load allowed tickers and create ticker-to-ID mapping."""
        tickers = []
        with open(file_path, 'r') as f:
            for line in f:
                ticker = line.strip()
                if ticker:
                    tickers.append(ticker)
        
        self.allowed_tickers = set(tickers)
        self.ticker_to_id = {ticker: idx for idx, ticker in enumerate(sorted(tickers))}
        logger.info(f"Loaded {len(self.allowed_tickers)} tickers with ID mapping")
    
    def _find_files(self) -> List[Path]:
        """Find data files for this stream."""
        files_with_dates = []
        
        for f in self.data_path.glob('*.parquet'):
            if not f.name.startswith('._'):
                date_str = f.stem.split('.')[0]
                try:
                    dt = pd.to_datetime(date_str)
                    files_with_dates.append((dt, f))
                except:
                    continue
        
        files_with_dates.sort(key=lambda x: x[0])
        
        if self.split == 'train':
            return [f for dt, f in files_with_dates if dt <= self.train_end]
        elif self.split == 'val':
            return [f for dt, f in files_with_dates 
                    if dt > self.train_end and dt <= self.val_end]
        elif self.split == 'test':
            return [f for dt, f in files_with_dates if dt > self.val_end]
        return [f for _, f in files_with_dates]
    
    def _load_rv(self, rv_file: str):
        """Load precomputed forward RV."""
        rv_df = pd.read_parquet(rv_file)
        rv_df['date'] = pd.to_datetime(rv_df['date']).dt.date
        
        rv_col = f'rv_{self.rv_horizon_days}d_forward'
        if rv_col not in rv_df.columns:
            rv_cols = [c for c in rv_df.columns if 'forward' in c]
            if rv_cols:
                rv_col = rv_cols[0]
                logger.warning(f"Using {rv_col} instead of rv_{self.rv_horizon_days}d_forward")
            else:
                logger.warning("No forward RV column found")
                return
        
        for _, row in rv_df.iterrows():
            if pd.notna(row[rv_col]):
                self._forward_rv_lookup[row['date']] = float(row[rv_col])
        
        logger.info(f"Loaded {len(self._forward_rv_lookup)} RV targets")
    
    def _start_prefetch(self):
        """Start background prefetch thread."""
        self._prefetch_queue = queue.Queue(maxsize=self.prefetch_files)
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self._prefetch_thread.start()
    
    def _stop_prefetch_thread(self):
        """Stop background prefetch thread."""
        self._stop_prefetch.set()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
    
    def _prefetch_worker(self):
        """Background worker that loads and filters files."""
        for file_path in self.files:
            if self._stop_prefetch.is_set():
                break
            
            try:
                df = pd.read_parquet(file_path)
                
                # Apply ticker filter
                if self.filter_tickers and self.allowed_tickers:
                    df = df[df['ticker'].isin(self.allowed_tickers)]
                
                if len(df) > 0:
                    date_str = file_path.stem.split('.')[0]
                    self._prefetch_queue.put((date_str, df))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue
        
        self._prefetch_queue.put(None)
    
    def _process_file_to_frames(self, df: pd.DataFrame, date_str: str) -> Iterator[Dict]:
        """Convert file data to 5-minute frames of bar data."""
        # Ensure required columns exist
        required_cols = ['ticker', 'bar_timestamp'] + [f for f in self.features if f in df.columns]
        missing = [c for c in ['ticker', 'bar_timestamp'] if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing} in {date_str}")
            return
        
        # Parse timestamp
        df['_ts'] = pd.to_datetime(df['bar_timestamp'])
        df = df.sort_values('_ts')
        
        # Group into frames
        df['_frame'] = df['_ts'].dt.floor(self.frame_interval)
        
        for frame_ts, frame_df in df.groupby('_frame'):
            if len(frame_df) == 0:
                continue
            
            # Extract features (use available ones)
            available_features = [f for f in self.features if f in frame_df.columns]
            if not available_features:
                continue
            
            # Get feature matrix
            features = frame_df[available_features].values.astype(np.float32)
            
            # Handle NaN and inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize features per frame (z-score)
            feat_mean = features.mean(axis=0, keepdims=True)
            feat_std = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - feat_mean) / feat_std
            
            # Get ticker IDs
            ticker_ids = None
            if self.ticker_to_id:
                ticker_ids = np.array([
                    self.ticker_to_id.get(t, 0) for t in frame_df['ticker'].values
                ], dtype=np.int64)
            
            yield {
                'features': features,
                'ticker_ids': ticker_ids,
                'n_bars': len(frame_df),
                'frame_ts': frame_ts,
                'date': date_str,
            }
    
    def __iter__(self) -> Iterator[BarBatch]:
        """Iterate over batches."""
        self._start_prefetch()
        
        try:
            yield from self._generate_batches()
        finally:
            self._stop_prefetch_thread()
    
    def _generate_batches(self) -> Iterator[BarBatch]:
        """Generate batches from prefetched files."""
        all_frames = []
        all_ticker_ids = []
        all_masks = []
        sample_boundaries = []
        
        current_frame_id = 0
        current_date = None
        date_start_frame = 0
        
        while True:
            item = self._prefetch_queue.get()
            if item is None:
                break
            
            date_str, df = item
            
            # Check for date change → new sample
            if current_date is not None and date_str != current_date:
                rv = self._forward_rv_lookup.get(
                    pd.to_datetime(current_date).date(), 
                    np.nan
                )
                if not np.isnan(rv):
                    sample_boundaries.append((date_start_frame, current_frame_id, rv))
                
                date_start_frame = current_frame_id
            
            current_date = date_str
            
            # Process file into frames
            for frame_data in self._process_file_to_frames(df, date_str):
                n_bars = frame_data['n_bars']
                features = frame_data['features']
                ticker_ids = frame_data.get('ticker_ids')
                
                # Pad or truncate to max_bars_per_frame
                if n_bars > self.max_bars_per_frame:
                    features = features[:self.max_bars_per_frame]
                    if ticker_ids is not None:
                        ticker_ids = ticker_ids[:self.max_bars_per_frame]
                    n_bars = self.max_bars_per_frame
                
                # Create padded arrays
                padded_features = np.zeros((self.max_bars_per_frame, self.num_features), dtype=np.float32)
                padded_features[:n_bars] = features
                
                mask = np.zeros(self.max_bars_per_frame, dtype=np.float32)
                mask[:n_bars] = 1.0
                
                all_frames.append(padded_features)
                all_masks.append(mask)
                
                if ticker_ids is not None:
                    padded_ticker_ids = np.zeros(self.max_bars_per_frame, dtype=np.int64)
                    padded_ticker_ids[:n_bars] = ticker_ids
                    all_ticker_ids.append(padded_ticker_ids)
                
                current_frame_id += 1
                
                # Check if batch is full
                if len(all_frames) >= self.max_frames_per_batch:
                    if current_frame_id > date_start_frame:
                        rv = self._forward_rv_lookup.get(
                            pd.to_datetime(current_date).date(),
                            np.nan
                        )
                        if not np.isnan(rv):
                            sample_boundaries.append((date_start_frame, current_frame_id, rv))
                    
                    yield self._create_batch(
                        all_frames, all_ticker_ids, all_masks, sample_boundaries
                    )
                    
                    all_frames = []
                    all_ticker_ids = []
                    all_masks = []
                    sample_boundaries = []
                    current_frame_id = 0
                    date_start_frame = 0
        
        # Final batch
        if all_frames:
            if current_date is not None and current_frame_id > date_start_frame:
                rv = self._forward_rv_lookup.get(
                    pd.to_datetime(current_date).date(),
                    np.nan
                )
                if not np.isnan(rv):
                    sample_boundaries.append((date_start_frame, current_frame_id, rv))
            
            yield self._create_batch(
                all_frames, all_ticker_ids, all_masks, sample_boundaries
            )
    
    def _create_batch(
        self,
        frames: List[np.ndarray],
        ticker_ids: List[np.ndarray],
        masks: List[np.ndarray],
        sample_boundaries: List[Tuple[int, int, float]],
    ) -> BarBatch:
        """Create a BarBatch from accumulated data."""
        batch = BarBatch('stock_1s')
        
        batch.bars = torch.from_numpy(np.stack(frames))
        batch.frame_mask = torch.from_numpy(np.stack(masks))
        
        if ticker_ids:
            batch.ticker_ids = torch.from_numpy(np.stack(ticker_ids))
        
        batch.sample_boundaries = sample_boundaries
        batch.num_frames = len(frames)
        
        return batch
