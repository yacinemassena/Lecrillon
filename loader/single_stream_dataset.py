"""
Single-Stream Dataset for Per-Stream TCN Pretraining.

Loads one data stream at a time (stocks, options, or index) with:
- Background thread prefetching files to RAM
- On-the-fly ticker filtering for stocks (O(1) set lookup)
- Chunk-level batching for strict VRAM control
- Sample boundary tracking for loss computation

Usage:
    dataset = SingleStreamDataset(
        stream='stocks',  # or 'options' or 'index'
        data_path='path/to/stream/data',
        ...
    )
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


@dataclass
class SingleStreamBatch:
    """
    A batch of chunks from a single stream with sample boundary tracking.
    
    Attributes:
        chunks: [N_chunks, chunk_len, dim_in] - All chunks in batch
        weights: [N_chunks] - Weight per chunk (tick count)
        frame_id: [N_chunks] - Which frame each chunk belongs to
        frame_scalars: [N_frames, num_scalars] - Per-frame scalars
        sample_boundaries: List of (start_frame, end_frame, rv_target) tuples
        num_frames: Total frames in batch
        num_chunks: Total chunks in batch
        stream: Name of the stream ('stocks', 'options', 'index')
    """
    __slots__ = ['chunks', 'weights', 'frame_id', 'frame_scalars',
                 'sample_boundaries', 'num_frames', 'num_chunks', 'stream']
    
    def __init__(self, stream: str):
        self.chunks = None
        self.weights = None
        self.frame_id = None
        self.frame_scalars = None
        self.sample_boundaries = []
        self.num_frames = 0
        self.num_chunks = 0
        self.stream = stream
    
    def to_device(self, device: torch.device) -> 'SingleStreamBatch':
        """Move tensors to device."""
        self.chunks = self.chunks.to(device)
        self.weights = self.weights.to(device)
        self.frame_id = self.frame_id.to(device)
        self.frame_scalars = self.frame_scalars.to(device)
        return self
    
    def get_rv_targets(self, device: torch.device) -> torch.Tensor:
        """Get RV targets for each sample in batch."""
        targets = [sb[2] for sb in self.sample_boundaries]
        return torch.tensor(targets, dtype=torch.float32, device=device)


# Stream-specific column mappings
STREAM_COLUMNS = {
    'stocks': {
        'price': 'price',
        'size': 'size', 
        'time': 'sip_timestamp',
        'ticker': 'ticker',
    },
    'options': {
        'price': 'price',
        'size': 'size',
        'time': 'sip_timestamp',
        'ticker': 'ticker',
    },
    'index': {
        'price': 'price',  # Index data only has price, no volume
        'size': None,       # No size column in index data
        'time': 'timestamp',
        'ticker': 'ticker',
    },
}


class SingleStreamDataset(IterableDataset):
    """
    Single-stream dataset with background prefetching and on-the-fly filtering.
    
    Memory model:
    - RAM: prefetch_files files in background queue
    - VRAM: Only current batch of chunks (controlled by max_chunks_per_batch)
    
    Filtering:
    - Stock filtering uses precomputed top-N list (O(1) set lookup)
    - Filtering happens in background thread while GPU trains
    """
    
    def __init__(
        self,
        stream: str,  # 'stocks', 'options', 'index'
        data_path: str,
        rv_file: str,
        split: str = 'train',
        frame_interval: str = '10s',
        chunk_len: int = 256,
        dim_in: int = 3,
        max_chunks_per_batch: int = 1500,   # Reduced from 4000 - RTX 5080 OOM with TCN-12
        prefetch_files: int = 16,
        rv_horizon_days: int = 30,
        train_end: str = '2022-12-31',
        val_end: str = '2023-06-30',
        filter_tickers: bool = False,
        allowed_tickers_file: Optional[str] = None,
    ):
        assert stream in ('stocks', 'options', 'index'), f"Unknown stream: {stream}"
        
        self.stream = stream
        self.data_path = Path(data_path)
        self.split = split
        self.frame_interval = frame_interval
        self.chunk_len = chunk_len
        self.dim_in = dim_in
        self.max_chunks_per_batch = max_chunks_per_batch
        self.prefetch_files = prefetch_files
        self.rv_horizon_days = rv_horizon_days
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.filter_tickers = filter_tickers
        
        # Column mapping for this stream
        self.columns = STREAM_COLUMNS[stream]
        
        # Load ticker filter (O(1) lookup)
        self.allowed_tickers: Set[str] = set()
        if filter_tickers and allowed_tickers_file and Path(allowed_tickers_file).exists():
            self._load_ticker_filter(allowed_tickers_file)
        
        # Find data files
        self.files = self._find_files()
        logger.info(f"SingleStreamDataset [{stream}/{split}]: {len(self.files)} files")
        
        # Load precomputed RV
        self._forward_rv_lookup = {}
        if rv_file and Path(rv_file).exists():
            self._load_rv(rv_file)
        
        # Prefetch state
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()
    
    def _load_ticker_filter(self, file_path: str):
        """Load allowed tickers from precomputed file."""
        with open(file_path, 'r') as f:
            self.allowed_tickers = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(self.allowed_tickers)} tickers for filtering")
    
    def _find_files(self) -> List[Path]:
        """Find data files for this stream."""
        files_with_dates = []
        
        if self.stream == 'index':
            # Index has year subdirectories
            for year_dir in self.data_path.iterdir():
                if year_dir.is_dir():
                    for f in year_dir.glob('*.parquet'):
                        if not f.name.startswith('._'):
                            date_str = f.stem.split('.')[0]
                            try:
                                dt = pd.to_datetime(date_str)
                                files_with_dates.append((dt, f))
                            except:
                                continue
        else:
            # Stocks and options are flat directories
            for f in self.data_path.glob('*.parquet'):
                if not f.name.startswith('._'):
                    date_str = f.stem.split('.')[0]
                    try:
                        dt = pd.to_datetime(date_str)
                        files_with_dates.append((dt, f))
                    except:
                        continue
        
        # Sort by date
        files_with_dates.sort(key=lambda x: x[0])
        
        # Filter by split
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
                
                # Apply ticker filter if enabled
                ticker_col = self.columns.get('ticker')
                if self.filter_tickers and self.allowed_tickers and ticker_col in df.columns:
                    df = df[df[ticker_col].isin(self.allowed_tickers)]
                
                if len(df) > 0:
                    # Extract date from filename
                    date_str = file_path.stem.split('.')[0]
                    self._prefetch_queue.put((date_str, df))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue
        
        # Signal end of data
        self._prefetch_queue.put(None)
    
    def _process_file_to_frames(
        self,
        df: pd.DataFrame,
        date_str: str,
    ) -> Iterator[Dict]:
        """Convert file data to 10-second frames."""
        price_col = self.columns['price']
        size_col = self.columns['size']
        time_col = self.columns['time']
        
        # Ensure required columns exist (size_col can be None for index data)
        required = [price_col, time_col]
        if size_col is not None:
            required.append(size_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing} in {date_str}")
            return
        
        # Convert time column
        if df[time_col].dtype == 'int64':
            # Nanosecond timestamps
            df['_ts'] = pd.to_datetime(df[time_col], unit='ns')
        else:
            df['_ts'] = pd.to_datetime(df[time_col])
        
        # Sort by time
        df = df.sort_values('_ts')
        
        # Group into frames
        df['_frame'] = df['_ts'].dt.floor(self.frame_interval)
        
        for frame_ts, frame_df in df.groupby('_frame'):
            if len(frame_df) == 0:
                continue
            
            # Extract features: [price, size, dt]
            prices = frame_df[price_col].values.astype(np.float32)
            
            # For index data (no size column), use price changes as proxy
            if size_col is not None and size_col in frame_df.columns:
                sizes = frame_df[size_col].values.astype(np.float32)
            else:
                # Use price differences as size proxy for index data
                sizes = np.abs(np.diff(prices, prepend=prices[0]))
            features = np.stack([prices, sizes, dt], axis=1)  # [N_ticks, 3]
            
            # Scalars: n_ticks, total_notional, total_volume
            n_ticks = len(frame_df)
            if size_col is not None and size_col in frame_df.columns:
                total_notional = (frame_df[price_col] * frame_df[size_col]).sum()
                total_volume = frame_df[size_col].sum()
            else:
                # For index data, use price variance as proxy
                total_notional = prices.var() * n_ticks
                total_volume = np.abs(sizes).sum()
            
            yield {
                'features': features,
                'n_ticks': n_ticks,
                'scalars': np.array([
                    np.log1p(n_ticks),
                    np.log1p(total_notional),
                    np.log1p(total_volume),
                ], dtype=np.float32),
                'frame_ts': frame_ts,
                'date': date_str,
            }
    
    def _chunk_frame(self, features: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """Split frame features into fixed-size chunks with tick counts."""
        chunks = []
        n_ticks = len(features)
        
        for i in range(0, n_ticks, self.chunk_len):
            chunk = features[i:i + self.chunk_len]
            tick_count = len(chunk)
            
            # Pad if needed
            if len(chunk) < self.chunk_len:
                pad = np.zeros((self.chunk_len - len(chunk), self.dim_in), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            
            chunks.append((chunk, tick_count))
        
        return chunks
    
    def __iter__(self) -> Iterator[SingleStreamBatch]:
        """Iterate over batches."""
        self._start_prefetch()
        
        try:
            yield from self._generate_batches()
        finally:
            self._stop_prefetch_thread()
    
    def _generate_batches(self) -> Iterator[SingleStreamBatch]:
        """Generate batches from prefetched files."""
        # Accumulators for current batch
        all_chunks = []
        all_weights = []
        all_frame_ids = []
        all_scalars = []
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
                # Record sample boundary for previous date
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
                chunks = self._chunk_frame(frame_data['features'])
                
                for chunk, tick_count in chunks:
                    all_chunks.append(chunk)
                    all_weights.append(tick_count)
                    all_frame_ids.append(current_frame_id)
                
                all_scalars.append(frame_data['scalars'])
                current_frame_id += 1
                
                # Check if batch is full
                if len(all_chunks) >= self.max_chunks_per_batch:
                    # Finalize current date's sample if it has frames
                    if current_frame_id > date_start_frame:
                        rv = self._forward_rv_lookup.get(
                            pd.to_datetime(current_date).date(),
                            np.nan
                        )
                        if not np.isnan(rv):
                            sample_boundaries.append((date_start_frame, current_frame_id, rv))
                    
                    # Yield batch
                    yield self._create_batch(
                        all_chunks, all_weights, all_frame_ids,
                        all_scalars, sample_boundaries
                    )
                    
                    # Reset accumulators
                    all_chunks = []
                    all_weights = []
                    all_frame_ids = []
                    all_scalars = []
                    sample_boundaries = []
                    current_frame_id = 0
                    date_start_frame = 0
        
        # Final batch
        if all_chunks:
            # Finalize last date's sample
            if current_date is not None and current_frame_id > date_start_frame:
                rv = self._forward_rv_lookup.get(
                    pd.to_datetime(current_date).date(),
                    np.nan
                )
                if not np.isnan(rv):
                    sample_boundaries.append((date_start_frame, current_frame_id, rv))
            
            yield self._create_batch(
                all_chunks, all_weights, all_frame_ids,
                all_scalars, sample_boundaries
            )
    
    def _create_batch(
        self,
        chunks: List[np.ndarray],
        weights: List[int],
        frame_ids: List[int],
        scalars: List[np.ndarray],
        sample_boundaries: List[Tuple[int, int, float]],
    ) -> SingleStreamBatch:
        """Create a SingleStreamBatch from accumulated data."""
        batch = SingleStreamBatch(self.stream)
        
        batch.chunks = torch.from_numpy(np.stack(chunks))
        batch.weights = torch.tensor(weights, dtype=torch.float32)
        batch.frame_id = torch.tensor(frame_ids, dtype=torch.long)
        batch.frame_scalars = torch.from_numpy(np.stack(scalars)) if scalars else torch.zeros(1, 3)
        batch.sample_boundaries = sample_boundaries
        batch.num_frames = len(scalars)
        batch.num_chunks = len(chunks)
        
        return batch
