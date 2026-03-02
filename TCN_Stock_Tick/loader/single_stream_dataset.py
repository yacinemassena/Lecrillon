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
        ticker_ids: [N_chunks, chunk_len] - Ticker IDs per tick (optional)
        frame_scalars: [N_frames, num_scalars] - Per-frame scalars
        sample_boundaries: List of (start_frame, end_frame, rv_target) tuples
        num_frames: Total frames in batch
        num_chunks: Total chunks in batch
        stream: Name of the stream ('stocks', 'options', 'index')
    """
    __slots__ = ['chunks', 'weights', 'frame_id', 'ticker_ids', 'frame_scalars',
                 'sample_boundaries', 'num_frames', 'num_chunks', 'stream']
    
    def __init__(self, stream: str):
        self.chunks = None
        self.weights = None
        self.frame_id = None
        self.ticker_ids = None
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
        if self.ticker_ids is not None:
            self.ticker_ids = self.ticker_ids.to(device)
        self.frame_scalars = self.frame_scalars.to(device)
        return self
    
    def get_rv_targets(self, device: torch.device) -> torch.Tensor:
        """Get RV targets for each sample in batch."""
        targets = [sb[2] for sb in self.sample_boundaries]
        return torch.tensor(targets, dtype=torch.float32, device=device)


# Stream-specific column mappings - Stock Tick only
STREAM_COLUMNS = {
    'stock_tick': {
        'price': 'price',
        'size': 'size',
        'time': 'sip_timestamp',
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
        max_chunks_per_batch: int = 2000,
        prefetch_files: int = 8,
        rv_horizon_days: int = 30,
        train_end: str = '2022-12-31',
        val_end: str = '2023-06-30',
        filter_tickers: bool = False,
        allowed_tickers_file: Optional[str] = None,
    ):
        assert stream in ('stock_tick',), f"Unknown stream: {stream}. This loader only supports stock_tick."
        
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
        
        # Load ticker filter and create ticker-to-ID mapping
        self.allowed_tickers: Set[str] = set()
        self.ticker_to_id: Dict[str, int] = {}
        if filter_tickers and allowed_tickers_file and Path(allowed_tickers_file).exists():
            self._load_ticker_filter(allowed_tickers_file)
        
        # Store num_tickers for model compatibility
        self.num_tickers = len(self.ticker_to_id) if self.ticker_to_id else 0
        
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
        """Load allowed tickers and create ticker-to-ID mapping."""
        tickers = []
        with open(file_path, 'r') as f:
            for line in f:
                ticker = line.strip()
                if ticker:
                    tickers.append(ticker)
        
        self.allowed_tickers = set(tickers)
        # Create ticker-to-ID mapping (sorted for consistency)
        self.ticker_to_id = {ticker: idx for idx, ticker in enumerate(sorted(tickers))}
        logger.info(f"Loaded {len(self.allowed_tickers)} tickers with ID mapping")
    
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
        
        # Ensure required columns exist (size_col and price_col can be None for index data)
        required = [time_col]
        if price_col is not None and price_col not in df.columns:
            # Price is optional for index data - skip this file if missing
            return
        if price_col is not None:
            required.append(price_col)
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
            
            # Extract raw features
            raw_prices = frame_df[price_col].values.astype(np.float32)
            n_ticks = len(raw_prices)
            
            # Raw sizes
            if size_col is not None and size_col in frame_df.columns:
                raw_sizes = frame_df[size_col].values.astype(np.float32)
            else:
                # Use absolute price changes as size proxy for index data
                raw_sizes = np.abs(np.diff(raw_prices, prepend=raw_prices[0]))
            
            # Raw dt (time since frame start, in seconds)
            frame_start = frame_df['_ts'].min()
            raw_dt = (frame_df['_ts'] - frame_start).dt.total_seconds().values.astype(np.float32)
            
            # --- Compute scalars from RAW values (before normalization) ---
            if size_col is not None and size_col in frame_df.columns:
                total_notional = float((raw_prices * raw_sizes).sum())
                total_volume = float(raw_sizes.sum())
            else:
                total_notional = float(raw_prices.var() * n_ticks)
                total_volume = float(np.abs(raw_sizes).sum())
            
            # --- Normalize features ---
            # 1. Price → log-returns from frame start (removes absolute level)
            # Handle zero prices in index data
            p0 = raw_prices[0]
            if p0 > 0 and np.all(raw_prices > 0):
                prices = np.log(raw_prices / p0)
            elif p0 > 0:
                # Some prices are zero - use safe division
                with np.errstate(divide='ignore', invalid='ignore'):
                    prices = np.log(np.where(raw_prices > 0, raw_prices / p0, 1.0))
                    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # First price is zero - skip normalization
                prices = np.zeros_like(raw_prices)
            
            # 2. Size → log1p + z-score per frame
            sizes = np.log1p(raw_sizes)
            if sizes.std() > 0:
                sizes = (sizes - sizes.mean()) / sizes.std()
            
            # 3. dt → relative to median inter-tick interval
            positive_dt = raw_dt[raw_dt > 0]
            median_dt = np.median(positive_dt) if len(positive_dt) > 0 else 1e-3
            dt = np.log1p(raw_dt / median_dt)
            
            # Stack features
            features = np.stack([prices, sizes, dt], axis=1)  # [N_ticks, 3]
            
            # Extract ticker IDs if ticker embeddings are enabled
            ticker_ids = None
            if self.ticker_to_id and 'ticker' in self.columns and self.columns['ticker'] in frame_df.columns:
                ticker_col = self.columns['ticker']
                tickers = frame_df[ticker_col].values
                # Map tickers to IDs (use 0 for unknown tickers)
                ticker_ids = np.array([self.ticker_to_id.get(t, 0) for t in tickers], dtype=np.int64)
            
            # Scalars: raw values (ChunkedFrameEncoder applies log1p)
            yield {
                'features': features,
                'ticker_ids': ticker_ids,
                'n_ticks': n_ticks,
                'scalars': np.array([
                    float(n_ticks),
                    total_notional,
                    total_volume,
                ], dtype=np.float32),
                'frame_ts': frame_ts,
                'date': date_str,
            }
    
    def _chunk_frame(self, features: np.ndarray, ticker_ids: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, int, Optional[np.ndarray]]]:
        """Split frame features into fixed-size chunks with tick counts and ticker IDs."""
        chunks = []
        n_ticks = len(features)
        
        for i in range(0, n_ticks, self.chunk_len):
            chunk = features[i:i + self.chunk_len]
            tick_count = len(chunk)
            
            # Handle ticker IDs for this chunk
            chunk_ticker_ids = None
            if ticker_ids is not None:
                chunk_ticker_ids = ticker_ids[i:i + self.chunk_len]
                # Pad ticker IDs if needed
                if len(chunk_ticker_ids) < self.chunk_len:
                    pad_ids = np.zeros(self.chunk_len - len(chunk_ticker_ids), dtype=np.int64)
                    chunk_ticker_ids = np.concatenate([chunk_ticker_ids, pad_ids])
            
            # Pad features if needed
            if len(chunk) < self.chunk_len:
                pad = np.zeros((self.chunk_len - len(chunk), self.dim_in), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            
            chunks.append((chunk, tick_count, chunk_ticker_ids))
        
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
        all_ticker_ids = []
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
                chunks = self._chunk_frame(frame_data['features'], frame_data.get('ticker_ids'))
                
                for chunk, tick_count, chunk_ticker_ids in chunks:
                    all_chunks.append(chunk)
                    all_weights.append(tick_count)
                    all_frame_ids.append(current_frame_id)
                    if chunk_ticker_ids is not None:
                        all_ticker_ids.append(chunk_ticker_ids)
                
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
                        all_ticker_ids, all_scalars, sample_boundaries
                    )
                    
                    # Reset accumulators
                    all_chunks = []
                    all_weights = []
                    all_frame_ids = []
                    all_ticker_ids = []
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
                all_ticker_ids, all_scalars, sample_boundaries
            )
    
    def _create_batch(
        self,
        chunks: List[np.ndarray],
        weights: List[int],
        frame_ids: List[int],
        ticker_ids: List[np.ndarray],
        scalars: List[np.ndarray],
        sample_boundaries: List[Tuple[int, int, float]],
    ) -> SingleStreamBatch:
        """Create a SingleStreamBatch from accumulated data."""
        batch = SingleStreamBatch(self.stream)
        
        batch.chunks = torch.from_numpy(np.stack(chunks))
        batch.weights = torch.tensor(weights, dtype=torch.float32)
        batch.frame_id = torch.tensor(frame_ids, dtype=torch.long)
        
        # Add ticker IDs if available
        if ticker_ids:
            batch.ticker_ids = torch.from_numpy(np.stack(ticker_ids))
        
        batch.frame_scalars = torch.from_numpy(np.stack(scalars)) if scalars else torch.zeros(1, 3)
        batch.sample_boundaries = sample_boundaries
        batch.num_frames = len(scalars)
        batch.num_chunks = len(chunks)
        
        return batch
