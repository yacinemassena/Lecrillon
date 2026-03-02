"""
Multi-Stream Dataset with Background Prefetching.

Handles stocks, options, and index data streams with:
- Background thread prefetches files to RAM
- On-the-fly ticker filtering (O(1) set lookup, no stalling)
- Chunk-level batching for strict VRAM control
- Sample boundary tracking for loss computation
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
class StreamConfig:
    """Configuration for a data stream."""
    name: str
    path: Path
    columns: List[str]  # [price_col, size_col, time_col]
    enabled: bool = True


class MultiStreamBatch:
    """
    A batch of chunks from multiple streams with sample boundary tracking.
    
    Attributes:
        chunks: [N_chunks, chunk_len, dim_in] - All chunks in batch
        weights: [N_chunks] - Weight per chunk (tick count)
        frame_id: [N_chunks] - Which frame each chunk belongs to
        stream_id: [N_chunks] - Which stream each chunk came from (0=stocks, 1=options, 2=index)
        frame_scalars: [N_frames, num_scalars] - Per-frame scalars
        sample_boundaries: List of (start_frame, end_frame, rv_target) tuples
        num_frames: Total frames in batch
        num_chunks: Total chunks in batch
    """
    __slots__ = ['chunks', 'weights', 'frame_id', 'stream_id', 'frame_scalars',
                 'sample_boundaries', 'num_frames', 'num_chunks']
    
    def __init__(self):
        self.chunks = None
        self.weights = None
        self.frame_id = None
        self.stream_id = None
        self.frame_scalars = None
        self.sample_boundaries = []
        self.num_frames = 0
        self.num_chunks = 0
    
    def to_device(self, device: torch.device) -> 'MultiStreamBatch':
        """Move tensors to device."""
        self.chunks = self.chunks.to(device)
        self.weights = self.weights.to(device)
        self.frame_id = self.frame_id.to(device)
        self.stream_id = self.stream_id.to(device)
        self.frame_scalars = self.frame_scalars.to(device)
        return self
    
    def get_rv_targets(self, device: torch.device) -> torch.Tensor:
        """Get RV targets for each sample in batch."""
        targets = [sb[2] for sb in self.sample_boundaries]
        return torch.tensor(targets, dtype=torch.float32, device=device)


class MultiStreamDataset(IterableDataset):
    """
    Multi-stream dataset with background prefetching and on-the-fly filtering.
    
    Memory model:
    - RAM: prefetch_files files per stream in background queue
    - VRAM: Only current batch of chunks (controlled by max_chunks_per_batch)
    
    Filtering:
    - Stock filtering uses precomputed top-N list (O(1) set lookup)
    - Filtering happens in background thread while GPU trains
    """
    
    STREAM_IDS = {'stocks': 0, 'options': 1, 'index': 2}
    
    def __init__(
        self,
        stocks_path: str,
        options_path: str,
        index_path: str,
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
        filter_stocks: bool = True,
        top_stocks_file: Optional[str] = None,
    ):
        self.stocks_path = Path(stocks_path)
        self.options_path = Path(options_path)
        self.index_path = Path(index_path)
        self.split = split
        self.frame_interval = frame_interval
        self.chunk_len = chunk_len
        self.dim_in = dim_in
        self.max_chunks_per_batch = max_chunks_per_batch
        self.prefetch_files = prefetch_files
        self.rv_horizon_days = rv_horizon_days
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.filter_stocks = filter_stocks
        
        # Load stock filter (O(1) lookup)
        self.allowed_stocks: Set[str] = set()
        if filter_stocks and top_stocks_file and Path(top_stocks_file).exists():
            self._load_stock_filter(top_stocks_file)
        
        # Find dates with all streams
        self.dates = self._find_common_dates()
        logger.info(f"MultiStreamDataset [{split}]: {len(self.dates)} dates")
        
        # Load precomputed RV
        self._forward_rv_lookup = {}
        if rv_file and Path(rv_file).exists():
            self._load_rv(rv_file)
        
        # Prefetch state
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()
    
    def _load_stock_filter(self, file_path: str):
        """Load allowed stocks from precomputed file."""
        with open(file_path, 'r') as f:
            self.allowed_stocks = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(self.allowed_stocks)} stocks for filtering")
    
    def _find_common_dates(self) -> List[str]:
        """Find dates that have data in all streams."""
        # Get dates from each stream
        stock_dates = set()
        for f in self.stocks_path.glob('*.parquet'):
            if not f.name.startswith('._'):
                stock_dates.add(f.stem.split('.')[0])
        
        option_dates = set()
        for f in self.options_path.glob('*.parquet'):
            if not f.name.startswith('._'):
                option_dates.add(f.stem.split('.')[0])
        
        # Index has year subdirectories
        index_dates = set()
        for year_dir in self.index_path.iterdir():
            if year_dir.is_dir():
                for f in year_dir.glob('*.parquet'):
                    if not f.name.startswith('._'):
                        index_dates.add(f.stem.split('.')[0])
        
        # Common dates
        common = stock_dates & option_dates & index_dates
        dates_with_dt = []
        
        for d in common:
            try:
                dt = pd.to_datetime(d)
                dates_with_dt.append((dt, d))
            except:
                continue
        
        dates_with_dt.sort(key=lambda x: x[0])
        
        # Filter by split
        if self.split == 'train':
            return [d for dt, d in dates_with_dt if dt <= self.train_end]
        elif self.split == 'val':
            return [d for dt, d in dates_with_dt 
                    if dt > self.train_end and dt <= self.val_end]
        elif self.split == 'test':
            return [d for dt, d in dates_with_dt if dt > self.val_end]
        return [d for _, d in dates_with_dt]
    
    def _load_rv(self, rv_file: str):
        """Load precomputed forward RV."""
        rv_df = pd.read_parquet(rv_file)
        rv_df['date'] = pd.to_datetime(rv_df['date']).dt.date
        
        rv_col = f'rv_{self.rv_horizon_days}d_forward'
        if rv_col not in rv_df.columns:
            rv_cols = [c for c in rv_df.columns if 'forward' in c]
            rv_col = rv_cols[0] if rv_cols else None
            if not rv_col:
                raise ValueError(f"No forward RV column in {rv_file}")
        
        self._forward_rv_lookup = dict(zip(rv_df['date'], rv_df[rv_col]))
        logger.info(f"Loaded RV for {len(self._forward_rv_lookup)} days")
    
    def _get_rv(self, date_str: str) -> float:
        """Get forward RV for a date."""
        try:
            date = pd.to_datetime(date_str).date()
            return self._forward_rv_lookup.get(date, np.nan)
        except:
            return np.nan
    
    def _get_file_paths(self, date_str: str) -> Dict[str, Path]:
        """Get file paths for all streams on a given date."""
        year = date_str[:4]
        return {
            'stocks': self.stocks_path / f'{date_str}.parquet',
            'options': self.options_path / f'{date_str}.parquet',
            'index': self.index_path / year / f'{date_str}.parquet',
        }
    
    def _load_stream(self, file_path: Path, stream_name: str) -> Optional[pd.DataFrame]:
        """Load and preprocess a single stream file."""
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        df.columns = df.columns.str.lower()
        
        # Apply stock filtering (O(1) set lookup)
        if stream_name == 'stocks' and self.filter_stocks and self.allowed_stocks:
            if 'ticker' in df.columns:
                df = df[df['ticker'].isin(self.allowed_stocks)]
                if df.empty:
                    return None
        
        # Standardize timestamp column
        time_col = None
        for col in ['sip_timestamp', 'timestamp', 'time', 't']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            return None
        
        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')
        
        # Bin into frames
        df['bin'] = df['timestamp'].dt.floor(self.frame_interval)
        
        return df
    
    def _process_frame(self, group: pd.DataFrame, stream_id: int) -> Optional[Dict]:
        """Process a single frame into chunks."""
        # Extract price
        price_col = next((c for c in ['price', 'close', 'last'] 
                          if c in group.columns), None)
        if not price_col:
            return None
        
        prices = group[price_col].values.astype(np.float32)
        n_ticks = len(prices)
        
        if n_ticks == 0:
            return None
        
        # Size
        if 'size' in group.columns:
            sizes = group['size'].values.astype(np.float32)
        else:
            sizes = np.ones(n_ticks, dtype=np.float32)
        
        # Time delta
        if 'timestamp' in group.columns:
            ts = group['timestamp']
            dt = ts.diff().dt.total_seconds().fillna(0.0).values.astype(np.float32)
        else:
            dt = np.zeros(n_ticks, dtype=np.float32)
        
        # Stack features [N, 3]
        ticks = np.stack([prices, sizes, dt], axis=1)
        
        # Normalize: log transform
        ticks = np.log1p(np.maximum(ticks, 0))
        
        # Scalars
        notional = np.sum(prices * sizes)
        vol = np.std(prices) if n_ticks > 1 else 0.0
        scalars = np.array([float(n_ticks), notional, vol], dtype=np.float32)
        
        # Chunk
        K = max(1, int(np.ceil(n_ticks / self.chunk_len)))
        pad_len = K * self.chunk_len - n_ticks
        
        if pad_len > 0:
            padding = np.zeros((pad_len, self.dim_in), dtype=np.float32)
            ticks = np.vstack([ticks, padding])
        
        chunks = ticks.reshape(K, self.chunk_len, self.dim_in)
        
        # Weights (tick count per chunk)
        weights = np.full(K, self.chunk_len, dtype=np.float32)
        if pad_len > 0:
            weights[-1] -= pad_len
        
        # Stream IDs
        stream_ids = np.full(K, stream_id, dtype=np.int64)
        
        return {
            'chunks': chunks,
            'weights': weights,
            'scalars': scalars,
            'stream_ids': stream_ids,
            'n_chunks': K,
        }
    
    def _load_and_process_date(self, date_str: str) -> Optional[Dict]:
        """Load all streams for a date and process into frames."""
        rv_target = self._get_rv(date_str)
        if np.isnan(rv_target):
            return None
        
        file_paths = self._get_file_paths(date_str)
        all_frames = []
        
        for stream_name, file_path in file_paths.items():
            stream_id = self.STREAM_IDS[stream_name]
            df = self._load_stream(file_path, stream_name)
            
            if df is None or df.empty:
                continue
            
            # Process each frame
            for bin_time, group in df.groupby('bin'):
                frame_data = self._process_frame(group, stream_id)
                if frame_data is not None:
                    frame_data['bin_time'] = bin_time
                    all_frames.append(frame_data)
        
        if not all_frames:
            return None
        
        # Sort frames by time
        all_frames.sort(key=lambda x: x['bin_time'])
        
        return {
            'frames': all_frames,
            'rv_target': rv_target,
            'date': date_str,
        }
    
    def _prefetch_worker(self, date_queue: queue.Queue, data_queue: queue.Queue):
        """Background thread that loads dates into RAM."""
        while not self._stop_prefetch.is_set():
            try:
                date_str = date_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if date_str is None:  # Poison pill
                break
            
            # Load and process date (filtering happens here, in background)
            date_data = self._load_and_process_date(date_str)
            
            if date_data is not None:
                data_queue.put(date_data)
            
            date_queue.task_done()
    
    def _start_prefetch(self, dates: List[str]) -> queue.Queue:
        """Start background prefetch thread."""
        date_queue = queue.Queue()
        data_queue = queue.Queue(maxsize=self.prefetch_files)
        
        for d in dates:
            date_queue.put(d)
        date_queue.put(None)  # Poison pill
        
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(date_queue, data_queue),
            daemon=True
        )
        self._prefetch_thread.start()
        
        return data_queue
    
    def _stop_prefetch_thread(self):
        """Stop the prefetch thread."""
        if self._prefetch_thread is not None:
            self._stop_prefetch.set()
            self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None
    
    def __iter__(self) -> Iterator[MultiStreamBatch]:
        """Iterate over chunk batches."""
        data_queue = self._start_prefetch(self.dates)
        
        # Accumulator for current batch
        batch_chunks = []
        batch_weights = []
        batch_scalars = []
        batch_frame_ids = []
        batch_stream_ids = []
        sample_boundaries = []
        
        current_frame_idx = 0
        current_chunk_count = 0
        dates_processed = 0
        
        try:
            while dates_processed < len(self.dates):
                try:
                    date_data = data_queue.get(timeout=10.0)
                except queue.Empty:
                    continue
                
                dates_processed += 1
                frames = date_data['frames']
                rv_target = date_data['rv_target']
                
                sample_start_frame = current_frame_idx
                
                for frame in frames:
                    n_chunks = frame['n_chunks']
                    
                    # Check if adding this frame would exceed batch limit
                    if current_chunk_count + n_chunks > self.max_chunks_per_batch:
                        if current_chunk_count > 0:
                            yield self._build_batch(
                                batch_chunks, batch_weights, batch_scalars,
                                batch_frame_ids, batch_stream_ids, sample_boundaries
                            )
                        
                        # Reset batch
                        batch_chunks = []
                        batch_weights = []
                        batch_scalars = []
                        batch_frame_ids = []
                        batch_stream_ids = []
                        sample_boundaries = []
                        current_frame_idx = 0
                        current_chunk_count = 0
                        sample_start_frame = 0
                    
                    # Add frame to batch
                    batch_chunks.append(frame['chunks'])
                    batch_weights.append(frame['weights'])
                    batch_scalars.append(frame['scalars'])
                    batch_stream_ids.append(frame['stream_ids'])
                    
                    frame_ids = np.full(n_chunks, current_frame_idx, dtype=np.int64)
                    batch_frame_ids.append(frame_ids)
                    
                    current_frame_idx += 1
                    current_chunk_count += n_chunks
                
                # End of date = end of sample
                sample_end_frame = current_frame_idx
                sample_boundaries.append((sample_start_frame, sample_end_frame, rv_target))
            
            # Yield final batch
            if current_chunk_count > 0:
                yield self._build_batch(
                    batch_chunks, batch_weights, batch_scalars,
                    batch_frame_ids, batch_stream_ids, sample_boundaries
                )
        
        finally:
            self._stop_prefetch_thread()
    
    def _build_batch(
        self,
        chunks_list: List[np.ndarray],
        weights_list: List[np.ndarray],
        scalars_list: List[np.ndarray],
        frame_ids_list: List[np.ndarray],
        stream_ids_list: List[np.ndarray],
        sample_boundaries: List[Tuple[int, int, float]],
    ) -> MultiStreamBatch:
        """Build a MultiStreamBatch from accumulated data."""
        batch = MultiStreamBatch()
        
        batch.chunks = torch.from_numpy(np.vstack(chunks_list))
        batch.weights = torch.from_numpy(np.concatenate(weights_list))
        batch.frame_scalars = torch.from_numpy(np.vstack(scalars_list))
        batch.frame_id = torch.from_numpy(np.concatenate(frame_ids_list))
        batch.stream_id = torch.from_numpy(np.concatenate(stream_ids_list))
        batch.sample_boundaries = sample_boundaries.copy()
        batch.num_frames = len(scalars_list)
        batch.num_chunks = len(batch.chunks)
        
        return batch


def create_multistream_dataloader(
    stocks_path: str,
    options_path: str,
    index_path: str,
    rv_file: str,
    split: str = 'train',
    max_chunks_per_batch: int = 2000,
    prefetch_files: int = 8,
    filter_stocks: bool = True,
    top_stocks_file: Optional[str] = None,
    **kwargs
) -> MultiStreamDataset:
    """
    Create a multi-stream dataloader.
    
    Args:
        stocks_path: Path to polygon_stock_trades
        options_path: Path to options_trades
        index_path: Path to index_data
        rv_file: Path to precomputed RV file
        split: 'train', 'val', or 'test'
        max_chunks_per_batch: Max chunks per batch (controls VRAM)
        prefetch_files: Dates to keep in RAM buffer
        filter_stocks: Whether to filter stocks to top-N
        top_stocks_file: Path to top stocks list
    
    Returns:
        MultiStreamDataset instance
    """
    return MultiStreamDataset(
        stocks_path=stocks_path,
        options_path=options_path,
        index_path=index_path,
        rv_file=rv_file,
        split=split,
        max_chunks_per_batch=max_chunks_per_batch,
        prefetch_files=prefetch_files,
        filter_stocks=filter_stocks,
        top_stocks_file=top_stocks_file,
        **kwargs
    )
