"""
SPY Tick Dataset for TCN Pretraining on Realized Volatility.
Computes 30-day forward RV targets from tick data.
"""

import os
import gzip
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple
from torch.utils.data import IterableDataset, get_worker_info
import logging


class SPYRVDataset(IterableDataset):
    """
    Streaming dataset for SPY tick data with 30-day forward RV targets.
    
    Features:
    - 10-second frames (matching architecture spec)
    - No truncation: ticks split into chunks
    - Per-frame scalars: n_ticks, notional, vol
    - 30-day forward Realized Volatility target
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        frame_interval: str = '10s',
        chunk_len: int = 256,
        num_frames: int = 360,  # 1 hour of 10s frames
        rv_horizon_days: int = 30,
        train_end: str = '2022-12-31',
        val_end: str = '2023-06-30',
        weight_mode: str = 'tick_count',
        dim_in: int = 3,
    ):
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_path)
        self.split = split
        self.frame_interval = frame_interval
        self.chunk_len = chunk_len
        self.num_frames = num_frames
        self.rv_horizon_days = rv_horizon_days
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.weight_mode = weight_mode
        self.dim_in = dim_in
        
        # Find all data files
        self.files = self._find_files()
        self.logger.info(f"SPYRVDataset [{split}]: Found {len(self.files)} files")
        
        # Precompute daily RV for 30-day forward targets
        self._daily_rv_cache = None
        
    def _find_files(self) -> List[Path]:
        """Find all parquet/csv files and filter by split."""
        files = []
        extensions = ['*.parquet', '*.csv', '*.csv.gz']
        
        for ext in extensions:
            files.extend(sorted(self.data_path.rglob(ext)))
        
        # Filter hidden files
        files = [f for f in files if not f.name.startswith('._')]
        
        # Parse dates and filter by split
        files_with_dates = []
        for f in files:
            try:
                # Try to parse date from filename
                base = f.stem.split('.')[0]
                date = pd.to_datetime(base, errors='coerce')
                if pd.isna(date):
                    # Try parent directory name
                    date = pd.to_datetime(f.parent.name, errors='coerce')
                if not pd.isna(date):
                    files_with_dates.append((date, f))
            except Exception:
                continue
        
        files_with_dates.sort(key=lambda x: x[0])
        
        # Split by date
        if self.split == 'train':
            return [f for d, f in files_with_dates if d <= self.train_end]
        elif self.split == 'val':
            return [f for d, f in files_with_dates 
                    if d > self.train_end and d <= self.val_end]
        elif self.split == 'test':
            return [f for d, f in files_with_dates if d > self.val_end]
        else:
            return [f for _, f in files_with_dates]
    
    def _shard_files(self, files: List[Path]) -> List[Path]:
        """Shard files across workers."""
        worker_info = get_worker_info()
        if worker_info is None:
            return files
        return files[worker_info.id::worker_info.num_workers]
    
    def _load_tick_data(self, file_path: Path) -> pd.DataFrame:
        """Load tick data from file."""
        try:
            if str(file_path).endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            return df
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path.name}: {e}")
            return pd.DataFrame()
    
    def _compute_daily_rv(self) -> pd.Series:
        """
        Precompute daily Realized Volatility for all dates.
        RV = sqrt(sum(log_returns^2)) for each day.
        Returns a Series indexed by date.
        """
        if self._daily_rv_cache is not None:
            return self._daily_rv_cache
        
        self.logger.info("Computing daily RV for all files...")
        daily_rvs = {}
        
        all_files = sorted(self.data_path.rglob('*.parquet'))
        all_files.extend(sorted(self.data_path.rglob('*.csv')))
        all_files = [f for f in all_files if not f.name.startswith('._')]
        
        for file_path in all_files:
            try:
                df = self._load_tick_data(file_path)
                if df.empty:
                    continue
                
                # Standardize columns
                df.columns = df.columns.str.lower()
                
                # Get price column
                price_col = next((c for c in ['price', 'close', 'last'] 
                                  if c in df.columns), None)
                if price_col is None:
                    continue
                
                prices = df[price_col].values
                if len(prices) < 2:
                    continue
                
                # Compute log returns
                log_returns = np.log(prices[1:] / prices[:-1])
                log_returns = log_returns[np.isfinite(log_returns)]
                
                # Daily RV
                rv = np.sqrt(np.sum(log_returns ** 2))
                
                # Parse date from filename
                base = file_path.stem.split('.')[0]
                date = pd.to_datetime(base, errors='coerce')
                if pd.isna(date):
                    date = pd.to_datetime(file_path.parent.name, errors='coerce')
                
                if not pd.isna(date):
                    daily_rvs[date.date()] = rv
                    
            except Exception as e:
                self.logger.warning(f"RV computation failed for {file_path}: {e}")
                continue
        
        self._daily_rv_cache = pd.Series(daily_rvs).sort_index()
        self.logger.info(f"Computed daily RV for {len(self._daily_rv_cache)} days")
        return self._daily_rv_cache
    
    def _get_forward_rv(self, current_date: pd.Timestamp) -> float:
        """
        Get 30-day forward Realized Volatility.
        Sum of daily RVs over next 30 calendar days (approx 21 trading days).
        """
        daily_rv = self._compute_daily_rv()
        
        start_date = current_date.date()
        end_date = (current_date + pd.Timedelta(days=self.rv_horizon_days)).date()
        
        # Get RVs in the forward window
        mask = (daily_rv.index > start_date) & (daily_rv.index <= end_date)
        forward_rvs = daily_rv[mask]
        
        if len(forward_rvs) < 10:  # Need at least 10 trading days
            return np.nan
        
        # Aggregate: sqrt(sum(daily_rv^2))
        rv_30d = np.sqrt(np.sum(forward_rvs.values ** 2))
        return rv_30d
    
    def _process_frame_ticks(self, group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process ticks for a single 10s frame.
        Returns: (chunks, weights, scalars)
        """
        # Extract features
        price_col = next((c for c in ['price', 'close', 'last'] 
                          if c in group.columns), None)
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
            sizes = np.ones(len(group), dtype=np.float32)
        
        # Time delta
        if 'timestamp' in group.columns:
            ts = group['timestamp']
            dt = ts.diff().dt.total_seconds().fillna(0.0).values.astype(np.float32)
        else:
            dt = np.zeros(len(group), dtype=np.float32)
        
        # Stack features: [N, 3]
        ticks = np.stack([prices, sizes, dt], axis=1)
        n_ticks = len(ticks)
        
        # Pad to dim_in if needed
        if ticks.shape[1] < self.dim_in:
            padding = np.zeros((n_ticks, self.dim_in - ticks.shape[1]), dtype=np.float32)
            ticks = np.hstack([ticks, padding])
        
        # Normalize: log transform + robust scaling
        ticks = np.log1p(np.maximum(ticks, 0))
        
        # Compute scalars
        n_ticks_scalar = float(n_ticks)
        notional = np.sum(prices * sizes)
        vol = np.std(prices) if n_ticks > 1 else 0.0
        scalars = np.array([n_ticks_scalar, notional, vol], dtype=np.float32)
        
        # Chunking
        K = max(1, int(np.ceil(n_ticks / self.chunk_len)))
        pad_len = K * self.chunk_len - n_ticks
        
        if pad_len > 0:
            padding = np.zeros((pad_len, self.dim_in), dtype=np.float32)
            ticks_padded = np.vstack([ticks, padding])
        else:
            ticks_padded = ticks
        
        chunks = ticks_padded.reshape(K, self.chunk_len, self.dim_in)
        
        # Weights
        if self.weight_mode == 'tick_count':
            weights = np.full(K, self.chunk_len, dtype=np.float32)
            if pad_len > 0:
                weights[-1] -= pad_len
        else:
            weights = np.ones(K, dtype=np.float32)
        
        return chunks, weights, scalars
    
    def _generate_frames_from_file(self, file_path: Path) -> Iterator[Tuple]:
        """
        Yields frames from a single file.
        Returns: (chunks, weights, scalars, timestamp, rv_target)
        """
        df = self._load_tick_data(file_path)
        if df.empty:
            return
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        
        # Handle timestamp
        if 'ts_ns' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_ns'], unit='ns', utc=True)
        elif 'sip_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['sip_timestamp'], errors='coerce')
        elif 'timestamp' not in df.columns:
            for col in ['time', 't', 'datetime']:
                if col in df.columns:
                    df['timestamp'] = df[col]
                    break
        
        if 'timestamp' not in df.columns:
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', drop=False, inplace=True)
        
        # Get file date for forward RV lookup
        try:
            base = file_path.stem.split('.')[0]
            file_date = pd.to_datetime(base, errors='coerce')
            if pd.isna(file_date):
                file_date = pd.to_datetime(file_path.parent.name, errors='coerce')
        except:
            file_date = df['timestamp'].iloc[0]
        
        # Get 30-day forward RV for this date
        rv_target = self._get_forward_rv(file_date)
        if np.isnan(rv_target):
            return  # Skip files without valid forward RV
        
        # Bin into 10s frames
        df['bin'] = df['timestamp'].dt.floor(self.frame_interval)
        
        for bin_time, group in df.groupby('bin'):
            chunks, weights, scalars = self._process_frame_ticks(group)
            yield (chunks, weights, scalars, bin_time, rv_target)
    
    def _pack_sample(self, buffer_list: List[Tuple]) -> Dict:
        """Pack frames into a batch sample."""
        all_chunks = [x[0] for x in buffer_list]
        all_weights = [x[1] for x in buffer_list]
        all_scalars = [x[2] for x in buffer_list]
        rv_targets = [x[4] for x in buffer_list]
        
        # Build frame_ptr for variable-length chunks
        frame_ptr = [0]
        current_idx = 0
        for chunks in all_chunks:
            current_idx += len(chunks)
            frame_ptr.append(current_idx)
        
        # Pack tensors
        packed_chunks = np.vstack(all_chunks)
        packed_weights = np.concatenate(all_weights)
        packed_scalars = np.vstack(all_scalars)
        packed_ptr = np.array(frame_ptr, dtype=np.int64)
        
        # RV target is the same for all frames in a file (30-day forward)
        # Use the last frame's target
        rv_target = np.array(rv_targets[-1], dtype=np.float32)
        
        return {
            'chunks': packed_chunks,
            'frame_ptr': packed_ptr,
            'weights': packed_weights,
            'frame_scalars': packed_scalars,
            'rv_target': rv_target,
        }
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over samples."""
        files = self._shard_files(self.files)
        
        for file_path in files:
            buffer = []
            
            for frame_data in self._generate_frames_from_file(file_path):
                buffer.append(frame_data)
                
                if len(buffer) >= self.num_frames:
                    yield self._pack_sample(buffer)
                    buffer = []
            
            # Yield remaining frames if enough
            if len(buffer) >= self.num_frames // 2:
                yield self._pack_sample(buffer)
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate multiple samples into a batch."""
        batch_chunks = []
        batch_weights = []
        batch_scalars = []
        batch_rv_targets = []
        batch_frame_id = []
        
        total_frames = 0
        
        for sample in batch:
            chunks = torch.from_numpy(sample['chunks'])
            weights = torch.from_numpy(sample['weights'])
            scalars = torch.from_numpy(sample['frame_scalars'])
            rv_target = torch.tensor(sample['rv_target'])
            frame_ptr = sample['frame_ptr']
            
            batch_chunks.append(chunks)
            batch_weights.append(weights)
            batch_scalars.append(scalars)
            batch_rv_targets.append(rv_target)
            
            # Build frame_id from frame_ptr
            n_chunks = len(chunks)
            frame_ids = torch.zeros(n_chunks, dtype=torch.long)
            for i in range(len(frame_ptr) - 1):
                start = frame_ptr[i]
                end = frame_ptr[i + 1]
                frame_ids[start:end] = i + total_frames
            
            batch_frame_id.append(frame_ids)
            total_frames += (len(frame_ptr) - 1)
        
        return {
            'chunks': torch.cat(batch_chunks, dim=0),
            'frame_id': torch.cat(batch_frame_id, dim=0),
            'weights': torch.cat(batch_weights, dim=0),
            'frame_scalars': torch.cat(batch_scalars, dim=0),
            'rv_target': torch.stack(batch_rv_targets),
            'num_frames': total_frames,
        }
