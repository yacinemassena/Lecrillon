"""
Bar-to-Mamba Dataset: Direct 1s bar sequences for Mamba-only VIX prediction.

No caching - just load files directly, train, discard.
Simple and clean.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Set
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


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
class BarMambaDataset(IterableDataset):
    """
    Direct bar-to-Mamba dataset. No caching.

    Each sample:
    - Input: lookback_days of 1s bars as a flat sequence [T, num_features]
    - Target: next-day VIX close (z-score normalized)
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

        # Index stock files by date
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

    def _build_anchor_dates(self) -> List[Dict]:
        """Build valid anchor dates for this split."""
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
                from datetime import timedelta
                for offset in range(1, 10):
                    candidate = d + timedelta(days=offset)
                    if candidate in vix_dates:
                        target_date = candidate
                        break

            if target_date is None:
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
                'target_date': target_date,
                'target_vix': self.vix_daily[target_date],
            })

        return valid

    def _load_day_bars(self, file_path: Path) -> Optional[np.ndarray]:
        """Load one day of bars as feature matrix. No caching."""
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

        # Sort by timestamp
        if 'bar_timestamp' in df.columns:
            df = df.sort('bar_timestamp')
        elif 'timestamp' in df.columns:
            df = df.sort('timestamp')

        # Extract features
        avail = [f for f in self.features if f in df.columns]
        if not avail:
            return None

        features = df.select(avail).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad missing features
        if len(avail) < self.num_features:
            padded = np.zeros((len(features), self.num_features), dtype=np.float32)
            padded[:, :len(avail)] = features
            features = padded

        # Cap bars per day
        if len(features) > self.max_bars_per_day:
            features = features[:self.max_bars_per_day]

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
        """Iterate over samples - load each file directly, no caching."""
        worker_info = torch.utils.data.get_worker_info()
        all_indices = list(range(len(self.anchor_dates)))
        
        if worker_info is not None:
            per_worker = len(all_indices) // worker_info.num_workers
            wid = worker_info.id
            start = wid * per_worker
            end = start + per_worker if wid < worker_info.num_workers - 1 else len(all_indices)
            all_indices = all_indices[start:end]

        # Shuffle for training
        if self.split == 'train':
            np.random.shuffle(all_indices)

        for idx in all_indices:
            sample = self.anchor_dates[idx]
            window_dates = sample['window_dates']
            target_vix = sample['target_vix']

            # Load bars for all lookback days
            all_bars = []
            for d in window_dates:
                file_path = self.stock_files.get(d)
                if file_path is None:
                    continue
                day_bars = self._load_day_bars(file_path)
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
            target_norm = self._normalize_vix(target_vix)

            yield {
                'bars': torch.from_numpy(bars),
                'vix_target': torch.tensor(target_norm, dtype=torch.float32),
                'num_bars': len(bars),
                'anchor_date': str(sample['anchor_date']),
            }

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
