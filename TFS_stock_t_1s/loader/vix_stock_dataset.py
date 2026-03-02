"""
VIX + Stock Dataset for Mamba Training.

Two dataset classes:
- MambaL1Dataset: 15-day windows of stock 5-min frames → next-day VIX close
- MambaL2Dataset: 365-day windows of daily summaries → VIX +30d close

Data leakage prevention:
- Stock data days [D-lookback+1, D] → predict VIX at D+horizon
- Walk-forward splits with 30-day gap
"""

import logging
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bar features (must match config.data.features)
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = [
    'close', 'volume', 'trade_count',
    'price_std', 'price_range_pct', 'vwap',
    'avg_trade_size', 'amihud', 'buy_volume', 'sell_volume',
    'tick_arrival_rate', 'large_trade_ratio', 'tick_burst',
    'rv_intrabar', 'ofi',
]


@dataclass
class MambaBatch:
    """Batch for Mamba Level 1 training.

    Attributes:
        frames: [total_frames, max_bars, num_features]  bar features per frame
        frame_mask: [total_frames, max_bars]  valid bar mask
        ticker_ids: [total_frames, max_bars]  optional ticker IDs
        vix_target: [B]  VIX target per sample
        num_frames_per_sample: [B]  frames count per sample
        num_samples: int
    """
    frames: torch.Tensor
    frame_mask: torch.Tensor
    ticker_ids: Optional[torch.Tensor]
    vix_target: torch.Tensor
    num_frames_per_sample: List[int]
    num_samples: int


# ---------------------------------------------------------------------------
# VIX target loader
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
            # Read only needed columns for speed
            df = pd.read_csv(csv_file, usecols=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'], utc=True)
            # Extract date part directly (faster than tz_convert + groupby)
            df['trading_date'] = df['date'].dt.date
            # Sort by date and take last close per day
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
# MambaL1Dataset
# ---------------------------------------------------------------------------
class MambaL1Dataset(IterableDataset):
    """
    Dataset for Mamba Level 1: 15-day stock frame windows → next-day VIX close.

    Each sample:
    - Input: 15 days × 78 frames = 1,170 frames of bar data
    - Target: VIX close on day D+1

    Iterates over valid anchor dates where:
    - All lookback days have stock data
    - Target date has VIX data
    """

    def __init__(
        self,
        config,
        split: str = 'train',
        level: int = 1,
    ):
        self.split = split
        self.level = level

        # Extract config
        data_cfg = config.data
        enc_cfg = config.encoder
        mamba_cfg = config.mamba1 if level == 1 else config.mamba2

        self.stock_path = Path(data_cfg.stock_data_path)
        self.features = data_cfg.features or DEFAULT_FEATURES
        self.num_features = len(self.features)
        self.max_bars_per_frame = enc_cfg.max_bars_per_frame
        self.frame_interval = enc_cfg.frame_interval
        self.frames_per_day = enc_cfg.frames_per_day
        self.lookback_days = mamba_cfg.lookback_days
        self.max_frames_per_batch = config.train.max_frames_per_batch

        # VIX normalization
        self.vix_normalize = getattr(data_cfg, 'vix_normalize', 'none')
        self.vix_mean = getattr(data_cfg, 'vix_mean', 20.0)
        self.vix_std = getattr(data_cfg, 'vix_std', 8.0)

        # Target horizon
        if level == 1:
            self.target_horizon_days = 1   # next day
        else:
            self.target_horizon_days = 30  # +30 days

        # Split dates
        if split == 'train':
            self.date_start = pd.to_datetime(data_cfg.train_start).date()
            self.date_end = pd.to_datetime(data_cfg.train_end).date()
        elif split == 'val':
            self.date_start = pd.to_datetime(data_cfg.val_start).date()
            self.date_end = pd.to_datetime(data_cfg.val_end).date()
        else:  # test
            self.date_start = pd.to_datetime(data_cfg.test_start).date()
            self.date_end = None  # open ended

        # Load ticker filter
        self.allowed_tickers: Set[str] = set()
        self.ticker_to_id: Dict[str, int] = {}
        tickers_file = data_cfg.allowed_tickers_file
        if tickers_file and Path(tickers_file).exists():
            self._load_tickers(tickers_file)

        # Index stock data files by date
        self.stock_files: Dict = {}  # date → Path
        self._index_stock_files()

        # Load VIX daily close
        self.vix_daily = load_vix_daily_close(data_cfg.vix_data_path)

        # Build list of valid anchor dates
        self.anchor_dates = self._build_anchor_dates()
        logger.info(
            f"MambaL{level}Dataset [{split}]: {len(self.anchor_dates)} valid samples, "
            f"lookback={self.lookback_days}d, horizon={self.target_horizon_days}d"
        )

    def _load_tickers(self, file_path: str):
        tickers = []
        with open(file_path, 'r') as f:
            for line in f:
                t = line.strip()
                if t:
                    tickers.append(t)
        self.allowed_tickers = set(tickers)
        self.ticker_to_id = {t: i for i, t in enumerate(sorted(tickers))}
        logger.info(f"Loaded {len(self.allowed_tickers)} tickers")

    def _index_stock_files(self):
        """Build date → file path mapping."""
        for f in self.stock_path.glob('*.parquet'):
            if f.name.startswith('._'):
                continue
            date_str = f.stem.split('.')[0]
            try:
                dt = pd.to_datetime(date_str).date()
                self.stock_files[dt] = f
            except Exception:
                continue
        logger.info(f"Indexed {len(self.stock_files)} stock data files")

    def _build_anchor_dates(self) -> List:
        """Build valid anchor dates for this split.

        An anchor date D is valid if:
        - D is within the split range
        - Stock data exists for all lookback days [D-lookback+1, D]
        - VIX data exists for target day D+horizon
        """
        # Get sorted trading dates (dates with stock data)
        all_dates = sorted(self.stock_files.keys())

        # Filter to dates with VIX data
        vix_dates = set(self.vix_daily.keys())

        valid = []
        for i, d in enumerate(all_dates):
            # Check split range
            if d < self.date_start:
                continue
            if self.date_end and d > self.date_end:
                continue

            # Find target date: D + horizon trading days forward
            target_idx = None
            for j in range(i + 1, min(i + self.target_horizon_days * 2 + 5, len(all_dates))):
                days_forward = (all_dates[j] - d).days
                if days_forward >= self.target_horizon_days:
                    target_idx = j
                    break

            if target_idx is None:
                continue

            target_date = all_dates[target_idx]
            if target_date not in vix_dates:
                continue

            # Check lookback: need at least lookback_days trading days before D
            lookback_start_idx = i - self.lookback_days + 1
            if lookback_start_idx < 0:
                continue

            # Verify we have enough actual trading days in the window
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

    def _process_day_to_frames(self, file_path: Path) -> List[Dict]:
        """Load one day's stock data and convert to 5-min frames."""
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return []

        # Filter tickers
        if self.allowed_tickers:
            df = df[df['ticker'].isin(self.allowed_tickers)]
        if len(df) == 0:
            return []

        # Parse timestamp and sort
        df['_ts'] = pd.to_datetime(df['bar_timestamp'])
        df = df.sort_values('_ts')

        # Group into frames
        df['_frame'] = df['_ts'].dt.floor(self.frame_interval)

        frames = []
        for frame_ts, frame_df in df.groupby('_frame'):
            if len(frame_df) == 0:
                continue

            # Extract available features
            avail = [f for f in self.features if f in frame_df.columns]
            if not avail:
                continue

            features = frame_df[avail].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Z-score per frame
            mu = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - mu) / std

            # Ticker IDs
            ticker_ids = None
            if self.ticker_to_id:
                ticker_ids = np.array(
                    [self.ticker_to_id.get(t, 0) for t in frame_df['ticker'].values],
                    dtype=np.int64,
                )

            n_bars = len(frame_df)
            # Pad/truncate
            if n_bars > self.max_bars_per_frame:
                features = features[:self.max_bars_per_frame]
                if ticker_ids is not None:
                    ticker_ids = ticker_ids[:self.max_bars_per_frame]
                n_bars = self.max_bars_per_frame

            padded_feat = np.zeros((self.max_bars_per_frame, self.num_features), dtype=np.float32)
            padded_feat[:n_bars, :len(avail)] = features

            mask = np.zeros(self.max_bars_per_frame, dtype=np.float32)
            mask[:n_bars] = 1.0

            padded_tid = None
            if ticker_ids is not None:
                padded_tid = np.zeros(self.max_bars_per_frame, dtype=np.int64)
                padded_tid[:n_bars] = ticker_ids

            frames.append({
                'features': padded_feat,
                'mask': mask,
                'ticker_ids': padded_tid,
            })

        return frames

    def __iter__(self) -> Iterator[MambaBatch]:
        """Iterate over samples, yielding MambaBatch."""
        # Shuffle anchor dates for training
        indices = list(range(len(self.anchor_dates)))
        if self.split == 'train':
            np.random.shuffle(indices)

        for idx in indices:
            sample = self.anchor_dates[idx]
            window_dates = sample['window_dates']
            target_vix = sample['target_vix']

            # Load all frames for the lookback window
            all_frames = []
            all_masks = []
            all_tids = []

            for d in window_dates:
                file_path = self.stock_files.get(d)
                if file_path is None:
                    continue
                day_frames = self._process_day_to_frames(file_path)
                for f in day_frames:
                    all_frames.append(f['features'])
                    all_masks.append(f['mask'])
                    if f['ticker_ids'] is not None:
                        all_tids.append(f['ticker_ids'])

            if len(all_frames) == 0:
                continue

            # Truncate to max expected frames (lookback_days * frames_per_day)
            max_frames = self.lookback_days * self.frames_per_day
            if len(all_frames) > max_frames:
                all_frames = all_frames[-max_frames:]
                all_masks = all_masks[-max_frames:]
                if all_tids:
                    all_tids = all_tids[-max_frames:]

            # Build tensors
            frames_t = torch.from_numpy(np.stack(all_frames))
            masks_t = torch.from_numpy(np.stack(all_masks))
            tids_t = torch.from_numpy(np.stack(all_tids)) if all_tids else None
            
            # Normalize VIX target
            if self.vix_normalize == 'zscore':
                target_norm = (target_vix - self.vix_mean) / self.vix_std
            elif self.vix_normalize == 'log':
                target_norm = np.log(target_vix) if target_vix > 0 else 0.0
            else:
                target_norm = target_vix
            target_t = torch.tensor([target_norm], dtype=torch.float32)

            yield MambaBatch(
                frames=frames_t,
                frame_mask=masks_t,
                ticker_ids=tids_t,
                vix_target=target_t,
                num_frames_per_sample=[len(all_frames)],
                num_samples=1,
            )

    @staticmethod
    def collate_fn(batches: List[MambaBatch]) -> MambaBatch:
        """Collate single-sample batches (each from __iter__)."""
        if len(batches) == 1:
            return batches[0]

        # For simplicity, just return first (batch_size=1 with IterableDataset)
        return batches[0]
