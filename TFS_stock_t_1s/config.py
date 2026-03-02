"""
Configuration for Stock 1s → Transformer → Mamba → VIX Prediction Pipeline.

GPU profiles, Mamba-1/2 configs, data paths, training hyperparameters.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

# ---------------------------------------------------------------------------
# Path detection
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

if platform.system() == 'Linux':
    # WSL: use /mnt/d/ path to Windows data
    DATA_ROOT = Path('/mnt/d/Mamba v2/datasets')
    VIX_ROOT = DATA_ROOT / 'VIX'
    STOCK_ROOT = DATA_ROOT / 'Stock_Data_1s'
else:
    DATA_ROOT = Path(r'D:\Mamba v2\datasets')
    VIX_ROOT = DATA_ROOT / 'VIX'
    STOCK_ROOT = DATA_ROOT / 'Stock_Data_1s'

# ---------------------------------------------------------------------------
# GPU Profiles  (name → {vram_gb, max_frames_per_batch, grad_accum})
# ---------------------------------------------------------------------------
GPU_PROFILES: Dict[str, dict] = {
    'rtx5080': {
        'vram_gb': 16,
        'max_frames_per_batch': 64,
        'grad_accum_steps': 8,
        'num_workers': 4,
    },
    'rtx5090': {
        'vram_gb': 32,
        'max_frames_per_batch': 128,
        'grad_accum_steps': 4,
        'num_workers': 4,
    },
    'a100': {
        'vram_gb': 80,
        'max_frames_per_batch': 256,
        'grad_accum_steps': 2,
        'num_workers': 8,
    },
    'b200': {
        'vram_gb': 180,
        'max_frames_per_batch': 512,
        'grad_accum_steps': 1,
        'num_workers': 8,
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TransformerEncoderConfig:
    """Level 0: Transformer frame encoder for 1s bar data."""
    num_features: int = 15
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    dropout: float = 0.1
    num_tickers: int = 100
    ticker_embed_dim: int = 16
    max_bars_per_frame: int = 500
    frame_interval: str = '5min'
    frames_per_day: int = 78


@dataclass
class MambaLayerConfig:
    """Config for a single Mamba level."""
    n_layers: int = 4
    d_model: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


@dataclass
class Mamba1Config(MambaLayerConfig):
    """Level 1: Mamba-1 short-term dynamics."""
    lookback_days: int = 15          # 15 × 78 = 1,170 steps
    target: str = 'vix_next_day'     # predict next-day VIX close


@dataclass
class Mamba2Config(MambaLayerConfig):
    """Level 2: Mamba-2 long-term regime."""
    lookback_days: int = 365         # 365 daily summary steps
    target: str = 'vix_plus_30d'     # predict VIX +30d close


@dataclass
class DataConfig:
    """Data paths and settings."""
    # Use /mnt/d/ paths when running in WSL, fallback to PROJECT_ROOT for Windows
    stock_data_path: str = '/mnt/d/Mamba v2/datasets/Stock_Data_1s' if os.path.exists('/mnt/d/Mamba v2/datasets') else str(PROJECT_ROOT / 'datasets' / 'Stock_Data_1s')
    vix_data_path: str = '/mnt/d/Mamba v2/datasets/VIX' if os.path.exists('/mnt/d/Mamba v2/datasets') else str(PROJECT_ROOT / 'datasets' / 'VIX')
    allowed_tickers_file: str = str(PROJECT_ROOT / 'TFS_stock_t_1s' / 'scripts' / 'top_100_stocks.txt')

    # Walk-forward splits (30-day gap between each)
    train_start: str = '2016-01-01'
    train_end: str = '2023-11-30'
    val_start: str = '2024-01-01'
    val_end: str = '2024-12-31'
    test_start: str = '2025-02-01'

    # VIX target normalization: 'none', 'zscore', 'log'
    # zscore: (vix - mean) / std
    # log: log(vix)
    vix_normalize: str = 'zscore'
    vix_mean: float = 19.14   # actual mean from VIX data
    vix_std: float = 8.24     # actual std from VIX data

    # Bar features to use
    features: List[str] = field(default_factory=lambda: [
        'close', 'volume', 'trade_count',
        'price_std', 'price_range_pct', 'vwap',
        'avg_trade_size', 'amihud', 'buy_volume', 'sell_volume',
        'tick_arrival_rate', 'large_trade_ratio', 'tick_burst',
        'rv_intrabar', 'ofi',
    ])


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 100
    steps_per_epoch: int = 500
    val_steps: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    clip_grad_norm: float = 1.0
    amp: bool = True
    amp_dtype: str = 'bfloat16'
    seed: int = 42

    # Scheduler
    scheduler: str = 'cosine'        # 'cosine' or 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    reduce_factor: float = 0.5
    schedule_patience: int = 10

    # Early stopping
    early_stopping_patience: int = 20

    # Loss
    loss_name: str = 'huber'         # 'huber', 'mse', 'l1'
    huber_delta: float = 1.0

    # Checkpointing
    checkpoint_dir: str = str(PROJECT_ROOT / 'checkpoints')
    checkpoint_period: int = 5
    resume_path: Optional[str] = None
    log_every: int = 50

    # Gradient accumulation (overridden by GPU profile)
    grad_accum_steps: int = 4
    # Max frames per batch (overridden by GPU profile)
    max_frames_per_batch: int = 64
    num_workers: int = 4


@dataclass
class Config:
    """Top-level configuration."""
    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    mamba1: Mamba1Config = field(default_factory=Mamba1Config)
    mamba2: Mamba2Config = field(default_factory=Mamba2Config)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = 'cuda'


def apply_gpu_profile(config: Config, profile_name: str) -> Config:
    """Apply GPU profile settings to config."""
    if profile_name not in GPU_PROFILES:
        raise ValueError(f"Unknown GPU profile: {profile_name}. "
                         f"Available: {list(GPU_PROFILES.keys())}")
    p = GPU_PROFILES[profile_name]
    config.train.max_frames_per_batch = p['max_frames_per_batch']
    config.train.grad_accum_steps = p['grad_accum_steps']
    config.train.num_workers = p['num_workers']
    return config
