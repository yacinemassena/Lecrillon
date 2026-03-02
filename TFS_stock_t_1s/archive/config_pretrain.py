"""
Configuration for TCN Pretraining on SPY Realized Volatility.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Project root directory (where this config file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Detect environment: VPS uses /TCN, local uses parent directory
import platform
if platform.system() == 'Linux' and Path('/TCN/datasets').exists():
    # VPS environment
    DATA_ROOT = Path('/TCN/datasets/2017-2025')
else:
    # Local Windows environment - data is in parent directory
    DATA_ROOT = PROJECT_ROOT.parent / 'datasets'


@dataclass
class GPUProfile:
    """GPU profile for VRAM-aware batching."""
    name: str = 'rtx5080'
    vram_gb: int = 16
    max_chunks_per_batch: int = 4000      # 16GB: 2000, 80GB: 11600
    filter_stocks: bool = False            # True for 16GB, False for 80GB
    top_n_stocks: int = 500               # 0 = all stocks
    prefetch_files: int = 8               # Files to keep in RAM


# Predefined GPU profiles
GPU_PROFILES = {
    'rtx5080': GPUProfile(
        name='rtx5080',
        vram_gb=16,
        max_chunks_per_batch=4000,
        filter_stocks=False,
        top_n_stocks=100,
        prefetch_files=8,
    ),
    'rtx5090': GPUProfile(
        name='rtx5090',
        vram_gb=32,
        max_chunks_per_batch=6000,
        filter_stocks=False,
        top_n_stocks=0,
        prefetch_files=12,
    ),
    'h100': GPUProfile(
        name='h100',
        vram_gb=80,
        max_chunks_per_batch=11600,
        filter_stocks=False,
        top_n_stocks=0,
        prefetch_files=16,
    ),
    'a100': GPUProfile(
        name='a100',
        vram_gb=80,
        max_chunks_per_batch=11600,
        filter_stocks=False,
        top_n_stocks=0,
        prefetch_files=16,
    ),
    'amd': GPUProfile(
        name='amd',
        vram_gb=192,
        max_chunks_per_batch=28000,  # 2.4x H100 capacity
        filter_stocks=False,
        top_n_stocks=0,
        prefetch_files=32,  # More RAM available
    ),
}


@dataclass
class StreamConfig:
    """Configuration for a single data stream with VRAM-aware batch sizing."""
    name: str
    data_path: str
    filter_tickers: bool = False
    allowed_tickers_file: Optional[str] = None
    num_tickers: int = 0  # For ticker embedding (0 = disabled)
    # Batch sizing per GPU (chunks per batch) - tuned to tick volume
    # Stocks (filtered): ~8M ticks/day → baseline
    # Options: ~5.4M ticks/day → 0.7x stocks
    # Index: ~347K ticks/day → 0.04x stocks (can fit more frames per batch)
    max_chunks_16gb: int = 2000   # RTX 5080
    max_chunks_32gb: int = 4000   # RTX 5090
    max_chunks_80gb: int = 11600  # H100/A100
    max_chunks_192gb: int = 28000 # AMD MI300X
    prefetch_files: int = 8
    # Stream-specific model architecture (complexity scales with data volume)
    hidden_dim: int = 512
    num_layers: int = 12
    dropout: float = 0.1


# Predefined stream configurations - Stock 1s Bar data only
STREAM_CONFIGS = {
    'stock_1s': StreamConfig(
        name='stock_1s',
        data_path=str(DATA_ROOT / 'Stock_Data_1s'),
        filter_tickers=True,  # Filter to top 100 on 16GB
        allowed_tickers_file=str(DATA_ROOT / 'top_100_stocks.txt'),
        num_tickers=100,  # Ticker embeddings for top 100 stocks
        max_chunks_16gb=4000,   # 1s bars: each bar is a unit (not chunked)
        max_chunks_32gb=8000,   # RTX 5090
        max_chunks_80gb=20000,
        max_chunks_192gb=50000, # AMD MI300X
        prefetch_files=12,
        # Medium model for bar data (simpler than tick data)
        hidden_dim=256,
        num_layers=6,
        dropout=0.1,
    ),
}


@dataclass
class PretrainDataConfig:
    """Dataset configuration for pretraining."""
    # Data paths - uses DATA_ROOT which auto-detects local vs VPS
    stocks_path: str = str(DATA_ROOT / 'polygon_stock_trades')
    options_path: str = str(DATA_ROOT / 'options_trades')
    index_path: str = str(DATA_ROOT / 'index_data')
    rv_file: str = str(DATA_ROOT / 'SPY_daily_rv' / 'spy_daily_rv_1d.parquet')
    top_stocks_file: str = str(DATA_ROOT / 'top_100_stocks.txt')
    
    # Frame settings
    frame_interval: str = '5min'
    chunk_len: int = 256
    num_frames: int = 78  # 6.5 hours / 5min = 78 frames per trading day
    
    # RV target
    rv_horizon_days: int = 1  # 1-day forward RV (next day)
    
    # Split dates
    train_end: str = '2023-12-31'
    val_end: str = '2024-12-31'
    # test: everything after val_end (2025-2026)
    
    # Features - Bar data uses 15 pre-computed features
    dim_in: int = 15  # BAR_FEATURES from bar_dataset.py
    weight_mode: str = 'bar_count'
    
    # Batching (set by GPU profile)
    max_chunks_per_batch: int = 2000
    prefetch_files: int = 8


@dataclass
class PretrainTCNConfig:
    """TCN encoder configuration."""
    dim_in: int = 3
    hidden_dim: int = 512
    num_layers: int = 12
    kernel_size: int = 3
    dropout: float = 0.1
    checkpoint_every: int = 3  # Checkpoint every 3 layers to save memory
    
    # Ticker embedding (disabled for SPY-only pretraining)
    num_tickers: int = 0
    ticker_embed_dim: int = 16


@dataclass
class PretrainRVHeadConfig:
    """RV prediction head configuration."""
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class PretrainTrainConfig:
    """Training configuration."""
    # Batch and accumulation
    batch_size: int = 4
    grad_accum_steps: int = 8
    effective_batch_size: int = 32  # batch_size * grad_accum_steps
    
    # Mixed precision
    amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    # Training duration
    epochs: int = 100
    steps_per_epoch: int = 500
    val_steps: int = 100
    
    # Optimizer
    optimizer: str = 'adamw'
    lr: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # LR Schedule
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Gradient clipping
    clip_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # Loss
    loss_type: str = 'huber'  # 'mse', 'huber', 'l1'
    huber_delta: float = 0.1
    
    # Checkpointing (relative to PROJECT_ROOT)
    checkpoint_dir: str = str(PROJECT_ROOT / 'checkpoints' / 'tcn_pretrain')
    save_every_epochs: int = 10
    
    # Logging
    log_every: int = 1  # Log every batch for full visibility
    wandb_project: Optional[str] = None  # Set to enable W&B logging
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = 'cuda'
    num_workers: int = 4


@dataclass 
class PretrainConfig:
    """Complete pretraining configuration."""
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
    tcn: PretrainTCNConfig = field(default_factory=PretrainTCNConfig)
    rv_head: PretrainRVHeadConfig = field(default_factory=PretrainRVHeadConfig)
    train: PretrainTrainConfig = field(default_factory=PretrainTrainConfig)
    
    def __post_init__(self):
        """Ensure consistency between configs."""
        # TCN input dim should match data dim
        self.tcn.dim_in = self.data.dim_in


def get_pretrain_config(**overrides) -> PretrainConfig:
    """
    Get pretraining config with optional overrides.
    
    Example:
        config = get_pretrain_config(
            data={'spy_data_path': '/path/to/spy'},
            train={'epochs': 50, 'lr': 3e-4}
        )
    """
    config = PretrainConfig()
    
    for section, values in overrides.items():
        if hasattr(config, section) and isinstance(values, dict):
            section_config = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
    
    return config
