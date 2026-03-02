from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Project root directory (where this config.py lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

@dataclass
class DatasetConfig:
    format: str = 'VIX-Ticks'
    name: str = 'vix_tick_dataset'
    task: str = 'regression'
    raw_ticks: bool = True
    resample_interval: str = '15s'
    chunk_len: int = 256
    weight_mode: str = 'tick_count'
    num_frames: int = 2880
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 7, 15, 30])
    
    # Paths (relative to PROJECT_ROOT)
    indices_path: str = str(PROJECT_ROOT / 'DataTraining' / 'INDEX')
    indices_pattern: str = '**/*.parquet' 
    vix_path: str = str(PROJECT_ROOT / 'datasets' / 'VIX')
    vix_pattern: str = '**/*.csv'
    
    # Split dates
    train_start: str = '2017-01-01'
    train_end: str = '2017-12-31'
    val_start: str = '2018-01-01'
    val_end: str = '2018-12-31'
    test_start: str = '2019-01-01'
    test_end: str = '2019-12-31'

    # Normalization
    norm_enable: bool = True
    norm_median: List[float] = field(default_factory=lambda: [3.0, 5.0])
    norm_iqr: List[float] = field(default_factory=lambda: [1.0, 2.0])
    
    # Streaming  # ADD
    num_workers: int = 4

    def __post_init__(self):
        import platform
        # Convert Windows paths to WSL if on Linux
        if platform.system() == 'Linux':
            for field in ['indices_path', 'vix_path']:
                path = getattr(self, field)
                if path and ':' in path:
                    # e.g. D:/foo -> /mnt/d/foo
                    drive, rest = path.split(':', 1)
                    new_path = f"/mnt/{drive.lower()}{rest}"
                    setattr(self, field, new_path)


@dataclass
class TCNConfig:
    dim_in: int = 3  # price, size, dt
    dim_out: int = 512
    layers: int = 12
    kernel_size: int = 3
    dropout: float = 0.1
    checkpoint_every: int = 0
    use_ticker_emb: bool = True
    ticker_embed_dim: int = 16
    pretrained_path: Optional[str] = None  # Path to pretrained encoder weights

@dataclass
class MambaConfig:
    n_layers: int = 4
    d_model: int = 768
    d_state: int = 256
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1

@dataclass
class RVConfig:
    predict_rv: bool = True
    horizon_min: int = 15
    head_hidden_dim: int = 64
    loss_weight: float = 1.0

@dataclass
class TickerConfig:
    # Map ticker symbol to ID
    tickers: List[str] = field(default_factory=lambda: ['VIX', 'SPY', 'QQQ', 'IWM'])
    # Which tickers to predict RV for
    predict_rv_tickers: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])
    
    def get_id(self, ticker: str) -> int:
        try:
            return self.tickers.index(ticker)
        except ValueError:
            return 0  # Default to 0 if unknown

@dataclass
class ModelConfig:
    dim_in: int = 3  # price, size, dt
    dim_out: int = 4
    
    # Hierarchical Configs
    tcn: TCNConfig = field(default_factory=TCNConfig)
    mamba_short: MambaConfig = field(default_factory=MambaConfig) # Mamba-1 (Short term)
    mamba_long: MambaConfig = field(default_factory=MambaConfig)  # Mamba-2 (Long term)
    rv: RVConfig = field(default_factory=RVConfig)
    tickers: TickerConfig = field(default_factory=TickerConfig)

    # Legacy fields to maintain compatibility during refactor (mapped to new configs in __post_init__ or property)
    # Encoder Streaming
    encoder_streaming: bool = True
    encoder_stream_chunk_size: int = 2048
    encoder_host_streaming_threshold: int = 20000

    # Head
    head_layers: int = 3
    head_dropout: float = 0.1
    head_activation: str = 'silu' 
    head_layer_norm: bool = True
    pooling_type: str = 'attention'
    
    @property
    def d_model(self): return self.mamba_short.d_model


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 4
    amp: bool = True  
    amp_dtype: str = 'bfloat16'  # 'float16' or 'bfloat16'
    epochs: int = 200  
    
    # Loop control (for IterableDataset)
    steps_per_epoch: int = 1000  # Training batches per epoch
    val_steps: int = 200         # Validation batches per epoch
    test_steps: int = 200        # Test batches for final eval
    
    # Reproducibility
    seed: int = 42
    
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-5
    clip_grad_norm: float = 1.0
    
    # Scheduler
    scheduler: str = 'plateau'  # 'cosine' or 'plateau'
    warmup_epochs: int = 10
    min_lr: float = 1e-6  
    reduce_factor: float = 0.1
    schedule_patience: int = 10 
    
    # Early stopping
    early_stopping_patience: int = 50 
    
    # Loss configuration
    loss_name: str = 'asymmetric'  # 'l1', 'mse', 'asymmetric', 'spike', 'combined'
    under_penalty: float = 2.0     # For asymmetric loss
    spike_threshold: float = 25.0  # VIX level considered a spike
    spike_penalty: float = 2.0     # Penalty multiplier for missing spikes
    
    # Gradient Checkpointing
    checkpoint_mode: str = "auto"  # "auto", "off", "all", "every_n"
    checkpoint_every_n_layers: int = 2
    checkpoint_encoder: bool = False

    # Logging/Checkpoints (relative to PROJECT_ROOT)
    out_dir: str = str(PROJECT_ROOT / 'results' / 'vix_mamba')
    ckpt_dir: str = str(PROJECT_ROOT / 'checkpoints')
    log_every: int = 100 
    checkpoint_period: int = 5
    resume_path: Optional[str] = None  # Checkpoint to resume from


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = 'cuda'