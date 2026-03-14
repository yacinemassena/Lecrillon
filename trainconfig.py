"""
Training configuration for Mamba-Only VIX Prediction.

Edit these values to customize training without modifying train.py.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data settings
    seq_len: int = 8000             # Max sequence length in 2-min bars (~41 trading days)
    
    # Training settings
    epochs: int = 50                # Number of training epochs
    train_steps: int = 0            # Steps per epoch (0 = full epoch)
    val_steps: int = 0              # Validation steps (0 = full validation)
    batch_size: int = 16            # Batch size (16 good for RTX 5080)
    num_workers: int = 4            # DataLoader workers (4 for VPS with 117GB RAM)
    
    # Model architecture
    d_model: int = 256              # Model hidden dimension (scaled down for ~5k samples)
    n_layers: int = 4               # Number of Mamba layers (stock/options streams)
    news_n_layers: int = 2          # Fewer layers for sparse news sequences
    d_state: int = 64               # Mamba state dimension
    checkpoint_interval: int = 300  # Fusion checkpoint every N bars (300 = 5 min)
    use_news: bool = True           # Enable news stream by default
    use_options: bool = True        # Enable options stream by default
    use_macro: bool = True          # Enable macro FiLM conditioning
    use_gdelt: bool = True          # Enable GDELT world state integration (default on)
    
    # Data source
    force_synthetic: bool = False   # Force synthetic data
    force_real: bool = False        # Force real data (fail if unavailable)
    
    # Learning rate (1e-4 for multimodal training with news/options injection)
    lr: float = 1e-4
    
    # Dates (2025+ reserved as untouched test set)
    train_end: str = '2024-03-31'   # End of training data
    val_end: str = '2024-12-31'     # End of validation data (9 months = ~190 samples)


# Default config instance
DEFAULT_CONFIG = TrainConfig()
