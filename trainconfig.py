"""
Training configuration for Mamba-Only VIX Prediction.

Edit these values to customize training without modifying train.py.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data settings
    seq_len: int = 351_000          # Max sequence length in 1-sec bars (~6 hours trading)
    lookback_days: int = 15         # Days of historical data per sample
    
    # Training settings
    epochs: int = 50                # Number of training epochs
    train_steps: int = 0            # Steps per epoch (0 = full epoch)
    val_steps: int = 0              # Validation steps (0 = full validation)
    batch_size: int = 16            # Batch size (16 good for RTX 5080)
    num_workers: int = 12           # DataLoader workers for parallel loading
    
    # Model architecture
    d_model: int = 256              # Model hidden dimension
    n_layers: int = 4               # Number of Mamba layers
    d_state: int = 64               # Mamba state dimension
    
    # Cache settings (for WSL/slow disk)
    cache_gb: float = 80.0          # RAM cache size in GB
    use_cache: bool = False         # Use chunked cache (set True for WSL)
    
    # Data source
    force_synthetic: bool = False   # Force synthetic data
    force_real: bool = False        # Force real data (fail if unavailable)
    
    # Dates
    train_end: str = '2023-11-30'   # End of training data
    val_end: str = '2024-12-31'     # End of validation data


# Default config instance
DEFAULT_CONFIG = TrainConfig()
