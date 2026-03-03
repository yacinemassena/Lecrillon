"""
Training configuration for Mamba-Only VIX Prediction.

Edit these values to customize training without modifying train.py.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data settings
    seq_len: int = 351_000          # Max sequence length in 1-sec bars (~6 hours trading)
    
    # Training settings
    epochs: int = 50                # Number of training epochs
    train_steps: int = 0            # Steps per epoch (0 = full epoch)
    val_steps: int = 0              # Validation steps (0 = full validation)
    batch_size: int = 16            # Batch size (16 good for RTX 5080)
    num_workers: int = 4            # DataLoader workers (4 for VPS with 117GB RAM)
    
    # Model architecture
    d_model: int = 256              # Model hidden dimension
    n_layers: int = 4               # Number of Mamba layers
    d_state: int = 64               # Mamba state dimension
    
    # Data source
    force_synthetic: bool = False   # Force synthetic data
    force_real: bool = False        # Force real data (fail if unavailable)
    
    # Dates
    train_end: str = '2023-11-30'   # End of training data
    val_end: str = '2024-12-31'     # End of validation data


# Default config instance
DEFAULT_CONFIG = TrainConfig()
