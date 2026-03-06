# VIX Prediction Architecture

> **Last Updated:** March 2026

## Overview

Pure **Mamba-only** architecture for VIX prediction. No TCN encoder - Mamba handles raw 1-second bar sequences directly.

## Current Implementation (v1)

### Mamba-1: Next-Day VIX Prediction

```
Input: 15,000 1-second bars (~4 hours of trading)
       └─ 6 features per bar: open, high, low, close, volume, vwap

Model: MambaOnlyVIX
       ├─ Input projection: 6 → 256
       ├─ Mamba layers: 4 layers, d_model=256, d_state=64
       └─ Prediction head: 256 → 128 → 1

Output: Next-day VIX close (single scalar)

Parameters: 2.17M
Training: LR=1e-6 (from HP sweep), 50 epochs, all data (2004-2024)
```

### Model Architecture

```python
class MambaOnlyVIX(nn.Module):
    def __init__(self, num_features=6, d_model=256, n_layers=4, d_state=64):
        self.input_proj = nn.Linear(num_features, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1)
        )
```

---

## Planned Extensions

### Option A: Separate Models per Horizon

Train 3 independent Mamba models, each specialized for a different prediction horizon:

| Model | Sequence Length | Target | Status |
|-------|-----------------|--------|--------|
| **Mamba-1** | 15,000 (1s bars) | VIX +1d | Training |
| **Mamba-2** | 50,000+ (1s bars) | VIX +7d | Planned |
| **Mamba-3** | 100,000+ (1s bars) | VIX +30d | Planned |

**Pros:** Each model optimized for its horizon, simpler training
**Cons:** 3x compute, no shared representations

### Option B: Multi-Target Single Model

One Mamba model with longer sequence, predicting all horizons simultaneously:

```
Input: 100,000+ 1-second bars (~28 hours / ~4 trading days)

Model: MambaMultiHorizon
       ├─ Shared Mamba backbone (6-8 layers, d_model=512)
       └─ Multi-head output:
           ├─ VIX +1d head
           ├─ VIX +7d head
           └─ VIX +30d head

Loss: weighted sum of all horizon losses
```

**Pros:** Shared representations, single model, potentially better generalization
**Cons:** Harder to train, may need curriculum learning

---

## System Requirements

**IMPORTANT: WSL (Windows Subsystem for Linux) Required for Training**

The Mamba SSM package requires CUDA compilation with Linux toolchain.

### Setup:
1. Install WSL: `wsl --install -d Ubuntu`
2. Run setup: `bash setupvps.sh` (VPS) or `bash setupenv.sh` (local)
3. Activate: `source /workspace/venv/bin/activate`

### Why WSL:
- Mamba uses CUDA C++ extensions via PyTorch's `CUDAExtension`
- Requires `nvcc` + Linux build tools
- The `.so` files are Linux binaries

---

## Training Configuration

From `trainconfig.py`:

```python
seq_len: int = 351_000      # Max sequence (~6 hours)
epochs: int = 50
batch_size: int = 16
lr: float = 1e-6            # From HP sweep (see docs/lr_sweep_results.md)

d_model: int = 256
n_layers: int = 4
d_state: int = 64

train_end: str = '2024-09-30'  # Use all data
val_end: str = '2024-12-31'
```

---

## Data Pipeline

```
Stock_Data_1s/
├─ 2004/
│  ├─ SPY_2004-01-02.parquet
│  └─ ...
├─ 2005/
│  └─ ...
└─ 2024/

VIX/
├─ vix_daily.parquet        # VIX close prices (targets)
└─ vix_intraday.parquet     # Optional: intraday VIX
```

### Features per 1-second bar:
1. `open` - Opening price
2. `high` - High price
3. `low` - Low price
4. `close` - Closing price
5. `volume` - Trade volume
6. `vwap` - Volume-weighted average price

---

## Hyperparameter Sweep Results

See `docs/lr_sweep_results.md` for full details.

**Winner: LR = 1e-6**
- Best validation loss: 0.1802
- Best validation MAE: 0.5409
- Most stable (no overfitting)

---

## Future Enhancements (Deferred)

These features are planned for later iterations:

### Additional Data Streams
- Options data (IV, skew, term structure)
- Index bars (SPX, NDX, RUT)
- News/sentiment embeddings
- Macro indicators via FiLM conditioning

### Event Memory
- Fed events (FOMC, minutes, speakers)
- GDELT geopolitical events
- Cross-attention for event retrieval

### Architecture Improvements
- Longer sequences (351k+ bars)
- Multi-resolution inputs
- Ensemble of horizons