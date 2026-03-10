# VIX Prediction Architecture

> **Last Updated:** March 2026

## Overview

**Multi-source Mamba architecture** for VIX prediction with cross-attention CLS fusion. Separate encoders for each data source (stock, options, news) fuse before the Mamba backbone.

## Current Implementation (v2)

### Architecture Diagram

```
Stock Bars [B, T, 15]  ─→ StockEncoder + CLS ─→ [B, T+1, d_model] ─┐
                                                                    │
Option Bars [B, T, 15] ─→ OptionEncoder + CLS ─→ [B, T+1, d_model] ─┼─→ CrossAttentionFusion
                                                                    │         │
News Embs [B, N, 3072] ─→ NewsEncoder + CLS ─→ [B, N+1, d_model] ───┘         │
                                                                              ↓
                                                              Fused CLS → MambaStack → Pool → VIXHead
                                                                                                │
                                                                              Output: VIX daily change [B]
```

### Key Components

| Component | Description |
|-----------|-------------|
| **SourceEncoder** | Projects features to d_model, prepends learnable CLS token |
| **CrossAttentionFusion** | Stock CLS attends to option/news sequences (4 heads) |
| **MambaStack** | 4 layers, d_model=256, d_state=64, selective scan |
| **VIXHead** | MLP regression: 256 → 128 → 1, zero-initialized |

### Data Sources

| Source | Features | Date Range | Path |
|--------|----------|------------|------|
| **Stock bars** | 15 | 2005+ | `datasets/Stock_Data_1s/` |
| **Options flow** | 15 | 2014-06+ | `datasets/opt_trade_1sec/` |
| **News embeddings** | 3072 | 2009+ | `datasets/benzinga_embeddings/` |

### Stock Features (15)
```python
['close', 'volume', 'trade_count', 'price_std', 'price_range_pct', 
 'vwap', 'avg_trade_size', 'amihud', 'buy_volume', 'sell_volume',
 'tick_arrival_rate', 'large_trade_ratio', 'tick_burst', 'rv_intrabar', 'ofi']
```

### Option Features (15 selected from 40)
```python
['put_call_ratio_volume', 'total_volume', 'total_trade_count',
 'term_skew', 'skew_proxy', 'atm_concentration',
 'sweep_intensity', 'net_large_flow',
 'pc_ratio_vs_20d', 'call_volume_vs_20d', 'put_volume_vs_20d',
 'near_pc_ratio', 'far_pc_ratio', 'uoa_call_count', 'uoa_put_count']
```

### Model Code

```python
class MambaOnlyVIX(nn.Module):
    def __init__(self, num_features=15, d_model=256, n_layers=4, d_state=64,
                 use_news=False, news_dim=3072, use_options=False, option_features=15):
        # Separate encoders per source
        self.stock_encoder = SourceEncoder(num_features, d_model)
        if use_options:
            self.option_encoder = SourceEncoder(option_features, d_model)
        if use_news:
            self.news_encoder = SourceEncoder(news_dim, d_model)
        
        # Cross-attention fusion (stock CLS attends to others)
        if use_news or use_options:
            self.fusion = CrossAttentionFusion(d_model, num_heads=4)
        
        # Mamba backbone
        self.mamba = MambaStack(n_layers, d_model, d_state)
        self.pool = SequencePooling(d_model, 'attention')
        self.head = VIXHead(d_model)
```

---

## Automatic Data Integration

Data sources are **automatically included based on sample date** within a single training run:

| Sample Date | Stock | Options | News |
|-------------|-------|---------|------|
| 2005-2008   | ✓     | ✗ (zeros, masked) | ✗ (zeros, masked) |
| 2009-2013   | ✓     | ✗ (zeros, masked) | ✓ |
| 2014+       | ✓     | ✓ | ✓ |

**How it works:**
1. Dataset loads data for each sample's date range
2. Missing sources (pre-2014 options, pre-2009 news) are zero-filled
3. `options_mask` / `news_mask` indicate where real data exists
4. Model checks `mask.sum() > 0` before adding source to cross-attention
5. Samples without data for a source use stock-only path

**Single run trains on all data with automatic source availability:**
```bash
python train.py --use-options --use-news --epochs 50
# 2005-2008: stock only
# 2009-2013: stock + news
# 2014+: stock + options + news
```

### A/B Comparison

```bash
python train.py                          # baseline (stock only)
python train.py --use-options            # stock + options (2014+ has options)
python train.py --use-news               # stock + news (2009+ has news)
python train.py --use-options --use-news # all sources (automatic per-sample)
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

## Future Enhancements

### Implemented 
- **Options flow data** - Put/call ratios, term skew, sweep intensity, UOA (38 features)
- **News embeddings** - Benzinga article embeddings (OpenAI 3072-dim)
- **Cross-attention fusion** - CLS tokens with multi-head attention
- **Parallel Mamba streams** - Stock/news/options with periodic fusion checkpoints
- **Macro FiLM conditioning** - Time-aware FiLM modulation from Fed/treasury/FOMC data
  - 15 macro features (fed funds, yields, CPI, unemployment, FOMC calendar)
  - Per-position gamma/beta via sinusoidal time-of-day encoding
  - Identity init (gamma=1, beta=0), 10x LR for FiLM params
  - Gamma/beta logging to verify macro signal is being used

### Planned
- Index bars (SPX, NDX, RUT)
- GDELT geopolitical events
- Longer sequences (351k+ bars)
- Multi-resolution inputs
- Ensemble of horizons