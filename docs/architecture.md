# VIX Prediction Architecture

> **Last Updated:** March 2026

## Overview

**Multi-stream Mamba architecture** for VIX prediction from 2-minute market data with periodic fusion across stock, options, and news-conditioned streams. GDELT is merged into the news stream, and macro data is applied through FiLM conditioning.

## Current Implementation (v2)

### Architecture Diagram

```
Stock Bars [B, T, 45] ───────────────────────────────→ Stock Mamba ───────────────┐
                                                                                   │
Option Flow [B, T, 47] ─→ OptionEncoder ─→ Option Mamba ───────────────────────────┤
                                                                                   │
News [B, N, 3072] + GDELT [B, M, 391] ─→ Merge + Type Embedding ─→ News Mamba ────┤
                                                                                   │
Macro Context [B, 15+] ─→ FiLM conditioning on stock stream ───────────────────────┤
                                                                                   │
Checkpoint fusion every N bars ─→ gated cross-stream fusion ─→ pooled head ─→ VIX targets [B, 4]
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Stock stream** | Main Mamba stream over 2-minute stock bars |
| **Option stream** | Encodes 47 option-flow features and fuses periodically with stock |
| **News stream** | Processes Benzinga embeddings and optional GDELT tokens in one shared news Mamba |
| **Macro FiLM** | Applies macro conditioning to stock representations |
| **ParallelMambaVIX** | 4 stock/options layers, 2 news layers, `d_model=256`, `d_state=64` |
| **Multi-horizon head** | Predicts VIX change for `+1d`, `+7d`, `+15d`, `+30d` |

### Data Sources

| Source | Features | Date Range | Path |
|--------|----------|------------|------|
| **Stock bars** | 45 | 2005+ | `datasets/Stock_Data_2min/` |
| **Options flow** | 47 | 2014-06+ | `datasets/opt_trade_2min/` |
| **News embeddings** | 3072 | 2009+ | `datasets/benzinga_embeddings/` |
| **GDELT world state** | 391 | configurable | `datasets/GDELT/` |
| **Macro conditioning** | dynamic | configurable | `datasets/macro/` |

### Stock Features (45)
```python
['open', 'high', 'low', 'close', 'volume', 'trade_count',
 'price_mean', 'price_std', 'price_range', 'price_range_pct', 'vwap',
 'avg_trade_size', 'std_trade_size', 'max_trade_size', 'amihud',
 'buy_volume', 'sell_volume', 'signed_trade_count',
 'tick_arrival_rate', 'large_trade_ratio', 'inter_trade_std', 'tick_burst',
 'rv_intrabar', 'bpv_intrabar', 'price_skew',
 'close_vs_vwap', 'high_vs_vwap', 'low_vs_vwap', 'ofi',
 'net_volume', 'volume_imbalance', 'trade_intensity', 'rv_bpv_ratio',
 'close_return', 'volume_per_trade', 'buy_sell_ratio', 'high_low_ratio',
 'close_position', 'day_of_week', 'days_to_friday', 'minute_of_day',
 'is_friday', 'days_to_monthly_expiry', 'days_to_weekly_expiry', 'is_expiration_day']
```

### Option Features (47)
```python
['call_volume', 'put_volume', 'call_trade_count', 'put_trade_count',
 'put_call_ratio_volume', 'put_call_ratio_count',
 'call_premium_total', 'put_premium_total',
 'near_volume', 'mid_volume', 'far_volume',
 'near_pc_ratio', 'far_pc_ratio', 'term_skew',
 'otm_put_volume', 'atm_volume', 'otm_call_volume',
 'skew_proxy', 'atm_concentration', 'deep_otm_put_volume',
 'total_large_trade_count', 'call_large_count', 'put_large_count',
 'net_large_flow', 'large_premium_total', 'sweep_intensity',
 'max_volume_surprise', 'avg_volume_surprise',
 'uoa_call_count', 'uoa_put_count',
 'total_volume', 'total_trade_count', 'unique_contracts', 'unique_strikes',
 'unique_expiries', 'pc_ratio_vs_20d', 'call_volume_vs_20d', 'put_volume_vs_20d',
 'net_premium_flow', 'premium_imbalance', 'large_trade_pct', 'put_large_pct',
 'uoa_total', 'uoa_put_bias', 'deep_otm_put_pct', 'near_far_ratio', 'sweep_to_trade_ratio']
```

### Model Code

```python
class ParallelMambaVIX(nn.Module):
    def __init__(self, num_features=45, d_model=256, n_layers=4, d_state=64,
                 use_news=True, news_dim=3072, news_n_layers=2,
                 use_options=True, option_features=47,
                 use_macro=True, macro_dim=15,
                 use_gdelt=False, gdelt_dim=391):
        ...
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

### Current Default Training Mode

```bash
python train.py
# defaults from trainconfig.py:
# - seq_len=1000  (~5 trading days of 2-min bars)
# - d_model=256
# - n_layers=4
# - news_n_layers=2
# - use_news=True
# - use_options=True
# - use_macro=True
# - use_gdelt=True
```

### Macro Features (25 total from FED/FRED)

Built from `datasets/FED/` via `tools/build_macro_from_fed.py`:

- **Rates**: `DFF`, `DGS3MO`, `DGS2`, `DGS5`, `DGS10`, `DGS30`
- **Yield curve**: `T10Y2Y`, `T10Y3M`, `ff_3m_spread`
- **Market stress**: `VIXCLS`, `BAMLH0A0HYM2`, `BAMLC0A0CM`, `credit_spread`
- **Inflation**: `CPIAUCSL`, `PCEPILFE`
- **Employment**: `UNRATE`, `ICSA`
- **Balance sheet**: `WALCL`
- **Economic activity**: `INDPRO`, `RSAFS`, `HOUST`, `UMCSENT`
- **FOMC calendar**: `days_since_fomc`, `days_until_fomc`, `is_fomc_week`

---

## Planned Extensions

### Implemented 
- **Options flow data** - Market-wide option flow from 2-minute bars (47 features)
- **News embeddings** - Benzinga article embeddings (OpenAI 3072-dim)
- **Parallel Mamba streams** - Stock/news/options with periodic fusion checkpoints
- **GDELT integration** - Merged into the news stream with token-type embeddings
- **Macro FiLM conditioning** - Time-aware FiLM modulation from Fed/treasury/FOMC data
  - 15 macro features (fed funds, yields, CPI, unemployment, FOMC calendar)
  - Per-position gamma/beta via sinusoidal time-of-day encoding
  - Identity init (gamma=1, beta=0), 10x LR for FiLM params
  - Gamma/beta logging to verify macro signal is being used

Train 3 independent Mamba models, each specialized for a different prediction horizon:

| Model | Sequence Length | Target | Status |
|-------|-----------------|--------|--------|
| **Mamba-1** | 1,000 (2-min bars) | VIX +1d | Training |
| **Mamba-2** | 2,000+ (2-min bars) | VIX +7d | Planned |
| **Mamba-3** | 4,000+ (2-min bars) | VIX +30d | Planned |

**Pros:** Each model optimized for its horizon, simpler training
**Cons:** 3x compute, no shared representations

### Option B: Multi-Target Single Model

One Mamba model with longer sequence, predicting all horizons simultaneously:

```
Input: 1,000+ 2-minute bars (~5 trading days)

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
seq_len: int = 1000
epochs: int = 50
batch_size: int = 16
lr: float = 1e-4

d_model: int = 256
n_layers: int = 4
d_state: int = 64
news_n_layers: int = 2
use_news: bool = True
use_options: bool = True
use_macro: bool = True

train_end: str = '2024-09-30'  # Use all data
val_end: str = '2024-12-31'
```

---

## Data Pipeline

```
Stock_Data_2min/
├─ 2024-01-02.parquet
├─ 2024-01-03.parquet
└─ ...

opt_trade_2min/
├─ 2024-01-02.parquet
├─ 2024-01-03.parquet
└─ ...

VIX/
├─ VIX_*.csv                # VIX close extracted from intraday CSVs
└─ ...
```

### Sequence Resolution

- **Stock/options cadence**: 2-minute bars
- **Bars per trading day**: ~195
- **Default sequence length**: 1000 bars
- **Default lookback**: ~5 trading days

---

## Hyperparameter Sweep Results

See `docs/lr_sweep_results.md` for full details.

These results were produced on an earlier configuration and should be treated as historical only. Re-tuning is recommended after the migration to 2-minute stock and options data.

---

## Future Enhancements

### Implemented 
- **Options flow data** - Market-wide option flow from 2-minute bars (47 features)
- **News embeddings** - Benzinga article embeddings (OpenAI 3072-dim)
- **Parallel Mamba streams** - Stock/news/options with periodic fusion checkpoints
- **GDELT integration** - Merged into the news stream with token-type embeddings
- **Macro FiLM conditioning** - Time-aware FiLM modulation from Fed/treasury/FOMC data
  - 15 macro features (fed funds, yields, CPI, unemployment, FOMC calendar)
  - Per-position gamma/beta via sinusoidal time-of-day encoding
  - Identity init (gamma=1, beta=0), 10x LR for FiLM params
  - Gamma/beta logging to verify macro signal is being used

### Planned
- Index bars (SPX, NDX, RUT)
- GDELT geopolitical events
- Longer sequences (2-min multi-week windows)
- Multi-resolution inputs
- Ensemble of horizons