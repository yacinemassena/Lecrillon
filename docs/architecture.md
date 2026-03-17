# VIX Prediction Architecture

> **Last Updated:** March 17, 2026

## Overview

**Multi-stream Mamba architecture** for VIX prediction from 2-minute market data with periodic fusion across stock, options, news, and VIX streams. GDELT is merged into the news stream, macro data is applied through FiLM conditioning, and VIX features provide extended-hours market context.

## Current Implementation (v2)

### Architecture Diagram

```
Stock Bars [B, T, 51] ───────────────────────────────→ Stock Mamba (d=256) ────────┐
                                                                                   │
Option Flow [B, T, 48] ─→ OptionEncoder ─→ Option Mamba (d=256) ───────────────────┤
                                                                                   │
News [B, N, 3072] + GDELT [B, M, 391] ─→ Merge + Type Embed ─→ News Mamba (d=256) ─┤
                                                                                   │
VIX Features [B, V, 25] ─→ VixEncoder ─→ VIX Mamba (d=64) ─→ proj ─────────────────┤
                                                                                   │
Macro Context [B, 55] ─→ FiLM conditioning on stock stream ────────────────────────┤
                                                                                   │
Fundamentals [B, 130] ─→ Linear projection ─→ cross-attention state ───────────────┤
                                                                                   │
Checkpoint fusion every N bars ─→ gated cross-stream fusion ─→ pooled head ─→ VIX targets [B, 4]
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Stock stream** | Main Mamba stream over 2-minute stock bars (d=256, 4 layers) |
| **Option stream** | Encodes 48 option-flow features, fuses with stock (d=256, 4 layers) |
| **News stream** | Benzinga + GDELT + Econ tokens merged by timestamp (d=256, 2 layers) |
| **VIX stream** | Extended hours VIX/VVIX features (25 dims), ~540 bars/day (d=64, 2 layers) |
| **Macro FiLM** | 55-feature time-aware conditioning on stock representations |
| **Fundamentals** | 130-dim sector state via cross-attention |
| **Multi-horizon head** | Predicts VIX change for `+1d`, `+7d`, `+15d`, `+30d` |

### Data Sources

| Source | Features | Date Range | Path |
|--------|----------|------------|------|
| **Stock bars** | 51 | 2005+ | `datasets/Stock_Data_2min/` |
| **Options flow** | 48 | 2014-06+ | `datasets/opt_trade_2min/` |
| **News embeddings** | 3072 | 2009+ | `datasets/benzinga_embeddings/` |
| **GDELT world state** | 391 | configurable | `datasets/GDELT/` |
| **VIX features** | 25 | 2005+ | `datasets/VIX/Vix_features/` |
| **Macro conditioning** | 55 | 2000+ | `datasets/MACRO/macro_daily_enhanced.parquet` |
| **Econ calendar** | 13 | 2007+ | `datasets/econ_calendar/` |
| **Fundamentals state** | 130 | 2010+ | `datasets/fundamentals/fundamentals_state.parquet` |
| **Cross-asset** | ~30 | 2000+ | `datasets/cross_asset/` (optional, via FRED API) |

### Stock Features (51)
```python
# Core OHLCV + microstructure (46 from parquet)
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
 'is_friday', 'days_to_monthly_expiry', 'days_to_weekly_expiry', 'is_expiration_day',

# Derived features (computed on-the-fly in dataloader, 5 new)
 'liquidity_stress',    # zscore(amihud) + zscore(inter_trade_std) - zscore(tick_arrival_rate)
 'ofi_acceleration',    # OFI(t) - OFI(t-10)
 'abs_ofi',             # |OFI|
 'intraday_vol_skew',   # rv_last_hour / rv_first_hour
 'ticker_dispersion']   # std(close_return) across tickers per timestamp
```

### VIX Features (21)

Extended hours VIX/VVIX features (~540 bars/day, 04:00-22:00 ET):

```python
['open', 'high', 'low', 'close',           # VIX OHLC
 'vvix',                                    # VVIX close (market hours only, ~18% coverage)
 '5dMA', '10dMA', '20dMA',                  # Moving averages (trading bar windows)
 'rv_5m', 'rv_30m', 'rv_2h',                # Realized volatility windows
 'rv_acceleration', 'rv_change_5', 'rv_change_30', 'rv_ratio',  # RV dynamics
 'vix_vvix_ratio',                          # VIX/VVIX relationship
 'vix_zscore_20d', 'vix_percentile_252d',   # VIX level context
 'distance_from_20dMA',                     # Mean reversion signal
 'vix_velocity_15', 'vix_velocity_75',      # 30min and 2.5h momentum
 'vix_acceleration_15', 'vix_acceleration_75',  # Second derivatives
 'rv_ratio_to_vix']                         # Variance risk premium proxy (SPY RV / VIX)
```

**VIX Mamba Architecture**:
- **d_model=64** (smaller than stock's 256, since only 21 features)
- **d_state=16** (smaller state dimension)
- **2 layers** (lightweight)
- Projects to main d_model (256) via linear layer for fusion

**Extended Hours Processing**:
- VIX trades 04:00-22:00 ET (~18h vs stock's 6.5h)
- ~540 bars/day vs stock's 195 bars
- At each checkpoint, VIX Mamba processes all VIX bars within timestamp window
- **Pre-market accumulation**: First checkpoint (09:30) processes ALL overnight VIX bars

### Option Features (48)
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
 'uoa_total', 'uoa_put_bias', 'deep_otm_put_pct', 'near_far_ratio', 'sweep_to_trade_ratio',

# Derived (computed on-the-fly)
 'skew_change']          # skew_proxy(t) - skew_proxy(t-30)
```

### Model Code

```python
class ParallelMambaVIX(nn.Module):
    def __init__(self, num_features=51, d_model=256, n_layers=4, d_state=64,
                 use_news=True, news_dim=3072, news_n_layers=2,
                 use_options=True, option_features=48,
                 use_macro=True, macro_dim=58,  # 54 base + 4 derived
                 use_gdelt=False, gdelt_dim=391,
                 use_vix_features=True, vix_features_dim=21, vix_n_layers=2,
                 vix_d_model=64, vix_d_state=16,
                 use_fundamentals=True, fundamentals_dim=130):
        ...
```

---

## Automatic Data Integration

Data sources are **automatically included based on sample date** within a single training run:

| Sample Date | Stock | Options | News |
|-------------|-------|---------|------|
| 2003-2008   | ✓     | ✗ (zeros, masked) | ✗ (zeros, masked) |
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
# 2003-2008: stock only
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
# - vix_n_layers=2
# - use_news=True
# - use_options=True
# - use_macro=True (auto-detects enhanced macro)
# - use_gdelt=True
# - use_vix_features=True
```

### Enhanced Macro Features (58 total from FED/FRED)

Built from `datasets/FED/` via `tools/build_enhanced_macro.py` + derived features:

- **Treasury yields**: `DGS1MO`, `DGS3MO`, `DGS2`, `DGS5`, `DGS10`, `DGS30`
- **Yield curve**: `yield_2s10s`, `yield_3m10y`, `yield_curve_steepness`, `T10Y2Y`, `T10Y3M`
- **Yield curve inversions**: `yc_2s10s_inverted`, `yc_3m10y_inverted`
- **Fed funds rates**: `DFF`, `DFEDTARL`, `DFEDTARU`, `FEDFUNDS`, `IORB`, `fed_funds_vs_target`
- **Inflation**: `CPIAUCSL`, `CPILFESL`, `PCEPI`, `PCEPILFE`, `MICH` + YoY changes
- **Employment**: `UNRATE`, `PAYEMS`, `ICSA`, `JTSJOL`, `CLF16OV`, `ICSA_4wma`, `ICSA_vs_4wma`
- **Balance sheet**: `WALCL`, `WSHOSHO`, `WRESBAL`, `RESPPLLOPNWW`, `WALCL_mom`
- **Market stress**: `VIXCLS`, `BAMLH0A0HYM2` (HY OAS), `BAMLC0A0CM` (IG OAS), `STLFSI4`
- **Economic activity**: `INDPRO`, `RSAFS`, `HOUST`, `UMCSENT`
- **FOMC calendar**: `days_since_fomc`, `days_until_fomc`, `is_fomc_week`
- **Spreads**: `ff_3m_spread`

**Derived features (computed on-the-fly in dataloader):**
- `credit_spread`: `BAMLH0A0HYM2 - BAMLC0A0CM` (HY-IG spread)
- `yield_curve_velocity`: `T10Y2Y(t) - T10Y2Y(t-5)` (5-day change)
- `stlfsi4_change`: `STLFSI4(t) - STLFSI4(t-1)` (daily Δ stress index)
- `fomc_proximity`: `exp(-days_until_fomc / 5)` (exponential decay)

**Data shift**: All macro features are shifted by T-1 to avoid lookahead bias.

### Cross-Asset Data (Optional)

Downloaded from FRED via `tools/download_cross_asset.py` (requires `FRED_API_KEY`):

- **Commodities**: Gold price, oil (WTI), gold/oil momentum
- **Credit spreads**: HY OAS, IG OAS, BBB spread, AAA spread, HY-IG differential
- **Real yields**: TIPS 5Y/10Y, breakeven inflation
- **Market stress**: VIX, STL Financial Stress Index, Chicago Fed NFCI
- **Dollar/FX**: DXY index, USD momentum
- **Derived**: Z-scores, momentum (5d/20d), regime flags

---

## Fundamentals Architecture

Comprehensive fundamentals integration with **130-dim cross-attention state** and **event tokens** for mega-caps.

### Cross-Attention State (130 dims, updates daily)

```
fundamentals_state.parquet (3,794 daily files)
├── Income/Earnings (~60 dims)
│   ├── Revenue growth by sector (QoQ)
│   ├── Margin trends (gross, operating, net) by sector
│   └── Profitability rate by sector
├── Balance Sheet/Leverage (~30 dims)
│   ├── Debt/equity ratio by sector
│   ├── Current ratio by sector
│   └── Cash ratio by sector
└── Short Interest (~10 dims)
    ├── SI level (days to cover) by sector
    └── SI change by sector
```

**IMPORTANT**: Uses `filing_date` (when data became public), NOT `period_date` (leakage prevention).

### News Mamba Tokens (Event-Driven)

| Token Type | Files | Trigger |
|------------|-------|---------|
| **Mega-cap earnings** | 771 | Top 50 quarterly filings |
| **Mega-cap SI changes** | 192 | SI shifts >10% |

**Earnings token features**: `ticker`, `sector`, `revenue_z`, `eps_z`, `gross_margin`, `net_margin`

**SI token features**: `ticker`, `sector`, `si_level_z`, `si_change`, `si_percentile`

### Data Sources

| Source | Coverage | Cadence | Used For |
|--------|----------|---------|----------|
| `income_statements.csv` | 2010-2026 | Quarterly | Margins, profitability |
| `balance_sheets.csv` | 2010-2026 | Quarterly | Leverage, liquidity |
| `cash_flow_statements.csv` | 2010-2026 | Quarterly | FCF, buybacks |
| `short_interest.csv` | 2017-2025 | Bi-weekly | SI aggregates + tokens |

### Build Scripts

```bash
python tools/build_fundamentals_state.py  # Main builder
```

### Output Files

```
datasets/fundamentals/
├── fundamentals_state.parquet       # 130-dim daily state
├── state_daily/YYYY-MM-DD.parquet   # Daily cross-attention files
├── earnings_tokens/YYYY-MM-DD.parquet  # Mega-cap earnings events
└── si_tokens/YYYY-MM-DD.parquet     # Mega-cap SI change events
```

---

## Earnings Data Architecture

Dual-path earnings integration for capturing both **events** (individual reports) and **state** (sector aggregates):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EARNINGS DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SEC EDGAR (19K companies)                                                   │
│  └─ companyfacts/*.json ─────────┐                                          │
│  └─ submissions/*.json ──────────┼──→ build_sec_fundamentals.py             │
│                                  │                                          │
│  Polygon REST API                │                                          │
│  └─ rest_data/income_statements ─┤                                          │
│  └─ rest_data/balance_sheets ────┤                                          │
│  └─ rest_data/cash_flow_statements                                          │
│                                  ↓                                          │
│                    ┌─────────────────────────────┐                          │
│                    │  company_fundamentals.parquet│ (all metrics, all cos)  │
│                    └─────────────────────────────┘                          │
│                                  │                                          │
│                    ┌─────────────┴─────────────┐                            │
│                    ↓                           ↓                            │
│     ┌──────────────────────────┐  ┌──────────────────────────┐              │
│     │  EARNINGS TOKENS (events) │  │  SECTOR STATE (context)  │              │
│     │  Top 50 mega-caps         │  │  All 19K companies       │              │
│     │  → News Mamba stream      │  │  → Cross-attention state │              │
│     │  Daily: YYYY-MM-DD.parquet│  │  Daily: sector_daily/    │              │
│     └──────────────────────────┘  └──────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Earnings Tokens (News Mamba Stream)

**Purpose**: Capture immediate market reaction to individual earnings events.

| Field | Description |
|-------|-------------|
| `ticker` | Company ticker |
| `timestamp` | Filing datetime (Unix seconds) |
| `embedding` | Normalized fundamentals vector (15-dim) |
| `revenues`, `net_income`, `eps` | Raw values |
| `revenues_yoy`, `net_income_yoy` | YoY surprise |

**Model integration**: Fed into news Mamba as tokens alongside Benzinga/GDELT.

### Sector State (Cross-Attention)

**Purpose**: Provide broader context that persists across sequence.

| Feature | Description |
|---------|-------------|
| `n_reported` | Companies reported in sector |
| `total_revenue`, `avg_revenue` | Sector revenue aggregates |
| `total_net_income`, `pct_profitable` | Earnings aggregates |
| `avg_gross_margin`, `avg_net_margin` | Margin aggregates |

**Model integration**: Cross-attention conditioning at checkpoint intervals (every 300 steps).

### Raw Data Sources

#### SEC EDGAR (`datasets/MACRO/sec_data/`)
- **companyfacts/** - 19K JSON files with US-GAAP metrics per company
- **submissions/** - 951K JSON files with CIK→ticker mapping + filing metadata
- **Excluded from upload** - raw files not needed after processing

#### Polygon REST API (`datasets/MACRO/rest_data/`)
- Pre-processed financial statements from Polygon.io:
  - `income_statements/` - Revenue, net income, EPS, margins
  - `balance_sheets/` - Assets, liabilities, equity
  - `cash_flow_statements/` - Operating, investing, financing flows
  - `ratios/` - P/E, P/B, ROE, ROA
  - `short_interest/`, `short_volume/` - Short selling data
  - `dividends/`, `splits/` - Corporate actions
  - `treasury_yields/`, `inflation/` - Macro indicators
- **Cleaner format** than SEC EDGAR, easier to parse
- **Excluded from upload** - use processed parquets instead

---

## Data Normalization

All inputs are normalized to prevent gradient scale imbalances across different data sources.

| Source | Method | Details |
|--------|--------|---------|
| **Stock bars** | Per-sequence z-score | `(x - μ) / σ` computed over sequence |
| **Options** | Per-sequence z-score | Same as stock, with mask for valid data |
| **VIX features** | Per-sequence z-score | Same as stock, with mask for valid data |
| **News embeddings** | Pre-normalized | OpenAI embeddings already unit-normalized |
| **GDELT embeddings** | Pre-normalized | MiniLM embeddings already unit-normalized |
| **Macro (FiLM)** | Global z-score | Stats from training period only |
| **Fundamentals** | Global z-score | Stats from training period only |
| **Econ calendar** | Field-specific scaling | See below |
| **VIX targets** | Raw points | Intentionally unscaled |

### Macro & Fundamentals Global Z-Score

```python
# During dataset init (training period only)
train_mask = data.index <= train_end
self.macro_mean = train_data.mean(axis=0)
self.macro_std = train_data.std(axis=0) + 1e-8

# At load time
macro_normalized = (macro_vec - self.macro_mean) / self.macro_std
```

This prevents features like Fed funds rate (5.0), credit spread (0.04), and STLFSI (-1.5) from having wildly different gradient contributions.

### Econ Calendar Field Scaling

| Field | Raw Range | Scaling | Output Range |
|-------|-----------|---------|--------------|
| `impact_ord` | 0-3 | `/3.0` | [0, 1] |
| `days_until` | ~-40 to +15 | `/15.0` | [~-3, 1] |
| `time_of_day` | 0-24 | `/24.0` | [0, 1] |
| `actual_z`, `forecast_z`, `previous_z` | z-scored | none | [~-3, 3] |
| Binary flags | 0-1 | none | [0, 1] |

---

## Planned Extensions

### Implemented 
- **Options flow data** - Market-wide option flow from 2-minute bars (48 features)
- **News embeddings** - Benzinga article embeddings (OpenAI 3072-dim)
- **Parallel Mamba streams** - Stock/news/options/VIX with periodic fusion checkpoints
- **GDELT integration** - Merged into the news stream with token-type embeddings
- **Enhanced Macro FiLM conditioning** - 58 features from FED/FRED data + 4 derived
- **VIX Mamba stream** - Dedicated stream for VIX features (21 dims) including extended hours
- **Cross-asset data pipeline** - Gold, bonds, credit spreads from FRED API
- **Derived stock features** - liquidity_stress, ofi_acceleration, abs_ofi, intraday_vol_skew, ticker_dispersion
- **Derived option features** - skew_change
- **Global z-score normalization** - Macro and fundamentals normalized using training period stats

### Planned
- Index bars (SPX, NDX, RUT)
- Longer sequences (2-min multi-week windows)
- Multi-resolution inputs
- Ensemble of horizons

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
vix_n_layers: int = 2
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

### Planned
- Index bars (SPX, NDX, RUT)
- Longer sequences (2-min multi-week windows)
- Multi-resolution inputs
- Ensemble of horizons

---

## Build Tools

| Tool | Purpose |
|------|---------|
| `tools/build_enhanced_macro.py` | Merge FED data into enhanced macro parquet (54 features) |
| `tools/download_cross_asset.py` | Download cross-asset data from FRED API |
| `tools/build_intraday_rv.py` | Add rolling RV features to stock bars |
| `tools/build_vix_features.py` | Build VIX feature parquets from raw CSVs |
| `tools/aggregate_2min.py` | Aggregate 1-second bars to 2-minute bars |
| `tools/build_fundamentals_state.py` | Build fundamentals cross-attention state |
| `tools/preprocess_dataset.py` | Build preprocessed numpy memmaps for fast loading |