# Changelog

## 2026-03-02 - Stock → Transformer → Mamba → VIX Pipeline (WSL Required)

### Major Architecture Change
- **Skipped pretraining**: TCN/Transformer pretraining on RV prediction proved ineffective (too complex for next-day RV, too easy for autoregression on 5-min RV)
- **End-to-end training**: Stock 1s bars → Transformer (Level 0) → Mamba-1 (Level 1, 15-day lookback) → Mamba-2 (Level 2, 365-day lookback) → VIX prediction
- **Single horizon per level**: Level 1 predicts next-day VIX close, Level 2 predicts VIX +30d close

### New Pipeline Components
- **`TFS_stock_t_1s/config.py`**: Unified config with GPU profiles (rtx5080/5090/a100/b200), Mamba-1/2 configs, VIX paths
- **`TFS_stock_t_1s/loader/vix_stock_dataset.py`**: MambaL1Dataset (15-day windows → next-day VIX) with leakage prevention
- **`TFS_stock_t_1s/mamba_model.py`**: StockMambaL1 (Transformer → Mamba-1 → VIX head), StockMambaL2 (daily summaries → Mamba-2 → VIX +30d)
- **`TFS_stock_t_1s/train_mamba.py`**: Full training script with DDP, GPU profiles, AMP bfloat16, early stopping, VPS data validation
- **`TFS_stock_t_1s/hyperparam_search_mamba.py`**: Optuna multi-GPU search (up to 8 GPUs parallel)

### Architecture Details
- **Level 0 (Transformer)**: 1s bar data → 5-min frames (78/day) → Transformer encoder (6L, 256 dim, 4 heads) → frame embeddings [256]
- **Level 1 (Mamba-1)**: 15 days × 78 frames = 1,170 steps → Mamba-1 (4-8L, d_model 256-512) → next-day VIX close + daily summaries
- **Level 2 (Mamba-2)**: 365 daily summaries → Mamba-2 (4-8L, d_model 256-512) → VIX +30d close
- **Data splits**: Train 2016-2023, Val 2024, Test 2025+ (30-day gap between splits)

### WSL Requirement (CRITICAL)
- **Mamba SSM requires Linux**: Cannot build on native Windows due to CUDA C++ extension compilation
- **All training must be in WSL**: Uses custom `mamba_blackwell` package (sm_120 Blackwell support)
- **Setup**: `wsl --install -d Ubuntu`, then `cd /mnt/d/Mamba\ v2 && bash setupenv.sh`
- **Why**: Mamba uses PyTorch `CUDAExtension` requiring `nvcc` + Linux build tools (`gcc`, `ninja`)
- **Fix guide**: Created `TFS_stock_t_1s/FIX_WSL.md` for WSL repair/reinstall instructions

### Archived Pretraining Scripts
- Moved pretraining-only files to `archive/` in both `TCN_Stock_Tick` and `TFS_stock_t_1s`:
  - `pretrain_tcn_rv.py`, `pretrain_bar_rv.py`, `hyperparam_search.py`, `config_pretrain.py`
  - `check_data_availability.py`, `setup_vps.sh`, `upload_to_r2.py`, etc.
- Kept `encoder/` and `loader/` for reuse in Mamba pipeline

### Data Handling
- **Stock data**: 5,625 parquet files (2003-2026), 1s bars with 31 features, top-100 tickers
- **VIX data**: 23 CSV files (2005-2026), 1-min bars, extract daily close for targets
- **Leakage prevention**: Stock days [D-14, D] → VIX day D+1 (Level 1), daily summaries [D-364, D] → VIX D+30 (Level 2)

### Usage
```bash
# In WSL with venv activated:
cd /mnt/d/Mamba\ v2/TFS_stock_t_1s
source ../venv/bin/activate

# Check data
python check_data_availability.py

# Train Level 1 (next-day VIX)
python train_mamba.py --profile rtx5080 --level 1

# Train Level 2 (VIX +30d)
python train_mamba.py --profile rtx5080 --level 2

# Smoke test
python train_mamba.py --profile rtx5080 --level 1 --smoke

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train_mamba.py --profile a100 --level 1 --gpus 4

# Hyperparameter search (8 GPUs)
python hyperparam_search_mamba.py --profile a100 --gpus 8 --n_trials 50
```

---

## 2026-03-01 - Multi-GPU DDP & RV Precompute Tools

### Multi-GPU Training (DDP)
- **Implemented DistributedDataParallel (DDP)** for multi-GPU training
- New CLI argument: `--gpus N` (supports 1-8 GPUs)
- Automatic process spawning with NCCL backend
- Gradient synchronization across GPUs
- Checkpointing only on rank 0 (main process)
- Usage: `python pretrain_tcn_rv.py --profile rtx5080 --stream index --gpus 4`

### RV Precompute Pipeline (`tools/rv_precompute/`)
- **`extract_spy.py`**: Extract SPY ticker from Polygon stock trades (parallel processing)
- **`compute_rv.py`**: Compute daily RV and 30-day forward RV
- **`run_full_pipeline.py`**: Combined runner for full RV precomputation
- Supports 22+ years of data (2003-2026)
- Output: `datasets/SPY_daily_rv/spy_daily_rv_30d.parquet`

### GPU Benchmark Results (Additional)
| GPU | Throughput | Epoch Time | Notes |
|-----|-----------|------------|-------|
| NVIDIA B200 | 7.4 it/s | 42s | Disappointing - slower than RTX 5090 |
| AMD MI300X | 12 it/s | 31s | Good, but RTX 5090 still better value |

### Key Insight
**Large datacenter GPUs (B200, H100) underperform** for this workload because:
- TCN model (18.6M params) is too small to saturate compute
- Memory bandwidth bottlenecks on large batches
- Consumer GPUs (RTX 5090) have lower latency for small batches

---

## 2026-03-01 - GPU Benchmarking & Training Optimizations

### Performance Optimizations
- **Fixed gradient checkpointing** - was configured but never used in forward pass
- **Fixed validation memory leak** - replaced tensor accumulation with running stats
- **Reduced validation time** from ~60s to ~20s per epoch

### GPU Profile Additions
- Added **RTX 5090** profile (32GB VRAM, 400 chunks for index)
- Added **AMD MI300X** profile (192GB VRAM, 3600 chunks for index)
- Added `max_chunks_32gb` and `max_chunks_192gb` to StreamConfig

### CLI Enhancements
- Added `--no-checkpoint` flag to disable gradient checkpointing (faster, more VRAM)
- Profiles now available: rtx5080, rtx5090, h100, a100, amd

### VPS Deployment
- Created `setup_vps.sh` for automated VPS setup (PyTorch, repo, data download)
- Created `download_index_data.py` for R2 data download

### GPU Benchmark Results (rtx5080 profile, 200 chunks)
| GPU | Throughput | Cost/Hour | Value |
|-----|-----------|-----------|-------|
| RTX 5090 | ~15 it/s | $1.50 | **Best** |
| AMD MI300X | 12 it/s | $1.99 | Good |
| H100 | 10 it/s | $2.50 | Decent |
| B200 | 7.4 it/s | $4.99 | Poor |
| RTX 5080 (local) | 5.29 it/s | $0 | Free |

### Key Finding
**Smaller batch sizes (200 chunks) are faster than large batches (1500+)** due to:
- Gradient checkpointing overhead
- Memory bandwidth bottlenecks
- Cache hierarchy mismatches on large GPUs

### Documentation
- Created `GPU_BENCHMARKS.md` with comprehensive performance analysis

---

## 2026-02-28 - Per-Stream TCN Pretraining
- **Refactored to separate TCN per stream** (stocks, options, index) per `architecture.md` spec
- Created `loader/single_stream_dataset.py` for single-stream loading with background prefetch
- Added `StreamConfig` dataclass and `STREAM_CONFIGS` to `config_pretrain.py`
- **Stream-specific batch sizing** tuned to tick volume:
  - Stocks (filtered ~8M ticks/day): 2K chunks (16GB) / 11.6K chunks (80GB)
  - Options (~5.4M ticks/day): 2.8K chunks (16GB) / 16K chunks (80GB)
  - Index (~347K ticks/day): 4K chunks (16GB) / 24K chunks (80GB)
- Updated `pretrain_tcn_rv.py` with `--stream` argument for stream selection
- Stream-specific checkpoint naming: `tcn_{stream}_best.pt`, `tcn_{stream}_encoder.pt`
- Usage: `python pretrain_tcn_rv.py --profile rtx5080 --stream stocks`

## 2026-02-28 - Multi-Stream Dataloader & GPU Profiles
- Created `loader/multistream_dataset.py` for combined stream loading (kept for Mamba training)
- Added `GPUProfile` dataclass with predefined profiles (rtx5080/h100/a100)
- Created `scripts/compute_top_stocks.py` to precompute top 100 stocks by volume

## 2026-02-28 - TCN Pretraining Setup
- Implemented TCN pretraining pipeline for SPY 30-day Realized Volatility prediction (`loader/spy_rv_dataset.py`, `encoder/rv_head.py`, `config_pretrain.py`, `pretrain_tcn_rv.py`)
