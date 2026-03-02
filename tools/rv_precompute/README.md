# SPY Realized Volatility Precompute

Scripts to extract SPY tick data and compute Realized Volatility (RV) for TCN pretraining.

## Quick Start

Run the full pipeline:

```bash
cd "D:\Mamba v2\tools\rv_precompute"
python run_full_pipeline.py
```

This will:
1. Extract SPY ticks from Polygon stock trades → `D:\Mamba v2\datasets\SPY_trades\`
2. Compute daily RV and 30-day forward RV → `D:\Mamba v2\datasets\SPY_daily_rv\`

## Scripts

### 1. `extract_spy.py` - Extract SPY from Polygon data

```bash
python extract_spy.py \
    --input "D:/polygon stock data/trades" \
    --output "D:/Mamba v2/datasets/SPY_trades" \
    --ticker SPY \
    --workers 8 \
    --skip-existing
```

**Arguments:**
- `--input`: Directory with daily parquet files (all stocks)
- `--output`: Output directory for SPY-only files
- `--ticker`: Ticker to extract (default: SPY)
- `--workers`: Parallel workers (default: 8)
- `--skip-existing`: Skip files that already exist

### 2. `compute_rv.py` - Compute RV from tick data

```bash
python compute_rv.py \
    --input "D:/Mamba v2/datasets/SPY_trades" \
    --output "D:/Mamba v2/datasets/SPY_daily_rv" \
    --horizon 30 \
    --workers 8
```

**Arguments:**
- `--input`: Directory with SPY parquet files
- `--output`: Output directory for RV parquet
- `--horizon`: Forward RV horizon in days (default: 30)
- `--min-days`: Minimum trading days for forward RV (default: 10)
- `--workers`: Parallel workers (default: 8)

## Output Format

`spy_daily_rv_30d.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Trading date |
| `daily_rv` | float64 | Daily RV = sqrt(sum(log_returns²)) |
| `n_ticks` | int64 | Number of ticks that day |
| `n_returns` | int64 | Number of valid log returns |
| `rv_30d_forward` | float64 | 30-day forward RV (NaN if < 10 trading days ahead) |

## RV Computation

**Daily RV:**
```python
log_returns = log(price[t] / price[t-1])
daily_rv = sqrt(sum(log_returns²))
```

**30-day Forward RV:**
```python
# Sum of squared daily RVs over next 30 calendar days (~21 trading days)
rv_30d_forward = sqrt(sum(daily_rv²))
```

## Expected Runtime

| Step | Files | Time (8 workers) |
|------|-------|------------------|
| Extract SPY | 5500 files | ~15 minutes |
| Compute RV | 5500 files | ~5 minutes |
| **Total** | - | **~20 minutes** |

## Integration with TCN Training

After running the pipeline, update `config_pretrain.py`:

```python
rv_file: str = 'D:/Mamba v2/datasets/SPY_daily_rv/spy_daily_rv_30d.parquet'
```

Or pass via CLI:
```bash
python pretrain_tcn_rv.py --rv_file "D:/Mamba v2/datasets/SPY_daily_rv/spy_daily_rv_30d.parquet"
```
