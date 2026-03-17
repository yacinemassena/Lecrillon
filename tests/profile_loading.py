"""Profile where time is actually spent loading one sample.
Tests per-feed parallel loading vs sequential."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np

DATA_ROOT = Path(r"D:\Mamba v2\datasets")

print("="*60)
print("LOADING PROFILER - per-feed parallel vs sequential")
print("="*60)

# ---- Phase 1: Raw parquet I/O threading test ----
print("\n--- Phase 1: Raw parquet threading (does GIL release?) ---")
import polars as pl
from concurrent.futures import ThreadPoolExecutor

stock_dir = DATA_ROOT / "Stock_Data_2min"
stock_files = sorted(stock_dir.glob("*.parquet"))[-42:]
print(f"Files: {len(stock_files)} stock parquets")

t0 = time.time()
for f in stock_files:
    pl.read_parquet(f)
seq_time = time.time() - t0
print(f"  Sequential:  {seq_time:.2f}s")

for n in [4, 16, 31]:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as p:
        list(p.map(lambda f: pl.read_parquet(f), stock_files))
    t = time.time() - t0
    print(f"  {n:2d} threads:  {t:.2f}s  (speedup {seq_time/t:.1f}x)")

# ---- Phase 2: Per-feed timing ----
print("\n--- Phase 2: Per-feed timing (sequential, cold cache) ---")

from loader.bar_mamba_dataset import (
    BarMambaDataset, _BARS_CACHE, _NEWS_CACHE, _GDELT_CACHE,
    _VIX_CACHE, _RAW_PARQUET_CACHE, _IO_THREADS
)

ds = BarMambaDataset(
    stock_data_path=str(DATA_ROOT / "Stock_Data_2min"),
    vix_data_path=str(DATA_ROOT / "VIX"),
    max_total_bars=8000, split='train',
    news_data_path=str(DATA_ROOT / "benzinga_embeddings"),
    use_news=True,
    options_data_path=str(DATA_ROOT / "opt_trade_2min"),
    use_options=True,
    gdelt_data_path=str(DATA_ROOT / "GDELT"),
    use_gdelt=True,
    vix_features_path=str(DATA_ROOT / "VIX" / "Vix_features"),
    use_vix_features=True,
    econ_calendar_path=str(DATA_ROOT / "econ_calendar"),
    use_econ=True,
    macro_data_path=str(DATA_ROOT / "MACRO" / "macro_daily_enhanced.parquet"),
    use_macro=True,
    fundamentals_data_path=str(DATA_ROOT / "fundamentals" / "fundamentals_state.parquet"),
    use_fundamentals=True,
)
print(f"Dataset: {len(ds)} samples, IO threads: {_IO_THREADS}")
if len(ds) == 0:
    print("ERROR: No samples"); sys.exit(1)

sample = ds.anchor_dates[0]
wd = sample['window_dates']
print(f"Sample 0: {len(wd)} window dates\n")

def clear_all_caches():
    _BARS_CACHE.clear(); _NEWS_CACHE.clear(); _GDELT_CACHE.clear()
    _VIX_CACHE.clear(); _RAW_PARQUET_CACHE.clear()

# Time each feed individually (sequential)
feeds = {}

clear_all_caches()
t0 = time.time()
stock = ds._feed_stock(wd)
feeds['Stock'] = time.time() - t0

clear_all_caches()
t0 = time.time()
news = ds._feed_news(wd)
feeds['News'] = time.time() - t0

clear_all_caches()
t0 = time.time()
gdelt = ds._feed_gdelt(wd)
feeds['GDELT'] = time.time() - t0

clear_all_caches()
t0 = time.time()
vix = ds._feed_vix(wd)
feeds['VIX'] = time.time() - t0

clear_all_caches()
t0 = time.time()
econ = ds._feed_econ(wd, sample['anchor_date'])
feeds['Econ'] = time.time() - t0

# Options needs loaded_days from stock
clear_all_caches()
stock2 = ds._feed_stock(wd)
t0 = time.time()
opts = ds._feed_options(stock2['loaded_days'], 8000)
feeds['Options'] = time.time() - t0

total_seq = sum(feeds.values())
print("Per-feed times (sequential, cold cache):")
for name, t in sorted(feeds.items(), key=lambda x: -x[1]):
    pct = t / total_seq * 100 if total_seq > 0 else 0
    print(f"  {name:10s}: {t:6.2f}s  ({pct:4.1f}%)")
print(f"  {'SUM':10s}: {total_seq:6.2f}s")

# ---- Phase 3: Full __getitem__ parallel (new architecture) ----
print("\n--- Phase 3: Full __getitem__ (parallel feeds) ---")

clear_all_caches()
BarMambaDataset._samples_loaded = 0
t0 = time.time()
result = ds[0]
parallel_time = time.time() - t0
print(f"  Cold cache:  {parallel_time:.2f}s")

t0 = time.time()
result = ds[0]
warm_time = time.time() - t0
print(f"  Warm cache:  {warm_time:.2f}s")

print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"  Sequential feeds (sum):  {total_seq:.1f}s")
print(f"  Parallel __getitem__:    {parallel_time:.1f}s")
if total_seq > 0 and parallel_time > 0:
    print(f"  Speedup:                 {total_seq/parallel_time:.1f}x")
print(f"  Warm cache:              {warm_time:.2f}s")
print(f"{'='*60}")
