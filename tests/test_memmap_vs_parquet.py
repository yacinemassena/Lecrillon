"""
Compare memmap vs parquet loading paths.

1. Correctness: same sample via both paths should produce matching tensors.
2. Speed: time N samples in each mode.

Usage:
    python tests/test_memmap_vs_parquet.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from loader.bar_mamba_dataset import BarMambaDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path("datasets")
STOCK_PATH = str(DATA_ROOT / "Stock_Data_2min")
VIX_PATH = str(DATA_ROOT / "VIX")
NEWS_PATH = str(DATA_ROOT / "benzinga_embeddings")
OPTIONS_PATH = str(DATA_ROOT / "opt_trade_2min")
MACRO_PATH = str(DATA_ROOT / "MACRO" / "macro_daily_enhanced.parquet")
GDELT_PATH = str(DATA_ROOT / "GDELT")
ECON_PATH = str(DATA_ROOT / "econ_calendar")
FUNDAMENTALS_PATH = str(DATA_ROOT / "fundamentals" / "fundamentals_state.parquet")
VIX_FEATURES_PATH = str(DATA_ROOT / "VIX" / "Vix_features")
PREPROCESSED_PATH = str(DATA_ROOT / "preprocessed")

SEQ_LEN = 1000  # small for test speed
TEST_INDICES = [0, 100, 500]  # sample indices to compare
BENCH_COUNT = 10  # samples for speed test

COMMON_KWARGS = dict(
    stock_data_path=STOCK_PATH,
    vix_data_path=VIX_PATH,
    split="train",
    max_total_bars=SEQ_LEN,
    train_start="2005-01-01",
    train_end="2023-11-30",
    val_end="2024-12-31",
    news_data_path=NEWS_PATH,
    use_news=True,
    options_data_path=OPTIONS_PATH,
    use_options=True,
    macro_data_path=MACRO_PATH,
    use_macro=True,
    gdelt_data_path=GDELT_PATH,
    use_gdelt=True,
    econ_calendar_path=ECON_PATH,
    use_econ=True,
    fundamentals_data_path=FUNDAMENTALS_PATH,
    use_fundamentals=True,
    vix_features_path=VIX_FEATURES_PATH,
    use_vix_features=True,
)


def compare_tensors(key: str, a, b, atol=1e-4):
    """Compare two tensors, return (passed, message)."""
    if a is None and b is None:
        return True, f"  {key:25s} both None"
    if a is None or b is None:
        return False, f"  {key:25s} MISMATCH: one is None"
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False, f"  {key:25s} SHAPE MISMATCH: {a.shape} vs {b.shape}"
        if a.dtype != b.dtype:
            # Allow float comparison across dtypes
            a, b = a.float(), b.float()
        close = torch.allclose(a, b, atol=atol, equal_nan=True)
        if close:
            return True, f"  {key:25s} OK  shape={list(a.shape)}"
        else:
            diff = (a - b).abs()
            return False, (f"  {key:25s} VALUES DIFFER  shape={list(a.shape)}  "
                          f"max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")
    # Scalar comparison
    if a == b:
        return True, f"  {key:25s} OK  value={a}"
    return False, f"  {key:25s} MISMATCH: {a} vs {b}"


def run_correctness_test():
    """Compare outputs of parquet vs memmap for the same samples."""
    print("=" * 70)
    print("CORRECTNESS TEST: parquet vs memmap")
    print("=" * 70)

    print("\nLoading parquet dataset...")
    t0 = time.time()
    ds_pq = BarMambaDataset(**COMMON_KWARGS, preprocessed_path=None)
    print(f"  Parquet dataset ready: {len(ds_pq)} samples ({time.time()-t0:.1f}s)")

    print("Loading memmap dataset...")
    t0 = time.time()
    ds_mm = BarMambaDataset(**COMMON_KWARGS, preprocessed_path=PREPROCESSED_PATH)
    print(f"  Memmap dataset ready: {len(ds_mm)} samples ({time.time()-t0:.1f}s)")
    print(f"  Memmap active: {ds_mm._use_memmaps}")

    if not ds_mm._use_memmaps:
        print("ERROR: memmap mode not activated!")
        return False

    # Verify same anchor dates
    assert len(ds_pq) == len(ds_mm), f"Dataset size mismatch: {len(ds_pq)} vs {len(ds_mm)}"

    all_passed = True
    keys_to_compare = [
        'bars', 'vix_targets', 'horizon_mask', 'num_bars',
        'news_embs', 'news_timestamps', 'num_news',
        'gdelt_embs', 'gdelt_timestamps', 'num_gdelt',
        'options', 'options_mask',
        'vix_features', 'vix_timestamps', 'vix_mask', 'num_vix',
        'macro_context', 'fundamentals_context',
        'econ_event_ids', 'econ_currency_ids', 'econ_numeric', 'econ_timestamps', 'num_econ',
    ]

    for idx in TEST_INDICES:
        if idx >= len(ds_pq):
            print(f"\nSkipping idx={idx} (only {len(ds_pq)} samples)")
            continue

        print(f"\n--- Sample idx={idx} (anchor={ds_pq.anchor_dates[idx]['anchor_date']}) ---")

        # Parquet
        t0 = time.time()
        sample_pq = ds_pq[idx]
        t_pq = time.time() - t0

        # Memmap
        t0 = time.time()
        sample_mm = ds_mm[idx]
        t_mm = time.time() - t0

        print(f"  Parquet: {t_pq:.3f}s  |  Memmap: {t_mm:.6f}s  |  Speedup: {t_pq/max(t_mm,1e-9):.0f}x")

        for key in keys_to_compare:
            a = sample_pq.get(key)
            b = sample_mm.get(key)
            passed, msg = compare_tensors(key, a, b)
            if not passed:
                all_passed = False
                print(f"  FAIL {msg}")
            else:
                print(msg)

    return all_passed


def run_speed_benchmark():
    """Time N samples in memmap mode."""
    print("\n" + "=" * 70)
    print(f"SPEED BENCHMARK: {BENCH_COUNT} samples (memmap only)")
    print("=" * 70)

    ds = BarMambaDataset(**COMMON_KWARGS, preprocessed_path=PREPROCESSED_PATH)
    if not ds._use_memmaps:
        print("ERROR: memmap mode not activated!")
        return

    # Warmup
    _ = ds[0]

    indices = np.linspace(0, len(ds) - 1, BENCH_COUNT, dtype=int)
    times = []
    for i, idx in enumerate(indices):
        t0 = time.time()
        sample = ds[idx]
        elapsed = time.time() - t0
        times.append(elapsed)
        bars = sample['bars'].shape[0] if 'bars' in sample else 0
        print(f"  [{i+1}/{BENCH_COUNT}] idx={idx:5d}  {elapsed*1000:8.2f} ms  bars={bars}")

    times = np.array(times)
    print(f"\n  Mean:   {times.mean()*1000:.2f} ms")
    print(f"  Median: {np.median(times)*1000:.2f} ms")
    print(f"  Min:    {times.min()*1000:.2f} ms")
    print(f"  Max:    {times.max()*1000:.2f} ms")
    print(f"  Total:  {times.sum():.3f} s for {BENCH_COUNT} samples")


if __name__ == "__main__":
    passed = run_correctness_test()
    run_speed_benchmark()
    print("\n" + "=" * 70)
    if passed:
        print("ALL CORRECTNESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED — check output above")
    print("=" * 70)
