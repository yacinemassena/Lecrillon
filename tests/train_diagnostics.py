"""
Verbose Train Diagnostics — step-by-step init checker for train.py.

Replays the full training initialization sequence with try/except,
timing, and verbose output at each phase. Run standalone:

    python train_diagnostics.py
    python train_diagnostics.py --seq-len 2000 --batch-size 4   # override config

Or via train.py:

    python train.py --diagnose
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_phase_num = 0
_total_passed = 0
_total_failed = 0


def _phase(name: str):
    """Print phase header."""
    global _phase_num
    _phase_num += 1
    print(f"\n{'='*60}")
    print(f"  Phase {_phase_num}: {name}")
    print(f"{'='*60}")


def _ok(msg: str):
    global _total_passed
    _total_passed += 1
    print(f"  [PASS] {msg}")


def _fail(msg: str):
    global _total_failed
    _total_failed += 1
    print(f"  [FAIL] {msg}")


def _info(msg: str):
    print(f"  [INFO] {msg}")


def _timed(label: str):
    """Context manager that prints elapsed time."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            _info(f"{label} took {elapsed:.2f}s")
    return _Timer()


# ---------------------------------------------------------------------------
# Phase 1: Environment
# ---------------------------------------------------------------------------
def phase_environment():
    _phase("Environment")

    _info(f"Python: {sys.version}")
    _info(f"CWD: {os.getcwd()}")
    _info(f"Script dir: {Path(__file__).parent.resolve()}")

    # PyTorch
    try:
        import torch
        _ok(f"torch {torch.__version__}")
    except ImportError as e:
        _fail(f"torch import: {e}")
        return False

    # CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"CUDA available: {gpu_name} ({vram:.1f} GB)")
        _info(f"CUDA version: {torch.version.cuda}")
        _info(f"cuDNN version: {torch.backends.cudnn.version()}")
        bf16 = torch.cuda.is_bf16_supported()
        _info(f"BF16 supported: {bf16}")
    else:
        _fail("CUDA not available — training requires GPU")
        return False

    # numpy / pandas / polars
    for pkg in ['numpy', 'pandas', 'polars']:
        try:
            mod = __import__(pkg)
            _ok(f"{pkg} {mod.__version__}")
        except ImportError as e:
            _fail(f"{pkg}: {e}")

    # mamba_ssm
    try:
        from mamba_ssm import Mamba
        _ok("mamba_ssm imported (Mamba class)")
    except ImportError:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            _ok("mamba_ssm imported (mamba_simple fallback)")
        except ImportError as e:
            _fail(f"mamba_ssm: {e}")
            return False

    # rich
    try:
        import rich
        try:
            from importlib.metadata import version as pkg_version
            rich_ver = pkg_version("rich")
        except Exception:
            rich_ver = "(version unknown)"
        _ok(f"rich {rich_ver}")
    except ImportError as e:
        _fail(f"rich: {e}")

    return True


# ---------------------------------------------------------------------------
# Phase 2: Config
# ---------------------------------------------------------------------------
def phase_config(args):
    _phase("Config")

    important = [
        'seq_len', 'batch_size', 'num_workers', 'epochs', 'train_steps',
        'val_steps', 'd_model', 'n_layers', 'd_state', 'lr',
        'use_news', 'use_options', 'use_macro', 'use_gdelt',
        'use_econ', 'use_fundamentals', 'use_vix_features',
    ]
    for k in important:
        v = getattr(args, k, '???')
        _info(f"{k} = {v}")
    _ok("Config parsed")
    return True


# ---------------------------------------------------------------------------
# Phase 3: Data Paths
# ---------------------------------------------------------------------------
def phase_data_paths() -> Optional[Dict[str, Path]]:
    _phase("Data Paths")

    try:
        from train import get_data_paths
        paths = get_data_paths()
    except Exception as e:
        _fail(f"get_data_paths(): {e}")
        traceback.print_exc()
        return None

    if not paths:
        _fail("No data directory found")
        return None

    _ok(f"Base detected with {len(paths)} path keys")
    for name, p in paths.items():
        exists = p.exists()
        if exists:
            # Count files
            n_files = sum(1 for _ in p.glob('*')) if p.is_dir() else 1
            _ok(f"{name}: {p}  ({n_files} items)")
        else:
            _fail(f"{name}: {p}  (MISSING)")

    return paths


# ---------------------------------------------------------------------------
# Phase 4: Data Overlap
# ---------------------------------------------------------------------------
def phase_data_overlap(paths: Dict[str, Path]) -> bool:
    _phase("Data Overlap Check")

    try:
        from train import check_data_overlap
        with _timed("check_data_overlap"):
            has_overlap = check_data_overlap(paths)
        if has_overlap:
            _ok("Stock-VIX overlap confirmed")
        else:
            _fail("No stock-VIX overlap found")
        return has_overlap
    except Exception as e:
        _fail(f"check_data_overlap crashed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Phase 5: Dataset Build
# ---------------------------------------------------------------------------
def phase_dataset_build(args, paths):
    _phase("Dataset Build")

    try:
        from loader.bar_mamba_dataset import BarMambaDataset, NUM_STOCK_FEATURES
    except Exception as e:
        _fail(f"Import BarMambaDataset: {e}")
        traceback.print_exc()
        return None, None

    stock_path = str(paths['stock'])
    vix_path = str(paths['vix'])

    # Resolve optional paths (simplified from train.py)
    news_path = _resolve_optional_path(args, 'use_news', 'news_path', paths, [
        lambda p: p.get('stock', Path('.')).parent / 'benzinga_embeddings',
    ])
    options_path = _resolve_optional_path(args, 'use_options', 'options_path', paths, [
        lambda p: p.get('options'),
        lambda p: p.get('stock', Path('.')).parent / 'opt_trade_2min',
    ])
    macro_path = _resolve_optional_path(args, 'use_macro', 'macro_path', paths, [
        lambda p: p.get('stock', Path('.')).parent / 'MACRO' / 'macro_daily_enhanced.parquet',
        lambda p: p.get('stock', Path('.')).parent / 'MACRO' / 'macro_daily.parquet',
    ])
    gdelt_path = _resolve_optional_path(args, 'use_gdelt', 'gdelt_path', paths, [
        lambda p: p.get('stock', Path('.')).parent / 'GDELT',
    ])
    econ_path = _resolve_optional_path(args, 'use_econ', 'econ_path', paths, [
        lambda p: p.get('stock', Path('.')).parent / 'econ_calendar',
    ])
    fundamentals_path = _resolve_optional_path(args, 'use_fundamentals', 'fundamentals_path', paths, [
        lambda p: p.get('stock', Path('.')).parent / 'fundamentals' / 'fundamentals_state.parquet',
    ])
    vix_features_path = _resolve_optional_path(args, 'use_vix_features', 'vix_features_path', paths, [
        lambda p: p.get('stock', Path('.')).parent / 'VIX' / 'Vix_features',
    ])

    _info(f"Resolved paths:")
    for name, val in [('news', news_path), ('options', options_path), ('macro', macro_path),
                      ('gdelt', gdelt_path), ('econ', econ_path),
                      ('fundamentals', fundamentals_path), ('vix_features', vix_features_path)]:
        _info(f"  {name}: {val or '(none)'}")

    # Build train dataset
    train_ds = None
    val_ds = None
    try:
        with _timed("Train dataset"):
            train_ds = BarMambaDataset(
                stock_data_path=stock_path, vix_data_path=vix_path,
                split='train', max_total_bars=args.seq_len,
                train_start=args.train_start, train_end=args.train_end,
                val_end=args.val_end,
                news_data_path=news_path, use_news=args.use_news,
                options_data_path=options_path, use_options=args.use_options,
                macro_data_path=macro_path, use_macro=args.use_macro,
                gdelt_data_path=gdelt_path, use_gdelt=args.use_gdelt,
                econ_calendar_path=econ_path, use_econ=args.use_econ,
                fundamentals_data_path=fundamentals_path, use_fundamentals=args.use_fundamentals,
                vix_features_path=vix_features_path, use_vix_features=args.use_vix_features,
            )
        _ok(f"Train dataset: {len(train_ds)} samples, {train_ds.num_features} features")
        _info(f"  macro_dim={getattr(train_ds, 'macro_dim', 0)}, "
              f"fundamentals_dim={getattr(train_ds, 'fundamentals_dim', 0)}, "
              f"econ_event_types={getattr(train_ds, 'econ_num_event_types', 0)}")
    except Exception as e:
        _fail(f"Train dataset build: {e}")
        traceback.print_exc()

    try:
        with _timed("Val dataset"):
            val_ds = BarMambaDataset(
                stock_data_path=stock_path, vix_data_path=vix_path,
                split='val', max_total_bars=args.seq_len,
                train_start=args.train_start, train_end=args.train_end,
                val_end=args.val_end,
                news_data_path=news_path, use_news=args.use_news,
                options_data_path=options_path, use_options=args.use_options,
                macro_data_path=macro_path, use_macro=args.use_macro,
                gdelt_data_path=gdelt_path, use_gdelt=args.use_gdelt,
                econ_calendar_path=econ_path, use_econ=args.use_econ,
                fundamentals_data_path=fundamentals_path, use_fundamentals=args.use_fundamentals,
                vix_features_path=vix_features_path, use_vix_features=args.use_vix_features,
                shared_state=train_ds.get_shared_state() if train_ds else None,
            )
        _ok(f"Val dataset: {len(val_ds)} samples")
    except Exception as e:
        _fail(f"Val dataset build: {e}")
        traceback.print_exc()

    return train_ds, val_ds


def _resolve_optional_path(args, use_flag, path_attr, paths, candidates):
    """Resolve an optional data path using same logic as train.py."""
    if not getattr(args, use_flag, False):
        return None
    explicit = getattr(args, path_attr, None)
    if explicit:
        return explicit
    for fn in candidates:
        try:
            p = fn(paths)
            if p is not None and Path(p).exists():
                return str(p)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Phase 6: DataLoader Smoke Test
# ---------------------------------------------------------------------------
def phase_dataloader_smoke(train_ds, args):
    _phase("DataLoader Smoke Test (1 batch)")

    import torch
    from torch.utils.data import DataLoader
    from loader.bar_mamba_dataset import BarMambaDataset

    # Use fewer workers for diagnostics to avoid masking errors
    num_workers = min(args.num_workers, 2)
    _info(f"Using {num_workers} workers (diagnostic mode)")

    try:
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=BarMambaDataset.collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            timeout=300 if num_workers > 0 else 0,  # 5min, matching train.py
        )
    except Exception as e:
        _fail(f"DataLoader creation: {e}")
        traceback.print_exc()
        return None

    _info("Fetching first batch...")
    try:
        with _timed("First batch"):
            batch = next(iter(loader))
    except Exception as e:
        _fail(f"First batch fetch: {e}")
        traceback.print_exc()
        return None

    _ok(f"Batch fetched successfully")

    # Report all tensor shapes
    for key, val in sorted(batch.items()):
        if isinstance(val, torch.Tensor):
            _info(f"  {key}: {val.shape}  dtype={val.dtype}")
        elif isinstance(val, list):
            _info(f"  {key}: list[{len(val)}] = {val[:4]}{'...' if len(val) > 4 else ''}")
        else:
            _info(f"  {key}: {type(val).__name__} = {val}")

    # Sanity checks on batch
    bars = batch.get('bars')
    if bars is not None:
        if torch.isnan(bars).any():
            _fail("NaN detected in bars tensor!")
        elif torch.isinf(bars).any():
            _fail("Inf detected in bars tensor!")
        else:
            _ok(f"bars: no NaN/Inf, range=[{bars.min():.2f}, {bars.max():.2f}]")

    targets = batch.get('vix_targets')
    if targets is not None:
        _info(f"  vix_targets range: [{targets.min():.2f}, {targets.max():.2f}]")

    hmask = batch.get('horizon_mask')
    if hmask is not None:
        for i, h in enumerate([1, 7, 15, 30]):
            pct = hmask[:, i].mean().item() * 100
            _info(f"  +{h}d target available: {pct:.0f}%")

    return batch


# ---------------------------------------------------------------------------
# Phase 7: Model Build
# ---------------------------------------------------------------------------
def phase_model_build(args, train_ds):
    _phase("Model Build")

    import torch

    try:
        from mamba_only_model import ParallelMambaVIX, NUM_OPTION_FEATURES
    except Exception as e:
        _fail(f"Import ParallelMambaVIX: {e}")
        traceback.print_exc()
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = train_ds.num_features

    macro_dim = getattr(train_ds, 'macro_dim', 15) if args.use_macro else 15
    gdelt_dim = getattr(train_ds, 'gdelt_dim', 391) if args.use_gdelt else 391
    econ_num_event_types = getattr(train_ds, 'econ_num_event_types', 412) + 1 if args.use_econ else 413
    econ_num_currencies = getattr(train_ds, 'econ_num_currencies', 4) + 1 if args.use_econ else 5
    fundamentals_dim = getattr(train_ds, 'fundamentals_dim', 130) if args.use_fundamentals else 130
    vix_features_dim = getattr(train_ds, 'num_vix_features', 25) if args.use_vix_features else 25

    _info(f"num_features={num_features}, macro_dim={macro_dim}, gdelt_dim={gdelt_dim}")
    _info(f"econ_event_types={econ_num_event_types}, econ_currencies={econ_num_currencies}")
    _info(f"fundamentals_dim={fundamentals_dim}, vix_features_dim={vix_features_dim}")

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9

    try:
        with _timed("Model init"):
            model = ParallelMambaVIX(
                num_features=num_features,
                d_model=args.d_model,
                n_layers=args.n_layers,
                d_state=args.d_state,
                d_conv=4, expand=2, dropout=0.1,
                checkpoint_interval=args.checkpoint_interval,
                use_news=args.use_news, news_dim=3072,
                news_n_layers=args.news_n_layers,
                use_options=args.use_options,
                option_features=NUM_OPTION_FEATURES,
                head_hidden=128,
                use_macro=args.use_macro, macro_dim=macro_dim,
                use_gdelt=args.use_gdelt, gdelt_dim=gdelt_dim,
                use_econ=args.use_econ,
                econ_num_event_types=econ_num_event_types,
                econ_num_currencies=econ_num_currencies,
                use_fundamentals=args.use_fundamentals,
                fundamentals_dim=fundamentals_dim,
                use_vix_features=args.use_vix_features,
                vix_features_dim=vix_features_dim,
                vix_n_layers=args.vix_n_layers,
                vix_d_model=args.vix_d_model,
                vix_d_state=args.vix_d_state,
            ).to(device)
    except Exception as e:
        _fail(f"Model build: {e}")
        traceback.print_exc()
        return None

    mem_after = torch.cuda.memory_allocated() / 1e9
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _ok(f"Model built: {n_params:,} params, VRAM={mem_after:.2f}GB (+{mem_after - mem_before:.2f}GB)")

    # Check for NaN in initial params
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    if has_nan:
        _fail("NaN in initial model parameters!")
    else:
        _ok("No NaN in initial params")

    return model


# ---------------------------------------------------------------------------
# Phase 8: Forward Pass
# ---------------------------------------------------------------------------
def phase_forward(model, batch, args):
    _phase("Forward Pass (1 batch)")

    import torch

    device = next(model.parameters()).device

    try:
        from train import batch_to_device
    except Exception as e:
        _fail(f"Import batch_to_device: {e}")
        return False

    # Transfer batch to GPU
    try:
        bd = batch_to_device(batch, device)
        _ok("Batch transferred to GPU")
    except Exception as e:
        _fail(f"batch_to_device: {e}")
        traceback.print_exc()
        return False

    # Report GPU memory after transfer
    mem = torch.cuda.memory_allocated() / 1e9
    _info(f"VRAM after batch transfer: {mem:.2f}GB")

    # Forward
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _info(f"AMP dtype: {amp_dtype}")

    model.eval()
    torch.cuda.reset_peak_memory_stats()

    try:
        with _timed("Forward pass"), torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype):
            outputs = model(
                bd['bars'], bd['bar_mask'],
                options=bd['options'], options_mask=bd['options_mask'],
                news_embs=bd['news_embs'], news_mask=bd['news_mask'],
                news_timestamps=bd['news_timestamps'],
                gdelt_embs=bd['gdelt_embs'], gdelt_mask=bd['gdelt_mask'],
                gdelt_timestamps=bd['gdelt_timestamps'],
                macro_context=bd['macro_context'],
                bar_timestamps=bd['bar_timestamps'],
                econ_event_ids=bd['econ_event_ids'],
                econ_currency_ids=bd['econ_currency_ids'],
                econ_numeric=bd['econ_numeric'],
                econ_mask=bd['econ_mask'],
                econ_timestamps=bd['econ_timestamps'],
                fundamentals_context=bd['fundamentals_context'],
                vix_features=bd['vix_features'],
                vix_mask=bd['vix_mask'],
                vix_timestamps=bd['vix_timestamps'],
            )
    except Exception as e:
        _fail(f"Forward pass: {e}")
        traceback.print_exc()
        return False

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    _ok(f"Forward pass succeeded, peak VRAM={peak_mem:.2f}GB")

    # Report outputs
    for key, val in sorted(outputs.items()):
        if isinstance(val, torch.Tensor):
            has_nan = torch.isnan(val).any().item()
            _info(f"  {key}: {val.shape}  range=[{val.min():.4f}, {val.max():.4f}]"
                  f"{'  ⚠ NaN!' if has_nan else ''}")

    return True


# ---------------------------------------------------------------------------
# Phase 9: Backward Pass
# ---------------------------------------------------------------------------
def phase_backward(model, batch, args):
    _phase("Backward Pass (1 step)")

    import torch
    from torch.amp import GradScaler

    device = next(model.parameters()).device

    try:
        from train import batch_to_device, SpikeWeightedHuberLoss
    except Exception as e:
        _fail(f"Import: {e}")
        return False

    bd = batch_to_device(batch, device)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    criterion = SpikeWeightedHuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    try:
        with _timed("Forward + backward"):
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                outputs = model(
                    bd['bars'], bd['bar_mask'],
                    options=bd['options'], options_mask=bd['options_mask'],
                    news_embs=bd['news_embs'], news_mask=bd['news_mask'],
                    news_timestamps=bd['news_timestamps'],
                    gdelt_embs=bd['gdelt_embs'], gdelt_mask=bd['gdelt_mask'],
                    gdelt_timestamps=bd['gdelt_timestamps'],
                    macro_context=bd['macro_context'],
                    bar_timestamps=bd['bar_timestamps'],
                    econ_event_ids=bd['econ_event_ids'],
                    econ_currency_ids=bd['econ_currency_ids'],
                    econ_numeric=bd['econ_numeric'],
                    econ_mask=bd['econ_mask'],
                    econ_timestamps=bd['econ_timestamps'],
                    fundamentals_context=bd['fundamentals_context'],
                    vix_features=bd['vix_features'],
                    vix_mask=bd['vix_mask'],
                    vix_timestamps=bd['vix_timestamps'],
                )
                pred = outputs['vix_pred']
                loss = criterion(pred, bd['target'], bd['horizon_mask'])

            scaler.scale(loss).backward()
            torch.cuda.synchronize()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

    except Exception as e:
        _fail(f"Backward pass: {e}")
        traceback.print_exc()
        return False

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    _ok(f"Backward pass succeeded")
    _info(f"  loss={loss.item():.6f}, grad_norm={grad_norm:.4f}, peak VRAM={peak_mem:.2f}GB")

    # Check for NaN in gradients
    nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    if nan_grads:
        _fail(f"NaN in gradients for {nan_grads} parameter tensors!")
    else:
        _ok("No NaN in gradients")

    # Count parameters with zero gradients (might indicate dead streams)
    zero_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0)
    total_with_grads = sum(1 for p in model.parameters() if p.grad is not None)
    if zero_grads > 0:
        _info(f"  ⚠ {zero_grads}/{total_with_grads} param tensors have zero gradients")
    else:
        _ok(f"All {total_with_grads} param tensors have non-zero gradients")

    return True


# ---------------------------------------------------------------------------
# Phase 10: Validation Checks (from train.py)
# ---------------------------------------------------------------------------
def phase_validation_checks(model, args, train_ds):
    _phase("Pre-Training Validation Checks (from train.py)")

    import torch

    device = next(model.parameters()).device
    num_features = train_ds.num_features
    macro_dim = getattr(train_ds, 'macro_dim', 15) if args.use_macro else 15
    gdelt_dim = getattr(train_ds, 'gdelt_dim', 391) if args.use_gdelt else 391
    fundamentals_dim = getattr(train_ds, 'fundamentals_dim', 130) if args.use_fundamentals else 130
    vix_features_dim = getattr(train_ds, 'num_vix_features', 25) if args.use_vix_features else 25

    try:
        from train import run_validation_checks
    except Exception as e:
        _fail(f"Import run_validation_checks: {e}")
        return False

    try:
        with _timed("run_validation_checks"):
            passed, total = run_validation_checks(
                model, args, device, num_features,
                macro_dim=macro_dim, gdelt_dim=gdelt_dim,
                fundamentals_dim=fundamentals_dim,
                vix_features_dim=vix_features_dim,
                phase="pre", is_distributed=False,
            )
        if passed == total:
            _ok(f"All validation checks passed: {passed}/{total}")
        else:
            _fail(f"Validation checks: {passed}/{total} passed")
    except Exception as e:
        _fail(f"run_validation_checks crashed: {e}")
        traceback.print_exc()
        return False

    return True


# ---------------------------------------------------------------------------
# Argument parser (mirrors train.py)
# ---------------------------------------------------------------------------
def build_parser():
    from trainconfig import DEFAULT_CONFIG as cfg

    parser = argparse.ArgumentParser(description='Train Diagnostics')
    parser.add_argument('--seq-len', type=int, default=cfg.seq_len)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--train-steps', type=int, default=cfg.train_steps)
    parser.add_argument('--val-steps', type=int, default=cfg.val_steps)
    parser.add_argument('--d-model', type=int, default=cfg.d_model)
    parser.add_argument('--n-layers', type=int, default=cfg.n_layers)
    parser.add_argument('--news-n-layers', type=int, default=cfg.news_n_layers)
    parser.add_argument('--d-state', type=int, default=cfg.d_state)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--checkpoint-interval', type=int, default=cfg.checkpoint_interval)
    parser.add_argument('--train-start', type=str, default='2005-01-01')
    parser.add_argument('--train-end', type=str, default=cfg.train_end)
    parser.add_argument('--val-end', type=str, default=cfg.val_end)
    parser.add_argument('--use-news', action='store_true', default=cfg.use_news)
    parser.add_argument('--no-news', action='store_false', dest='use_news')
    parser.add_argument('--news-path', type=str, default=None)
    parser.add_argument('--use-options', action='store_true', default=cfg.use_options)
    parser.add_argument('--no-options', action='store_false', dest='use_options')
    parser.add_argument('--options-path', type=str, default=None)
    parser.add_argument('--use-macro', action='store_true', default=cfg.use_macro)
    parser.add_argument('--macro-path', type=str, default=None)
    parser.add_argument('--use-gdelt', action='store_true', default=cfg.use_gdelt)
    parser.add_argument('--gdelt-path', type=str, default=None)
    parser.add_argument('--use-econ', action='store_true', default=cfg.use_econ)
    parser.add_argument('--econ-path', type=str, default=None)
    parser.add_argument('--use-fundamentals', action='store_true', default=cfg.use_fundamentals)
    parser.add_argument('--fundamentals-path', type=str, default=None)
    parser.add_argument('--use-vix-features', action='store_true', default=cfg.use_vix_features)
    parser.add_argument('--no-vix-features', action='store_false', dest='use_vix_features')
    parser.add_argument('--vix-features-path', type=str, default=None)
    parser.add_argument('--vix-n-layers', type=int, default=cfg.vix_n_layers)
    parser.add_argument('--vix-d-model', type=int, default=cfg.vix_d_model)
    parser.add_argument('--vix-d-state', type=int, default=cfg.vix_d_state)
    parser.add_argument('--spike-thresh', type=float, default=2.0)
    parser.add_argument('--extreme-thresh', type=float, default=4.0)
    parser.add_argument('--spike-weight', type=float, default=3.0)
    parser.add_argument('--extreme-weight', type=float, default=5.0)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_diagnostics(args=None):
    """Run all diagnostic phases. Can be called from train.py or standalone."""
    global _phase_num, _total_passed, _total_failed
    _phase_num = 0
    _total_passed = 0
    _total_failed = 0

    print("\n" + "=" * 60)
    print("  MAMBA VIX TRAINING DIAGNOSTICS")
    print("=" * 60)
    t0 = time.time()

    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    # Phase 1
    if not phase_environment():
        _summary(t0)
        return 1

    # Phase 2
    phase_config(args)

    # Phase 3
    paths = phase_data_paths()
    if paths is None:
        _summary(t0)
        return 1

    # Phase 4
    if not phase_data_overlap(paths):
        _summary(t0)
        return 1

    # Phase 5
    train_ds, val_ds = phase_dataset_build(args, paths)
    if train_ds is None:
        _summary(t0)
        return 1

    # Phase 6
    batch = phase_dataloader_smoke(train_ds, args)
    if batch is None:
        _summary(t0)
        return 1

    # Phase 7
    model = phase_model_build(args, train_ds)
    if model is None:
        _summary(t0)
        return 1

    # Phase 8
    phase_forward(model, batch, args)

    # Phase 9
    phase_backward(model, batch, args)

    # Phase 10
    phase_validation_checks(model, args, train_ds)

    _summary(t0)
    return 0 if _total_failed == 0 else 1


def _summary(t0):
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS COMPLETE  ({elapsed:.1f}s)")
    print(f"  Passed: {_total_passed}  |  Failed: {_total_failed}")
    if _total_failed == 0:
        print(f"  ✅ All checks passed — training should work")
    else:
        print(f"  ❌ {_total_failed} check(s) failed — see above for details")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    sys.exit(run_diagnostics())
