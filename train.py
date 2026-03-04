"""
Mamba-Only VIX Prediction Training.

1s stock bars → Linear → Mamba (selective scan) → Pool → VIX prediction.
No Transformer encoder.

Usage:
    python train.py                    # default settings from trainconfig.py
    python train.py --epochs 100       # override epochs
    python train.py --seq-len 50000    # override sequence length
    
    # Multi-GPU training (6 GPUs)
    torchrun --nproc_per_node=6 train.py --seq-len 15000
    
Edit trainconfig.py to change default settings.
"""

import os
import sys
import argparse
import logging
import datetime
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from trainconfig import DEFAULT_CONFIG as cfg
from dashboard import SimpleDashboard

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise, dashboard handles display
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Global dashboard instance
dashboard = SimpleDashboard()

# ---------------------------------------------------------------------------
# Distributed Training Utilities
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    return 0, 1, 0  # Single GPU fallback


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Path detection (WSL or Windows)
# ---------------------------------------------------------------------------
def get_data_paths() -> Dict[str, Path]:
    """Detect data paths for WSL, Windows, or Linux VPS."""
    if os.path.exists('/workspace/datasets'):
        # Linux VPS (Docker)
        base = Path('/workspace/datasets')
    elif os.path.exists('/mnt/d/Mamba v2/datasets'):
        # WSL
        base = Path('/mnt/d/Mamba v2/datasets')
    elif os.path.exists(r'D:\Mamba v2\datasets'):
        # Windows
        base = Path(r'D:\Mamba v2\datasets')
    else:
        return {}

    return {
        'stock': base / 'Stock_Data_1s',
        'vix': base / 'VIX',
        'rv': base / 'SPY_daily_rv',
    }


def check_data_overlap(paths: Dict[str, Path]) -> bool:
    """Check if stock and VIX data have overlapping dates."""
    stock_path = paths.get('stock')
    vix_path = paths.get('vix')

    if not stock_path or not stock_path.exists():
        return False
    if not vix_path or not vix_path.exists():
        return False

    # Get stock date range
    import pandas as pd
    stock_dates = set()
    for f in stock_path.glob('*.parquet'):
        try:
            dt = pd.to_datetime(f.stem.split('.')[0]).date()
            stock_dates.add(dt)
        except Exception:
            continue

    if not stock_dates:
        return False

    # Get VIX date range
    vix_dates = set()
    for csv_file in sorted(vix_path.glob('VIX_*.csv')):
        try:
            df = pd.read_csv(csv_file, usecols=['date', 'close'], nrows=1)
            year = int(csv_file.stem.split('_')[1])
            # Approximate: if stock dates are in this year range
            for sd in stock_dates:
                if sd.year == year:
                    vix_dates.add(sd)
        except Exception:
            continue

    # Check for dates where stock has data and VIX has next-day data
    stock_max = max(stock_dates)
    stock_min = min(stock_dates)
    logger.info(f"Stock data range: {stock_min} to {stock_max} ({len(stock_dates)} days)")

    # Quick check: load VIX for the stock date range
    from loader.bar_mamba_dataset import load_vix_daily_close
    vix_daily = load_vix_daily_close(str(vix_path))
    vix_date_set = set(vix_daily.keys())

    # Find overlapping dates (stock day D where VIX exists at D+1..D+5)
    from datetime import timedelta
    overlap_count = 0
    for d in sorted(stock_dates):
        for offset in range(1, 6):
            if (d + timedelta(days=offset)) in vix_date_set:
                overlap_count += 1
                break

    logger.info(f"VIX data: {len(vix_daily)} days total")
    logger.info(f"Stock-VIX overlap (anchors with targets): {overlap_count}")
    return overlap_count >= 5  # Need at least 5 for smoke test


# ---------------------------------------------------------------------------
# Synthetic Dataset (fallback)
# ---------------------------------------------------------------------------
class SyntheticBarDataset(IterableDataset):
    """Generate synthetic bar data for smoke testing when real data unavailable."""

    def __init__(self, num_samples: int = 50, seq_len: int = 2000, num_features: int = 15):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features

    def __iter__(self):
        for _ in range(self.num_samples):
            bars = np.random.randn(self.seq_len, self.num_features).astype(np.float32)
            # Synthetic VIX target: loosely correlated with bar volatility
            vol = np.std(bars[:, 0])  # std of "close" feature
            vix = vol * 5 + np.random.randn() * 0.5
            yield {
                'bars': torch.from_numpy(bars),
                'vix_target': torch.tensor(vix, dtype=torch.float32),
                'num_bars': self.seq_len,
                'anchor_date': '2005-01-01',
            }

    @staticmethod
    def collate_fn(batch):
        """Same collate as BarMambaDataset."""
        if not batch:
            return {}
        max_len = max(b['num_bars'] for b in batch)
        num_features = batch[0]['bars'].shape[1]
        B = len(batch)

        bars_padded = torch.zeros(B, max_len, num_features)
        bar_mask = torch.zeros(B, max_len)
        targets = torch.zeros(B)

        for i, b in enumerate(batch):
            T = b['num_bars']
            bars_padded[i, :T, :] = b['bars']
            bar_mask[i, :T] = 1.0
            targets[i] = b['vix_target']

        return {
            'bars': bars_padded,
            'bar_mask': bar_mask,
            'vix_target': targets,
            'num_bars': [b['num_bars'] for b in batch],
        }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_steps(model, loader, optimizer, criterion, scaler, device, num_steps,
                amp_dtype=torch.bfloat16, grad_accum=1, epoch=1):
    """Run training iterations with detailed timing.
    
    Args:
        num_steps: Number of steps to run. 0 = full epoch (all samples).
        epoch: Current epoch number for display.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    
    # If num_steps=0, iterate through full epoch
    if num_steps == 0:
        data_iter = enumerate(loader)
    else:
        loader_iter = iter(loader)
        data_iter = range(num_steps)

    for step_data in data_iter:
        if num_steps == 0:
            step, batch = step_data
        else:
            step = step_data
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
        
        torch.cuda.synchronize()
        t_start = time.time()
        
        # Data loading timing (already loaded if num_steps=0)
        t_data_start = time.time()
        t_data = time.time() - t_data_start

        # GPU transfer
        t_gpu_start = time.time()
        bars = batch['bars'].to(device, non_blocking=True)
        bar_mask = batch['bar_mask'].to(device, non_blocking=True)
        target = batch['vix_target'].to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_gpu = time.time() - t_gpu_start
        
        seq_len = bars.shape[1]

        # Forward pass
        t_fwd_start = time.time()
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(bars, bar_mask)
            pred = outputs['vix_pred']
            loss = criterion(pred, target) / grad_accum
        torch.cuda.synchronize()
        t_fwd = time.time() - t_fwd_start

        # Backward pass
        t_bwd_start = time.time()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t_bwd_start

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)

        step_loss = loss.item() * grad_accum
        total_loss += step_loss
        num_batches += 1
        
        torch.cuda.synchronize()
        t_total = time.time() - t_start

        # VRAM stats - use reserved memory which shows actual GPU allocation
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9

        # Update dashboard with per-step timing
        step_display = f"E{epoch} Step {step + 1}"
        if num_steps > 0:
            step_display += f"/{num_steps}"
        
        dashboard.console.print(
            f"[green]{step_display}[/] | "
            f"loss={step_loss:.4f} | seq={seq_len:,} | "
            f"data={t_data:.2f}s fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s | "
            f"[cyan]{t_total:.2f}s/it[/] | "
            f"VRAM={mem_reserved:.1f}GB"
        )

    epoch_time = time.time() - epoch_start_time
    avg_iter_time = epoch_time / max(num_batches, 1)
    return total_loss / max(num_batches, 1), num_batches, avg_iter_time


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_steps(model, loader, criterion, device, num_steps, amp_dtype=torch.bfloat16):
    """Run validation iterations.
    
    Args:
        num_steps: Number of steps to run. 0 = full validation (all samples).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    # If num_steps=0, iterate through full validation set
    if num_steps == 0:
        data_iter = enumerate(loader)
    else:
        loader_iter = iter(loader)
        data_iter = range(num_steps)

    for step_data in data_iter:
        if num_steps == 0:
            step, batch = step_data
        else:
            step = step_data
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

        bars = batch['bars'].to(device, non_blocking=True)
        bar_mask = batch['bar_mask'].to(device, non_blocking=True)
        target = batch['vix_target'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(bars, bar_mask)
            pred = outputs['vix_pred']
            loss = criterion(pred, target)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    if num_batches == 0:
        return {'loss': 0.0, 'mae': 0.0}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = (preds - targets).abs().mean().item()

    return {
        'loss': total_loss / num_batches,
        'mae': mae,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Mamba-Only VIX Prediction Training')
    # All defaults come from trainconfig.py - edit that file to change defaults
    parser.add_argument('--synthetic', action='store_true', default=cfg.force_synthetic,
                        help='Force synthetic data')
    parser.add_argument('--real', action='store_true', default=cfg.force_real,
                        help='Force real data (fail if unavailable)')
    parser.add_argument('--train-steps', type=int, default=cfg.train_steps,
                        help='Training steps per epoch (0 = full epoch)')
    parser.add_argument('--val-steps', type=int, default=cfg.val_steps,
                        help='Validation steps per epoch (0 = full validation)')
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--seq-len', type=int, default=cfg.seq_len,
                        help='Max sequence length in 1-sec bars')
    parser.add_argument('--d-model', type=int, default=cfg.d_model)
    parser.add_argument('--n-layers', type=int, default=cfg.n_layers)
    parser.add_argument('--d-state', type=int, default=cfg.d_state)
    # No caching - direct file loading
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers,
                        help='DataLoader workers for parallel data loading')
    parser.add_argument('--train-end', type=str, default=cfg.train_end,
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--val-end', type=str, default=cfg.val_end,
                        help='End date for validation data (YYYY-MM-DD)')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    is_main = is_main_process()

    # Start dashboard (only on main process)
    if is_main:
        dashboard.start()
        dashboard.log(f"[bold cyan]📦 Config:[/] Batch: {args.batch_size} | Workers: {args.num_workers}")
        dashboard.log(f"[bold cyan]📊 Config:[/] Seq={args.seq_len:,} | Epochs={args.epochs}")
        if is_distributed:
            dashboard.log(f"[bold magenta]🚀 Distributed:[/] {world_size} GPUs")
        dashboard.state.total_epochs = args.epochs
        dashboard.state.batch_size = args.batch_size

    seed_everything(42 + rank)  # Different seed per rank for data augmentation

    # Device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available() and is_main:
        gpu_name = torch.cuda.get_device_name(local_rank)
        vram_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        dashboard.log(f"[bold green]🖥️ Device:[/] {gpu_name} ({vram_gb:.1f}GB)")
        dashboard.state.vram_total = vram_gb
    elif is_main:
        dashboard.log(f"[yellow]🖥️ Device:[/] CPU")

    # Data source selection
    use_synthetic = args.synthetic
    data_paths = get_data_paths()

    if not use_synthetic and not args.real:
        # Auto-detect
        if data_paths:
            if is_main:
                dashboard.log("[dim]Checking data availability...[/]")
            try:
                has_overlap = check_data_overlap(data_paths)
                if not has_overlap:
                    if is_main:
                        dashboard.log("[yellow]No stock-VIX overlap → synthetic data[/]")
                    use_synthetic = True
                else:
                    if is_main:
                        dashboard.log("[green]✓ Real data available[/]")
            except Exception as e:
                if is_main:
                    dashboard.log(f"[yellow]Data check failed: {e} → synthetic[/]")
                use_synthetic = True
        else:
            if is_main:
                dashboard.log("[yellow]No data directory → synthetic data[/]")
            use_synthetic = True

    if args.real and use_synthetic:
        logger.error("--real requested but no usable data found")
        sys.exit(1)

    # Build datasets
    num_features = 15
    train_sampler = None
    val_sampler = None
    
    if use_synthetic:
        if is_main:
            dashboard.log(f"[yellow]Using SYNTHETIC data[/] (seq_len={args.seq_len})")
        train_dataset = SyntheticBarDataset(
            num_samples=50 * world_size, seq_len=args.seq_len, num_features=num_features
        )
        val_dataset = SyntheticBarDataset(
            num_samples=20 * world_size, seq_len=args.seq_len, num_features=num_features
        )
        collate_fn = SyntheticBarDataset.collate_fn
    else:
        if is_main:
            dashboard.log("[green]Using REAL data[/]")
        from loader.bar_mamba_dataset import BarMambaDataset

        stock_path = str(data_paths['stock'])
        vix_path = str(data_paths['vix'])

        train_dataset = BarMambaDataset(
            stock_data_path=stock_path,
            vix_data_path=vix_path,
            split='train',
            max_total_bars=args.seq_len,
            train_end=args.train_end,
            val_end=args.val_end,
        )
        val_dataset = BarMambaDataset(
            stock_data_path=stock_path,
            vix_data_path=vix_path,
            split='val',
            max_total_bars=args.seq_len,
            train_end=args.train_end,
            val_end=args.val_end,
        )
        num_features = train_dataset.num_features
        collate_fn = BarMambaDataset.collate_fn

    # Note: DistributedSampler not used with IterableDataset
    # For real distributed training with map-style datasets, add sampler here
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Build model
    if is_main:
        dashboard.log(f"[dim]Building model: d={args.d_model}, layers={args.n_layers}, state={args.d_state}[/]")

    from mamba_only_model import MambaOnlyVIX
    model = MambaOnlyVIX(
        num_features=num_features,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=4,
        expand=2,
        dropout=0.1,
        pooling='attention',
        head_hidden=128,
    ).to(device)

    # Wrap model in DDP for multi-GPU training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            dashboard.log(f"[dim]Model wrapped in DistributedDataParallel[/]")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    # AMP
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    if is_main:
        dashboard.log(f"[dim]AMP: {amp_dtype}[/]")
        dashboard.log("─" * 50)

    # Training loop
    for epoch in range(args.epochs):
        if is_main:
            dashboard.state.epoch = epoch + 1
            dashboard.log(f"\n[bold cyan]═══ Epoch {epoch+1}/{args.epochs} ═══[/]")
            dashboard.state.step = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Sync all processes before epoch
        if is_distributed:
            dist.barrier()

        # Train
        train_loss, train_steps_count, avg_iter_time = train_steps(
            model, train_loader, optimizer, criterion, scaler,
            device, args.train_steps, amp_dtype, epoch=epoch+1,
        )

        # Validate (only on main process)
        if is_main:
            val_metrics = val_steps(
                model.module if is_distributed else model,
                val_loader, criterion, device,
                args.val_steps, amp_dtype,
            )

            # Memory stats
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.max_memory_allocated() / 1e9
                mem_res = torch.cuda.max_memory_reserved() / 1e9
            else:
                mem_alloc = mem_res = 0.0

            dashboard.state.val_loss = val_metrics['loss']
            dashboard.state.val_mae = val_metrics['mae']
            dashboard.log(
                f"[green]✓ Epoch {epoch+1}/{args.epochs}:[/] "
                f"steps={train_steps_count} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"iter={avg_iter_time:.2f}s/it | "
                f"VRAM={mem_alloc:.1f}GB"
            )

    # Final checks (only on main process)
    checks_passed = 0
    total_checks = 5
    
    if is_main:
        dashboard.log("\n" + "═" * 50)
        dashboard.log("[bold]VALIDATION CHECKS[/]")
        dashboard.log("═" * 50)

        # Get underlying model for DDP
        eval_model = model.module if is_distributed else model

        # Check 1: Model has parameters
        n_params = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)
        ok = n_params > 0
        dashboard.log(f"  [{'[green]✓' if ok else '[red]✗'}] Model has {n_params:,} parameters[/]")
        checks_passed += ok

        # Check 2: Forward pass produces output
        eval_model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 100, num_features).to(device)
            out = eval_model(dummy)
            ok = 'vix_pred' in out and out['vix_pred'].shape == (1,)
            dashboard.log(f"  [{'[green]✓' if ok else '[red]✗'}] Forward pass correct[/]")
            checks_passed += ok

        # Check 3: Backward pass works
        eval_model.train()
        dummy = torch.randn(1, 100, num_features, device=device, requires_grad=False)
        out = eval_model(dummy)
        loss = out['vix_pred'].sum()
        loss.backward()
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in eval_model.parameters())
        dashboard.log(f"  [{'[green]✓' if has_grads else '[red]✗'}] Gradients flow[/]")
        checks_passed += has_grads

        # Check 4: Loss is finite
        ok = np.isfinite(train_loss) and train_loss > 0
        dashboard.log(f"  [{'[green]✓' if ok else '[red]✗'}] Loss finite ({train_loss:.4f})[/]")
        checks_passed += ok

        # Check 5: No NaN in parameters
        has_nan = any(torch.isnan(p).any() for p in eval_model.parameters())
        ok = not has_nan
        dashboard.log(f"  [{'[green]✓' if ok else '[red]✗'}] No NaN in params[/]")
        checks_passed += ok

        dashboard.log(f"\n[bold]Result: {checks_passed}/{total_checks} checks passed[/]")
        dashboard.stop()

    # Cleanup distributed
    cleanup_distributed()

    if is_main:
        if checks_passed == total_checks:
            print("\n✅ TRAINING COMPLETE")
            return 0
        else:
            print("\n❌ TRAINING FAILED")
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
