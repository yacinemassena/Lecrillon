"""
Mamba-Only VIX Prediction Training.

1s stock bars → Linear → Mamba (selective scan) → Pool → VIX prediction.
No Transformer encoder.

Usage:
    python train.py                    # default settings from trainconfig.py
    python train.py --epochs 100       # override epochs
    python train.py --seq-len 50000    # override sequence length
    python train.py --resume           # resume from checkpoint
    
    # Multi-GPU training (6 GPUs)
    torchrun --nproc_per_node=6 train.py --seq-len 15000
    torchrun --nproc_per_node=6 train.py --seq-len 15000 --resume  # resume multi-GPU
    
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
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from trainconfig import DEFAULT_CONFIG as cfg
from dashboard import SimpleDashboard

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: str = 'logs') -> Path:
    """Setup file logging for this training run. Returns log file path."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'train_{timestamp}.log'
    
    # File handler for this run
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler (minimal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

logger = logging.getLogger(__name__)

# Global dashboard instance
dashboard = SimpleDashboard()

# Log file path (set in main)
LOG_FILE: Optional[Path] = None

# ---------------------------------------------------------------------------
# Distributed Training Utilities
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', 
            rank=rank, 
            world_size=world_size,
            device_id=torch.device(f'cuda:{local_rank}')  # Fixes barrier() warning
        )
        
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
# Checkpoint Utilities
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scaler, epoch, val_loss, checkpoint_dir, is_distributed):
    """Save training checkpoint (overwrites previous)."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Get underlying model for DDP
    model_state = model.module.state_dict() if is_distributed else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
    }
    
    checkpoint_file = checkpoint_path / 'checkpoint.pt'
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_dir, model, optimizer, scaler, device, is_distributed):
    """Load training checkpoint if exists. Returns start_epoch."""
    checkpoint_file = Path(checkpoint_dir) / 'checkpoint.pt'
    
    if not checkpoint_file.exists():
        return 0, None  # Start from epoch 0
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Load model state
    if is_distributed:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', None)
    
    return start_epoch, val_loss


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
class SyntheticBarDataset(Dataset):
    """Generate synthetic bar data for smoke testing when real data unavailable."""

    def __init__(self, num_samples: int = 50, seq_len: int = 2000, num_features: int = 15):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        import bisect
        from loader.bar_mamba_dataset import VIX_BUCKET_BOUNDARIES
        bars = np.random.randn(self.seq_len, self.num_features).astype(np.float32)
        # Synthetic VIX change target: loosely correlated with bar volatility
        vol = np.std(bars[:, 0])  # std of "close" feature
        vix_change = (vol - 1.0) * 2.0 + np.random.randn() * 0.5  # centered near 0
        bucket = bisect.bisect_right(VIX_BUCKET_BOUNDARIES, vix_change)
        return {
            'bars': torch.from_numpy(bars),
            'vix_target': torch.tensor(vix_change, dtype=torch.float32),
            'vix_bucket': torch.tensor(bucket, dtype=torch.long),
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
        buckets = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            T = b['num_bars']
            bars_padded[i, :T, :] = b['bars']
            bar_mask[i, :T] = 1.0
            targets[i] = b['vix_target']
            buckets[i] = b['vix_bucket']

        return {
            'bars': bars_padded,
            'bar_mask': bar_mask,
            'vix_target': targets,
            'vix_bucket': buckets,
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
                amp_dtype=torch.bfloat16, grad_accum=1, epoch=1,
                cls_criterion=None, cls_weight=0.5):
    """Run training iterations with detailed timing.
    
    Args:
        num_steps: Number of steps to run. 0 = full epoch (all samples).
        epoch: Current epoch number for display.
        cls_criterion: Classification loss (CrossEntropyLoss) for bucket head.
        cls_weight: Weight for classification loss in combined objective.
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
        bucket_target = batch['vix_bucket'].to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_gpu = time.time() - t_gpu_start
        
        seq_len = bars.shape[1]

        # Forward pass
        t_fwd_start = time.time()
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(bars, bar_mask)
            pred = outputs['vix_pred']
            reg_loss = criterion(pred, target)
            # Combined loss: regression + classification
            if cls_criterion is not None:
                bucket_logits = outputs['bucket_logits']
                cls_loss = cls_criterion(bucket_logits, bucket_target)
                loss = (reg_loss + cls_weight * cls_loss) / grad_accum
            else:
                loss = reg_loss / grad_accum
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
def val_steps(model, loader, criterion, device, num_steps, amp_dtype=torch.bfloat16,
             cls_criterion=None, cls_weight=0.5):
    """Run validation iterations.
    
    Args:
        num_steps: Number of steps to run. 0 = full validation (all samples).
        cls_criterion: Classification loss for bucket head.
        cls_weight: Weight for classification loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    correct_buckets = 0
    total_samples = 0

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
        bucket_target = batch['vix_bucket'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(bars, bar_mask)
            pred = outputs['vix_pred']
            reg_loss = criterion(pred, target)
            if cls_criterion is not None:
                bucket_logits = outputs['bucket_logits']
                cls_loss = cls_criterion(bucket_logits, bucket_target)
                loss = reg_loss + cls_weight * cls_loss
            else:
                loss = reg_loss
                bucket_logits = outputs.get('bucket_logits')

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

        # Bucket accuracy
        if bucket_logits is not None:
            pred_buckets = bucket_logits.argmax(dim=-1)
            correct_buckets += (pred_buckets == bucket_target).sum().item()
            total_samples += bucket_target.shape[0]

    if num_batches == 0:
        return {'loss': 0.0, 'mae': 0.0, 'bucket_acc': 0.0}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = (preds - targets).abs().mean().item()  # now in VIX points directly
    bucket_acc = correct_buckets / max(total_samples, 1)

    return {
        'loss': total_loss / num_batches,
        'mae': mae,
        'bucket_acc': bucket_acc,
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
    parser.add_argument('--lr', type=float, default=cfg.lr,
                        help='Learning rate (default: 1e-6, from HP sweep)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name for logging (default: auto-generated)')
    parser.add_argument('--results-csv', type=str, default=None,
                        help='CSV file to append results (for HP sweep)')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    is_main = is_main_process()

    # Setup logging (only on main process)
    global LOG_FILE
    if is_main:
        LOG_FILE = setup_logging()
        logger.info(f"Training started - logs saved to {LOG_FILE}")

    # Start dashboard (only on main process)
    if is_main:
        dashboard.start()
        dashboard.log(f"[dim]📝 Logs: {LOG_FILE}[/]")
        dashboard.log(f"[bold cyan]📦 Config:[/] Batch: {args.batch_size} | Workers: {args.num_workers}")
        dashboard.log(f"[bold cyan]📊 Config:[/] Seq={args.seq_len:,} | Epochs={args.epochs}")
        logger.info(f"Config: batch={args.batch_size}, workers={args.num_workers}, seq_len={args.seq_len}, epochs={args.epochs}, lr={args.lr}")
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

    # Map-style dataset now supports native shuffling
    # For distributed training, could add DistributedSampler here
    
    # Reduce workers for multi-GPU to prevent OOM (6 GPUs × 4 workers = 24 processes)
    effective_workers = max(1, args.num_workers // world_size) if is_distributed else args.num_workers
    if is_main and is_distributed and effective_workers != args.num_workers:
        dashboard.log(f"[dim]Workers reduced: {args.num_workers} → {effective_workers} per GPU[/]")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=effective_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(effective_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=effective_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(effective_workers > 0),
    )

    # Build model
    if is_main:
        dashboard.log(f"[dim]Building model: d={args.d_model}, layers={args.n_layers}, state={args.d_state}[/]")

    from mamba_only_model import MambaOnlyVIX
    from loader.bar_mamba_dataset import NUM_VIX_BUCKETS
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
        num_buckets=NUM_VIX_BUCKETS,
    ).to(device)

    # Wrap model in DDP for multi-GPU training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            dashboard.log(f"[dim]Model wrapped in DistributedDataParallel[/]")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if is_main:
        dashboard.log(f"[dim]LR: {args.lr}[/]")
    criterion = nn.HuberLoss(delta=0.25)  # ~0.5 VIX points — MSE for small errors, MAE for large
    cls_criterion = nn.CrossEntropyLoss()  # bucket classification head

    # AMP
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        start_epoch, prev_val_loss = load_checkpoint(
            args.checkpoint_dir, model, optimizer, scaler, device, is_distributed
        )
        if is_main:
            if start_epoch > 0:
                dashboard.log(f"[bold yellow]🔄 Resumed from epoch {start_epoch}[/] (val_loss={prev_val_loss:.4f if prev_val_loss else 'N/A'})")
            else:
                dashboard.log(f"[dim]No checkpoint found, starting fresh[/]")

    if is_main:
        dashboard.log(f"[dim]AMP: {amp_dtype}[/]")
        dashboard.log("─" * 50)

    # Best model tracking
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
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
            cls_criterion=cls_criterion, cls_weight=0.5,
        )

        # Validate (only on main process)
        if is_main:
            val_metrics = val_steps(
                model.module if is_distributed else model,
                val_loader, criterion, device,
                args.val_steps, amp_dtype,
                cls_criterion=cls_criterion, cls_weight=0.5,
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
                f"[green]\u2713 Epoch {epoch+1}/{args.epochs}:[/] "
                f"steps={train_steps_count} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_mae={val_metrics['mae']:.2f}pt | "
                f"bucket_acc={val_metrics['bucket_acc']:.1%} | "
                f"iter={avg_iter_time:.2f}s/it | "
                f"VRAM={mem_alloc:.1f}GB"
            )
            # Log to file
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                f"val_mae={val_metrics['mae']:.2f}pt, bucket_acc={val_metrics['bucket_acc']:.1%}, "
                f"iter={avg_iter_time:.2f}s/it, VRAM={mem_alloc:.1f}GB"
            )

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_ckpt = save_checkpoint(
                    model, optimizer, scaler, epoch + 1, val_metrics['loss'],
                    args.checkpoint_dir + '/best', is_distributed
                )
                dashboard.log(f"[bold green]🏆 New best model![/] val_loss={best_val_loss:.4f}")
                logger.info(f"New best model saved: val_loss={best_val_loss:.4f}")

            # Save checkpoint every N epochs (only on main process)
            if (epoch + 1) % args.save_every == 0:
                ckpt_file = save_checkpoint(
                    model, optimizer, scaler, epoch + 1, val_metrics['loss'],
                    args.checkpoint_dir, is_distributed
                )
                dashboard.log(f"[bold blue]💾 Checkpoint saved:[/] {ckpt_file}")

    # Save final checkpoint
    if is_main and args.epochs > 0:
        ckpt_file = save_checkpoint(
            model, optimizer, scaler, args.epochs, val_metrics['loss'],
            args.checkpoint_dir, is_distributed
        )
        dashboard.log(f"[bold blue]💾 Final checkpoint saved:[/] {ckpt_file}")

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

        # Check 2: Forward pass produces output (both heads)
        eval_model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 100, num_features).to(device)
            out = eval_model(dummy)
            ok = ('vix_pred' in out and out['vix_pred'].shape == (1,)
                  and 'bucket_logits' in out and out['bucket_logits'].shape[0] == 1)
            dashboard.log(f"  [{'[green]✓' if ok else '[red]✗'}] Forward pass correct (dual-head)[/]")
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
        
        # Log results to CSV for HP sweep
        if args.results_csv:
            import csv
            from pathlib import Path
            csv_path = Path(args.results_csv)
            write_header = not csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['exp_name', 'lr', 'd_model', 'n_layers', 'd_state', 
                                   'seq_len', 'epochs', 'final_train_loss', 'final_val_loss', 
                                   'final_val_mae_pt', 'final_bucket_acc', 'checks_passed'])
                exp_name = args.exp_name or f"lr_{args.lr}"
                writer.writerow([exp_name, args.lr, args.d_model, args.n_layers, args.d_state,
                               args.seq_len, args.epochs, f"{train_loss:.6f}", 
                               f"{val_metrics['loss']:.6f}", f"{val_metrics['mae']:.4f}",
                               f"{val_metrics['bucket_acc']:.4f}",
                               f"{checks_passed}/{total_checks}"])
            dashboard.log(f"[bold green]📊 Results appended to:[/] {args.results_csv}")
        
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
