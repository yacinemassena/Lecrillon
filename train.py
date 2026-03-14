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
# Multi-Horizon Spike-Weighted Loss
# ---------------------------------------------------------------------------
HORIZONS = [1, 7, 15, 30]  # +1d, +7d, +15d, +30d
NUM_HORIZONS = len(HORIZONS)


class SpikeWeightedHuberLoss(nn.Module):
    """
    Multi-horizon HuberLoss with tiered spike weighting.
    
    Tiers (points-based):
      - Normal: |change| < spike_thresh → weight = 1.0
      - Spike: |change| >= spike_thresh → weight = spike_weight (3×)
      - Extreme: |change| >= extreme_thresh → weight = extreme_weight (5×)
    
    This encourages the model to pay more attention to VIX spikes/crushes
    without distorting the data distribution.
    """
    
    def __init__(self, delta: float = 0.25, spike_thresh: float = 2.0, 
                 extreme_thresh: float = 4.0, spike_weight: float = 3.0, 
                 extreme_weight: float = 5.0):
        super().__init__()
        self.delta = delta
        self.spike_thresh = spike_thresh
        self.extreme_thresh = extreme_thresh
        self.spike_weight = spike_weight
        self.extreme_weight = extreme_weight
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                horizon_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: [B, 4] predictions for each horizon
            target: [B, 4] targets for each horizon
            horizon_mask: [B, 4] validity mask (1=valid, 0=missing horizon)
        
        Returns:
            Scalar loss
        """
        # Base Huber loss per element
        base_loss = self.huber(pred, target)  # [B, 4]
        
        # Tiered weights based on absolute target change
        abs_target = target.abs()
        weights = torch.ones_like(target)
        weights = torch.where(abs_target >= self.spike_thresh, 
                              torch.tensor(self.spike_weight, device=target.device), weights)
        weights = torch.where(abs_target >= self.extreme_thresh, 
                              torch.tensor(self.extreme_weight, device=target.device), weights)
        
        # Apply spike weights
        weighted_loss = base_loss * weights
        
        # Apply horizon mask if provided (mask out missing horizons)
        if horizon_mask is not None:
            weighted_loss = weighted_loss * horizon_mask
            # Normalize by valid horizons only
            valid_count = horizon_mask.sum()
            if valid_count > 0:
                return weighted_loss.sum() / valid_count
            else:
                return weighted_loss.sum()
        
        return weighted_loss.mean()
    
    def get_spike_stats(self, target: torch.Tensor, horizon_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Get statistics about spike distribution in batch."""
        abs_target = target.abs()
        
        if horizon_mask is not None:
            valid = horizon_mask.bool()
            abs_target = abs_target[valid]
        
        if abs_target.numel() == 0:
            return {'normal_pct': 0, 'spike_pct': 0, 'extreme_pct': 0}
        
        total = abs_target.numel()
        normal = (abs_target < self.spike_thresh).sum().item()
        spike = ((abs_target >= self.spike_thresh) & (abs_target < self.extreme_thresh)).sum().item()
        extreme = (abs_target >= self.extreme_thresh).sum().item()
        
        return {
            'normal_pct': normal / total * 100,
            'spike_pct': spike / total * 100,
            'extreme_pct': extreme / total * 100,
        }


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
        'stock': base / 'Stock_Data_2min',
        'options': base / 'opt_trade_2min',
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

    def __init__(self, num_samples: int = 50, seq_len: int = 2000, num_features: int = 29):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        bars = np.random.randn(self.seq_len, self.num_features).astype(np.float32)
        # Synthetic VIX change target: loosely correlated with bar volatility
        vol = np.std(bars[:, 0])  # std of "close" feature
        vix_change = (vol - 1.0) * 2.0 + np.random.randn() * 0.5  # centered near 0
        return {
            'bars': torch.from_numpy(bars),
            'vix_target': torch.tensor(vix_change, dtype=torch.float32),
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
        target = batch['vix_targets'].to(device, non_blocking=True)  # [B, 4] multi-horizon
        horizon_mask = batch['horizon_mask'].to(device, non_blocking=True)  # [B, 4]
        
        # Optional news data
        news_embs = batch.get('news_embs')
        news_mask = batch.get('news_mask')
        news_indices = batch.get('news_indices')
        
        if news_embs is not None:
            news_embs = news_embs.to(device, non_blocking=True)
        if news_mask is not None:
            news_mask = news_mask.to(device, non_blocking=True)
        if news_indices is not None:
            news_indices = news_indices.to(device, non_blocking=True)
        
        # Optional options data
        options = batch.get('options')
        options_mask = batch.get('options_mask')
        
        if options is not None:
            options = options.to(device, non_blocking=True)
        if options_mask is not None:
            options_mask = options_mask.to(device, non_blocking=True)
        
        # Optional macro context and bar timestamps (for FiLM)
        macro_context = batch.get('macro_context')
        bar_timestamps = batch.get('bar_timestamps')
        
        if macro_context is not None:
            macro_context = macro_context.to(device, non_blocking=True)
        if bar_timestamps is not None:
            bar_timestamps = bar_timestamps.to(device, non_blocking=True)
        
        # Optional GDELT data
        gdelt_embs = batch.get('gdelt_embs')
        gdelt_mask = batch.get('gdelt_mask')
        gdelt_timestamps = batch.get('gdelt_timestamps')
        news_timestamps = batch.get('news_timestamps')
        
        if gdelt_embs is not None:
            gdelt_embs = gdelt_embs.to(device, non_blocking=True)
        if gdelt_mask is not None:
            gdelt_mask = gdelt_mask.to(device, non_blocking=True)
        if gdelt_timestamps is not None:
            gdelt_timestamps = gdelt_timestamps.to(device, non_blocking=True)
        if news_timestamps is not None:
            news_timestamps = news_timestamps.to(device, non_blocking=True)
        
        torch.cuda.synchronize()
        t_gpu = time.time() - t_gpu_start
        
        seq_len = bars.shape[1]

        # Forward pass
        t_fwd_start = time.time()
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(
                bars, bar_mask,
                options=options,
                options_mask=options_mask,
                news_embs=news_embs,
                news_mask=news_mask,
                news_indices=news_indices,
                news_timestamps=news_timestamps,
                gdelt_embs=gdelt_embs,
                gdelt_mask=gdelt_mask,
                gdelt_timestamps=gdelt_timestamps,
                macro_context=macro_context,
                bar_timestamps=bar_timestamps,
            )
            pred = outputs['vix_pred']  # [B, 4] multi-horizon
            
            # Combined loss from all streams (with spike weighting and horizon mask)
            loss = criterion(pred, target, horizon_mask)
            
            # Per-stream auxiliary losses (for monitoring) - use +1d target only
            stream_losses = {'combined': loss.item()}
            target_1d = target[:, 0]  # +1d target for aux losses
            
            if 'stock_pred' in outputs:
                stock_loss = F.huber_loss(outputs['stock_pred'], target_1d, delta=0.25)
                loss = loss + 0.3 * stock_loss  # Auxiliary weight
                stream_losses['stock'] = stock_loss.item()
            
            if 'news_pred' in outputs:
                news_loss = F.huber_loss(outputs['news_pred'], target_1d, delta=0.25)
                loss = loss + 0.3 * news_loss
                stream_losses['news'] = news_loss.item()
            
            if 'options_pred' in outputs:
                options_loss = F.huber_loss(outputs['options_pred'], target_1d, delta=0.25)
                loss = loss + 0.3 * options_loss
                stream_losses['options'] = options_loss.item()
            
            loss = loss / grad_accum
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

        # Update dashboard with per-step timing and per-stream losses
        step_display = f"E{epoch} Step {step + 1}"
        if num_steps > 0:
            step_display += f"/{num_steps}"
        
        # Build per-stream loss display with colors
        loss_parts = []
        if 'stock' in stream_losses:
            loss_parts.append(f"[cyan]S:{stream_losses['stock']:.3f}[/]")
        if 'news' in stream_losses:
            loss_parts.append(f"[yellow]N:{stream_losses['news']:.3f}[/]")
        if 'options' in stream_losses:
            loss_parts.append(f"[magenta]O:{stream_losses['options']:.3f}[/]")
        loss_parts.append(f"[green]C:{stream_losses['combined']:.3f}[/]")
        
        stream_display = " ".join(loss_parts) if loss_parts else f"loss={step_loss:.4f}"
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        dashboard.console.print(
            f"[bold]{step_display}[/] | {stream_display} | seq={seq_len:,} | "
            f"fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s | "
            f"[dim]{t_total:.2f}s/it[/] | VRAM={mem_reserved:.1f}GB | {timestamp}"
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
    all_horizon_masks = []
    stock_preds = []
    news_preds = []
    options_preds = []

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
        target = batch['vix_targets'].to(device, non_blocking=True)  # [B, 4] multi-horizon
        horizon_mask = batch['horizon_mask'].to(device, non_blocking=True)  # [B, 4]
        
        # Optional news data
        news_embs = batch.get('news_embs')
        news_mask = batch.get('news_mask')
        news_indices = batch.get('news_indices')
        
        if news_embs is not None:
            news_embs = news_embs.to(device, non_blocking=True)
        if news_mask is not None:
            news_mask = news_mask.to(device, non_blocking=True)
        if news_indices is not None:
            news_indices = news_indices.to(device, non_blocking=True)
        
        # Optional options data
        options = batch.get('options')
        options_mask = batch.get('options_mask')
        
        if options is not None:
            options = options.to(device, non_blocking=True)
        if options_mask is not None:
            options_mask = options_mask.to(device, non_blocking=True)
        
        # Optional macro context and bar timestamps (for FiLM)
        macro_context = batch.get('macro_context')
        bar_timestamps = batch.get('bar_timestamps')
        
        if macro_context is not None:
            macro_context = macro_context.to(device, non_blocking=True)
        if bar_timestamps is not None:
            bar_timestamps = bar_timestamps.to(device, non_blocking=True)
        
        # Optional GDELT data
        gdelt_embs = batch.get('gdelt_embs')
        gdelt_mask = batch.get('gdelt_mask')
        gdelt_timestamps = batch.get('gdelt_timestamps')
        news_timestamps = batch.get('news_timestamps')
        
        if gdelt_embs is not None:
            gdelt_embs = gdelt_embs.to(device, non_blocking=True)
        if gdelt_mask is not None:
            gdelt_mask = gdelt_mask.to(device, non_blocking=True)
        if gdelt_timestamps is not None:
            gdelt_timestamps = gdelt_timestamps.to(device, non_blocking=True)
        if news_timestamps is not None:
            news_timestamps = news_timestamps.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(
                bars, bar_mask,
                options=options,
                options_mask=options_mask,
                news_embs=news_embs,
                news_mask=news_mask,
                news_indices=news_indices,
                news_timestamps=news_timestamps,
                gdelt_embs=gdelt_embs,
                gdelt_mask=gdelt_mask,
                gdelt_timestamps=gdelt_timestamps,
                macro_context=macro_context,
                bar_timestamps=bar_timestamps,
            )
            pred = outputs['vix_pred']  # [B, 4] multi-horizon
            loss = criterion(pred, target, horizon_mask)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
        all_horizon_masks.append(horizon_mask.cpu())
        
        # Track per-stream predictions for MAE (auxiliary heads output +1d only)
        if 'stock_pred' in outputs:
            stock_preds.append(outputs['stock_pred'].cpu())
        if 'news_pred' in outputs:
            news_preds.append(outputs['news_pred'].cpu())
        if 'options_pred' in outputs:
            options_preds.append(outputs['options_pred'].cpu())

    if num_batches == 0:
        return {'loss': 0.0, 'mae_1d': 0.0}

    preds = torch.cat(all_preds)          # [N, 4]
    targets = torch.cat(all_targets)      # [N, 4]
    h_masks = torch.cat(all_horizon_masks)  # [N, 4]
    
    # Per-horizon metrics
    result = {'loss': total_loss / num_batches}
    
    for i, h in enumerate(HORIZONS):
        valid = h_masks[:, i].bool()
        if valid.sum() == 0:
            continue
        
        pred_h = preds[:, i][valid]
        target_h = targets[:, i][valid]
        
        # MAE for this horizon
        mae_h = (pred_h - target_h).abs().mean().item()
        result[f'mae_{h}d'] = mae_h
        
        # Directional accuracy for this horizon
        pred_sign = (pred_h > 0).float()
        target_sign = (target_h > 0).float()
        dir_acc_h = (pred_sign == target_sign).float().mean().item() * 100
        result[f'dir_acc_{h}d'] = dir_acc_h
        
        # Within-threshold accuracy for this horizon
        abs_error = (pred_h - target_h).abs()
        within_1pt_h = (abs_error < 1.0).float().mean().item() * 100
        within_2pt_h = (abs_error < 2.0).float().mean().item() * 100
        result[f'within_1pt_{h}d'] = within_1pt_h
        result[f'within_2pt_{h}d'] = within_2pt_h
    
    # Legacy compatibility: overall MAE is +1d MAE
    result['mae'] = result.get('mae_1d', 0.0)
    result['dir_acc'] = result.get('dir_acc_1d', 0.0)
    result['within_1pt'] = result.get('within_1pt_1d', 0.0)
    result['within_2pt'] = result.get('within_2pt_1d', 0.0)
    
    # Per-stream MAE (auxiliary heads use +1d target)
    target_1d = targets[:, 0]
    if stock_preds:
        stock_mae = (torch.cat(stock_preds) - target_1d).abs().mean().item()
        result['stock_mae'] = stock_mae
    if news_preds:
        news_mae = (torch.cat(news_preds) - target_1d).abs().mean().item()
        result['news_mae'] = news_mae
    if options_preds:
        options_mae = (torch.cat(options_preds) - target_1d).abs().mean().item()
        result['options_mae'] = options_mae

    return result


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
    parser.add_argument('--news-n-layers', type=int, default=cfg.news_n_layers,
                        help='Number of Mamba layers for news stream (default: 2)')
    parser.add_argument('--d-state', type=int, default=cfg.d_state)
    # No caching - direct file loading
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers,
                        help='DataLoader workers for parallel data loading')
    parser.add_argument('--train-start', type=str, default='2005-01-01',
                        help='Start date for training data (YYYY-MM-DD, default: 2005-01-01)')
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
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement, default: 5)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name for logging (default: auto-generated)')
    parser.add_argument('--results-csv', type=str, default=None,
                        help='CSV file to append results (for HP sweep)')
    parser.add_argument('--use-news', action='store_true', default=cfg.use_news,
                        help='Enable news integration (Benzinga embeddings). Enabled by default.')
    parser.add_argument('--no-news', action='store_false', dest='use_news',
                        help='Disable news integration for A/B comparison.')
    parser.add_argument('--news-path', type=str, default=None,
                        help='Path to news embeddings directory (default: datasets/benzinga_embeddings/news)')
    parser.add_argument('--use-options', action='store_true', default=cfg.use_options,
                        help='Enable options flow integration. Enabled by default.')
    parser.add_argument('--no-options', action='store_false', dest='use_options',
                        help='Disable options flow integration for A/B comparison.')
    parser.add_argument('--options-path', type=str, default=None,
                        help='Path to options data directory (default: datasets/opt_trade_1sec)')
    parser.add_argument('--use-macro', action='store_true', default=cfg.use_macro,
                        help='Enable macro FiLM conditioning (Fed/treasury/FOMC)')
    parser.add_argument('--macro-path', type=str, default=None,
                        help='Path to macro_daily.parquet (default: datasets/macro/macro_daily.parquet)')
    parser.add_argument('--use-gdelt', action='store_true', default=cfg.use_gdelt,
                        help='Enable GDELT world state integration. Enabled by default.')
    parser.add_argument('--no-gdelt', action='store_false', dest='use_gdelt',
                        help='Disable GDELT integration for A/B comparison.')
    parser.add_argument('--gdelt-path', type=str, default=None,
                        help='Path to GDELT embeddings directory (default: datasets/GDELT)')
    parser.add_argument('--checkpoint-interval', type=int, default=cfg.checkpoint_interval,
                        help='Fusion checkpoint interval in bars (default: 300 = 5 min)')
    # Spike-weighted loss parameters
    parser.add_argument('--spike-thresh', type=float, default=2.0,
                        help='VIX point change threshold for spike weighting (default: 2.0)')
    parser.add_argument('--extreme-thresh', type=float, default=4.0,
                        help='VIX point change threshold for extreme spike weighting (default: 4.0)')
    parser.add_argument('--spike-weight', type=float, default=3.0,
                        help='Loss weight multiplier for spikes (default: 3.0)')
    parser.add_argument('--extreme-weight', type=float, default=5.0,
                        help='Loss weight multiplier for extreme spikes (default: 5.0)')
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
        logger.info(f"Config: batch={args.batch_size}, workers={args.num_workers}, seq_len={args.seq_len}, epochs={args.epochs}, lr={args.lr}, use_news={args.use_news}, use_options={args.use_options}")
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
    num_features = 45  # Stock features from Stock_Data_2min (47 cols - ticker - bar_timestamp)
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
        
        # News path: use provided path or default
        # Note: data loader looks for news_daily/ subdirectory internally
        news_path = args.news_path
        if args.use_news and news_path is None:
            # Try default paths (point to benzinga_embeddings parent dir)
            for candidate in [
                data_paths.get('stock', Path('.')).parent / 'benzinga_embeddings',
                Path('datasets/benzinga_embeddings'),
            ]:
                if candidate.exists():
                    news_path = str(candidate)
                    break
        
        # Options path: use provided path or default (2-min data)
        options_path = args.options_path
        if args.use_options and options_path is None:
            # Try auto-detected path first, then fallback candidates
            auto_path = data_paths.get('options')
            if auto_path and auto_path.exists():
                options_path = str(auto_path)
            else:
                for candidate in [
                    data_paths.get('stock', Path('.')).parent / 'opt_trade_2min',
                    Path('datasets/opt_trade_2min'),
                ]:
                    if candidate.exists():
                        options_path = str(candidate)
                        break
        
        if is_main:
            if args.use_news:
                if news_path:
                    dashboard.log(f"[bold cyan]📰 News:[/] {news_path}")
                    logger.info(f"News integration ENABLED: {news_path}")
                else:
                    dashboard.log("[yellow]⚠️ News enabled but no news path found[/]")
                    logger.warning("News enabled but no path found")
            else:
                dashboard.log("[dim]📰 News: disabled (baseline mode)[/]")
                logger.info("News integration DISABLED (baseline A/B mode)")
            
            if args.use_options:
                if options_path:
                    dashboard.log(f"[bold cyan]📊 Options:[/] {options_path}")
                    logger.info(f"Options integration ENABLED: {options_path}")
                else:
                    dashboard.log("[yellow]⚠️ Options enabled but no path found[/]")
                    logger.warning("Options enabled but no path found")
            else:
                dashboard.log("[dim]📊 Options: disabled (baseline mode)[/]")
                logger.info("Options integration DISABLED (baseline A/B mode)")
        
        # Macro path: use provided path or default
        macro_path = args.macro_path
        if args.use_macro and macro_path is None:
            for candidate in [
                data_paths.get('stock', Path('.')).parent / 'macro' / 'macro_daily.parquet',
                Path('datasets/macro/macro_daily.parquet'),
            ]:
                if candidate.exists():
                    macro_path = str(candidate)
                    break
        
        if is_main:
            if args.use_macro:
                if macro_path:
                    dashboard.log(f"[bold cyan]🏦 Macro FiLM:[/] {macro_path}")
                    logger.info(f"Macro FiLM ENABLED: {macro_path}")
                else:
                    dashboard.log("[yellow]⚠️ Macro enabled but no macro_daily.parquet found[/]")
                    logger.warning("Macro enabled but no path found")
            else:
                dashboard.log("[dim]🏦 Macro: disabled (baseline mode)[/]")
        
        # GDELT path: use provided path or default
        gdelt_path = args.gdelt_path
        if args.use_gdelt and gdelt_path is None:
            for candidate in [
                data_paths.get('stock', Path('.')).parent / 'GDELT',
                Path('datasets/GDELT'),
            ]:
                if candidate.exists():
                    gdelt_path = str(candidate)
                    break
        
        if is_main:
            if args.use_gdelt:
                if gdelt_path:
                    dashboard.log(f"[bold cyan]🌍 GDELT:[/] {gdelt_path}")
                    logger.info(f"GDELT world state ENABLED: {gdelt_path}")
                else:
                    dashboard.log("[yellow]⚠️ GDELT enabled but no GDELT path found[/]")
                    logger.warning("GDELT enabled but no path found")
            else:
                dashboard.log("[dim]🌍 GDELT: disabled (baseline mode)[/]")

        train_dataset = BarMambaDataset(
            stock_data_path=stock_path,
            vix_data_path=vix_path,
            split='train',
            max_total_bars=args.seq_len,
            train_start=args.train_start,
            train_end=args.train_end,
            val_end=args.val_end,
            news_data_path=news_path,
            use_news=args.use_news,
            options_data_path=options_path,
            use_options=args.use_options,
            macro_data_path=macro_path,
            use_macro=args.use_macro,
            gdelt_data_path=gdelt_path,
            use_gdelt=args.use_gdelt,
        )
        val_dataset = BarMambaDataset(
            stock_data_path=stock_path,
            vix_data_path=vix_path,
            split='val',
            max_total_bars=args.seq_len,
            train_start=args.train_start,
            train_end=args.train_end,
            val_end=args.val_end,
            news_data_path=news_path,
            use_news=args.use_news,
            options_data_path=options_path,
            use_options=args.use_options,
            macro_data_path=macro_path,
            use_macro=args.use_macro,
            gdelt_data_path=gdelt_path,
            use_gdelt=args.use_gdelt,
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

    # Auto-calculate steps if not specified (0 = full epoch)
    if args.train_steps == 0:
        args.train_steps = len(train_dataset) // args.batch_size
        if is_main:
            dashboard.log(f"[dim]Train steps auto-calculated: {args.train_steps} ({len(train_dataset)} samples / batch {args.batch_size})[/]")
    if args.val_steps == 0:
        args.val_steps = len(val_dataset) // args.batch_size
        if is_main:
            dashboard.log(f"[dim]Val steps auto-calculated: {args.val_steps} ({len(val_dataset)} samples / batch {args.batch_size})[/]")

    # Build model
    if is_main:
        dashboard.log(f"[dim]Building model: d={args.d_model}, layers={args.n_layers}, state={args.d_state}[/]")

    from mamba_only_model import ParallelMambaVIX, NUM_OPTION_FEATURES
    
    # Determine macro_dim from dataset if macro is enabled
    macro_dim = getattr(train_dataset, 'macro_dim', 15) if args.use_macro else 15
    
    # Determine gdelt_dim from dataset if gdelt is enabled
    gdelt_dim = getattr(train_dataset, 'gdelt_dim', 391) if args.use_gdelt else 391
    
    model = ParallelMambaVIX(
        num_features=num_features,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=4,
        expand=2,
        dropout=0.1,
        checkpoint_interval=args.checkpoint_interval,
        use_news=args.use_news,
        news_dim=3072,
        news_n_layers=args.news_n_layers,
        use_options=args.use_options,
        option_features=NUM_OPTION_FEATURES,
        head_hidden=128,
        use_macro=args.use_macro,
        macro_dim=macro_dim,
        use_gdelt=args.use_gdelt,
        gdelt_dim=gdelt_dim,
    ).to(device)
    if is_main:
        dashboard.log(f"[bold magenta]🔀 Parallel streams:[/] checkpoint every {args.checkpoint_interval} bars")

    # Wrap model in DDP for multi-GPU training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            dashboard.log(f"[dim]Model wrapped in DistributedDataParallel[/]")

    # Optimizer & loss — separate param group with 10x LR for FiLM generator
    if args.use_macro and hasattr(model, 'module'):
        # DDP wrapped
        base_model = model.module
    else:
        base_model = model
    
    if args.use_macro and hasattr(base_model, 'film_generator'):
        film_params = list(base_model.film_generator.parameters())
        film_param_ids = set(id(p) for p in film_params)
        other_params = [p for p in model.parameters() if id(p) not in film_param_ids]
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': args.lr},
            {'params': film_params, 'lr': args.lr * 10, 'weight_decay': 1e-4},
        ], weight_decay=1e-5)
        if is_main:
            dashboard.log(f"[dim]LR: {args.lr} (FiLM: {args.lr * 10})[/]")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        if is_main:
            dashboard.log(f"[dim]LR: {args.lr}[/]")
    # Spike-weighted multi-horizon loss
    criterion = SpikeWeightedHuberLoss(
        delta=0.25,
        spike_thresh=args.spike_thresh,
        extreme_thresh=args.extreme_thresh,
        spike_weight=args.spike_weight,
        extreme_weight=args.extreme_weight,
    )
    if is_main:
        dashboard.log(f"[dim]Loss: SpikeWeighted (spike≥{args.spike_thresh}pt→{args.spike_weight}×, extreme≥{args.extreme_thresh}pt→{args.extreme_weight}×)[/]")

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
    epochs_without_improvement = 0

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
            
            # Build per-stream MAE display
            stream_mae_parts = []
            if 'stock_mae' in val_metrics:
                stream_mae_parts.append(f"[cyan]S:{val_metrics['stock_mae']:.2f}[/]")
            if 'news_mae' in val_metrics:
                stream_mae_parts.append(f"[yellow]N:{val_metrics['news_mae']:.2f}[/]")
            if 'options_mae' in val_metrics:
                stream_mae_parts.append(f"[magenta]O:{val_metrics['options_mae']:.2f}[/]")
            stream_mae_display = " ".join(stream_mae_parts) if stream_mae_parts else ""
            
            # Per-horizon metrics display
            horizon_parts = []
            for h in HORIZONS:
                mae_h = val_metrics.get(f'mae_{h}d', 0)
                dir_h = val_metrics.get(f'dir_acc_{h}d', 0)
                horizon_parts.append(f"+{h}d:{mae_h:.2f}pt/{dir_h:.0f}%")
            horizon_display = " | ".join(horizon_parts)
            
            dashboard.log(
                f"[green]✓ Epoch {epoch+1}/{args.epochs}:[/] "
                f"steps={train_steps_count} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}"
            )
            dashboard.log(
                f"  [bold]Horizons:[/] {horizon_display}"
            )
            dashboard.log(
                f"  [dim]iter={avg_iter_time:.2f}s/it | VRAM={mem_alloc:.1f}GB[/]"
            )
            if stream_mae_parts:
                dashboard.log(f"  [dim]Stream MAE:[/] {stream_mae_display}")
            # Log to file with per-horizon details
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                f"horizons=[{horizon_display}], "
                f"iter={avg_iter_time:.2f}s/it, VRAM={mem_alloc:.1f}GB"
            )
            
            # Log scale parameters if present (news/options injection)
            eval_model = model.module if is_distributed else model
            scale_info = []
            if hasattr(eval_model, 'news_scale'):
                scale_info.append(f"news_scale={eval_model.news_scale.item():.4f}")
            if hasattr(eval_model, 'option_scale'):
                scale_info.append(f"option_scale={eval_model.option_scale.item():.4f}")
            if scale_info:
                logger.info(f"Scale params: {', '.join(scale_info)}")
                dashboard.log(f"[dim]📏 Scales: {', '.join(scale_info)}[/]")
            
            # Log FiLM gamma/beta statistics if macro is enabled
            if hasattr(eval_model, 'film_generator'):
                film_stats = eval_model.film_generator.get_film_stats()
                if film_stats:
                    film_parts = []
                    for i in range(eval_model.n_layers):
                        gm = film_stats.get(f'film_gamma_L{i}_mean', 1.0)
                        gs = film_stats.get(f'film_gamma_L{i}_std', 0.0)
                        bm = film_stats.get(f'film_beta_L{i}_mean', 0.0)
                        bs = film_stats.get(f'film_beta_L{i}_std', 0.0)
                        film_parts.append(f"L{i}:γ={gm:.3f}±{gs:.3f},β={bm:.3f}±{bs:.3f}")
                    film_display = " | ".join(film_parts)
                    dashboard.log(f"  [dim]🏦 FiLM:[/] {film_display}")
                    logger.info(f"FiLM stats: {film_display}")

            # Save best model and track patience
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
                best_ckpt = save_checkpoint(
                    model, optimizer, scaler, epoch + 1, val_metrics['loss'],
                    args.checkpoint_dir + '/best', is_distributed
                )
                dashboard.log(f"[bold green]🏆 New best model![/] val_loss={best_val_loss:.4f}")
                logger.info(f"New best model saved: val_loss={best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                dashboard.log(f"[dim]No improvement for {epochs_without_improvement}/{args.patience} epochs[/]")
                
                if epochs_without_improvement >= args.patience:
                    dashboard.log(f"[bold yellow]⏹ Early stopping triggered after {args.patience} epochs without improvement[/]")
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
                    break

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
            # Build proper dummy inputs for all streams
            dummy_bars = torch.randn(1, 100, num_features).to(device)
            dummy_mask = torch.ones(1, 100, dtype=torch.bool, device=device)
            dummy_opts = torch.randn(1, 100, NUM_OPTION_FEATURES).to(device) if args.use_options else None
            dummy_opts_mask = torch.ones(1, 100, dtype=torch.bool, device=device) if args.use_options else None
            dummy_news = torch.randn(1, 10, 3072).to(device) if args.use_news else None
            dummy_news_mask = torch.ones(1, 10, dtype=torch.bool, device=device) if args.use_news else None
            dummy_news_idx = torch.zeros(1, 10, dtype=torch.long, device=device) if args.use_news else None
            dummy_news_ts = torch.arange(10, device=device).unsqueeze(0) * 1000 if args.use_news else None
            dummy_gdelt = torch.randn(1, 5, gdelt_dim).to(device) if args.use_gdelt else None
            dummy_gdelt_mask = torch.ones(1, 5, dtype=torch.bool, device=device) if args.use_gdelt else None
            dummy_gdelt_ts = torch.arange(5, device=device).unsqueeze(0) * 1000 if args.use_gdelt else None
            dummy_macro = torch.randn(1, macro_dim).to(device) if args.use_macro else None
            dummy_bar_ts = torch.arange(100, device=device).unsqueeze(0)
            out = eval_model(
                dummy_bars, dummy_mask, dummy_opts, dummy_opts_mask,
                dummy_news, dummy_news_mask, dummy_news_idx, dummy_news_ts,
                dummy_gdelt, dummy_gdelt_mask, dummy_gdelt_ts,
                dummy_macro, dummy_bar_ts
            )
            ok = 'vix_pred' in out and out['vix_pred'].shape == (1, 4)
            dashboard.log(f"  [{'[green]✓' if ok else '[red]✗'}] Forward pass correct[/]")
            checks_passed += ok

        # Check 3: Backward pass works
        eval_model.train()
        out = eval_model(
            dummy_bars, dummy_mask, dummy_opts, dummy_opts_mask,
            dummy_news, dummy_news_mask, dummy_news_idx, dummy_news_ts,
            dummy_gdelt, dummy_gdelt_mask, dummy_gdelt_ts,
            dummy_macro, dummy_bar_ts
        )
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
            csv_path = Path(args.results_csv)
            write_header = not csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['exp_name', 'lr', 'd_model', 'n_layers', 'd_state', 
                                   'seq_len', 'epochs', 'final_train_loss', 'final_val_loss', 
                                   'final_val_mae_pt', 'checks_passed'])
                exp_name = args.exp_name or f"lr_{args.lr}"
                writer.writerow([exp_name, args.lr, args.d_model, args.n_layers, args.d_state,
                               args.seq_len, args.epochs, f"{train_loss:.6f}", 
                               f"{val_metrics['loss']:.6f}", f"{val_metrics['mae']:.4f}",
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
