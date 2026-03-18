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
from torch.utils.data import DataLoader
from torch.amp import GradScaler
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
    # Check relative path first (inside repo)
    script_dir = Path(__file__).parent
    if (script_dir / 'datasets').exists():
        base = script_dir / 'datasets'
    elif os.path.exists('/workspace/datasets'):
        # Linux VPS (Docker) - standalone datasets folder
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
# Batch-to-device helper (used by train_steps and val_steps)
# ---------------------------------------------------------------------------
def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Transfer all batch tensors to device. Returns dict of GPU tensors."""
    d = {}
    d['bars'] = batch['bars'].to(device, non_blocking=True)
    d['bar_mask'] = batch['bar_mask'].to(device, non_blocking=True)
    d['target'] = batch['vix_targets'].to(device, non_blocking=True)
    d['horizon_mask'] = batch['horizon_mask'].to(device, non_blocking=True)

    # Optional tensor keys: .get() + conditional .to()
    _optional_keys = [
        'news_embs', 'news_mask', 'news_timestamps',
        'options', 'options_mask',
        'macro_context', 'bar_timestamps',
        'gdelt_embs', 'gdelt_mask', 'gdelt_timestamps',
        'econ_event_ids', 'econ_currency_ids', 'econ_numeric',
        'econ_mask', 'econ_timestamps',
        'fundamentals_context',
        'vix_features', 'vix_mask', 'vix_timestamps',
    ]
    for key in _optional_keys:
        val = batch.get(key)
        d[key] = val.to(device, non_blocking=True) if val is not None else None

    return d


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
        logger.info(f"Epoch {epoch}: creating full-epoch iterator...")
        data_iter = enumerate(loader)
        logger.info(f"Epoch {epoch}: iterator ready, starting steps")
    else:
        logger.info(f"Epoch {epoch}: creating iterator for {num_steps} steps...")
        loader_iter = iter(loader)
        logger.info(f"Epoch {epoch}: iterator ready, starting steps")
        data_iter = range(num_steps)

    for step_data in data_iter:
        if num_steps == 0:
            step, batch = step_data
        else:
            step = step_data
            try:
                if step == 0:
                    logger.info(f"Epoch {epoch}: fetching first batch...")
                batch = next(loader_iter)
                if step == 0:
                    logger.info(f"Epoch {epoch}: first batch received, training active")
            except StopIteration:
                break
        
        t_start = time.time()
        
        # GPU transfer
        bd = batch_to_device(batch, device)
        bars = bd['bars']
        bar_mask = bd['bar_mask']
        target = bd['target']
        horizon_mask = bd['horizon_mask']
        news_embs = bd['news_embs']
        news_mask = bd['news_mask']
        news_timestamps = bd['news_timestamps']
        options = bd['options']
        options_mask = bd['options_mask']
        macro_context = bd['macro_context']
        bar_timestamps = bd['bar_timestamps']
        gdelt_embs = bd['gdelt_embs']
        gdelt_mask = bd['gdelt_mask']
        gdelt_timestamps = bd['gdelt_timestamps']
        econ_event_ids = bd['econ_event_ids']
        econ_currency_ids = bd['econ_currency_ids']
        econ_numeric = bd['econ_numeric']
        econ_mask = bd['econ_mask']
        econ_timestamps = bd['econ_timestamps']
        fundamentals_context = bd['fundamentals_context']
        vix_features = bd['vix_features']
        vix_mask = bd['vix_mask']
        vix_timestamps = bd['vix_timestamps']
        
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
                news_timestamps=news_timestamps,
                gdelt_embs=gdelt_embs,
                gdelt_mask=gdelt_mask,
                gdelt_timestamps=gdelt_timestamps,
                macro_context=macro_context,
                bar_timestamps=bar_timestamps,
                econ_event_ids=econ_event_ids,
                econ_currency_ids=econ_currency_ids,
                econ_numeric=econ_numeric,
                econ_mask=econ_mask,
                econ_timestamps=econ_timestamps,
                fundamentals_context=fundamentals_context,
                vix_features=vix_features,
                vix_mask=vix_mask,
                vix_timestamps=vix_timestamps,
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
            
            if 'vix_aux_pred' in outputs:
                vix_aux_loss = F.huber_loss(outputs['vix_aux_pred'], target_1d, delta=0.25)
                loss = loss + 0.3 * vix_aux_loss
                stream_losses['vix'] = vix_aux_loss.item()
            
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
        if 'vix' in stream_losses:
            loss_parts.append(f"[blue]V:{stream_losses['vix']:.3f}[/]")
        loss_parts.append(f"[green]C:{stream_losses['combined']:.3f}[/]")
        
        stream_display = " ".join(loss_parts) if loss_parts else f"loss={step_loss:.4f}"
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        dashboard.console.print(
            f"[bold]{step_display}[/] | {stream_display} | seq={seq_len:,} | "
            f"fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s | "
            f"[dim]{t_total:.2f}s/it[/] | VRAM={mem_reserved:.1f}GB | {timestamp}"
        )

        # Log to file every 15 steps + last step
        is_last = (num_steps > 0 and step + 1 == num_steps)
        if (step + 1) % 15 == 0 or is_last:
            avg_loss = total_loss / num_batches
            sl = " ".join(f"{k}={v:.3f}" for k, v in stream_losses.items())
            logger.info(
                f"Epoch {epoch} step {step + 1}{f'/{num_steps}' if num_steps > 0 else ''}: "
                f"{sl} avg={avg_loss:.4f} | "
                f"fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s total={t_total:.2f}s/it | "
                f"VRAM={mem_reserved:.1f}GB"
            )

    # Flush remaining accumulated gradients (partial group at end of epoch)
    if grad_accum > 1 and num_batches % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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

        bd = batch_to_device(batch, device)
        bars = bd['bars']
        bar_mask = bd['bar_mask']
        target = bd['target']
        horizon_mask = bd['horizon_mask']
        news_embs = bd['news_embs']
        news_mask = bd['news_mask']
        news_timestamps = bd['news_timestamps']
        options = bd['options']
        options_mask = bd['options_mask']
        macro_context = bd['macro_context']
        bar_timestamps = bd['bar_timestamps']
        gdelt_embs = bd['gdelt_embs']
        gdelt_mask = bd['gdelt_mask']
        gdelt_timestamps = bd['gdelt_timestamps']
        econ_event_ids = bd['econ_event_ids']
        econ_currency_ids = bd['econ_currency_ids']
        econ_numeric = bd['econ_numeric']
        econ_mask = bd['econ_mask']
        econ_timestamps = bd['econ_timestamps']
        fundamentals_context = bd['fundamentals_context']
        vix_features = bd['vix_features']
        vix_mask = bd['vix_mask']
        vix_timestamps = bd['vix_timestamps']

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(
                bars, bar_mask,
                options=options,
                options_mask=options_mask,
                news_embs=news_embs,
                news_mask=news_mask,
                news_timestamps=news_timestamps,
                gdelt_embs=gdelt_embs,
                gdelt_mask=gdelt_mask,
                gdelt_timestamps=gdelt_timestamps,
                macro_context=macro_context,
                bar_timestamps=bar_timestamps,
                econ_event_ids=econ_event_ids,
                econ_currency_ids=econ_currency_ids,
                econ_numeric=econ_numeric,
                econ_mask=econ_mask,
                econ_timestamps=econ_timestamps,
                fundamentals_context=fundamentals_context,
                vix_features=vix_features,
                vix_mask=vix_mask,
                vix_timestamps=vix_timestamps,
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
# Real-Data Preflight Checks
# ---------------------------------------------------------------------------
def run_real_data_preflight(
    model: nn.Module,
    train_dataset,
    collate_fn,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    batch_size: int = 4,
    amp_dtype=torch.bfloat16,
    is_distributed: bool = False,
) -> bool:
    """
    Quick preflight on 2 real samples: load → forward → backward → check.
    Returns True if all checks pass.
    """
    ok = 0
    total = 0
    
    dashboard.log("\n" + "═" * 50)
    dashboard.log("[bold]REAL-DATA PREFLIGHT[/]")
    dashboard.log("═" * 50)
    
    # 1. Load 2 samples directly
    total += 1
    t0 = time.time()
    try:
        s0 = train_dataset[0]
        t_s0 = time.time() - t0
        s1 = train_dataset[len(train_dataset) // 2]
        t_load = time.time() - t0
        batch = collate_fn([s0, s1])
        per_sample = t_load / 2
        dashboard.log(f"  [[green]✓[/]] Loaded 2 samples in {t_load:.1f}s "
                      f"(bars={batch['bars'].shape}, targets={batch['vix_targets'].shape})")
        ok += 1
    except Exception as e:
        per_sample = 0
        dashboard.log(f"  [[red]✗[/]] Dataset load failed: {e}")
        dashboard.log(f"\n[bold]Real-data preflight: {ok}/{total} passed[/]")
        return False
    
    # 2. Check inputs for NaN/Inf
    total += 1
    bad_keys = [k for k in ['bars', 'vix_targets'] 
                if k in batch and (torch.isnan(batch[k]).any() or torch.isinf(batch[k]).any())]
    if bad_keys:
        dashboard.log(f"  [[red]✗[/]] NaN/Inf in: {bad_keys}")
    else:
        dashboard.log(f"  [[green]✓[/]] Inputs clean")
        ok += 1
    
    # 3. Forward + loss
    total += 1
    bd = batch_to_device(batch, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            out = model(
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
            pred = out['vix_pred']
            loss = criterion(pred, bd['target'], bd['horizon_mask'])
        
        loss_ok = torch.isfinite(loss).item() and not torch.isnan(pred).any().item()
        if loss_ok:
            dashboard.log(f"  [[green]✓[/]] Forward OK: loss={loss.item():.4f}, "
                          f"pred=[{pred.min():.3f}, {pred.max():.3f}]")
            ok += 1
        else:
            dashboard.log(f"  [[red]✗[/]] Forward produced NaN/Inf")
    except Exception as e:
        dashboard.log(f"  [[red]✗[/]] Forward crashed: {e}")
        dashboard.log(f"\n[bold]Real-data preflight: {ok}/{total} passed[/]")
        return False
    
    # 4. Backward + grad check
    total += 1
    try:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nan_grads = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        
        if nan_grads:
            dashboard.log(f"  [[red]✗[/]] NaN in gradients")
        else:
            dashboard.log(f"  [[green]✓[/]] Backward OK: grad_norm={grad_norm:.4f}")
            ok += 1
        
        # Don't step optimizer — just clean up
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
    except Exception as e:
        dashboard.log(f"  [[red]✗[/]] Backward crashed: {e}")
    
    dashboard.log(f"\n[bold]Real-data preflight: {ok}/{total} passed[/]")
    logger.info(f"Real-data preflight: {ok}/{total} passed")
    return ok == total, per_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Mamba-Only VIX Prediction Training')
    # All defaults come from trainconfig.py - edit that file to change defaults
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
    parser.add_argument('--io-threads', type=int, default=0,
                        help='I/O threads for parallel file loading (0 = auto: cpu_count-1)')
    parser.add_argument('--preprocessed-path', type=str, default=None,
                        help='Path to preprocessed memmaps (auto-detected from datasets/preprocessed)')
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size,
                        help='Batch size for training')
    parser.add_argument('--grad-accum', type=int, default=cfg.grad_accum,
                        help='Gradient accumulation steps (effective batch = batch_size × grad_accum)')
    parser.add_argument('--train-start', type=str, default='2005-01-01',
                        help='Start date for training data (YYYY-MM-DD, default: 2005-01-01)')
    parser.add_argument('--train-end', type=str, default=cfg.train_end,
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--val-end', type=str, default=cfg.val_end,
                        help='End date for validation data (YYYY-MM-DD)')
    parser.add_argument('--lr', type=float, default=cfg.lr,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=cfg.weight_decay,
                        help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default=cfg.scheduler, choices=['none', 'cosine', 'plateau'],
                        help='LR scheduler: none, cosine, plateau (default: cosine)')
    parser.add_argument('--d-fusion', type=int, default=cfg.d_fusion,
                        help='Wider fusion output dimension (default: 512)')
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
    parser.add_argument('--use-econ', action='store_true', default=cfg.use_econ,
                        help='Enable economic calendar integration. Enabled by default.')
    parser.add_argument('--no-econ', action='store_false', dest='use_econ',
                        help='Disable economic calendar for A/B comparison.')
    parser.add_argument('--econ-path', type=str, default=None,
                        help='Path to econ_calendar directory (default: datasets/econ_calendar)')
    parser.add_argument('--use-fundamentals', action='store_true', default=cfg.use_fundamentals,
                        help='Enable fundamentals cross-attention (sector state). Enabled by default.')
    parser.add_argument('--no-fundamentals', action='store_false', dest='use_fundamentals',
                        help='Disable fundamentals for A/B comparison.')
    parser.add_argument('--fundamentals-path', type=str, default=None,
                        help='Path to fundamentals_state.parquet (default: datasets/fundamentals/fundamentals_state.parquet)')
    parser.add_argument('--use-vix-features', action='store_true', default=cfg.use_vix_features,
                        help='Enable VIX Mamba stream (extended hours). Enabled by default.')
    parser.add_argument('--no-vix-features', action='store_false', dest='use_vix_features',
                        help='Disable VIX features for A/B comparison.')
    parser.add_argument('--vix-features-path', type=str, default=None,
                        help='Path to VIX features directory (default: datasets/VIX/Vix_features)')
    parser.add_argument('--vix-n-layers', type=int, default=cfg.vix_n_layers,
                        help='Number of VIX Mamba layers (default: 2)')
    parser.add_argument('--vix-d-model', type=int, default=cfg.vix_d_model,
                        help='VIX Mamba hidden dimension (default: 64)')
    parser.add_argument('--vix-d-state', type=int, default=cfg.vix_d_state,
                        help='VIX Mamba state dimension (default: 16)')
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
    parser.add_argument('--diagnose', action='store_true',
                        help='Run verbose init diagnostics (train_diagnostics.py) then exit')
    args = parser.parse_args()

    # Run diagnostics if requested
    if args.diagnose:
        from train_diagnostics import run_diagnostics
        sys.exit(run_diagnostics(args))

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    is_main = is_main_process()

    # Setup logging (only on main process)
    global LOG_FILE
    if is_main:
        LOG_FILE = setup_logging()
        logger.info(f"Training started - logs saved to {LOG_FILE}")

    # Configure I/O thread pool for parallel file loading
    from loader.bar_mamba_dataset import set_io_threads, _IO_THREADS
    if args.io_threads > 0:
        set_io_threads(args.io_threads)
        io_label = f"{args.io_threads} (manual)"
    else:
        io_label = f"{_IO_THREADS} (auto)"

    # Start dashboard (only on main process)
    if is_main:
        dashboard.start()
        dashboard.log(f"[dim]📝 Logs: {LOG_FILE}[/]")
        eff_batch = args.batch_size * args.grad_accum
        dashboard.log(f"[bold cyan]📦 Config:[/] Batch: {args.batch_size} × {args.grad_accum} accum = {eff_batch} effective | IO threads: {io_label}")
        dashboard.log(f"[bold cyan]📊 Config:[/] Seq={args.seq_len:,} | Epochs={args.epochs}")
        logger.info(f"Config: batch={args.batch_size}, grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}, seq_len={args.seq_len}, epochs={args.epochs}, lr={args.lr}, use_news={args.use_news}, use_options={args.use_options}")
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

    # Data paths - fail fast if not available
    data_paths = get_data_paths()
    
    if not data_paths:
        logger.error("No data directory found. Set up datasets/ or specify paths.")
        sys.exit(1)
    
    if is_main:
        dashboard.log("[dim]Checking data availability...[/]")
    
    try:
        has_overlap = check_data_overlap(data_paths)
        if not has_overlap:
            logger.error("No stock-VIX overlap found. Check your data paths.")
            sys.exit(1)
        if is_main:
            dashboard.log("[green]✓ Real data available[/]")
    except Exception as e:
        logger.error(f"Data check failed: {e}")
        sys.exit(1)

    # Build datasets
    from loader.bar_mamba_dataset import NUM_STOCK_FEATURES, BarMambaDataset
    train_sampler = None
    val_sampler = None

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
    # Priority: enhanced macro (with FED/cross-asset) > standard macro
    macro_path = args.macro_path
    if args.use_macro and macro_path is None:
        for candidate in [
            # Enhanced macro with FED yields, rates, credit spreads
            data_paths.get('stock', Path('.')).parent / 'MACRO' / 'macro_daily_enhanced.parquet',
            Path('datasets/MACRO/macro_daily_enhanced.parquet'),
            # Standard macro (sector fundamentals only)
            data_paths.get('stock', Path('.')).parent / 'MACRO' / 'macro_daily.parquet',
            Path('datasets/MACRO/macro_daily.parquet'),
            # Legacy paths
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
    
    # Econ calendar path: use provided path or default
    econ_path = args.econ_path
    if args.use_econ and econ_path is None:
        for candidate in [
            data_paths.get('stock', Path('.')).parent / 'econ_calendar',
            Path('datasets/econ_calendar'),
        ]:
            if candidate.exists():
                econ_path = str(candidate)
                break
    
    if is_main:
        if args.use_econ:
            if econ_path:
                dashboard.log(f"[bold cyan]📅 Econ Calendar:[/] {econ_path}")
                logger.info(f"Econ calendar ENABLED: {econ_path}")
            else:
                dashboard.log("[yellow]⚠️ Econ enabled but no econ_calendar path found[/]")
                logger.warning("Econ enabled but no path found")
        else:
            dashboard.log("[dim]📅 Econ: disabled (baseline mode)[/]")
    
    # Fundamentals path: use provided path or default
    fundamentals_path = args.fundamentals_path
    if args.use_fundamentals and fundamentals_path is None:
        for candidate in [
            data_paths.get('stock', Path('.')).parent / 'fundamentals' / 'fundamentals_state.parquet',
            Path('datasets/fundamentals/fundamentals_state.parquet'),
        ]:
            if candidate.exists():
                fundamentals_path = str(candidate)
                break
    
    # VIX features path: use provided path or default
    vix_features_path = args.vix_features_path
    if args.use_vix_features and vix_features_path is None:
        for candidate in [
            data_paths.get('stock', Path('.')).parent / 'VIX' / 'Vix_features',
            Path('datasets/VIX/Vix_features'),
        ]:
            if candidate.exists():
                vix_features_path = str(candidate)
                break
    
    if is_main:
        if args.use_fundamentals:
            if fundamentals_path:
                dashboard.log(f"[bold cyan]📊 Fundamentals:[/] {fundamentals_path}")
                logger.info(f"Fundamentals cross-attention ENABLED: {fundamentals_path}")
            else:
                dashboard.log("[yellow]⚠️ Fundamentals enabled but no fundamentals_state.parquet found[/]")
                logger.warning("Fundamentals enabled but no path found")
        else:
            dashboard.log("[dim]📊 Fundamentals: disabled (baseline mode)[/]")
        
        if args.use_vix_features:
            if vix_features_path:
                dashboard.log(f"[bold cyan]📈 VIX Mamba:[/] {vix_features_path} (d={args.vix_d_model}, {args.vix_n_layers}L)")
                logger.info(f"VIX Mamba ENABLED: {vix_features_path}, d_model={args.vix_d_model}, d_state={args.vix_d_state}, {args.vix_n_layers} layers")
            else:
                dashboard.log("[yellow]⚠️ VIX features enabled but no Vix_features dir found[/]")
                logger.warning("VIX features enabled but no path found")
        else:
            dashboard.log("[dim]📈 VIX Mamba: disabled (baseline mode)[/]")

    # Auto-detect preprocessed memmaps
    pp_path = args.preprocessed_path
    if pp_path is None:
        # Auto-detect from datasets/preprocessed
        for candidate in [
            data_paths.get('stock', Path('.')).parent / 'preprocessed',
            Path('datasets/preprocessed'),
        ]:
            if (candidate / 'stock_index.json').exists():
                pp_path = str(candidate)
                break
    if is_main:
        if pp_path:
            dashboard.log(f"[bold green]⚡ Memmap mode:[/] {pp_path}")
            logger.info(f"Preprocessed memmaps: {pp_path}")
        else:
            dashboard.log("[dim]⚡ Memmap: not found (using parquet loading)[/]")

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
        econ_calendar_path=econ_path,
        use_econ=args.use_econ,
        fundamentals_data_path=fundamentals_path,
        use_fundamentals=args.use_fundamentals,
        vix_features_path=vix_features_path,
        use_vix_features=args.use_vix_features,
        preprocessed_path=pp_path,
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
        econ_calendar_path=econ_path,
        use_econ=args.use_econ,
        fundamentals_data_path=fundamentals_path,
        use_fundamentals=args.use_fundamentals,
        vix_features_path=vix_features_path,
        use_vix_features=args.use_vix_features,
        shared_state=train_dataset.get_shared_state(),
        preprocessed_path=pp_path,
    )
    num_features = train_dataset.num_features
    collate_fn = BarMambaDataset.collate_fn

    # Map-style dataset now supports native shuffling
    # For distributed training, could add DistributedSampler here
    
    # num_workers=0: single process shares one RAM cache (from _BARS_CACHE etc.)
    # Multi-worker fork after CUDA init deadlocks; spawn works but duplicates caches.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
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
    
    # Determine econ vocab sizes from dataset if econ is enabled
    econ_num_event_types = getattr(train_dataset, 'econ_num_event_types', 412) + 1 if args.use_econ else 413  # +1 for padding idx 0
    econ_num_currencies = getattr(train_dataset, 'econ_num_currencies', 4) + 1 if args.use_econ else 5  # +1 for padding idx 0
    
    # Determine fundamentals_dim from dataset if fundamentals is enabled
    fundamentals_dim = getattr(train_dataset, 'fundamentals_dim', 130) if args.use_fundamentals else 130
    
    # Determine vix_features_dim from dataset if vix features is enabled
    vix_features_dim = getattr(train_dataset, 'num_vix_features', 25) if args.use_vix_features else 25
    
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
        d_fusion=args.d_fusion,
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
            {'params': film_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        ], weight_decay=args.weight_decay)
        if is_main:
            dashboard.log(f"[dim]LR: {args.lr} (FiLM: {args.lr}, wd={args.weight_decay})[/]")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if is_main:
            dashboard.log(f"[dim]LR: {args.lr}, wd={args.weight_decay}[/]")
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

    # LR Scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )
        if is_main:
            dashboard.log(f"[dim]Scheduler: CosineAnnealing (T_max={args.epochs}, eta_min={args.lr * 0.01:.2e})[/]")
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=args.lr * 0.01,
        )
        if is_main:
            dashboard.log(f"[dim]Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr={args.lr * 0.01:.2e})[/]")
    else:
        if is_main:
            dashboard.log(f"[dim]Scheduler: none (fixed LR)[/]")

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

    # --- REAL-DATA PREFLIGHT ---
    if is_main:
        preflight_ok, per_sample_secs = run_real_data_preflight(
            model, train_dataset, collate_fn, criterion, optimizer, scaler,
            device, amp_dtype=amp_dtype, is_distributed=is_distributed,
        )
        if not preflight_ok:
            dashboard.log("[bold yellow]⚠ Real-data preflight had failures — check above[/]")
        if per_sample_secs > 0:
            est_batch = per_sample_secs * args.batch_size
            dashboard.log(f"[dim]⏱ First batch estimate: {per_sample_secs:.0f}s/sample × {args.batch_size} = ~{est_batch/60:.1f} min (cold cache, faster after)[/]")
        dashboard.log("")  # Spacer

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
            device, args.train_steps, amp_dtype, grad_accum=args.grad_accum, epoch=epoch+1,
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
            
            # Log FiLM gamma/beta statistics if macro is enabled
            eval_model = model.module if is_distributed else model
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

            # Step LR scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                dashboard.log(f"  [dim]LR: {current_lr:.2e}[/]")
                logger.info(f"LR after scheduler step: {current_lr:.2e}")

            # Save checkpoint every N epochs (only on main process)
            if (epoch + 1) % args.save_every == 0:
                ckpt_file = save_checkpoint(
                    model, optimizer, scaler, epoch + 1, val_metrics['loss'],
                    args.checkpoint_dir, is_distributed
                )
                dashboard.log(f"[bold blue]💾 Checkpoint saved:[/] {ckpt_file}")

    # Save final checkpoint
    if is_main and args.epochs > 0 and 'val_metrics' in dir():
        ckpt_file = save_checkpoint(
            model, optimizer, scaler, args.epochs, val_metrics['loss'],
            args.checkpoint_dir, is_distributed
        )
        dashboard.log(f"[bold blue]💾 Final checkpoint saved:[/] {ckpt_file}")

    # Log results to CSV for HP sweep
    if is_main and args.results_csv:
        import csv
        csv_path = Path(args.results_csv)
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['exp_name', 'lr', 'd_model', 'n_layers', 'd_state', 
                               'seq_len', 'epochs', 'final_train_loss', 'final_val_loss', 
                               'final_val_mae_pt'])
            exp_name = args.exp_name or f"lr_{args.lr}"
            writer.writerow([exp_name, args.lr, args.d_model, args.n_layers, args.d_state,
                           args.seq_len, args.epochs, f"{train_loss:.6f}", 
                           f"{val_metrics['loss']:.6f}", f"{val_metrics['mae']:.4f}"])
        dashboard.log(f"[bold green]📊 Results appended to:[/] {args.results_csv}")

    if is_main:
        dashboard.stop()
        print("\n✅ TRAINING COMPLETE")

    cleanup_distributed()
    return 0


if __name__ == '__main__':
    sys.exit(main())
