"""
Training Script: Stock 1s → Transformer → Mamba → VIX Prediction.

Features:
- DDP multi-GPU support
- GPU profiles (rtx5080, rtx5090, a100, b200)
- AMP bfloat16
- Gradient accumulation + clipping
- CosineAnnealing / Plateau scheduler
- Early stopping + checkpointing
- VPS data availability validation
- CLI: --profile, --epochs, --level, --gpus, --resume, --smoke
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config, GPU_PROFILES, apply_gpu_profile, PROJECT_ROOT
from loader.vix_stock_dataset import MambaL1Dataset, MambaBatch
from mamba_model import build_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


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
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_loss = val_loss
            self.counter = 0
            return False

    def state_dict(self):
        return {'best_loss': self.best_loss, 'counter': self.counter}

    def load_state_dict(self, d):
        self.best_loss = d.get('best_loss')
        self.counter = d.get('counter', 0)


# ---------------------------------------------------------------------------
# Data Availability Check
# ---------------------------------------------------------------------------
def validate_data_availability(config: Config, rank: int = 0) -> bool:
    """Check that required data exists before training."""
    stock_path = Path(config.data.stock_data_path)
    vix_path = Path(config.data.vix_data_path)

    issues = []
    if not stock_path.exists():
        issues.append(f"Stock data not found: {stock_path}")
    else:
        n_files = len(list(stock_path.glob('*.parquet')))
        if n_files == 0:
            issues.append(f"No parquet files in {stock_path}")
        elif rank == 0:
            logger.info(f"Stock data: {n_files} files in {stock_path}")

    if not vix_path.exists():
        issues.append(f"VIX data not found: {vix_path}")
    else:
        n_csv = len(list(vix_path.glob('VIX_*.csv')))
        if n_csv == 0:
            issues.append(f"No VIX CSV files in {vix_path}")
        elif rank == 0:
            logger.info(f"VIX data: {n_csv} CSV files in {vix_path}")

    # Check available_streams.json if exists
    streams_file = PROJECT_ROOT / 'available_streams.json'
    if streams_file.exists():
        try:
            with open(streams_file) as f:
                streams = json.load(f)
            if rank == 0:
                logger.info(f"Available streams config: {streams_file}")
        except Exception:
            pass

    if issues:
        for issue in issues:
            logger.error(f"DATA ISSUE: {issue}")
        return False

    return True


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def get_loss_fn(config: Config) -> nn.Module:
    name = config.train.loss_name.lower()
    if name == 'huber':
        return nn.HuberLoss(delta=config.train.huber_delta)
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'l1':
        return nn.L1Loss()
    else:
        return nn.HuberLoss(delta=config.train.huber_delta)


# ---------------------------------------------------------------------------
# Training Epoch
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    config: Config,
    device: torch.device,
    scheduler=None,
    global_step: int = 0,
    amp_dtype=torch.bfloat16,
) -> tuple:
    """Train for steps_per_epoch batches. Returns (loss, global_step, samples/sec)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    grad_accum = config.train.grad_accum_steps

    loader_iter = iter(loader)
    pbar = tqdm(range(config.train.steps_per_epoch), desc='Train', leave=False)
    optimizer.zero_grad(set_to_none=True)
    start_time = datetime.datetime.now()

    for step_idx in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            try:
                batch = next(loader_iter)
            except StopIteration:
                logger.error("Dataset exhausted. Check data paths and split dates.")
                break

        # Move to device
        frames = batch.frames.to(device, non_blocking=True)
        frame_mask = batch.frame_mask.to(device, non_blocking=True)
        ticker_ids = batch.ticker_ids.to(device, non_blocking=True) if batch.ticker_ids is not None else None
        target = batch.vix_target.to(device, non_blocking=True)

        num_samples += 1

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=config.train.amp):
            # Use larger chunk_size and disable checkpointing for speed
            outputs = model(frames, frame_mask, ticker_ids, chunk_size=128)
            pred = outputs['vix_pred']
            loss = criterion(pred, target) / grad_accum

        scaler.scale(loss).backward()

        if (step_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and config.train.scheduler.lower() == 'cosine':
                scheduler.step()

            global_step += 1

        total_loss += loss.item() * grad_accum
        num_batches += 1
        pbar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'pred': f'{pred.item():.2f}',
            'tgt': f'{target.item():.2f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    sps = num_samples / elapsed if elapsed > 0 else 0

    return total_loss / max(num_batches, 1), global_step, sps


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    config: Config,
    device: torch.device,
    num_steps: int,
    amp_dtype=torch.bfloat16,
) -> Dict[str, float]:
    """Validate for num_steps batches. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    loader_iter = iter(loader)

    for _ in tqdm(range(num_steps), desc='Val', leave=False):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

        frames = batch.frames.to(device, non_blocking=True)
        frame_mask = batch.frame_mask.to(device, non_blocking=True)
        ticker_ids = batch.ticker_ids.to(device, non_blocking=True) if batch.ticker_ids is not None else None
        target = batch.vix_target.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=config.train.amp):
            outputs = model(frames, frame_mask, ticker_ids)
            pred = outputs['vix_pred']
            loss = criterion(pred, target)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    if num_batches == 0:
        return {'loss': 0.0, 'mae': 0.0, 'corr': 0.0}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = (preds - targets).abs().mean().item()

    # Correlation
    if len(preds) > 2:
        corr = torch.corrcoef(torch.stack([preds.flatten(), targets.flatten()]))[0, 1].item()
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return {
        'loss': total_loss / num_batches,
        'mae': mae,
        'corr': corr,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scaler, scheduler, epoch, global_step,
                    best_val_loss, early_stopping, path):
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'early_stopping': early_stopping.state_dict(),
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scaler, scheduler, early_stopping, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'early_stopping' in ckpt:
        early_stopping.load_state_dict(ckpt['early_stopping'])
    return ckpt.get('epoch', 0), ckpt.get('global_step', 0), ckpt.get('best_val_loss', float('inf'))


# ---------------------------------------------------------------------------
# DDP Setup
# ---------------------------------------------------------------------------
def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------
def main(args):
    # Config
    config = Config()

    # Apply GPU profile
    if args.profile:
        config = apply_gpu_profile(config, args.profile)

    # CLI overrides
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.lr = args.lr

    level = args.level

    # DDP setup
    rank = 0
    world_size = 1
    use_ddp = args.gpus and args.gpus > 1

    if use_ddp:
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', args.gpus))
        setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Smoke test overrides
    if args.smoke:
        config.train.steps_per_epoch = 5
        config.train.val_steps = 3
        config.train.epochs = 2
        if rank == 0:
            logger.info("SMOKE TEST MODE: 5 train + 3 val steps, 2 epochs")

    # Seed
    seed_everything(config.train.seed + rank)

    # Validate data
    if not validate_data_availability(config, rank):
        logger.error("Data validation failed. Aborting.")
        if use_ddp:
            cleanup_ddp()
        sys.exit(1)

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Stock → Transformer → Mamba-{level} → VIX Training")
        logger.info("=" * 60)
        if args.profile:
            p = GPU_PROFILES[args.profile]
            logger.info(f"GPU Profile: {args.profile} ({p['vram_gb']}GB VRAM)")
        logger.info(f"Level: {level}")
        logger.info(f"Device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(rank)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(rank).total_memory / 1e9:.1f}GB")
        logger.info("=" * 60)

    # Directories
    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    if rank == 0:
        logger.info("Building datasets...")

    train_dataset = MambaL1Dataset(config, split='train', level=level)
    val_dataset = MambaL1Dataset(config, split='val', level=level)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,       # IterableDataset yields single samples
        shuffle=False,
        num_workers=0,      # Threading handled internally
        collate_fn=MambaL1Dataset.collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=MambaL1Dataset.collate_fn,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Train samples: {len(train_dataset.anchor_dates)}")
        logger.info(f"Val samples: {len(val_dataset.anchor_dates)}")
        logger.info("Dataloaders ready")

    # Model
    model = build_model(config, level=level).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    # Scheduler
    scheduler = None
    if config.train.scheduler.lower() == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        total_steps = config.train.epochs * config.train.steps_per_epoch
        warmup_steps = config.train.warmup_epochs * config.train.steps_per_epoch

        # Linear warmup + cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(config.train.min_lr / config.train.lr, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if rank == 0:
            logger.info(f"Cosine scheduler: {warmup_steps} warmup / {total_steps} total steps")
    elif config.train.scheduler.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.train.reduce_factor,
            patience=config.train.schedule_patience, min_lr=config.train.min_lr,
        )
        if rank == 0:
            logger.info(f"Plateau scheduler: patience={config.train.schedule_patience}")

    # Loss + AMP
    criterion = get_loss_fn(config)

    amp_dtype = torch.float32
    use_scaler = False
    if config.train.amp:
        if config.train.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            amp_dtype = torch.float16
            use_scaler = True

    scaler = GradScaler(enabled=use_scaler)
    early_stopping = EarlyStopping(patience=config.train.early_stopping_patience)

    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume and Path(args.resume).exists():
        if rank == 0:
            logger.info(f"Resuming from {args.resume}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, scheduler, early_stopping, device
        )

    # Training loop
    if rank == 0:
        logger.info(f"Training: {config.train.epochs} epochs, "
                     f"{config.train.steps_per_epoch} steps/epoch, "
                     f"grad_accum={config.train.grad_accum_steps}")

    for epoch in range(start_epoch, config.train.epochs):
        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch + 1}/{config.train.epochs}")
            logger.info(f"{'=' * 60}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Train
        train_loss, global_step, sps = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            config, device, scheduler, global_step, amp_dtype,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, config, device,
            config.train.val_steps, amp_dtype,
        )

        # Plateau scheduler
        if scheduler is not None and config.train.scheduler.lower() == 'plateau':
            scheduler.step(val_metrics['loss'])

        if rank == 0:
            # GPU stats
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.max_memory_allocated() / 1e9
                mem_res = torch.cuda.max_memory_reserved() / 1e9
            else:
                mem_alloc = mem_res = 0.0

            logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val Corr: {val_metrics['corr']:.4f}"
            )
            logger.info(
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Throughput: {sps:.1f} samples/sec | "
                f"GPU: {mem_alloc:.2f}/{mem_res:.2f} GB"
            )

            # Save best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, scaler, scheduler, epoch, global_step,
                    best_val_loss, early_stopping,
                    ckpt_dir / f'mamba_l{args.level}_best.pt',
                )
                logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % config.train.checkpoint_period == 0:
                save_checkpoint(
                    model, optimizer, scaler, scheduler, epoch, global_step,
                    best_val_loss, early_stopping,
                    ckpt_dir / f'mamba_l{args.level}_epoch_{epoch + 1}.pt',
                )

        # Early stopping
        if early_stopping(val_metrics['loss']):
            if rank == 0:
                logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if rank == 0:
        logger.info(f"\nTraining complete: {datetime.datetime.now()}")
        if args.smoke:
            logger.info("SMOKE TEST PASSED!")

    if use_ddp:
        cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stock → Mamba → VIX')
    parser.add_argument('--profile', type=str, default='rtx5080',
                        choices=list(GPU_PROFILES.keys()),
                        help='GPU profile name')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2],
                        help='Mamba level (1=short-term, 2=long-term)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs for DDP')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: 2 epochs, 5+3 steps')
    args = parser.parse_args()

    if args.gpus > 1:
        # Launch with torchrun
        logger.info(f"Use: torchrun --nproc_per_node={args.gpus} train_mamba.py ...")
        # If launched directly via torchrun, LOCAL_RANK is set
        if 'LOCAL_RANK' not in os.environ:
            logger.error("For multi-GPU, use: torchrun --nproc_per_node=N train_mamba.py --gpus N ...")
            sys.exit(1)

    main(args)
