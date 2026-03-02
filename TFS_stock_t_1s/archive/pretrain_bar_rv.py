"""
Bar-Based Pretraining Script for 1-Second Stock Data.

Uses Transformer encoder for bar sequences (not TCN chunking).
Predicts next-day realized volatility from intraday bar patterns.

Usage:
    python pretrain_bar_rv.py --profile rtx5080
    python pretrain_bar_rv.py --profile h100 --epochs 50
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from config_pretrain import PretrainConfig, get_pretrain_config, GPU_PROFILES, STREAM_CONFIGS
from loader.bar_dataset import BarDataset, BarBatch, BAR_FEATURES
from encoder.bar_encoder import BarEncoder, BarFrameEncoder
from encoder.rv_head import RVPredictionHead, RVLoss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BarPretrainModel(nn.Module):
    """Bar-based model for RV pretraining."""
    
    def __init__(self, bar_encoder: BarFrameEncoder, rv_head: nn.Module):
        super().__init__()
        self.bar_encoder = bar_encoder
        self.rv_head = rv_head
    
    def forward(self, batch: BarBatch) -> torch.Tensor:
        """Forward pass with sample-level RV prediction."""
        # Encode all frames
        frame_vecs = self.bar_encoder(
            bars=batch.bars,
            mask=batch.frame_mask,
            ticker_ids=batch.ticker_ids,
        )
        
        # Aggregate frames per sample and predict RV
        rv_preds = []
        valid_targets = []
        for start_frame, end_frame, rv_target in batch.sample_boundaries:
            if end_frame > start_frame:
                sample_frames = frame_vecs[start_frame:end_frame]
                sample_emb = sample_frames.mean(dim=0, keepdim=True)
                rv_pred = self.rv_head(sample_emb)
                rv_preds.append(rv_pred)
                valid_targets.append(rv_target)
        
        if rv_preds:
            targets_tensor = torch.tensor(valid_targets, dtype=torch.float32, device=batch.bars.device)
            return torch.cat(rv_preds), targets_tensor
        else:
            dummy = torch.zeros(1, device=batch.bars.device)
            return dummy, dummy
    
    def save_encoder(self, path: str):
        """Save encoder weights for transfer."""
        torch.save({
            'bar_encoder_state_dict': self.bar_encoder.state_dict(),
        }, path)


def build_model(config: PretrainConfig) -> BarPretrainModel:
    """Build the bar-based pretraining model."""
    stream_config = STREAM_CONFIGS['stock_1s']
    
    hidden_dim = stream_config.hidden_dim
    num_layers = stream_config.num_layers
    dropout = stream_config.dropout
    
    bar_encoder = BarEncoder(
        num_features=len(BAR_FEATURES),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=4,
        dropout=dropout,
        num_tickers=stream_config.num_tickers,
        ticker_embed_dim=config.tcn.ticker_embed_dim,
    )
    
    frame_encoder = BarFrameEncoder(
        bar_encoder=bar_encoder,
        d_model=hidden_dim,
    )
    
    rv_head = RVPredictionHead(
        in_dim=hidden_dim,
        hidden_dim=config.rv_head.hidden_dim,
        dropout=config.rv_head.dropout,
        num_layers=config.rv_head.num_layers,
    )
    
    return BarPretrainModel(frame_encoder, rv_head)


def build_dataloaders(config: PretrainConfig, gpu_profile):
    """Build bar dataloaders."""
    stream_config = STREAM_CONFIGS['stock_1s']
    
    # Determine max frames per batch based on GPU
    if gpu_profile.vram_gb <= 16:
        max_frames = 80
    elif gpu_profile.vram_gb <= 32:
        max_frames = 160
    else:
        max_frames = 400
    
    train_loader = BarDataset(
        data_path=stream_config.data_path,
        rv_file=config.data.rv_file,
        split='train',
        frame_interval=config.data.frame_interval,
        max_bars_per_frame=500,
        max_frames_per_batch=max_frames,
        prefetch_files=stream_config.prefetch_files,
        rv_horizon_days=config.data.rv_horizon_days,
        train_end=config.data.train_end,
        val_end=config.data.val_end,
        filter_tickers=stream_config.filter_tickers,
        allowed_tickers_file=stream_config.allowed_tickers_file,
    )
    
    val_loader = BarDataset(
        data_path=stream_config.data_path,
        rv_file=config.data.rv_file,
        split='val',
        frame_interval=config.data.frame_interval,
        max_bars_per_frame=500,
        max_frames_per_batch=max_frames,
        prefetch_files=stream_config.prefetch_files,
        rv_horizon_days=config.data.rv_horizon_days,
        train_end=config.data.train_end,
        val_end=config.data.val_end,
        filter_tickers=stream_config.filter_tickers,
        allowed_tickers_file=stream_config.allowed_tickers_file,
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: BarDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    config: PretrainConfig,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    device = torch.device(config.train.device)
    
    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    
    grad_accum_steps = config.train.grad_accum_steps
    optimizer.zero_grad()
    
    epoch_start = time.time()
    
    pbar = tqdm(total=config.train.steps_per_epoch, desc=f"Epoch {epoch}", ncols=100)
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= config.train.steps_per_epoch:
            break
        
        if batch_idx == 0:
            logger.info(f"Batch 0: {batch.num_frames} frames, bars shape: {batch.bars.shape}")
        
        batch.to_device(device)
        
        n_samples = len(batch.sample_boundaries)
        if n_samples == 0:
            continue
        
        amp_dtype = torch.bfloat16 if config.train.amp_dtype == 'bfloat16' else torch.float16
        with autocast('cuda', enabled=config.train.amp, dtype=amp_dtype):
            rv_preds, rv_targets = model(batch)
            if rv_preds.numel() == 0 or rv_targets.numel() == 0:
                continue
            loss = criterion(rv_preds, rv_targets)
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        total_samples += n_samples
        total_batches += 1
        
        avg_loss = total_loss / total_batches
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'samples': total_samples})
        pbar.update(1)
    
    pbar.close()
    
    epoch_time = time.time() - epoch_start
    
    return {
        'loss': total_loss / max(total_batches, 1),
        'samples': total_samples,
        'batches': total_batches,
        'time': epoch_time,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: BarDataset,
    criterion: nn.Module,
    config: PretrainConfig,
) -> dict:
    """Validate the model."""
    model.eval()
    device = torch.device(config.train.device)
    
    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= config.train.val_steps:
            break
        
        batch.to_device(device)
        
        n_samples = len(batch.sample_boundaries)
        if n_samples == 0:
            continue
        
        amp_dtype = torch.bfloat16 if config.train.amp_dtype == 'bfloat16' else torch.float16
        with autocast('cuda', enabled=config.train.amp, dtype=amp_dtype):
            rv_preds, rv_targets = model(batch)
            if rv_preds.numel() == 0 or rv_targets.numel() == 0:
                continue
            loss = criterion(rv_preds, rv_targets)
        
        total_loss += loss.item()
        total_samples += n_samples
        total_batches += 1
    
    return {
        'loss': total_loss / max(total_batches, 1),
        'samples': total_samples,
    }


def save_checkpoint(
    model: BarPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: PretrainConfig,
    is_best: bool = False,
):
    """Save model checkpoint."""
    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    torch.save(checkpoint, ckpt_dir / 'bar_stock_1s_latest.pt')
    
    if is_best:
        torch.save(checkpoint, ckpt_dir / 'bar_stock_1s_best.pt')
        model.save_encoder(str(ckpt_dir / 'bar_stock_1s_encoder.pt'))
        logger.info(f"Saved best model (val_loss={metrics['val_loss']:.4f})")
    
    if epoch % config.train.save_every_epochs == 0:
        torch.save(checkpoint, ckpt_dir / f'bar_stock_1s_epoch_{epoch}.pt')


def main(args):
    """Main training function."""
    config = get_pretrain_config()
    
    # Get GPU profile
    profile_name = args.profile.lower()
    if profile_name not in GPU_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    gpu_profile = GPU_PROFILES[profile_name]
    
    # Apply CLI overrides
    if args.epochs:
        config.train.epochs = args.epochs
    
    device = torch.device(args.device if args.device else 'cuda')
    config.train.device = str(device)
    
    set_seed(config.train.seed)
    
    logger.info("=" * 60)
    logger.info("Bar-Based Pretraining - Stock 1s Data")
    logger.info("=" * 60)
    logger.info(f"GPU Profile: {gpu_profile.name} ({gpu_profile.vram_gb}GB VRAM)")
    logger.info(f"Features: {len(BAR_FEATURES)} bar features")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Build dataloaders
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config, gpu_profile)
    logger.info("Dataloaders ready")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=config.train.betas,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.train.min_lr,
    )
    
    # Loss
    criterion = RVLoss(
        loss_type=config.train.loss_type,
        delta=config.train.huber_delta,
    )
    
    # AMP scaler
    scaler = GradScaler('cuda', enabled=config.train.amp)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config.train.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.train.epochs}")
        logger.info("="*60)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, config, epoch
        )
        
        val_metrics = validate(model, val_loader, criterion, config)
        
        if scheduler:
            scheduler.step()
        
        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Time: {train_metrics['time']:.1f}s"
        )
        
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss']},
            config, is_best
        )
        
        if patience_counter >= config.train.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("=" * 60)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bar-Based Pretraining - Stock 1s Data')
    parser.add_argument('--profile', type=str, default='rtx5080',
                        choices=['rtx5080', 'rtx5090', 'h100', 'a100', 'amd'],
                        help='GPU profile')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    main(args)
