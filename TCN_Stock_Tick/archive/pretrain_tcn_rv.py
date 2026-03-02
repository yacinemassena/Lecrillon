"""
TCN Pretraining Script - Per-Stream with GPU Profile Selection.

Features:
- Single-stream training (options, stock_1s, or stock_tick separately)
- GPU profile selection (rtx5080/rtx5090/h100/a100/amd)
- Multi-GPU training with DistributedDataParallel (DDP)
- Background prefetching with on-the-fly filtering
- Chunk-level batching for strict VRAM control
- Separate TCN checkpoint per stream for transfer to Mamba architecture

Usage:
    # Train 1-second stock bars TCN on RTX 5080 (filtered top 100)
    python pretrain_tcn_rv.py --profile rtx5080 --stream stock_1s
    
    # Train raw stock tick TCN
    python pretrain_tcn_rv.py --profile rtx5090 --stream stock_tick
    
    # Train options TCN on H100
    python pretrain_tcn_rv.py --profile h100 --stream options
    
    # Multi-GPU training (4 GPUs)
    python pretrain_tcn_rv.py --profile rtx5080 --stream stock_1s --gpus 4
    
    # 8 GPU training on large cluster
    python pretrain_tcn_rv.py --profile h100 --stream stock_tick --gpus 8
    
    # Enable torch.compile() for faster training (10-20% speedup)
    python pretrain_tcn_rv.py --profile rtx5090 --stream stock_1s --compile
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
import numpy as np
from tqdm import tqdm

from config_pretrain import PretrainConfig, get_pretrain_config, GPU_PROFILES, GPUProfile, STREAM_CONFIGS, StreamConfig
from loader.single_stream_dataset import SingleStreamDataset, SingleStreamBatch
from encoder.frame_encoder import FrameEncoder
from encoder.chunked_encoder import ChunkedFrameEncoder
from encoder.rv_head import RVPredictionHead, RVLoss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = None) -> bool:
    """Check if this is the main process (rank 0)."""
    if rank is not None:
        return rank == 0
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


class TCNPretrainModel(nn.Module):
    """TCN model for RV pretraining with single-stream chunk-level batching."""
    
    def __init__(self, frame_encoder: nn.Module, chunked_encoder: nn.Module, rv_head: nn.Module, stream: str):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.chunked_encoder = chunked_encoder
        self.rv_head = rv_head
        self.stream = stream
    
    def forward(self, batch: SingleStreamBatch, use_checkpoint: bool = False) -> torch.Tensor:
        """Forward pass with sample-level RV prediction."""
        frame_vecs = self.chunked_encoder(
            chunks=batch.chunks,
            frame_id=batch.frame_id,
            weights=batch.weights,
            frame_scalars=batch.frame_scalars,
            num_frames=batch.num_frames,
            ticker_ids=batch.ticker_ids,
            use_checkpoint=use_checkpoint,
        )
        
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
            targets_tensor = torch.tensor(valid_targets, dtype=torch.float32, device=batch.chunks.device)
            return torch.cat(rv_preds), targets_tensor
        else:
            dummy = torch.zeros(1, device=batch.chunks.device)
            return dummy, dummy
    
    def save_encoder(self, path: str):
        """Save encoder weights for transfer to Mamba architecture."""
        torch.save({
            'stream': self.stream,
            'frame_encoder_state_dict': self.frame_encoder.state_dict(),
            'chunked_encoder_state_dict': self.chunked_encoder.state_dict(),
        }, path)


def build_model(config: PretrainConfig, stream: str) -> TCNPretrainModel:
    """Build the pretraining model for a specific stream.
    
    Uses stream-specific architecture from STREAM_CONFIGS:
    - Index: 4 layers, 128 hidden (smallest - low data volume)
    - Options: 8 layers, 256 hidden (medium)
    - Stocks: 12 layers, 512 hidden (largest - highest data volume)
    """
    stream_config = STREAM_CONFIGS[stream]
    
    # Use stream-specific architecture
    hidden_dim = stream_config.hidden_dim
    num_layers = stream_config.num_layers
    dropout = stream_config.dropout
    
    frame_encoder = FrameEncoder(
        kind='tcn',
        in_features=config.tcn.dim_in,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        checkpoint_every=config.tcn.checkpoint_every,
        num_tickers=stream_config.num_tickers,
        ticker_embed_dim=config.tcn.ticker_embed_dim,
    )
    
    chunked_encoder = ChunkedFrameEncoder(
        frame_encoder=frame_encoder,
        d_model=hidden_dim,
        num_scalars=3,
        stream_chunks=True,        # Process chunks in batches to reduce memory
        stream_chunk_size=512,     # Process 512 chunks at a time
    )
    
    rv_head = RVPredictionHead(
        in_dim=hidden_dim,
        hidden_dim=config.rv_head.hidden_dim,
        dropout=config.rv_head.dropout,
        num_layers=config.rv_head.num_layers,
    )
    
    return TCNPretrainModel(frame_encoder, chunked_encoder, rv_head, stream)


def build_dataloaders(config: PretrainConfig, gpu_profile: GPUProfile, stream: str):
    """Build single-stream dataloaders with stream-specific batch sizing."""
    stream_config = STREAM_CONFIGS[stream]
    
    # For stocks on 16GB GPU, use filtering; otherwise no filtering
    filter_tickers = (stream == 'stocks' and gpu_profile.filter_stocks)
    
    # Use stream-specific batch sizing based on GPU VRAM
    if gpu_profile.vram_gb <= 16:
        max_chunks = stream_config.max_chunks_16gb
    elif gpu_profile.vram_gb <= 32:
        max_chunks = stream_config.max_chunks_32gb
    elif gpu_profile.vram_gb <= 80:
        max_chunks = stream_config.max_chunks_80gb
    else:
        max_chunks = stream_config.max_chunks_192gb
    
    prefetch_files = stream_config.prefetch_files
    
    train_loader = SingleStreamDataset(
        stream=stream,
        data_path=stream_config.data_path,
        rv_file=config.data.rv_file,
        split='train',
        frame_interval=config.data.frame_interval,
        chunk_len=config.data.chunk_len,
        dim_in=config.data.dim_in,
        max_chunks_per_batch=max_chunks,
        prefetch_files=prefetch_files,
        rv_horizon_days=config.data.rv_horizon_days,
        train_end=config.data.train_end,
        val_end=config.data.val_end,
        filter_tickers=filter_tickers,
        allowed_tickers_file=stream_config.allowed_tickers_file if filter_tickers else None,
    )
    
    val_loader = SingleStreamDataset(
        stream=stream,
        data_path=stream_config.data_path,
        rv_file=config.data.rv_file,
        split='val',
        frame_interval=config.data.frame_interval,
        chunk_len=config.data.chunk_len,
        dim_in=config.data.dim_in,
        max_chunks_per_batch=max_chunks,
        prefetch_files=prefetch_files,
        rv_horizon_days=config.data.rv_horizon_days,
        train_end=config.data.train_end,
        val_end=config.data.val_end,
        filter_tickers=filter_tickers,
        allowed_tickers_file=stream_config.allowed_tickers_file if filter_tickers else None,
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: SingleStreamDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    config: PretrainConfig,
    epoch: int,
    use_checkpoint: bool = True,
) -> dict:
    """Train for one epoch."""
    model.train()
    device = torch.device(config.train.device)
    
    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    # Use running stats instead of accumulating all predictions
    sum_preds = 0.0
    sum_targets = 0.0
    sum_preds_sq = 0.0
    sum_targets_sq = 0.0
    sum_preds_targets = 0.0
    n_for_corr = 0
    
    grad_accum_steps = config.train.grad_accum_steps
    optimizer.zero_grad()
    
    epoch_start = time.time()
    
    pbar = tqdm(total=config.train.steps_per_epoch, desc=f"Epoch {epoch}", ncols=100)
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= config.train.steps_per_epoch:
            break
        
        # Debug: log actual batch size on first batch
        if batch_idx == 0:
            logger.info(f"Batch 0: {batch.num_chunks} chunks, {batch.num_frames} frames, chunks shape: {batch.chunks.shape}")
        
        batch.to_device(device)
        
        n_samples = len(batch.sample_boundaries)
        if n_samples == 0:
            continue
        
        amp_dtype = torch.bfloat16 if config.train.amp_dtype == 'bfloat16' else torch.float16
        with autocast('cuda', enabled=config.train.amp, dtype=amp_dtype):
            rv_preds, rv_targets = model(batch, use_checkpoint=use_checkpoint)
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
        
        # Update running correlation stats (no memory accumulation)
        with torch.no_grad():
            preds_cpu = rv_preds.detach().float().cpu()
            targs_cpu = rv_targets.detach().float().cpu()
            sum_preds += preds_cpu.sum().item()
            sum_targets += targs_cpu.sum().item()
            sum_preds_sq += (preds_cpu ** 2).sum().item()
            sum_targets_sq += (targs_cpu ** 2).sum().item()
            sum_preds_targets += (preds_cpu * targs_cpu).sum().item()
            n_for_corr += len(preds_cpu)
        
        # Update progress bar every batch
        avg_loss = total_loss / total_batches
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'samples': total_samples})
        pbar.update(1)
    
    pbar.close()
    
    # Compute correlation from running stats
    if n_for_corr > 1:
        mean_p = sum_preds / n_for_corr
        mean_t = sum_targets / n_for_corr
        var_p = sum_preds_sq / n_for_corr - mean_p ** 2
        var_t = sum_targets_sq / n_for_corr - mean_t ** 2
        cov = sum_preds_targets / n_for_corr - mean_p * mean_t
        if var_p > 0 and var_t > 0:
            corr = cov / (np.sqrt(var_p) * np.sqrt(var_t))
        else:
            corr = 0.0
    else:
        corr = 0.0
    
    epoch_time = time.time() - epoch_start
    
    return {
        'loss': total_loss / max(total_batches, 1),
        'correlation': corr if not np.isnan(corr) else 0.0,
        'samples': total_samples,
        'batches': total_batches,
        'time': epoch_time,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: SingleStreamDataset,
    criterion: nn.Module,
    config: PretrainConfig,
) -> dict:
    """Validate the model."""
    model.eval()
    device = torch.device(config.train.device)
    
    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    # Use running stats instead of accumulating tensors
    sum_preds = 0.0
    sum_targets = 0.0
    sum_preds_sq = 0.0
    sum_targets_sq = 0.0
    sum_preds_targets = 0.0
    n_for_corr = 0
    
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
        
        # Update running stats
        preds_cpu = rv_preds.detach().float().cpu()
        targs_cpu = rv_targets.detach().float().cpu()
        sum_preds += preds_cpu.sum().item()
        sum_targets += targs_cpu.sum().item()
        sum_preds_sq += (preds_cpu ** 2).sum().item()
        sum_targets_sq += (targs_cpu ** 2).sum().item()
        sum_preds_targets += (preds_cpu * targs_cpu).sum().item()
        n_for_corr += len(preds_cpu)
    
    # Compute correlation and R2 from running stats
    if n_for_corr > 1:
        mean_p = sum_preds / n_for_corr
        mean_t = sum_targets / n_for_corr
        var_p = sum_preds_sq / n_for_corr - mean_p ** 2
        var_t = sum_targets_sq / n_for_corr - mean_t ** 2
        cov = sum_preds_targets / n_for_corr - mean_p * mean_t
        
        # Correlation
        if var_p > 0 and var_t > 0:
            corr = cov / (np.sqrt(var_p) * np.sqrt(var_t))
        else:
            corr = 0.0
        
        # R2 = 1 - (SS_res / SS_tot) = 1 - (var_residual / var_target)
        # SS_res = sum((y - y_pred)^2) = sum(y^2) - 2*sum(y*y_pred) + sum(y_pred^2)
        ss_res = sum_targets_sq - 2 * sum_preds_targets + sum_preds_sq
        ss_tot = sum_targets_sq - n_for_corr * mean_t ** 2
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
    else:
        corr, r2 = 0.0, 0.0
    
    return {
        'loss': total_loss / max(total_batches, 1),
        'correlation': corr if not np.isnan(corr) else 0.0,
        'r2': r2 if not np.isnan(r2) else 0.0,
        'samples': total_samples,
    }


def save_checkpoint(
    model: TCNPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: PretrainConfig,
    is_best: bool = False,
):
    """Save model checkpoint with stream-specific naming."""
    stream = model.stream
    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Full checkpoint
    checkpoint = {
        'epoch': epoch,
        'stream': stream,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': {
            'tcn_hidden_dim': config.tcn.hidden_dim,
            'tcn_num_layers': config.tcn.num_layers,
            'tcn_dim_in': config.tcn.dim_in,
        },
    }
    
    torch.save(checkpoint, ckpt_dir / f'tcn_{stream}_latest.pt')
    
    if is_best:
        torch.save(checkpoint, ckpt_dir / f'tcn_{stream}_best.pt')
        # Also save encoder-only weights for transfer
        model.save_encoder(str(ckpt_dir / f'tcn_{stream}_encoder.pt'))
        logger.info(f"Saved best {stream} model (val_loss={metrics['val_loss']:.4f})")
    
    if epoch % config.train.save_every_epochs == 0:
        torch.save(checkpoint, ckpt_dir / f'tcn_{stream}_epoch_{epoch}.pt')


def validate_data_availability(stream: str, rank: int = 0) -> bool:
    """Check if required data is available before training."""
    config_file = Path('available_streams.json')
    
    # Only check on main process
    if not is_main_process(rank):
        return True
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                data = json.load(f)
                if not data.get('can_train', {}).get(stream, False):
                    logger.error("=" * 60)
                    logger.error(f"Data for stream '{stream}' is not available!")
                    logger.error("=" * 60)
                    logger.error(f"Available streams: {', '.join(data.get('available_streams', []))}")
                    logger.error("")
                    logger.error("To download data from R2:")
                    logger.error(f"  python check_data_availability.py --download --stream {stream}")
                    logger.error("")
                    logger.error("Or check what's available:")
                    logger.error("  python check_data_availability.py")
                    logger.error("=" * 60)
                    return False
        except Exception as e:
            logger.warning(f"Could not read available_streams.json: {e}")
            logger.warning("Proceeding with training (assuming data is available)...")
    
    return True


def main(rank: int, world_size: int, args):
    """Main training function with DDP support.
    
    Args:
        rank: GPU rank (0 to world_size-1). For single GPU, rank=0.
        world_size: Total number of GPUs. For single GPU, world_size=1.
        args: CLI arguments.
    """
    # Setup DDP if using multiple GPUs
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device(args.device if args.device else 'cuda')
    
    config = get_pretrain_config()
    
    # Get GPU profile
    profile_name = args.profile.lower()
    if profile_name not in GPU_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(GPU_PROFILES.keys())}")
    gpu_profile = GPU_PROFILES[profile_name]
    
    # Get stream config
    stream = args.stream.lower()
    if stream not in STREAM_CONFIGS:
        raise ValueError(f"Unknown stream: {stream}. Available: {list(STREAM_CONFIGS.keys())}")
    stream_config = STREAM_CONFIGS[stream]
    
    # Validate data availability before proceeding
    if not validate_data_availability(stream, rank):
        if use_ddp:
            cleanup_ddp()
        sys.exit(1)
    
    # Apply CLI overrides
    if args.rv_file:
        config.data.rv_file = args.rv_file
    if args.epochs:
        config.train.epochs = args.epochs
    config.train.device = str(device)
    
    # Setup
    set_seed(config.train.seed, rank)
    
    # Compute stream-specific batch size
    if gpu_profile.vram_gb <= 16:
        max_chunks = stream_config.max_chunks_16gb
    else:
        max_chunks = stream_config.max_chunks_80gb
    filter_enabled = (stream == 'stocks' and gpu_profile.filter_stocks)
    
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info(f"TCN Pretraining - {stream.upper()} Stream")
        logger.info("=" * 60)
        logger.info(f"GPU Profile: {gpu_profile.name} ({gpu_profile.vram_gb}GB VRAM)")
        logger.info(f"Stream: {stream}")
        logger.info(f"Data path: {stream_config.data_path}")
        logger.info(f"Filter tickers: {filter_enabled}")
        logger.info(f"Max chunks/batch: {max_chunks} (stream-tuned)")
        logger.info(f"Prefetch files: {stream_config.prefetch_files}")
        logger.info(f"TCN: {stream_config.num_layers} layers, {stream_config.hidden_dim} hidden, dropout={stream_config.dropout}")
        logger.info(f"Device: {device}")
        if use_ddp:
            logger.info(f"DDP: {world_size} GPUs (rank {rank})")
        logger.info("=" * 60)
    
    # Build model
    model = build_model(config, stream)
    model = model.to(device)
    
    # Apply torch.compile() for faster training
    use_compile = getattr(args, 'compile', False)
    if use_compile:
        if is_main_process(rank):
            logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        if is_main_process(rank):
            logger.info("Model compiled successfully")
    
    # Wrap model with DDP if using multiple GPUs
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    num_params = sum(p.numel() for p in model.parameters())
    if is_main_process(rank):
        logger.info(f"Model parameters: {num_params:,}")
    
    # Build dataloaders
    if is_main_process(rank):
        logger.info("Building dataloaders - this may take a moment...")
    train_loader, val_loader = build_dataloaders(config, gpu_profile, stream)
    if is_main_process(rank):
        logger.info("Dataloaders ready - starting training loop...")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=config.train.betas,
    )
    
    # Scheduler
    if config.train.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.min_lr,
        )
    else:
        scheduler = None
    
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
    
    epoch_times = []
    
    for epoch in range(1, config.train.epochs + 1):
        if is_main_process(rank):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{config.train.epochs}")
            logger.info("="*60)
        
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, config, epoch,
            use_checkpoint=not args.no_checkpoint
        )
        
        # Synchronize across GPUs before validation
        if use_ddp:
            dist.barrier()
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate ETA
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.train.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_hours = eta_seconds / 3600
        
        # Log with ETA (only main process)
        if is_main_process(rank):
            logger.info(
                f"Epoch {epoch}/{config.train.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Train Corr: {train_metrics['correlation']:.3f} | "
                f"Val Corr: {val_metrics['correlation']:.3f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Avg: {avg_epoch_time:.1f}s | "
                f"ETA: {eta_hours:.1f}h ({remaining_epochs} epochs left)"
            )
        
        # Checkpointing
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Only save checkpoints on main process
        if is_main_process(rank):
            # Get the underlying model for DDP
            model_to_save = model.module if use_ddp else model
            save_checkpoint(
                model_to_save, optimizer, scheduler, epoch,
                {'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss']},
                config, is_best
            )
        
        # Early stopping
        if patience_counter >= config.train.early_stopping_patience:
            if is_main_process(rank):
                logger.info(f"Early stopping at epoch {epoch}")
            break
    
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        logger.info(f"Encoder saved to: {config.train.checkpoint_dir}/tcn_{stream}_best.pt")
        logger.info("=" * 60)
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()


def run_ddp(rank: int, world_size: int, args):
    """Wrapper function for DDP spawning."""
    main(rank, world_size, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCN Pretraining - Per-Stream')
    parser.add_argument('--profile', type=str, default='rtx5080',
                        choices=['rtx5080', 'rtx5090', 'h100', 'a100', 'amd'],
                        help='GPU profile (rtx5080=16GB, rtx5090=32GB, h100/a100=80GB, amd=192GB)')
    parser.add_argument('--stream', type=str, default='stock_tick',
                        choices=['stock_tick'],
                        help='Data stream to train on (stock_tick only)')
    parser.add_argument('--rv_file', type=str, help='Path to precomputed RV file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--no-checkpoint', action='store_true', 
                        help='Disable gradient checkpointing (faster but uses more VRAM)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (1-8). Use 1 for single GPU training.')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Validate GPU count
    num_gpus = min(args.gpus, torch.cuda.device_count())
    if num_gpus < 1:
        raise ValueError("No GPUs available")
    
    if num_gpus > 1:
        # Multi-GPU training with DDP
        print(f"Starting DDP training on {num_gpus} GPUs...")
        mp.spawn(run_ddp, args=(num_gpus, args), nprocs=num_gpus, join=True)
    else:
        # Single GPU training
        main(rank=0, world_size=1, args=args)
