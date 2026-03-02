"""
Optuna Hyperparameter Search for TCN Pretraining.

Features:
- Multi-GPU parallel trials (1 GPU per trial)
- Stream-specific search spaces
- Early pruning of bad trials
- SQLite storage for resumable searches
- Best hyperparameters saved to JSON

Usage:
    # Search with 4 GPUs, 20 trials
    python hyperparam_search.py --stream index --gpus 4 --n_trials 20
    
    # Resume previous search
    python hyperparam_search.py --stream index --gpus 4 --n_trials 20 --resume
    
    # Quick search (fewer epochs per trial)
    python hyperparam_search.py --stream index --gpus 4 --n_trials 50 --epochs 10
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import numpy as np
import optuna
from optuna.trial import TrialState
from tqdm import tqdm

from config_pretrain import (
    PretrainConfig, get_pretrain_config, GPU_PROFILES, STREAM_CONFIGS,
    PROJECT_ROOT
)
from loader.single_stream_dataset import SingleStreamDataset
from encoder.frame_encoder import FrameEncoder
from encoder.chunked_encoder import ChunkedFrameEncoder
from encoder.rv_head import RVPredictionHead, RVLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Search spaces per stream (complexity scales with data volume)
SEARCH_SPACES = {
    'index': {
        'num_layers': [2, 3, 4, 5, 6],
        'hidden_dim': [64, 128, 192, 256],
        'dropout': (0.05, 0.3),
        'lr': (1e-5, 1e-3),
        'weight_decay': (1e-6, 1e-3),
        'warmup_epochs': [2, 3, 5, 8],
    },
    'options': {
        'num_layers': [4, 6, 8, 10],
        'hidden_dim': [128, 256, 384, 512],
        'dropout': (0.05, 0.25),
        'lr': (1e-5, 5e-4),
        'weight_decay': (1e-6, 1e-4),
        'warmup_epochs': [3, 5, 8, 10],
    },
    'stocks': {
        'num_layers': [8, 10, 12, 14, 16],
        'hidden_dim': [256, 384, 512, 768],
        'dropout': (0.05, 0.2),
        'lr': (1e-5, 3e-4),
        'weight_decay': (1e-6, 1e-4),
        'warmup_epochs': [3, 5, 8, 10],
    },
}


class TCNPretrainModel(nn.Module):
    """TCN model for RV pretraining."""
    
    def __init__(self, frame_encoder: nn.Module, chunked_encoder: nn.Module, rv_head: nn.Module):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.chunked_encoder = chunked_encoder
        self.rv_head = rv_head
    
    def forward(self, batch, use_checkpoint: bool = False):
        frame_vecs = self.chunked_encoder(
            chunks=batch.chunks,
            frame_id=batch.frame_id,
            weights=batch.weights,
            frame_scalars=batch.frame_scalars,
            num_frames=batch.num_frames,
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


def build_model_from_params(
    dim_in: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_tickers: int = 0,
    ticker_embed_dim: int = 16,
    checkpoint_every: int = 3,
    rv_head_hidden: int = 256,
    rv_head_layers: int = 2,
) -> TCNPretrainModel:
    """Build model with specified hyperparameters."""
    frame_encoder = FrameEncoder(
        kind='tcn',
        in_features=dim_in,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        checkpoint_every=checkpoint_every,
        num_tickers=num_tickers,
        ticker_embed_dim=ticker_embed_dim,
    )
    
    chunked_encoder = ChunkedFrameEncoder(
        frame_encoder=frame_encoder,
        d_model=hidden_dim,
        num_scalars=3,
        stream_chunks=True,
        stream_chunk_size=512,
    )
    
    rv_head = RVPredictionHead(
        in_dim=hidden_dim,
        hidden_dim=rv_head_hidden,
        dropout=dropout,
        num_layers=rv_head_layers,
    )
    
    return TCNPretrainModel(frame_encoder, chunked_encoder, rv_head)


def train_one_epoch(model, loader, optimizer, criterion, scaler, config, device, steps_per_epoch):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    optimizer.zero_grad()
    grad_accum_steps = config.train.grad_accum_steps
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= steps_per_epoch:
            break
        
        batch.to_device(device)
        n_samples = len(batch.sample_boundaries)
        if n_samples == 0:
            continue
        
        amp_dtype = torch.bfloat16 if config.train.amp_dtype == 'bfloat16' else torch.float16
        with autocast('cuda', enabled=config.train.amp, dtype=amp_dtype):
            rv_preds, rv_targets = model(batch, use_checkpoint=True)
            if rv_preds.numel() == 0 or rv_targets.numel() == 0:
                continue
            loss = criterion(rv_preds, rv_targets) / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        total_batches += 1
    
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, config, device, val_steps):
    """Validate and return loss + correlation."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    sum_preds = 0.0
    sum_targets = 0.0
    sum_preds_sq = 0.0
    sum_targets_sq = 0.0
    sum_preds_targets = 0.0
    n_for_corr = 0
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= val_steps:
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
        total_batches += 1
        
        preds_cpu = rv_preds.detach().float().cpu()
        targs_cpu = rv_targets.detach().float().cpu()
        sum_preds += preds_cpu.sum().item()
        sum_targets += targs_cpu.sum().item()
        sum_preds_sq += (preds_cpu ** 2).sum().item()
        sum_targets_sq += (targs_cpu ** 2).sum().item()
        sum_preds_targets += (preds_cpu * targs_cpu).sum().item()
        n_for_corr += len(preds_cpu)
    
    # Compute correlation
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
    
    return {
        'loss': total_loss / max(total_batches, 1),
        'correlation': corr if not np.isnan(corr) else 0.0,
    }


def run_trial(
    trial_params: Dict[str, Any],
    gpu_id: int,
    stream: str,
    profile: str,
    epochs: int,
    steps_per_epoch: int,
    val_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run a single trial on specified GPU.
    
    Returns dict with 'val_loss', 'val_corr', 'best_epoch', 'status'.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    config = get_pretrain_config()
    gpu_profile = GPU_PROFILES[profile]
    stream_config = STREAM_CONFIGS[stream]
    
    # Apply trial hyperparameters
    hidden_dim = trial_params['hidden_dim']
    num_layers = trial_params['num_layers']
    dropout = trial_params['dropout']
    lr = trial_params['lr']
    weight_decay = trial_params['weight_decay']
    warmup_epochs = trial_params['warmup_epochs']
    
    # Build model
    model = build_model_from_params(
        dim_in=config.tcn.dim_in,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_tickers=stream_config.num_tickers,
    )
    model = model.to(device)
    
    # Build dataloaders
    if gpu_profile.vram_gb <= 16:
        max_chunks = stream_config.max_chunks_16gb
    elif gpu_profile.vram_gb <= 32:
        max_chunks = stream_config.max_chunks_32gb
    else:
        max_chunks = stream_config.max_chunks_80gb
    
    filter_tickers = (stream == 'stocks' and gpu_profile.filter_stocks)
    
    train_loader = SingleStreamDataset(
        stream=stream,
        data_path=stream_config.data_path,
        rv_file=config.data.rv_file,
        split='train',
        frame_interval=config.data.frame_interval,
        chunk_len=config.data.chunk_len,
        dim_in=config.data.dim_in,
        max_chunks_per_batch=max_chunks,
        prefetch_files=stream_config.prefetch_files,
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
        prefetch_files=stream_config.prefetch_files,
        rv_horizon_days=config.data.rv_horizon_days,
        train_end=config.data.train_end,
        val_end=config.data.val_end,
        filter_tickers=filter_tickers,
        allowed_tickers_file=stream_config.allowed_tickers_file if filter_tickers else None,
    )
    
    # Optimizer with trial LR
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=config.train.betas,
    )
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = RVLoss(loss_type=config.train.loss_type, delta=config.train.huber_delta)
    scaler = GradScaler('cuda', enabled=config.train.amp)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_corr = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 10  # Reduced patience for hyperparam search
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, config, device, steps_per_epoch
        )
        val_metrics = validate(model, val_loader, criterion, config, device, val_steps)
        scheduler.step()
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_corr = val_metrics['correlation']
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            break
    
    return {
        'val_loss': best_val_loss,
        'val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'status': 'completed',
    }


def objective_wrapper(
    trial: optuna.Trial,
    stream: str,
    profile: str,
    gpu_id: int,
    epochs: int,
    steps_per_epoch: int,
    val_steps: int,
) -> float:
    """Optuna objective function that samples hyperparameters and runs trial."""
    search_space = SEARCH_SPACES[stream]
    
    # Sample hyperparameters
    trial_params = {
        'num_layers': trial.suggest_categorical('num_layers', search_space['num_layers']),
        'hidden_dim': trial.suggest_categorical('hidden_dim', search_space['hidden_dim']),
        'dropout': trial.suggest_float('dropout', *search_space['dropout']),
        'lr': trial.suggest_float('lr', *search_space['lr'], log=True),
        'weight_decay': trial.suggest_float('weight_decay', *search_space['weight_decay'], log=True),
        'warmup_epochs': trial.suggest_categorical('warmup_epochs', search_space['warmup_epochs']),
    }
    
    seed = 42 + trial.number
    
    try:
        result = run_trial(
            trial_params=trial_params,
            gpu_id=gpu_id,
            stream=stream,
            profile=profile,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            seed=seed,
        )
        
        # Report intermediate values for pruning
        trial.set_user_attr('val_corr', result['val_corr'])
        trial.set_user_attr('best_epoch', result['best_epoch'])
        
        return result['val_loss']
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_parallel_search(
    stream: str,
    profile: str,
    n_gpus: int,
    n_trials: int,
    epochs: int,
    steps_per_epoch: int,
    val_steps: int,
    resume: bool = False,
):
    """Run parallel hyperparameter search across multiple GPUs."""
    
    # Storage for resumable search
    storage_path = PROJECT_ROOT / 'checkpoints' / 'optuna'
    storage_path.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}/optuna_{stream}.db"
    
    study_name = f"tcn_{stream}_hyperparam_search"
    
    # Create or load study
    if resume:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            logger.info(f"Resuming study '{study_name}' with {len(study.trials)} existing trials")
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
            logger.info(f"Created new study '{study_name}'")
    else:
        # Delete existing study if not resuming
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
        except KeyError:
            pass
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        logger.info(f"Created new study '{study_name}'")
    
    logger.info("=" * 60)
    logger.info(f"Optuna Hyperparameter Search - {stream.upper()} Stream")
    logger.info("=" * 60)
    logger.info(f"GPUs: {n_gpus}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Epochs per trial: {epochs}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {val_steps}")
    logger.info(f"Search space: {SEARCH_SPACES[stream]}")
    logger.info("=" * 60)
    
    # Run trials with GPU assignment
    # Each trial gets assigned to a GPU in round-robin fashion
    completed_trials = 0
    
    # Process trials sequentially but assign GPUs round-robin
    # For true parallelism, we use optuna's built-in parallel optimization
    def objective_with_gpu(trial):
        gpu_id = trial.number % n_gpus
        return objective_wrapper(
            trial=trial,
            stream=stream,
            profile=profile,
            gpu_id=gpu_id,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
        )
    
    # Run optimization with n_jobs for parallelism
    study.optimize(
        objective_with_gpu,
        n_trials=n_trials,
        n_jobs=n_gpus,  # Parallel trials = number of GPUs
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("Search Complete!")
    logger.info("=" * 60)
    
    # Best trial
    best_trial = study.best_trial
    logger.info(f"\nBest Trial: {best_trial.number}")
    logger.info(f"Best Val Loss: {best_trial.value:.6f}")
    logger.info(f"Best Val Corr: {best_trial.user_attrs.get('val_corr', 'N/A')}")
    logger.info(f"Best Epoch: {best_trial.user_attrs.get('best_epoch', 'N/A')}")
    logger.info("\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save best params to JSON
    results = {
        'stream': stream,
        'best_trial_number': best_trial.number,
        'best_val_loss': best_trial.value,
        'best_val_corr': best_trial.user_attrs.get('val_corr'),
        'best_epoch': best_trial.user_attrs.get('best_epoch'),
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'search_space': SEARCH_SPACES[stream],
    }
    
    results_path = storage_path / f'best_params_{stream}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Top 5 trials
    logger.info("\nTop 5 Trials:")
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    top_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
    for i, trial in enumerate(top_trials, 1):
        logger.info(f"  {i}. Trial {trial.number}: loss={trial.value:.6f}, "
                   f"corr={trial.user_attrs.get('val_corr', 'N/A'):.3f}, "
                   f"layers={trial.params['num_layers']}, "
                   f"hidden={trial.params['hidden_dim']}, "
                   f"lr={trial.params['lr']:.2e}")
    
    return study


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Search for TCN')
    parser.add_argument('--stream', type=str, required=True,
                        choices=['stocks', 'options', 'index'],
                        help='Data stream to optimize')
    parser.add_argument('--profile', type=str, default='rtx5080',
                        choices=['rtx5080', 'rtx5090', 'h100', 'a100', 'amd'],
                        help='GPU profile for batch sizing')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use for parallel trials')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Epochs per trial (reduced for faster search)')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Training steps per epoch (reduced for faster search)')
    parser.add_argument('--val_steps', type=int, default=50,
                        help='Validation steps per epoch')
    parser.add_argument('--resume', action='store_true',
                        help='Resume previous search from SQLite storage')
    
    args = parser.parse_args()
    
    # Validate GPU count
    available_gpus = torch.cuda.device_count()
    n_gpus = min(args.gpus, available_gpus)
    if n_gpus < 1:
        raise ValueError("No GPUs available")
    if n_gpus < args.gpus:
        logger.warning(f"Requested {args.gpus} GPUs but only {available_gpus} available. Using {n_gpus}.")
    
    logger.info(f"Starting hyperparameter search with {n_gpus} GPU(s)...")
    
    study = run_parallel_search(
        stream=args.stream,
        profile=args.profile,
        n_gpus=n_gpus,
        n_trials=args.n_trials,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        resume=args.resume,
    )
    
    logger.info("\nTo train with best hyperparameters, update STREAM_CONFIGS in config_pretrain.py")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
