"""
Optuna Hyperparameter Search for Stock → Mamba → VIX Pipeline.

Features:
- Multi-GPU parallel trials (1 GPU per trial, up to 8 GPUs)
- Search: Mamba layers/dims, Transformer layers/dims, lr, dropout
- SQLite storage for resumability
- MedianPruner for early stopping
- Save best params to JSON
"""

import os
import sys
import json
import argparse
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    print("Install optuna: pip install optuna")
    sys.exit(1)

from config import Config, GPU_PROFILES, apply_gpu_profile, PROJECT_ROOT
from loader.vix_stock_dataset import MambaL1Dataset, MambaBatch
from mamba_model import build_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search Space
# ---------------------------------------------------------------------------
def suggest_config(trial: Trial, base_config: Config, level: int = 1) -> Config:
    """Build a Config with trial-suggested hyperparameters."""
    config = Config()

    # Transformer encoder (Level 0)
    config.encoder.hidden_dim = trial.suggest_categorical('enc_hidden_dim', [128, 192, 256, 384])
    config.encoder.num_layers = trial.suggest_int('enc_num_layers', 2, 8)
    config.encoder.num_heads = trial.suggest_categorical('enc_num_heads', [2, 4, 8])
    config.encoder.dropout = trial.suggest_float('enc_dropout', 0.05, 0.3)

    # Mamba-1
    config.mamba1.d_model = trial.suggest_categorical('m1_d_model', [128, 192, 256, 384, 512])
    config.mamba1.n_layers = trial.suggest_int('m1_n_layers', 2, 8)
    config.mamba1.d_state = trial.suggest_categorical('m1_d_state', [16, 32, 64, 128, 256])
    config.mamba1.expand = trial.suggest_categorical('m1_expand', [1, 2, 4])
    config.mamba1.dropout = trial.suggest_float('m1_dropout', 0.05, 0.3)

    if level == 2:
        # Mamba-2
        config.mamba2.d_model = trial.suggest_categorical('m2_d_model', [128, 192, 256, 384, 512])
        config.mamba2.n_layers = trial.suggest_int('m2_n_layers', 2, 8)
        config.mamba2.d_state = trial.suggest_categorical('m2_d_state', [16, 32, 64, 128, 256])
        config.mamba2.expand = trial.suggest_categorical('m2_expand', [1, 2, 4])
        config.mamba2.dropout = trial.suggest_float('m2_dropout', 0.05, 0.3)

    # Training
    config.train.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    config.train.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # Copy data config from base
    config.data = base_config.data
    config.train.max_frames_per_batch = base_config.train.max_frames_per_batch
    config.train.grad_accum_steps = base_config.train.grad_accum_steps
    config.train.num_workers = base_config.train.num_workers

    # Shorter for search
    config.train.epochs = 15
    config.train.steps_per_epoch = 200
    config.train.val_steps = 50
    config.train.early_stopping_patience = 5

    return config


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial: Trial, base_config: Config, level: int, gpu_id: int) -> float:
    """Single Optuna trial: build model, train short, return val loss."""
    device = torch.device(f'cuda:{gpu_id}')
    config = suggest_config(trial, base_config, level)

    try:
        # Build model
        model = build_model(config, level=level).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trial {trial.number}: {num_params:,} params on GPU {gpu_id}")

        # Data
        train_dataset = MambaL1Dataset(config, split='train', level=level)
        val_dataset = MambaL1Dataset(config, split='val', level=level)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=MambaL1Dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=MambaL1Dataset.collate_fn)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr,
                                      weight_decay=config.train.weight_decay)
        criterion = nn.HuberLoss(delta=config.train.huber_delta)

        # AMP
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        from torch.amp import GradScaler
        scaler = GradScaler(enabled=(amp_dtype == torch.float16))

        best_val = float('inf')
        grad_accum = config.train.grad_accum_steps

        for epoch in range(config.train.epochs):
            # Train
            model.train()
            total_loss = 0.0
            n_batches = 0
            loader_iter = iter(train_loader)
            optimizer.zero_grad(set_to_none=True)

            for step in range(config.train.steps_per_epoch):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        break

                frames = batch.frames.to(device, non_blocking=True)
                mask = batch.frame_mask.to(device, non_blocking=True)
                tids = batch.ticker_ids.to(device) if batch.ticker_ids is not None else None
                target = batch.vix_target.to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                    out = model(frames, mask, tids)
                    loss = criterion(out['vix_pred'], target) / grad_accum

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * grad_accum
                n_batches += 1

            # Validate
            model.eval()
            val_loss = 0.0
            val_n = 0
            val_iter = iter(val_loader)

            with torch.no_grad():
                for _ in range(config.train.val_steps):
                    try:
                        batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        try:
                            batch = next(val_iter)
                        except StopIteration:
                            break

                    frames = batch.frames.to(device)
                    mask = batch.frame_mask.to(device)
                    tids = batch.ticker_ids.to(device) if batch.ticker_ids is not None else None
                    target = batch.vix_target.to(device)

                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                        out = model(frames, mask, tids)
                        loss = criterion(out['vix_pred'], target)

                    val_loss += loss.item()
                    val_n += 1

            avg_val = val_loss / max(val_n, 1)

            # Report to Optuna
            trial.report(avg_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if avg_val < best_val:
                best_val = avg_val

            logger.info(
                f"Trial {trial.number} Epoch {epoch + 1}: "
                f"train={total_loss / max(n_batches, 1):.4f} val={avg_val:.4f}"
            )

        return best_val

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU wrapper
# ---------------------------------------------------------------------------
def run_search(args):
    """Run Optuna search with multi-GPU support."""
    base_config = Config()
    if args.profile:
        base_config = apply_gpu_profile(base_config, args.profile)

    # SQLite for resumability
    results_dir = PROJECT_ROOT / 'results' / 'hyperparam_search'
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / f'optuna_mamba_l{args.level}.db'
    study_name = f'mamba_l{args.level}_vix'

    storage = f'sqlite:///{db_path}'

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
    )

    n_gpus = min(args.gpus, torch.cuda.device_count())
    logger.info(f"Running {args.n_trials} trials on {n_gpus} GPU(s)")
    logger.info(f"Study: {study_name}, DB: {db_path}")

    if n_gpus <= 1:
        # Sequential on single GPU
        gpu_id = 0
        study.optimize(
            lambda trial: objective(trial, base_config, args.level, gpu_id),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
    else:
        # Parallel: launch workers via subprocess
        import multiprocessing as mp

        def gpu_worker(gpu_id, n_trials_per_gpu):
            study = optuna.load_study(study_name=study_name, storage=storage)
            study.optimize(
                lambda trial: objective(trial, base_config, args.level, gpu_id),
                n_trials=n_trials_per_gpu,
            )

        trials_per_gpu = max(1, args.n_trials // n_gpus)
        processes = []
        for gpu_id in range(n_gpus):
            p = mp.Process(target=gpu_worker, args=(gpu_id, trials_per_gpu))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Reload study to get results
        study = optuna.load_study(study_name=study_name, storage=storage)

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_trial.value:.4f}")
    logger.info(f"Best params:")
    for k, v in study.best_trial.params.items():
        logger.info(f"  {k}: {v}")

    # Save best params
    best_path = results_dir / f'best_params_l{args.level}.json'
    with open(best_path, 'w') as f:
        json.dump({
            'trial_number': study.best_trial.number,
            'value': study.best_trial.value,
            'params': study.best_trial.params,
            'datetime': str(datetime.datetime.now()),
        }, f, indent=2)
    logger.info(f"Saved best params to {best_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Search')
    parser.add_argument('--profile', type=str, default='rtx5080',
                        choices=list(GPU_PROFILES.keys()))
    parser.add_argument('--level', type=int, default=1, choices=[1, 2])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs for parallel trials (up to 8)')
    args = parser.parse_args()

    run_search(args)
