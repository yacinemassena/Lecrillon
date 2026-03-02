"""
VIX Mamba Training Script

Complete training loop with:
- AMP + gradient accumulation + clipping
- Masked loss for missing targets
- Step-based cosine warmup OR plateau scheduler
- Bounded train/val loops (steps_per_epoch, val_steps)
- Per-horizon metrics
- Checkpointing with resume
- Smoke test mode
"""

import os
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler  # Deprecated
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config, PROJECT_ROOT
from loader.vix_tick_dataset import VIXTickDataset
from vix_mamba_model import VIXMambaModel
from loss.vix_losses import get_loss, build_loss
from tools.schedulers import get_cosine_schedule_with_warmup
from scripts.preprocess_index import run_preprocessing
from utils.vram_policy import choose_checkpoint_policy


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Reproducibility
# =============================================================================
def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Masked Loss Wrapper
# =============================================================================
def masked_loss(pred: torch.Tensor, target: torch.Tensor, 
                base_criterion: nn.Module) -> torch.Tensor:
    """
    Compute loss only on valid (non-zero) targets.
    Target == 0.0 is treated as missing.
    
    Args:
        pred: [B, H] predictions
        target: [B, H] targets (0.0 indicates missing)
        base_criterion: Base loss function
        
    Returns:
        Masked loss scalar
    """
    valid_mask = target != 0.0
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return base_criterion(pred[valid_mask], target[valid_mask])


# =============================================================================
# Early Stopping
# =============================================================================
class EarlyStopping:
    """Early stopping based on validation loss."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
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
    
    def state_dict(self) -> Dict:
        return {'best_loss': self.best_loss, 'counter': self.counter}
    
    def load_state_dict(self, state: Dict):
        self.best_loss = state.get('best_loss')
        self.counter = state.get('counter', 0)


# =============================================================================
# Training Step
# =============================================================================
def train_epoch(model, loader, optimizer, base_criterion, scaler, config, 
                device, scheduler=None, global_step=0, amp_dtype=torch.bfloat16):
    """
    Train for exactly steps_per_epoch batches.
    
    Returns:
        train_loss, global_step, samples_per_sec
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    
    steps_per_epoch = config.train.steps_per_epoch
    grad_accum_steps = config.train.grad_accum_steps
    
    loader_iter = iter(loader)
    pbar = tqdm(range(steps_per_epoch), desc='Train', leave=False)
    optimizer.zero_grad(set_to_none=True)
    
    start_time = datetime.datetime.now()
    
    for step_idx in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            # Restart the loader if it runs out
            loader_iter = iter(loader)
            try:
                batch = next(loader_iter)
            except StopIteration:
                logger.error("Dataset exhausted and empty on restart. Check data paths and split dates.")
                raise RuntimeError("DataLoader is empty. No data found for training.")
        
        # Host Streaming Logic:
        # If total chunks exceed threshold, keep chunks/weights/frame_id on CPU.
        # The encoder will move slices to GPU on-demand.
        keys_on_cpu = set()
        if config.model.encoder_streaming and 'chunks' in batch:
            total_chunks = batch['chunks'].size(0)
            if total_chunks > config.model.encoder_host_streaming_threshold:
                keys_on_cpu = {'chunks', 'frame_id', 'weights'}

        # Move tensors to device (skip those marked for CPU)
        batch = {
            k: v.to(device, non_blocking=True) 
            if torch.is_tensor(v) and k not in keys_on_cpu else v 
            for k, v in batch.items()
        }
        
        B = len(batch['target'])
        num_samples += B
        
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=config.train.amp):
            outputs = model(batch)
            
            # Handle Dictionary Output
            if isinstance(outputs, dict):
                logits = outputs['logits']
                rv_pred = outputs.get('rv_pred', None)
            else:
                logits = outputs
                rv_pred = None
            
            # Safety assert: pred shape == target shape
            assert logits.shape == batch['target'].shape, \
                f"Shape mismatch: pred={logits.shape}, target={batch['target'].shape}"
            
            # Main VIX Loss
            main_loss = masked_loss(logits, batch['target'], base_criterion)
            
            # RV Loss (Auxiliary)
            loss = main_loss
            rv_loss_val = 0.0
            
            if rv_pred is not None and 'rv_targets' in batch:
                rv_targets = batch['rv_targets']
                # Mask missing targets (0.0)
                rv_mask = rv_targets != 0.0
                
                if rv_mask.any():
                    # Use MSE for RV
                    rv_loss = F.mse_loss(rv_pred[rv_mask], rv_targets[rv_mask])
                    # Add to total loss with weight
                    loss = loss + config.model.rv.loss_weight * rv_loss
                    rv_loss_val = rv_loss.item()
            
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (step_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.clip_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Step-based scheduler (if cosine)
            if scheduler is not None and config.train.scheduler.lower() == 'cosine':
                scheduler.step()
            
            global_step += 1
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'rv_loss': f'{rv_loss_val:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    samples_per_sec = num_samples / elapsed if elapsed > 0 else 0
    
    return total_loss / num_batches, global_step, samples_per_sec


# =============================================================================
# Validation 
# =============================================================================
@torch.no_grad()
def validate(model, loader, base_criterion, config, device, num_steps: int, amp_dtype=torch.bfloat16):
    """
    Validate for exactly num_steps batches.
    Compute masked MAE overall and per-horizon.
    
    Returns:
        metrics dict: loss, mae, mae_1d, mae_7d, mae_15d, mae_30d, rv_mae
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    # Track RV metrics
    total_rv_mae = 0.0
    rv_batches = 0
    
    loader_iter = iter(loader)
    
    # Check if loader is empty
    try:
        first_batch = next(loader_iter)
    except StopIteration:
        logger.warning("Validation/Test loader is empty. Skipping.")
        return {'loss': 0.0, 'mae': 0.0}
        
    # Put back the first batch effectively by chaining
    import itertools
    loader_iter = itertools.chain([first_batch], loader_iter)
    
    for _ in tqdm(range(num_steps), desc='Val', leave=False):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            try:
                batch = next(loader_iter)
            except StopIteration:
                 # Should be caught above, but safety
                 break
        
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=config.train.amp):
            outputs = model(batch)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
                rv_pred = outputs.get('rv_pred', None)
            else:
                logits = outputs
                rv_pred = None
            
            # Main Loss
            main_loss = masked_loss(logits, batch['target'], base_criterion)
            
            # RV Loss/MAE
            rv_loss = 0.0
            if rv_pred is not None and 'rv_targets' in batch:
                rv_targets = batch['rv_targets']
                rv_mask = rv_targets != 0.0
                if rv_mask.any():
                    rv_l = F.mse_loss(rv_pred[rv_mask], rv_targets[rv_mask])
                    rv_loss = rv_l
                    
                    # MAE for metrics
                    rv_mae = (rv_pred[rv_mask] - rv_targets[rv_mask]).abs().mean().item()
                    total_rv_mae += rv_mae
                    rv_batches += 1
            
            # Combined loss for reporting (weighted)
            loss = main_loss + config.model.rv.loss_weight * rv_loss
        
        total_loss += loss.item()
        num_batches += 1
        
        all_preds.append(logits.cpu())
        all_targets.append(batch['target'].cpu())
    
    # Aggregate predictions
    preds = torch.cat(all_preds)     # [N, H]
    targets = torch.cat(all_targets) # [N, H]
    
    # Compute masked overall MAE
    valid_mask = targets != 0.0
    if valid_mask.any():
        mae = (preds[valid_mask] - targets[valid_mask]).abs().mean().item()
    else:
        mae = 0.0
    
    # Compute per-horizon MAE
    horizons = config.dataset.prediction_horizons
    horizon_names = [f'mae_{h}d' for h in horizons]
    horizon_maes = {}
    
    for h_idx, h_name in enumerate(horizon_names):
        if h_idx < preds.shape[1]:
            h_mask = targets[:, h_idx] != 0.0
            if h_mask.any():
                h_mae = (preds[h_mask, h_idx] - targets[h_mask, h_idx]).abs().mean().item()
            else:
                h_mae = 0.0
            horizon_maes[h_name] = h_mae
            
    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'mae': mae,
        'rv_mae': total_rv_mae / max(rv_batches, 1),
        **horizon_maes
    }
    
    return metrics


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(model, optimizer, scaler, scheduler, epoch, global_step, 
                    best_val_loss, early_stopping, path):
    """Save full training state."""
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
    """Load training state from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if 'early_stopping' in checkpoint:
        early_stopping.load_state_dict(checkpoint['early_stopping'])
    
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('global_step', 0),
        checkpoint.get('best_val_loss', float('inf'))
    )


# =============================================================================
# Main Training
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train VIX Mamba Model')
    parser.add_argument('--smoke', action='store_true', 
                        help='Smoke test: 5 train + 5 val steps then exit')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Config
    config = Config()
    
    # Ensure dim_out matches prediction horizons
    if len(config.dataset.prediction_horizons) != config.model.dim_out:
        logger.info(f"Overriding model.dim_out ({config.model.dim_out}) to match prediction_horizons length ({len(config.dataset.prediction_horizons)})")
        config.model.dim_out = len(config.dataset.prediction_horizons)
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # Override for smoke test
    if args.smoke:
        config.train.steps_per_epoch = 5
        config.train.val_steps = 5
        config.train.epochs = 1
        # Reduce window size for fast smoke test
        config.dataset.num_frames = 50 
        logger.info("🔥 SMOKE TEST MODE: 5 train + 5 val steps (num_frames=50)")
    
    # Override resume path from CLI
    if args.resume:
        config.train.resume_path = args.resume
    
    # Seed
    seed_everything(config.train.seed)
    
    # Verify CUDA
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        total_vram_gb = device_props.total_memory / (1024**3)
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        logger.info(f"GPU Memory: {total_vram_gb:.1f} GB")
        
        # Select checkpointing policy
        mode, every_n = choose_checkpoint_policy(
            total_vram_gb, 
            config.train.checkpoint_mode, 
            config.train.checkpoint_every_n_layers
        )
        
        # Apply policy to config
        config.train.checkpoint_mode = mode
        config.train.checkpoint_every_n_layers = every_n
        
        logger.info(f"Checkpointing Policy: mode={mode}, every_n={every_n}, encoder={config.train.checkpoint_encoder}")
        
    else:
        logger.warning("CUDA not available, using CPU")
        config.train.checkpoint_mode = "off"
    
    logger.info(f"Starting training: {datetime.datetime.now()}")
    
    # Output directories
    Path(config.train.out_dir).mkdir(parents=True, exist_ok=True)
    Path(config.train.ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------
    logger.info("Running preprocessing step...")
    # Preprocess INDEX data
    # Input: datasets/INDEX (original CSVs)
    # Output: DataTraining/INDEX (Parquet)
    # Timezone: America/New_York (US Equity Markets)
    run_preprocessing(
        input_path=str(PROJECT_ROOT / "datasets" / "INDEX"),
        output_path=str(PROJECT_ROOT / "DataTraining" / "INDEX"),
        timezone="America/New_York",
        partition="none", # Keep file structure
        skip_existing=True,
        verify=False,      # Skip verify for speed in loop
        smoke_test=args.smoke
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    logger.info("Loading datasets...")
    train_dataset = VIXTickDataset(config, split='train')
    val_dataset = VIXTickDataset(config, split='val')
    test_dataset = VIXTickDataset(config, split='test')
    
    num_workers = config.dataset.num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,  # IterableDataset
        num_workers=num_workers,
        collate_fn=VIXTickDataset.collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=VIXTickDataset.collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=VIXTickDataset.collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    logger.info("Creating model...")
    model = VIXMambaModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )
    
    scheduler = None
    if config.train.scheduler.lower() == 'cosine':
        # Step-based cosine warmup
        num_training_steps = config.train.epochs * config.train.steps_per_epoch
        num_warmup_steps = config.train.warmup_epochs * config.train.steps_per_epoch
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        logger.info(f"Using cosine scheduler: {num_warmup_steps} warmup / {num_training_steps} total steps")
        
    elif config.train.scheduler.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.train.reduce_factor,
            patience=config.train.schedule_patience,
            min_lr=config.train.min_lr
        )
        logger.info(f"Using plateau scheduler: patience={config.train.schedule_patience}")
    
    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------
    base_criterion = build_loss(config)
    
    # -------------------------------------------------------------------------
    # AMP Scaler & Early Stopping
    # -------------------------------------------------------------------------
    # Determine dtype and scaler usage
    amp_dtype = torch.float32
    use_scaler = False
    
    if config.train.amp:
        if config.train.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False  # BF16 doesn't need it
            logger.info("Using bfloat16 for AMP (Scaler disabled)")
        else:
            amp_dtype = torch.float16
            use_scaler = True
            logger.info("Using float16 for AMP (Scaler enabled)")
    else:
        logger.info("AMP disabled")

    scaler = GradScaler(enabled=use_scaler) 
    early_stopping = EarlyStopping(patience=config.train.early_stopping_patience)
    
    # Smoke check for checkpointing if enabled
    if config.train.checkpoint_mode != "off" and torch.cuda.is_available():
        logger.info("Running checkpointing smoke check...")
        try:
            model.train()
            # Create a dummy batch
            dummy_chunks = torch.randn(10, config.dataset.chunk_len, config.model.dim_in).to(device)
            dummy_frame_id = torch.zeros(10, dtype=torch.long).to(device)
            dummy_weights = torch.ones(10).to(device)
            dummy_scalars = torch.randn(1, 3).to(device) # B=1
            dummy_batch = {
                'chunks': dummy_chunks,
                'frame_id': dummy_frame_id,
                'weights': dummy_weights,
                'frame_scalars': dummy_scalars,
                'num_frames': 1,
                'target': torch.randn(1, config.model.dim_out).to(device)
            }
            
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=config.train.amp):
                out = model(dummy_batch)
                loss = out.sum()
            
            scaler.scale(loss).backward()
            optimizer.zero_grad()
            logger.info("✓ Checkpointing smoke check passed")
        except Exception as e:
            logger.error(f"Checkpointing smoke check FAILED: {e}")
            raise e

    # -------------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if config.train.resume_path and Path(config.train.resume_path).exists():
        logger.info(f"Resuming from {config.train.resume_path}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            config.train.resume_path, model, optimizer, scaler, 
            scheduler, early_stopping, device
        )
        logger.info(f"Resumed: epoch={start_epoch}, step={global_step}, best_val={best_val_loss:.4f}")
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting training...")
    logger.info(f"Epochs: {config.train.epochs} | Steps/epoch: {config.train.steps_per_epoch}")
    logger.info(f"Batch size: {config.train.batch_size} | Grad accum: {config.train.grad_accum_steps}")
    
    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = datetime.datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config.train.epochs}")
        
        # Reset GPU peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Train
        train_loss, global_step, samples_per_sec = train_epoch(
            model, train_loader, optimizer, base_criterion, scaler, 
            config, device, scheduler, global_step, amp_dtype=amp_dtype
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, base_criterion, config, device, 
            config.train.val_steps, amp_dtype=amp_dtype
        )
        
        # Plateau scheduler step (uses val loss)
        if scheduler is not None and config.train.scheduler.lower() == 'plateau':
            scheduler.step(val_metrics['loss'])
        
        # GPU memory stats
        if torch.cuda.is_available():
            max_mem_alloc = torch.cuda.max_memory_allocated() / 1e9
            max_mem_reserved = torch.cuda.max_memory_reserved() / 1e9
        else:
            max_mem_alloc = max_mem_reserved = 0.0
        
        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        
        # Logging
        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f}"
        )
        
        # Dynamic horizon logging
        horizon_log = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k.startswith('mae_')])
        logger.info(horizon_log)

        logger.info(
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Throughput: {samples_per_sec:.1f} samples/sec | "
            f"GPU Mem: {max_mem_alloc:.2f}/{max_mem_reserved:.2f} GB"
        )
        logger.info(f"Epoch time: {epoch_time:.1f}s")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scaler, scheduler, epoch, global_step,
                best_val_loss, early_stopping,
                Path(config.train.ckpt_dir) / 'best_model.pt'
            )
            logger.info(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % config.train.checkpoint_period == 0:
            save_checkpoint(
                model, optimizer, scaler, scheduler, epoch, global_step,
                best_val_loss, early_stopping,
                Path(config.train.ckpt_dir) / f'checkpoint_epoch_{epoch+1}.pt'
            )
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # -------------------------------------------------------------------------
    # Final Test Evaluation
    # -------------------------------------------------------------------------
    if not args.smoke:
        logger.info("\n" + "="*60)
        logger.info("Evaluating on test set...")
        
        # Load best model
        best_ckpt = Path(config.train.ckpt_dir) / 'best_model.pt'
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = validate(
            model, test_loader, base_criterion, config, device,
            config.train.test_steps, amp_dtype=amp_dtype
        )
        
        logger.info(f"Test Loss: {test_metrics['loss']:.4f} | Test MAE: {test_metrics['mae']:.4f}")
        
        horizon_log_test = " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items() if k.startswith('mae_')])
        logger.info(horizon_log_test)
    
    logger.info(f"\nTraining complete: {datetime.datetime.now()}")
    
    if args.smoke:
        logger.info("🔥 Smoke test PASSED!")


if __name__ == '__main__':
    main()
