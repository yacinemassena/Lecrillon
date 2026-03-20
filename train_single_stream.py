"""
Phase 1: Single-Stream Pre-Training for Mamba VIX Prediction.

Trains ONE Mamba stream + macro FiLM against all 4 VIX horizons (+1d, +7d, +15d, +30d).
Each stream learns its own representation independently before fusion in Phase 2.

Usage:
    python train_single_stream.py --stream stock --seq-len 8000 --epochs 30
    python train_single_stream.py --stream options --seq-len 8000 --epochs 30
    python train_single_stream.py --stream news --seq-len 8000 --epochs 30
    python train_single_stream.py --stream vix --seq-len 8000 --epochs 30
    
    # Summarize all pre-trained streams
    python train_single_stream.py --summarize
"""

import os
import sys
import argparse
import logging
import datetime
import time
import math
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from trainconfig import DEFAULT_CONFIG as cfg
from dashboard import SimpleDashboard

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(stream_type: str, log_dir: str = 'logs') -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'pretrain_{stream_type}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return log_file

logger = logging.getLogger(__name__)
dashboard = SimpleDashboard()

HORIZONS = [1, 7, 15, 30]
NUM_HORIZONS = len(HORIZONS)


# ---------------------------------------------------------------------------
# SingleStreamModel
# ---------------------------------------------------------------------------
class SingleStreamModel(nn.Module):
    """Single Mamba stream + macro FiLM + MultiHorizonVIXHead.
    
    Wraps existing components from mamba_only_model.py for independent
    pre-training of one data stream.
    """

    def __init__(
        self,
        stream_type: str,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        input_dim: int = 50,
        macro_dim: int = 55,
        checkpoint_interval: int = 300,
        # News-specific
        news_dim: int = 3072,
        use_gdelt: bool = False,
        gdelt_dim: int = 391,
        use_econ: bool = False,
        econ_num_event_types: int = 413,
        econ_num_currencies: int = 5,
        # VIX-specific
        vix_d_model: int = 64,
        vix_d_state: int = 16,
        vix_n_layers: int = 2,
    ):
        super().__init__()
        from mamba_only_model import (
            StreamEncoder, StreamMamba, FiLMGenerator,
            MultiHorizonVIXHead, SequencePooling, EconEncoder,
        )

        self.stream_type = stream_type
        self.d_model = d_model
        self.checkpoint_interval = checkpoint_interval
        self.use_gdelt = use_gdelt
        self.use_econ = use_econ

        if stream_type == 'vix':
            actual_d = vix_d_model
            actual_layers = vix_n_layers
            actual_state = vix_d_state
        elif stream_type == 'news':
            actual_d = d_model
            actual_layers = 2  # News uses fewer layers
            actual_state = d_state
        else:
            actual_d = d_model
            actual_layers = n_layers
            actual_state = d_state

        self.actual_d = actual_d
        self.actual_layers = actual_layers

        # --- Stream-specific encoder ---
        if stream_type == 'stock':
            self.encoder = StreamEncoder(input_dim, d_model, dropout)
        elif stream_type == 'options':
            self.encoder = StreamEncoder(input_dim, d_model, dropout)
        elif stream_type == 'news':
            # Benzinga encoder
            self.news_encoder = StreamEncoder(news_dim, d_model, dropout, normalize_input=True)
            # Type embeddings: 0=GDELT, 1=Benzinga, 2=Econ
            self.news_type_embedding = nn.Embedding(3, d_model)
            nn.init.normal_(self.news_type_embedding.weight, mean=0.0, std=0.02)
            # GDELT encoder
            if use_gdelt:
                self.gdelt_embed_dim = 384
                self.gdelt_stats_dim = 7
                self.gdelt_embed_norm = nn.LayerNorm(self.gdelt_embed_dim)
                self.gdelt_stats_norm = nn.LayerNorm(self.gdelt_stats_dim)
                self.gdelt_encoder = nn.Sequential(
                    nn.Linear(gdelt_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
            # Econ encoder
            if use_econ:
                self.econ_encoder = EconEncoder(
                    d_model=d_model,
                    num_event_types=econ_num_event_types,
                    num_currencies=econ_num_currencies,
                    dropout=dropout,
                )
        elif stream_type == 'vix':
            self.encoder = StreamEncoder(input_dim, vix_d_model, dropout)

        # --- Mamba ---
        self.mamba = StreamMamba(
            n_layers=actual_layers,
            d_model=actual_d,
            d_state=actual_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # --- FiLM ---
        self.film = FiLMGenerator(
            macro_dim=macro_dim,
            d_model=actual_d,
            n_layers=actual_layers,
            dropout=dropout,
        )

        # --- Pooling + Head ---
        self.pool = SequencePooling(actual_d, 'attention')
        self.head = MultiHorizonVIXHead(actual_d, hidden_dim=128, dropout=dropout)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SingleStreamModel[{stream_type}]: {num_params:,} params "
                    f"(d={actual_d}, L={actual_layers}, state={actual_state})")

    def _merge_news_sources(
        self,
        benzinga_encoded, benzinga_ts, benzinga_mask,
        gdelt_encoded, gdelt_ts, gdelt_mask,
        device,
    ):
        """Merge Benzinga + GDELT encoded tokens sorted by timestamp."""
        if benzinga_encoded is None and gdelt_encoded is None:
            return None, None, None
        if benzinga_encoded is None:
            return gdelt_encoded, gdelt_ts, gdelt_mask
        if gdelt_encoded is None:
            return benzinga_encoded, benzinga_ts, benzinga_mask

        B = benzinga_encoded.shape[0]
        N1 = benzinga_encoded.shape[1]
        N2 = gdelt_encoded.shape[1]

        combined_encoded = torch.cat([benzinga_encoded, gdelt_encoded], dim=1)
        combined_ts = torch.cat([benzinga_ts, gdelt_ts], dim=1)

        if benzinga_mask is not None and gdelt_mask is not None:
            combined_mask = torch.cat([benzinga_mask, gdelt_mask], dim=1)
        elif benzinga_mask is not None:
            combined_mask = torch.cat([benzinga_mask, torch.ones(B, N2, device=device)], dim=1)
        elif gdelt_mask is not None:
            combined_mask = torch.cat([torch.ones(B, N1, device=device), gdelt_mask], dim=1)
        else:
            combined_mask = torch.ones(B, N1 + N2, device=device)

        sorted_indices = torch.argsort(combined_ts, dim=1)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        combined_encoded = torch.gather(combined_encoded, 1, sorted_indices_expanded)
        combined_ts = torch.gather(combined_ts, 1, sorted_indices)
        combined_mask = torch.gather(combined_mask, 1, sorted_indices)

        return combined_encoded, combined_ts, combined_mask

    def forward(
        self,
        bars: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
        bar_timestamps: Optional[torch.Tensor] = None,
        macro_context: Optional[torch.Tensor] = None,
        # Options
        options: Optional[torch.Tensor] = None,
        options_mask: Optional[torch.Tensor] = None,
        # News
        news_embs: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
        news_timestamps: Optional[torch.Tensor] = None,
        gdelt_embs: Optional[torch.Tensor] = None,
        gdelt_mask: Optional[torch.Tensor] = None,
        gdelt_timestamps: Optional[torch.Tensor] = None,
        econ_event_ids: Optional[torch.Tensor] = None,
        econ_currency_ids: Optional[torch.Tensor] = None,
        econ_numeric: Optional[torch.Tensor] = None,
        econ_mask: Optional[torch.Tensor] = None,
        econ_timestamps: Optional[torch.Tensor] = None,
        # VIX
        vix_features: Optional[torch.Tensor] = None,
        vix_mask: Optional[torch.Tensor] = None,
        vix_timestamps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = bars.device
        B, T, _ = bars.shape

        # --- Generate FiLM params ---
        film_params = None
        if macro_context is not None and bar_timestamps is not None:
            if self.stream_type in ('stock', 'options'):
                film_params = self.film(macro_context, bar_timestamps, T)
            elif self.stream_type == 'news':
                # News doesn't use FiLM per-position (no bar-aligned timestamps)
                # We generate but use only for checkpoint-level conditioning
                film_params = None
            elif self.stream_type == 'vix':
                if vix_timestamps is not None and vix_timestamps.shape[1] > 0:
                    film_params = self.film(macro_context, vix_timestamps, vix_timestamps.shape[1])

        # --- STOCK stream ---
        if self.stream_type == 'stock':
            encoded = self.encoder(bars)
            # Process in checkpoint segments (same as ParallelMambaVIX)
            num_checkpoints = (T + self.checkpoint_interval - 1) // self.checkpoint_interval
            all_outputs = []
            for cp in range(num_checkpoints):
                s = cp * self.checkpoint_interval
                e = min((cp + 1) * self.checkpoint_interval, T)
                seg = encoded[:, s:e, :]
                seg_mask = bar_mask[:, s:e] if bar_mask is not None else None
                seg_film = None
                if film_params is not None:
                    seg_film = [(g[:, s:e, :], b[:, s:e, :]) for g, b in film_params]
                out = self.mamba(seg, seg_mask, seg_film)
                all_outputs.append(out)
            all_out = torch.cat(all_outputs, dim=1)
            pooled = self.pool(all_out)
            pred = self.head(pooled)
            return {'vix_pred': pred}

        # --- OPTIONS stream ---
        elif self.stream_type == 'options':
            if options is None:
                # Fallback: zero prediction
                return {'vix_pred': torch.zeros(B, NUM_HORIZONS, device=device)}
            encoded = self.encoder(options)
            num_checkpoints = (T + self.checkpoint_interval - 1) // self.checkpoint_interval
            all_outputs = []
            for cp in range(num_checkpoints):
                s = cp * self.checkpoint_interval
                e = min((cp + 1) * self.checkpoint_interval, T)
                seg = encoded[:, s:e, :]
                seg_mask = options_mask[:, s:e] if options_mask is not None else None
                seg_film = None
                if film_params is not None:
                    seg_film = [(g[:, s:e, :], b[:, s:e, :]) for g, b in film_params]
                out = self.mamba(seg, seg_mask, seg_film)
                all_outputs.append(out)
            all_out = torch.cat(all_outputs, dim=1)
            pooled = self.pool(all_out)
            pred = self.head(pooled)
            return {'vix_pred': pred}

        # --- NEWS stream ---
        elif self.stream_type == 'news':
            # Encode Benzinga
            benzinga_encoded = None
            benzinga_ts = None
            if news_embs is not None and news_embs.shape[1] > 0:
                benzinga_encoded = self.news_encoder(news_embs)
                type_emb = self.news_type_embedding(
                    torch.ones(benzinga_encoded.shape[:2], dtype=torch.long, device=device)
                )
                benzinga_encoded = benzinga_encoded + type_emb
                benzinga_ts = news_timestamps

            # Encode GDELT
            gdelt_encoded = None
            gdelt_ts = None
            if self.use_gdelt and gdelt_embs is not None and gdelt_embs.shape[1] > 0:
                gdelt_embed = self.gdelt_embed_norm(gdelt_embs[..., :self.gdelt_embed_dim])
                gdelt_stats = self.gdelt_stats_norm(gdelt_embs[..., self.gdelt_embed_dim:])
                gdelt_combined = torch.cat([gdelt_embed, gdelt_stats], dim=-1)
                gdelt_encoded = self.gdelt_encoder(gdelt_combined)
                type_emb = self.news_type_embedding(
                    torch.zeros(gdelt_encoded.shape[:2], dtype=torch.long, device=device)
                )
                gdelt_encoded = gdelt_encoded + type_emb
                gdelt_ts = gdelt_timestamps

            # Encode Econ
            econ_encoded = None
            econ_ts = None
            if self.use_econ and econ_event_ids is not None and econ_event_ids.shape[1] > 0:
                econ_encoded = self.econ_encoder(econ_event_ids, econ_currency_ids, econ_numeric)
                type_emb = self.news_type_embedding(
                    torch.full(econ_encoded.shape[:2], 2, dtype=torch.long, device=device)
                )
                econ_encoded = econ_encoded + type_emb
                econ_ts = econ_timestamps

            # Merge Benzinga + GDELT
            merged_encoded, merged_ts, merged_mask = self._merge_news_sources(
                benzinga_encoded, benzinga_ts, news_mask,
                gdelt_encoded, gdelt_ts, gdelt_mask if self.use_gdelt else None,
                device,
            )
            # Merge (Benzinga+GDELT) + Econ
            news_encoded, combined_ts, combined_mask = self._merge_news_sources(
                merged_encoded, merged_ts, merged_mask,
                econ_encoded, econ_ts, econ_mask if self.use_econ else None,
                device,
            )

            if news_encoded is None or news_encoded.shape[1] == 0:
                return {'vix_pred': torch.zeros(B, NUM_HORIZONS, device=device)}

            # Process through news Mamba (no checkpoint segments, news is sparse)
            news_out = self.mamba(news_encoded, combined_mask)
            # Pool over the full sequence
            pooled = self.pool(news_out)
            pred = self.head(pooled)
            return {'vix_pred': pred}

        # --- VIX stream ---
        elif self.stream_type == 'vix':
            if vix_features is None or vix_features.shape[1] == 0:
                return {'vix_pred': torch.zeros(B, NUM_HORIZONS, device=device)}
            encoded = self.encoder(vix_features)
            out = self.mamba(encoded, vix_mask, film_params)
            pooled = self.pool(out)
            pred = self.head(pooled)
            return {'vix_pred': pred}

        else:
            raise ValueError(f"Unknown stream type: {self.stream_type}")


# ---------------------------------------------------------------------------
# Helpers reused from train.py
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_paths():
    from train import get_data_paths as _gdp
    return _gdp()


def batch_to_device(batch, device):
    d = {}
    d['bars'] = batch['bars'].to(device, non_blocking=True)
    d['bar_mask'] = batch['bar_mask'].to(device, non_blocking=True)
    d['target'] = batch['vix_targets'].to(device, non_blocking=True)
    d['horizon_mask'] = batch['horizon_mask'].to(device, non_blocking=True)
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
def train_epoch(model, loader, optimizer, scaler, device, amp_dtype, epoch, stream_type):
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    for step, batch in enumerate(loader):
        t_start = time.time()
        bd = batch_to_device(batch, device)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(
                bd['bars'], bd['bar_mask'],
                bar_timestamps=bd['bar_timestamps'],
                macro_context=bd['macro_context'],
                options=bd['options'],
                options_mask=bd['options_mask'],
                news_embs=bd['news_embs'],
                news_mask=bd['news_mask'],
                news_timestamps=bd['news_timestamps'],
                gdelt_embs=bd['gdelt_embs'],
                gdelt_mask=bd['gdelt_mask'],
                gdelt_timestamps=bd['gdelt_timestamps'],
                econ_event_ids=bd['econ_event_ids'],
                econ_currency_ids=bd['econ_currency_ids'],
                econ_numeric=bd['econ_numeric'],
                econ_mask=bd['econ_mask'],
                econ_timestamps=bd['econ_timestamps'],
                vix_features=bd['vix_features'],
                vix_mask=bd['vix_mask'],
                vix_timestamps=bd['vix_timestamps'],
            )
            pred = outputs['vix_pred']  # [B, 4]
            target = bd['target']
            horizon_mask = bd['horizon_mask']

            # Plain Huber loss, equally weighted across 4 horizons
            huber = F.huber_loss(pred, target, delta=0.25, reduction='none')  # [B, 4]
            huber = huber * horizon_mask
            valid = horizon_mask.sum()
            loss = huber.sum() / valid.clamp(min=1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        num_batches += 1

        t_total = time.time() - t_start
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

        if (step + 1) % 15 == 0 or step == 0:
            avg = total_loss / num_batches
            logger.info(
                f"[{stream_type}] E{epoch} step {step+1}: "
                f"loss={loss.item():.4f} avg={avg:.4f} | "
                f"{t_total:.2f}s/it | VRAM={mem:.1f}GB"
            )
            dashboard.console.print(
                f"[bold][{stream_type}] E{epoch} Step {step+1}[/] | "
                f"loss={loss.item():.4f} avg={avg:.4f} | "
                f"[dim]{t_total:.2f}s/it | VRAM={mem:.1f}GB[/]"
            )

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, epoch_time / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_epoch(model, loader, device, amp_dtype, stream_type):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds, all_targets, all_masks = [], [], []

    for batch in loader:
        bd = batch_to_device(batch, device)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            outputs = model(
                bd['bars'], bd['bar_mask'],
                bar_timestamps=bd['bar_timestamps'],
                macro_context=bd['macro_context'],
                options=bd['options'],
                options_mask=bd['options_mask'],
                news_embs=bd['news_embs'],
                news_mask=bd['news_mask'],
                news_timestamps=bd['news_timestamps'],
                gdelt_embs=bd['gdelt_embs'],
                gdelt_mask=bd['gdelt_mask'],
                gdelt_timestamps=bd['gdelt_timestamps'],
                econ_event_ids=bd['econ_event_ids'],
                econ_currency_ids=bd['econ_currency_ids'],
                econ_numeric=bd['econ_numeric'],
                econ_mask=bd['econ_mask'],
                econ_timestamps=bd['econ_timestamps'],
                vix_features=bd['vix_features'],
                vix_mask=bd['vix_mask'],
                vix_timestamps=bd['vix_timestamps'],
            )
            pred = outputs['vix_pred']
            target = bd['target']
            horizon_mask = bd['horizon_mask']

            huber = F.huber_loss(pred, target, delta=0.25, reduction='none')
            huber = huber * horizon_mask
            valid = horizon_mask.sum()
            loss = huber.sum() / valid.clamp(min=1)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
        all_masks.append(horizon_mask.cpu())

    if num_batches == 0:
        return {'loss': 0.0}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    masks = torch.cat(all_masks)

    result = {'loss': total_loss / num_batches}
    for i, h in enumerate(HORIZONS):
        valid = masks[:, i].bool()
        if valid.sum() == 0:
            continue
        p = preds[:, i][valid]
        t = targets[:, i][valid]
        result[f'mae_{h}d'] = (p - t).abs().mean().item()
        pred_sign = (p > 0).float()
        target_sign = (t > 0).float()
        result[f'dir_acc_{h}d'] = (pred_sign == target_sign).float().mean().item() * 100

    return result


# ---------------------------------------------------------------------------
# Summarize pre-trained streams
# ---------------------------------------------------------------------------
def summarize_streams(checkpoint_dir='checkpoints'):
    ckpt_dir = Path(checkpoint_dir)
    streams = ['stock', 'options', 'news', 'vix']

    print("\n" + "=" * 100)
    print("PHASE 1 PRE-TRAINING SUMMARY")
    print("=" * 100)
    header = f"{'Stream':<10} {'Val Loss':<10} "
    for h in HORIZONS:
        header += f"+{h}d MAE  +{h}d Dir  "
    print(header)
    print("-" * 100)

    for s in streams:
        ckpt_path = ckpt_dir / f'pretrain_{s}.pt'
        if not ckpt_path.exists():
            print(f"{s:<10} NOT FOUND")
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        metrics = ckpt.get('metrics', {})
        val_loss = ckpt.get('best_val_loss', 0)
        row = f"{s:<10} {val_loss:<10.4f} "
        for h in HORIZONS:
            mae = metrics.get(f'+{h}d_mae', 0)
            dir_acc = metrics.get(f'+{h}d_dir', 0)
            row += f"{mae:<8.2f} {dir_acc:<8.1f} "
        print(row)
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Phase 1: Single-Stream Pre-Training')
    parser.add_argument('--stream', type=str, choices=['stock', 'options', 'news', 'vix'],
                        help='Stream type to pre-train')
    parser.add_argument('--seq-len', type=int, default=cfg.seq_len)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--film-lr', type=float, default=None,
                        help='FiLM LR (default: same as --lr)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d-model', type=int, default=cfg.d_model)
    parser.add_argument('--n-layers', type=int, default=cfg.n_layers)
    parser.add_argument('--d-state', type=int, default=64,
                        help='Mamba state dimension for pre-training (default: 64)')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=cfg.checkpoint_interval)
    parser.add_argument('--train-start', type=str, default='2005-01-01')
    parser.add_argument('--train-end', type=str, default=cfg.train_end)
    parser.add_argument('--val-end', type=str, default=cfg.val_end)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--summarize', action='store_true',
                        help='Print summary of all pre-trained streams and exit')
    parser.add_argument('--vix-d-model', type=int, default=cfg.vix_d_model)
    parser.add_argument('--vix-d-state', type=int, default=16)
    parser.add_argument('--vix-n-layers', type=int, default=cfg.vix_n_layers)
    parser.add_argument('--preprocessed-path', type=str, default=None)
    parser.add_argument('--predict-target', type=str, default=cfg.predict_target,
                        choices=['vix', 'vxx', 'spy'],
                        help='Prediction target ticker (default: vix)')
    args = parser.parse_args()

    if args.summarize:
        summarize_streams(args.checkpoint_dir)
        return 0

    if args.stream is None:
        parser.error("--stream is required (stock, options, news, or vix)")

    stream = args.stream
    film_lr = args.film_lr or args.lr

    # Setup
    log_file = setup_logging(stream)
    seed_everything(42)
    dashboard.start()

    logger.info(f"Phase 1 pre-training: stream={stream}")
    logger.info(f"Config: seq_len={args.seq_len}, epochs={args.epochs}, "
                f"batch_size={args.batch_size}, lr={args.lr}, film_lr={film_lr}")
    dashboard.log(f"[bold cyan]Phase 1 Pre-Training: {stream.upper()}[/]")
    dashboard.log(f"[dim]Logs: {log_file}[/]")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        dashboard.log(f"[bold green]GPU:[/] {gpu_name} ({vram:.1f}GB)")

    # --- Data ---
    data_paths = get_data_paths()
    if not data_paths:
        logger.error("No data directory found")
        return 1

    # Determine which data streams to enable
    use_news = (stream == 'news')
    use_options = (stream == 'options')
    use_gdelt = (stream == 'news')
    use_econ = (stream == 'news')
    use_vix_features = (stream == 'vix')
    use_macro = True  # Always for FiLM
    use_fundamentals = False  # Not used in single-stream pre-training

    from loader.bar_mamba_dataset import BarMambaDataset, NUM_STOCK_FEATURES, NUM_OPTION_FEATURES

    stock_path = str(data_paths['stock'])
    vix_path = str(data_paths['vix'])

    # Resolve optional paths
    news_path = None
    if use_news:
        for c in [data_paths.get('stock', Path('.')).parent / 'benzinga_embeddings',
                  Path('datasets/benzinga_embeddings')]:
            if c.exists():
                news_path = str(c)
                break

    options_path = None
    if use_options:
        auto = data_paths.get('options')
        if auto and auto.exists():
            options_path = str(auto)
        else:
            for c in [data_paths.get('stock', Path('.')).parent / 'opt_trade_2min',
                      Path('datasets/opt_trade_2min')]:
                if c.exists():
                    options_path = str(c)
                    break

    macro_path = None
    for c in [data_paths.get('stock', Path('.')).parent / 'MACRO' / 'macro_daily_enhanced.parquet',
              Path('datasets/MACRO/macro_daily_enhanced.parquet'),
              data_paths.get('stock', Path('.')).parent / 'MACRO' / 'macro_daily.parquet',
              Path('datasets/MACRO/macro_daily.parquet')]:
        if c.exists():
            macro_path = str(c)
            break

    gdelt_path = None
    if use_gdelt:
        for c in [data_paths.get('stock', Path('.')).parent / 'GDELT',
                  Path('datasets/GDELT')]:
            if c.exists():
                gdelt_path = str(c)
                break

    econ_path = None
    if use_econ:
        for c in [data_paths.get('stock', Path('.')).parent / 'econ_calendar',
                  Path('datasets/econ_calendar')]:
            if c.exists():
                econ_path = str(c)
                break

    vix_features_path = None
    if use_vix_features:
        for c in [data_paths.get('stock', Path('.')).parent / 'VIX' / 'Vix_features',
                  Path('datasets/VIX/Vix_features')]:
            if c.exists():
                vix_features_path = str(c)
                break

    # Auto-detect preprocessed memmaps
    pp_path = args.preprocessed_path
    if pp_path is None:
        for c in [data_paths.get('stock', Path('.')).parent / 'preprocessed',
                  Path('datasets/preprocessed')]:
            if (c / 'stock_index.json').exists():
                pp_path = str(c)
                break

    train_dataset = BarMambaDataset(
        stock_data_path=stock_path, vix_data_path=vix_path,
        split='train', max_total_bars=args.seq_len,
        train_start=args.train_start, train_end=args.train_end, val_end=args.val_end,
        news_data_path=news_path, use_news=use_news,
        options_data_path=options_path, use_options=use_options,
        macro_data_path=macro_path, use_macro=use_macro,
        gdelt_data_path=gdelt_path, use_gdelt=use_gdelt,
        econ_calendar_path=econ_path, use_econ=use_econ,
        use_fundamentals=False,
        vix_features_path=vix_features_path, use_vix_features=use_vix_features,
        preprocessed_path=pp_path,
        predict_target=args.predict_target,
    )
    val_dataset = BarMambaDataset(
        stock_data_path=stock_path, vix_data_path=vix_path,
        split='val', max_total_bars=args.seq_len,
        train_start=args.train_start, train_end=args.train_end, val_end=args.val_end,
        news_data_path=news_path, use_news=use_news,
        options_data_path=options_path, use_options=use_options,
        macro_data_path=macro_path, use_macro=use_macro,
        gdelt_data_path=gdelt_path, use_gdelt=use_gdelt,
        econ_calendar_path=econ_path, use_econ=use_econ,
        use_fundamentals=False,
        vix_features_path=vix_features_path, use_vix_features=use_vix_features,
        shared_state=train_dataset.get_shared_state(),
        preprocessed_path=pp_path,
        predict_target=args.predict_target,
    )

    collate_fn = BarMambaDataset.collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn, pin_memory=True)

    dashboard.log(f"[dim]Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples[/]")
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # --- Model ---
    macro_dim = getattr(train_dataset, 'macro_dim', 55)
    gdelt_dim = getattr(train_dataset, 'gdelt_dim', 391)
    econ_num_event_types = getattr(train_dataset, 'econ_num_event_types', 412) + 1 if use_econ else 413
    econ_num_currencies = getattr(train_dataset, 'econ_num_currencies', 4) + 1 if use_econ else 5

    # Determine input dim per stream
    if stream == 'stock':
        input_dim = train_dataset.num_features
    elif stream == 'options':
        input_dim = NUM_OPTION_FEATURES
    elif stream == 'news':
        input_dim = 3072  # Benzinga embedding dim
    elif stream == 'vix':
        input_dim = getattr(train_dataset, 'num_vix_features', 25)

    model = SingleStreamModel(
        stream_type=stream,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        dropout=args.dropout,
        input_dim=input_dim,
        macro_dim=macro_dim,
        checkpoint_interval=args.checkpoint_interval,
        news_dim=3072,
        use_gdelt=use_gdelt,
        gdelt_dim=gdelt_dim,
        use_econ=use_econ,
        econ_num_event_types=econ_num_event_types,
        econ_num_currencies=econ_num_currencies,
        vix_d_model=args.vix_d_model,
        vix_d_state=args.vix_d_state,
        vix_n_layers=args.vix_n_layers,
    ).to(device)

    # --- Optimizer ---
    film_params = list(model.film.parameters())
    film_ids = set(id(p) for p in film_params)
    other_params = [p for p in model.parameters() if id(p) not in film_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': film_params, 'lr': film_lr, 'weight_decay': 0},
    ], weight_decay=1e-4)

    dashboard.log(f"[dim]LR: {args.lr} (FiLM: {film_lr}, wd=0 for FiLM)[/]")
    logger.info(f"LR: {args.lr}, FiLM LR: {film_lr}")

    # AMP
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))
    dashboard.log(f"[dim]AMP: {amp_dtype}[/]")
    dashboard.log("─" * 50)

    # --- Training loop ---
    best_val_loss = float('inf')
    best_metrics = {}
    epochs_no_improve = 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        dashboard.log(f"\n[bold cyan]═══ [{stream.upper()}] Epoch {epoch}/{args.epochs} ═══[/]")

        train_loss, avg_iter = train_epoch(
            model, train_loader, optimizer, scaler, device, amp_dtype, epoch, stream
        )

        val_metrics = val_epoch(model, val_loader, device, amp_dtype, stream)
        val_loss = val_metrics['loss']

        # Display
        horizon_parts = []
        for h in HORIZONS:
            mae = val_metrics.get(f'mae_{h}d', 0)
            dir_acc = val_metrics.get(f'dir_acc_{h}d', 0)
            horizon_parts.append(f"+{h}d:{mae:.2f}pt/{dir_acc:.0f}%")
        horizon_display = " | ".join(horizon_parts)

        # FiLM stats
        film_stats = model.film.get_film_stats()
        film_parts = []
        for i in range(model.actual_layers):
            gm = film_stats.get(f'film_gamma_L{i}_mean', 1.0)
            gs = film_stats.get(f'film_gamma_L{i}_std', 0.0)
            bm = film_stats.get(f'film_beta_L{i}_mean', 0.0)
            bs = film_stats.get(f'film_beta_L{i}_std', 0.0)
            film_parts.append(f"L{i}:γ={gm:.3f}±{gs:.3f},β={bm:.3f}±{bs:.3f}")
        film_display = " | ".join(film_parts)

        dashboard.log(
            f"[green]✓ [{stream.upper()}] E{epoch}:[/] "
            f"train={train_loss:.4f} val={val_loss:.4f}"
        )
        dashboard.log(f"  [bold]Horizons:[/] {horizon_display}")
        dashboard.log(f"  [dim]FiLM: {film_display}[/]")
        dashboard.log(f"  [dim]iter={avg_iter:.2f}s/it[/]")

        logger.info(
            f"[{stream}] Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"horizons=[{horizon_display}]"
        )
        logger.info(f"FiLM stats: {film_display}")

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_metrics = {}
            for h in HORIZONS:
                best_metrics[f'+{h}d_mae'] = val_metrics.get(f'mae_{h}d', 0)
                best_metrics[f'+{h}d_dir'] = val_metrics.get(f'dir_acc_{h}d', 0)

            # Save checkpoint
            save_dict = {
                'stream_type': stream,
                'mamba': model.mamba.state_dict(),
                'film': model.film.state_dict(),
                'pool': model.pool.state_dict(),
                'config': {
                    'd_model': model.actual_d,
                    'n_layers': model.actual_layers,
                    'd_state': args.d_state if stream != 'vix' else args.vix_d_state,
                    'input_dim': input_dim,
                    'macro_dim': macro_dim,
                    'checkpoint_interval': args.checkpoint_interval,
                },
                'best_val_loss': best_val_loss,
                'metrics': best_metrics,
                'epoch': epoch,
            }
            # Stream-specific encoder saving
            if stream == 'news':
                save_dict['news_encoder'] = model.news_encoder.state_dict()
                save_dict['type_embedding'] = model.news_type_embedding.state_dict()
                if use_gdelt:
                    save_dict['gdelt_encoder'] = model.gdelt_encoder.state_dict()
                    save_dict['gdelt_embed_norm'] = model.gdelt_embed_norm.state_dict()
                    save_dict['gdelt_stats_norm'] = model.gdelt_stats_norm.state_dict()
                if use_econ:
                    save_dict['econ_encoder'] = model.econ_encoder.state_dict()
            else:
                save_dict['encoder'] = model.encoder.state_dict()

            save_path = ckpt_dir / f'pretrain_{stream}.pt'
            torch.save(save_dict, save_path)
            dashboard.log(f"[bold green]🏆 New best![/] val_loss={best_val_loss:.4f} → {save_path}")
            logger.info(f"New best model saved: val_loss={best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            dashboard.log(f"[dim]No improvement for {epochs_no_improve}/{args.patience} epochs[/]")
            if epochs_no_improve >= args.patience:
                dashboard.log(f"[bold yellow]⏹ Early stopping at epoch {epoch}[/]")
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final summary
    dashboard.log(f"\n[bold]{'=' * 60}[/]")
    dashboard.log(f"[bold cyan]STREAM: {stream.upper()} — COMPLETE[/]")
    dashboard.log(f"Best val_loss: {best_val_loss:.4f}")
    for h in HORIZONS:
        mae = best_metrics.get(f'+{h}d_mae', 0)
        dir_acc = best_metrics.get(f'+{h}d_dir', 0)
        dashboard.log(f"  +{h}d: MAE={mae:.2f}pt, Dir={dir_acc:.1f}%")
    dashboard.log(f"[bold]{'=' * 60}[/]")

    dashboard.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
