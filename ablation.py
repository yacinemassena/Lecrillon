"""
A/B Stream Ablation Study.

Loads a trained checkpoint and runs inference on the validation set with each
stream selectively disabled.  Quantifies marginal contribution of every stream
by comparing per-horizon MAE / directional-accuracy / loss against the
all-streams baseline.

Usage:
    python ablation.py --checkpoint downloaded_outputs_my_experiment/checkpoints/checkpoint.pt
    python ablation.py --checkpoint checkpoints/best/checkpoint.pt --batch-size 2
"""

import argparse
import csv
import datetime
import logging
import sys
import time
import types
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# If mamba_ssm is not installed (e.g. Windows), inject pure-PyTorch shim
# ---------------------------------------------------------------------------
try:
    import mamba_ssm  # noqa: F401 — real CUDA kernels (WSL/Linux)
except ImportError:
    from mamba_shim import Mamba as _ShimMamba
    _mamba_ssm = types.ModuleType('mamba_ssm')
    _mamba_ssm.Mamba = _ShimMamba
    _mamba_ssm_modules = types.ModuleType('mamba_ssm.modules')
    _mamba_ssm_modules_simple = types.ModuleType('mamba_ssm.modules.mamba_simple')
    _mamba_ssm_modules_simple.Mamba = _ShimMamba
    _mamba_ssm.modules = _mamba_ssm_modules
    _mamba_ssm_modules.mamba_simple = _mamba_ssm_modules_simple
    sys.modules['mamba_ssm'] = _mamba_ssm
    sys.modules['mamba_ssm.modules'] = _mamba_ssm_modules
    sys.modules['mamba_ssm.modules.mamba_simple'] = _mamba_ssm_modules_simple
    logging.getLogger(__name__).info('mamba_ssm not found, using pure-PyTorch shim')

from trainconfig import DEFAULT_CONFIG as cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Horizons (must match train.py)
# ---------------------------------------------------------------------------
HORIZONS = [1, 7, 15, 30]
NUM_HORIZONS = len(HORIZONS)

# ---------------------------------------------------------------------------
# Loss (copied from train.py to avoid import side-effects)
# ---------------------------------------------------------------------------
class SpikeWeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.25, spike_thresh=2.0, extreme_thresh=4.0,
                 spike_weight=3.0, extreme_weight=5.0):
        super().__init__()
        self.delta = delta
        self.spike_thresh = spike_thresh
        self.extreme_thresh = extreme_thresh
        self.spike_weight = spike_weight
        self.extreme_weight = extreme_weight
        self.huber = nn.HuberLoss(delta=delta, reduction='none')

    def forward(self, pred, target, horizon_mask=None):
        base_loss = self.huber(pred, target)
        abs_target = target.abs()
        weights = torch.ones_like(target)
        weights = torch.where(abs_target >= self.spike_thresh,
                              torch.tensor(self.spike_weight, device=target.device), weights)
        weights = torch.where(abs_target >= self.extreme_thresh,
                              torch.tensor(self.extreme_weight, device=target.device), weights)
        weighted_loss = base_loss * weights
        if horizon_mask is not None:
            weighted_loss = weighted_loss * horizon_mask
            valid_count = horizon_mask.sum()
            if valid_count > 0:
                return weighted_loss.sum() / valid_count
        return weighted_loss.mean()


# ---------------------------------------------------------------------------
# batch_to_device  (from train.py)
# ---------------------------------------------------------------------------
def batch_to_device(batch: Dict, device: torch.device) -> Dict:
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
# Ablation conditions
# ---------------------------------------------------------------------------
ABLATION_CONDITIONS: List[Dict] = [
    {
        'name': 'Baseline (all streams)',
        'null_keys': [],
    },
    {
        'name': '−Options',
        'null_keys': ['options', 'options_mask'],
    },
    {
        'name': '−News (Benzinga)',
        'null_keys': ['news_embs', 'news_mask', 'news_timestamps'],
    },
    {
        'name': '−GDELT',
        'null_keys': ['gdelt_embs', 'gdelt_mask', 'gdelt_timestamps'],
    },
    {
        'name': '−Econ Calendar',
        'null_keys': ['econ_event_ids', 'econ_currency_ids', 'econ_numeric',
                      'econ_mask', 'econ_timestamps'],
    },
    {
        'name': '−All News-like (Benz+GDELT+Econ)',
        'null_keys': ['news_embs', 'news_mask', 'news_timestamps',
                      'gdelt_embs', 'gdelt_mask', 'gdelt_timestamps',
                      'econ_event_ids', 'econ_currency_ids', 'econ_numeric',
                      'econ_mask', 'econ_timestamps'],
    },
    {
        'name': '−VIX Features',
        'null_keys': ['vix_features', 'vix_mask', 'vix_timestamps'],
    },
    {
        'name': '−Macro FiLM',
        'null_keys': ['macro_context'],
    },
    {
        'name': '−Fundamentals',
        'null_keys': ['fundamentals_context'],
    },
]


# ---------------------------------------------------------------------------
# Inference loop for one condition
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_condition(model, loader, criterion, device, null_keys,
                  amp_dtype=torch.bfloat16):
    """Run full validation with specific keys nulled.  Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    all_horizon_masks = []

    total_batches = len(loader)
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        bd = batch_to_device(batch, device)

        # Null out keys for this ablation condition
        for key in null_keys:
            bd[key] = None

        target = bd['target']
        horizon_mask = bd['horizon_mask']

        t_batch = time.time()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            outputs = model(
                bd['bars'], bd['bar_mask'],
                options=bd['options'],
                options_mask=bd['options_mask'],
                news_embs=bd['news_embs'],
                news_mask=bd['news_mask'],
                news_timestamps=bd['news_timestamps'],
                gdelt_embs=bd['gdelt_embs'],
                gdelt_mask=bd['gdelt_mask'],
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
            pred = outputs['vix_pred']
            loss = criterion(pred, target, horizon_mask)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
        all_horizon_masks.append(horizon_mask.cpu())

        # Progress logging
        elapsed = time.time() - t_start
        batch_time = time.time() - t_batch
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            remaining = batch_time * (total_batches - batch_idx - 1)
            logger.info(f'    batch {batch_idx+1}/{total_batches}  '
                        f'{batch_time:.1f}s/batch  ETA {remaining:.0f}s')

    if num_batches == 0:
        return {}

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    h_masks = torch.cat(all_horizon_masks)

    result = {'loss': total_loss / num_batches, 'n_samples': len(preds)}

    for i, h in enumerate(HORIZONS):
        valid = h_masks[:, i].bool()
        if valid.sum() == 0:
            continue
        pred_h = preds[:, i][valid]
        target_h = targets[:, i][valid]

        mae_h = (pred_h - target_h).abs().mean().item()
        result[f'mae_{h}d'] = mae_h

        pred_sign = (pred_h > 0).float()
        target_sign = (target_h > 0).float()
        dir_acc_h = (pred_sign == target_sign).float().mean().item() * 100
        result[f'dir_{h}d'] = dir_acc_h

    return result


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------
def print_results(all_results: List[Dict], conditions: List[Dict]):
    baseline = all_results[0]

    # Header
    sep = '─' * 120
    print(f'\n{sep}')
    print(f'{"STREAM ABLATION RESULTS":^120}')
    print(sep)

    # Column headers
    header = f'{"Condition":<35} {"Loss":>7}'
    for h in HORIZONS:
        header += f' {"MAE+" + str(h) + "d":>9} {"Dir+" + str(h) + "d":>9}'
    print(header)
    print(sep)

    for cond, result in zip(conditions, all_results):
        name = cond['name']
        loss_str = f'{result["loss"]:.4f}'

        # Delta vs baseline
        if cond['null_keys']:  # not baseline
            d_loss = result['loss'] - baseline['loss']
            loss_str += f' ({d_loss:+.4f})'

        row = f'{name:<35} {loss_str:>17}'

        for h in HORIZONS:
            mae_key = f'mae_{h}d'
            dir_key = f'dir_{h}d'
            mae_val = result.get(mae_key, float('nan'))
            dir_val = result.get(dir_key, float('nan'))

            if cond['null_keys']:
                d_mae = mae_val - baseline.get(mae_key, 0)
                d_dir = dir_val - baseline.get(dir_key, 0)
                row += f' {mae_val:7.3f}pt' + f'({d_mae:+.3f})'
                row += f' {dir_val:6.1f}%' + f'({d_dir:+.1f})'
            else:
                row += f' {mae_val:7.3f}pt' + '         '
                row += f' {dir_val:6.1f}%' + '       '

        print(row)

    print(sep)

    # Rank streams by +1d MAE impact (higher = more important)
    print(f'\n{"STREAM IMPORTANCE RANKING (+1d MAE impact)":^120}')
    print(sep)
    impacts = []
    for cond, result in zip(conditions[1:], all_results[1:]):
        d_mae = result.get('mae_1d', 0) - baseline.get('mae_1d', 0)
        impacts.append((cond['name'], d_mae))

    impacts.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, d_mae) in enumerate(impacts, 1):
        bar_len = int(abs(d_mae) * 200)  # scale for display
        bar = '█' * min(bar_len, 50)
        direction = '↑ WORSE' if d_mae > 0 else '↓ better'
        print(f'  {rank}. {name:<40} {d_mae:+.4f} pt  {direction}  {bar}')

    print(sep)


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
def save_csv(all_results: List[Dict], conditions: List[Dict], path: str):
    baseline = all_results[0]
    fieldnames = ['condition', 'loss', 'delta_loss']
    for h in HORIZONS:
        fieldnames += [f'mae_{h}d', f'delta_mae_{h}d', f'dir_{h}d', f'delta_dir_{h}d']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond, result in zip(conditions, all_results):
            row = {
                'condition': cond['name'],
                'loss': f'{result["loss"]:.6f}',
                'delta_loss': f'{result["loss"] - baseline["loss"]:+.6f}' if cond['null_keys'] else '0.000000',
            }
            for h in HORIZONS:
                mae_val = result.get(f'mae_{h}d', 0)
                dir_val = result.get(f'dir_{h}d', 0)
                row[f'mae_{h}d'] = f'{mae_val:.4f}'
                row[f'delta_mae_{h}d'] = f'{mae_val - baseline.get(f"mae_{h}d", 0):+.4f}' if cond['null_keys'] else '0.0000'
                row[f'dir_{h}d'] = f'{dir_val:.2f}'
                row[f'delta_dir_{h}d'] = f'{dir_val - baseline.get(f"dir_{h}d", 0):+.2f}' if cond['null_keys'] else '0.00'
            writer.writerow(row)

    logger.info(f'Results saved to {path}')


# ---------------------------------------------------------------------------
# Data path detection (from train.py)
# ---------------------------------------------------------------------------
def get_data_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).parent
    if (script_dir / 'datasets').exists():
        base = script_dir / 'datasets'
    elif Path(r'D:\Mamba v2\datasets').exists():
        base = Path(r'D:\Mamba v2\datasets')
    else:
        return {}
    return {
        'stock': base / 'Stock_Data_2min',
        'options': base / 'opt_trade_2min',
        'vix': base / 'VIX',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='A/B Stream Ablation Study')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint.pt')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Inference batch size (default: 4, safe for 16GB)')
    parser.add_argument('--seq-len', type=int, default=cfg.seq_len,
                        help='Max sequence length (must match training)')
    parser.add_argument('--output-csv', type=str, default='ablation_results.csv',
                        help='Output CSV path')
    parser.add_argument('--train-start', type=str, default='2005-01-01')
    parser.add_argument('--train-end', type=str, default=cfg.train_end)
    parser.add_argument('--val-end', type=str, default=cfg.val_end)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    args = parser.parse_args()

    # Device — detect CUDA compatibility (RTX 5080 SM 120 not supported by current PyTorch)
    if args.device == 'auto':
        use_cuda = False
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works by running a trivial op
                torch.zeros(1, device='cuda')
                use_cuda = True
            except RuntimeError:
                logger.warning('CUDA available but kernels incompatible (SM 120?), falling back to CPU')
        device = torch.device('cuda' if use_cuda else 'cpu')
    else:
        device = torch.device(args.device)

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f'Device: {gpu_name} ({vram_gb:.1f}GB)')
    else:
        logger.info('Device: CPU (7.5M param model — inference is fine on CPU)')

    # Load checkpoint to inspect saved config
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    ckpt_epoch = checkpoint.get('epoch', '?')
    ckpt_val_loss = checkpoint.get('val_loss', '?')
    logger.info(f'Checkpoint epoch={ckpt_epoch}, val_loss={ckpt_val_loss}')

    # Data paths
    data_paths = get_data_paths()
    if not data_paths:
        logger.error('No datasets/ directory found')
        return 1

    stock_path = str(data_paths['stock'])
    vix_path = str(data_paths['vix'])

    # Auto-detect optional data paths
    base = data_paths['stock'].parent

    news_path = None
    for c in [base / 'benzinga_embeddings', Path('datasets/benzinga_embeddings')]:
        if c.exists():
            news_path = str(c)
            break

    options_path = None
    for c in [data_paths.get('options'), base / 'opt_trade_2min']:
        if c and c.exists():
            options_path = str(c)
            break

    macro_path = None
    for c in [base / 'MACRO' / 'macro_daily_enhanced.parquet',
              base / 'MACRO' / 'macro_daily.parquet']:
        if c.exists():
            macro_path = str(c)
            break

    gdelt_path = None
    for c in [base / 'GDELT']:
        if c.exists():
            gdelt_path = str(c)
            break

    econ_path = None
    for c in [base / 'econ_calendar']:
        if c.exists():
            econ_path = str(c)
            break

    fundamentals_path = None
    for c in [base / 'fundamentals' / 'fundamentals_state.parquet']:
        if c.exists():
            fundamentals_path = str(c)
            break

    vix_features_path = None
    for c in [base / 'VIX' / 'Vix_features']:
        if c.exists():
            vix_features_path = str(c)
            break

    # Auto-detect preprocessed memmaps
    pp_path = None
    for c in [base / 'preprocessed']:
        if (c / 'stock_index.json').exists():
            pp_path = str(c)
            break

    logger.info(f'Data paths: stock={stock_path}, news={news_path}, options={options_path}')
    logger.info(f'  macro={macro_path}, gdelt={gdelt_path}, econ={econ_path}')
    logger.info(f'  fundamentals={fundamentals_path}, vix_features={vix_features_path}')
    logger.info(f'  preprocessed={pp_path}')

    # Build dataset (val split only — we share state from a dummy train dataset)
    from loader.bar_mamba_dataset import BarMambaDataset, NUM_STOCK_FEATURES

    train_dataset = BarMambaDataset(
        stock_data_path=stock_path,
        vix_data_path=vix_path,
        split='train',
        max_total_bars=args.seq_len,
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        news_data_path=news_path,
        use_news=True,
        options_data_path=options_path,
        use_options=True,
        macro_data_path=macro_path,
        use_macro=True,
        gdelt_data_path=gdelt_path,
        use_gdelt=True,
        econ_calendar_path=econ_path,
        use_econ=True,
        fundamentals_data_path=fundamentals_path,
        use_fundamentals=True,
        vix_features_path=vix_features_path,
        use_vix_features=True,
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
        use_news=True,
        options_data_path=options_path,
        use_options=True,
        macro_data_path=macro_path,
        use_macro=True,
        gdelt_data_path=gdelt_path,
        use_gdelt=True,
        econ_calendar_path=econ_path,
        use_econ=True,
        fundamentals_data_path=fundamentals_path,
        use_fundamentals=True,
        vix_features_path=vix_features_path,
        use_vix_features=True,
        shared_state=train_dataset.get_shared_state(),
        preprocessed_path=pp_path,
    )

    num_features = train_dataset.num_features
    collate_fn = BarMambaDataset.collate_fn

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    logger.info(f'Val dataset: {len(val_dataset)} samples, num_features={num_features}')

    # Build model with same architecture as training (v1 = pre-WideFusion)
    from mamba_only_model_v1 import ParallelMambaVIX
    from loader.bar_mamba_dataset import NUM_OPTION_FEATURES

    macro_dim = getattr(train_dataset, 'macro_dim', 15)
    gdelt_dim = getattr(train_dataset, 'gdelt_dim', 391)
    econ_num_event_types = getattr(train_dataset, 'econ_num_event_types', 412) + 1
    econ_num_currencies = getattr(train_dataset, 'econ_num_currencies', 4) + 1
    fundamentals_dim = getattr(train_dataset, 'fundamentals_dim', 130)
    vix_features_dim = getattr(train_dataset, 'num_vix_features', 25)

    # Architecture params matching the training checkpoint (pre-V2 config bump)
    # d_state was 64 (not 128), vix_d_state was 16 (not 32) during training
    CKPT_D_STATE = 64
    CKPT_VIX_D_STATE = 16

    model = ParallelMambaVIX(
        num_features=num_features,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=CKPT_D_STATE,
        d_conv=4,
        expand=2,
        dropout=0.1,
        checkpoint_interval=cfg.checkpoint_interval,
        use_news=True,
        news_dim=3072,
        news_n_layers=cfg.news_n_layers,
        use_options=True,
        option_features=NUM_OPTION_FEATURES,
        head_hidden=128,
        use_macro=True,
        macro_dim=macro_dim,
        use_gdelt=True,
        gdelt_dim=gdelt_dim,
        use_econ=True,
        econ_num_event_types=econ_num_event_types,
        econ_num_currencies=econ_num_currencies,
        use_fundamentals=True,
        fundamentals_dim=fundamentals_dim,
        use_vix_features=True,
        vix_features_dim=vix_features_dim,
        vix_n_layers=cfg.vix_n_layers,
        vix_d_model=cfg.vix_d_model,
        vix_d_state=CKPT_VIX_D_STATE,
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model loaded: {num_params:,} parameters')

    # Loss
    criterion = SpikeWeightedHuberLoss()

    # AMP dtype (only used on CUDA)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

    # Run ablation conditions
    all_results = []
    total_conditions = len(ABLATION_CONDITIONS)

    logger.info(f'\n{"="*60}')
    logger.info(f'STARTING ABLATION STUDY ({total_conditions} conditions)')
    logger.info(f'{"="*60}')

    for idx, cond in enumerate(ABLATION_CONDITIONS):
        t0 = time.time()
        logger.info(f'\n[{idx+1}/{total_conditions}] {cond["name"]}')
        if cond['null_keys']:
            logger.info(f'  Nulling: {cond["null_keys"]}')

        result = run_condition(
            model, val_loader, criterion, device,
            null_keys=cond['null_keys'],
            amp_dtype=amp_dtype,
        )

        elapsed = time.time() - t0
        result['time_s'] = elapsed

        # Log immediate results
        mae_1d = result.get('mae_1d', float('nan'))
        dir_1d = result.get('dir_1d', float('nan'))
        logger.info(f'  loss={result["loss"]:.4f}  MAE+1d={mae_1d:.3f}pt  '
                     f'Dir+1d={dir_1d:.1f}%  ({elapsed:.1f}s)')

        if idx > 0 and all_results:
            baseline = all_results[0]
            d_mae = mae_1d - baseline.get('mae_1d', 0)
            d_loss = result['loss'] - baseline['loss']
            logger.info(f'  Δloss={d_loss:+.4f}  ΔMAE+1d={d_mae:+.4f}pt')

        all_results.append(result)

        # VRAM report
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f'  Peak VRAM: {mem:.2f}GB')

    # Print summary table
    print_results(all_results, ABLATION_CONDITIONS)

    # Save CSV
    save_csv(all_results, ABLATION_CONDITIONS, args.output_csv)
    logger.info(f'\nResults saved to {args.output_csv}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
