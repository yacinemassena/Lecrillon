"""
Stock 1s → Transformer → Mamba → VIX Prediction Model.

Three model classes:
- StockMambaL1: Transformer frames → Mamba-1 (1,170 steps) → next-day VIX
- StockMambaL2: Daily summaries → Mamba-2 (365 steps) → VIX +30d
- StockMambaFull: End-to-end L0 → L1 → L2
"""

import math
import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from encoder.bar_encoder import BarEncoder, BarFrameEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mamba import (custom sm_120 build)
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba
except ImportError:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except ImportError:
        raise ImportError(
            "mamba_ssm not found. Install from custom_packages/mamba_blackwell "
            "or run setupenv.sh"
        )

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None
    logger.warning("Mamba2 not available, Level 2 will use Mamba-1 architecture")


# ---------------------------------------------------------------------------
# Sequence Pooling
# ---------------------------------------------------------------------------
class SequencePooling(nn.Module):
    """Pooling strategies for sequence embeddings."""

    def __init__(self, d_model: int, pooling_type: str = 'last'):
        super().__init__()
        self.pooling_type = pooling_type.lower()
        if self.pooling_type == 'attention':
            self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] optional (1=valid, 0=pad)
        Returns:
            [B, D]
        """
        if self.pooling_type == 'last':
            if mask is not None:
                # Find last valid position per batch
                lengths = mask.sum(dim=1).long().clamp(min=1)
                return x[torch.arange(x.size(0), device=x.device), lengths - 1]
            return x[:, -1, :]
        elif self.pooling_type == 'mean':
            if mask is not None:
                mask_exp = mask.unsqueeze(-1)
                return (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            return x.mean(dim=1)
        elif self.pooling_type == 'attention':
            scores = self.attn(x)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            weights = torch.softmax(scores, dim=1)
            return (weights * x).sum(dim=1)
        else:
            return x[:, -1, :]


# ---------------------------------------------------------------------------
# VIX Prediction Head
# ---------------------------------------------------------------------------
class VIXHead(nn.Module):
    """MLP head → single VIX scalar."""

    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize final layer to predict 0 (z-score normalized VIX mean)
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → [B]"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Mamba Block Stack
# ---------------------------------------------------------------------------
class MambaStack(nn.Module):
    """Stack of Mamba layers with residual + norm + dropout."""

    def __init__(self, config, use_mamba2: bool = False):
        super().__init__()
        MambaClass = Mamba2 if (use_mamba2 and Mamba2 is not None) else Mamba

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(config.n_layers):
            self.layers.append(
                MambaClass(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
            )
            self.norms.append(nn.LayerNorm(config.d_model))
            self.dropouts.append(
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            residual = x
            x = layer(x)
            x = norm(x)
            x = drop(x)
            x = x + residual
        return x


# ---------------------------------------------------------------------------
# StockMambaL1: Transformer → Mamba-1 → VIX next-day
# ---------------------------------------------------------------------------
class StockMambaL1(nn.Module):
    """
    Level 0 + Level 1: Stock bars → Transformer frames → Mamba-1 → VIX.

    Input: bar frames [total_frames, max_bars, num_features]
    Output: VIX prediction scalar
    """

    def __init__(self, config):
        super().__init__()
        enc_cfg = config.encoder
        m1_cfg = config.mamba1

        # Level 0: Transformer frame encoder
        self.bar_encoder = BarEncoder(
            num_features=enc_cfg.num_features,
            hidden_dim=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            num_heads=enc_cfg.num_heads,
            dropout=enc_cfg.dropout,
            num_tickers=enc_cfg.num_tickers,
            ticker_embed_dim=enc_cfg.ticker_embed_dim,
        )

        self.frame_encoder = BarFrameEncoder(
            bar_encoder=self.bar_encoder,
            d_model=enc_cfg.hidden_dim,
        )

        # Projection to Mamba d_model
        self.d_model = m1_cfg.d_model
        if enc_cfg.hidden_dim != m1_cfg.d_model:
            self.proj = nn.Linear(enc_cfg.hidden_dim, m1_cfg.d_model)
        else:
            self.proj = nn.Identity()

        # Level 1: Mamba-1
        self.mamba = MambaStack(m1_cfg, use_mamba2=False)

        # Pooling + head
        self.pooling = SequencePooling(m1_cfg.d_model, pooling_type='attention')
        self.vix_head = VIXHead(m1_cfg.d_model, dropout=m1_cfg.dropout)

        # Config for daily pooling (used by L2)
        self.frames_per_day = enc_cfg.frames_per_day

    def encode_frames(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode bar frames through Transformer.

        Args:
            frames: [N, max_bars, F]
            frame_mask: [N, max_bars]
            ticker_ids: [N, max_bars] optional

        Returns:
            [N, hidden_dim] frame embeddings
        """
        return self.frame_encoder(frames, frame_mask, ticker_ids)

    def encode_frames_chunked(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
        chunk_size: int = 128,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Encode frames in chunks to reduce VRAM usage.
        
        Instead of loading all 1170 frames at once (~4GB VRAM),
        process 128 frames at a time (~400MB VRAM).

        Args:
            frames: [N, max_bars, F]
            frame_mask: [N, max_bars]
            ticker_ids: [N, max_bars] optional
            chunk_size: frames per chunk (default 128)
            use_checkpoint: use gradient checkpointing (saves VRAM, slower)

        Returns:
            [N, hidden_dim] frame embeddings
        """
        N = frames.size(0)
        if N <= chunk_size:
            return self.encode_frames(frames, frame_mask, ticker_ids)
        
        embeddings = []
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_frames = frames[i:end_idx]
            chunk_mask = frame_mask[i:end_idx]
            chunk_tickers = ticker_ids[i:end_idx] if ticker_ids is not None else None
            
            if use_checkpoint and self.training:
                # Gradient checkpointing: recompute forward during backward
                emb = torch_checkpoint(
                    self.encode_frames,
                    chunk_frames, chunk_mask, chunk_tickers,
                    use_reentrant=False,
                )
            else:
                emb = self.encode_frames(chunk_frames, chunk_mask, chunk_tickers)
            embeddings.append(emb)
        
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        return_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward: frames → Transformer (chunked) → Mamba-1 → VIX.

        Args:
            frames: [N_total, max_bars, F]
            frame_mask: [N_total, max_bars]
            ticker_ids: [N_total, max_bars] optional
            chunk_size: frames per Transformer chunk (reduces VRAM)
            return_timing: if True, include timing breakdown in output

        Returns:
            dict with 'vix_pred' [1] and 'mamba_hidden' [1, T, D]
        """
        import time
        timings = {}
        
        # Level 0: encode frames in chunks (VRAM efficient)
        t0 = time.time()
        frame_emb = self.encode_frames_chunked(
            frames, frame_mask, ticker_ids, chunk_size=chunk_size
        )  # [N, hidden_dim]
        if return_timing:
            torch.cuda.synchronize()
            timings['transformer'] = time.time() - t0

        # Project to Mamba dim
        t0 = time.time()
        x = self.proj(frame_emb)  # [N, d_model]

        # Reshape to [1, T, d_model] (single sample)
        x = x.unsqueeze(0)  # [1, T, d_model]

        # Level 1: Mamba-1
        h = self.mamba(x)  # [1, T, d_model]
        if return_timing:
            torch.cuda.synchronize()
            timings['mamba'] = time.time() - t0

        # Pool → predict
        t0 = time.time()
        h_pool = self.pooling(h)  # [1, d_model]
        vix_pred = self.vix_head(h_pool)  # [1]
        if return_timing:
            torch.cuda.synchronize()
            timings['head'] = time.time() - t0

        result = {
            'vix_pred': vix_pred,
            'mamba_hidden': h,
        }
        if return_timing:
            result['timing'] = timings
        return result

    def get_daily_summaries(self, mamba_hidden: torch.Tensor) -> torch.Tensor:
        """Pool Mamba-1 hidden states into daily summaries for Level 2.

        Args:
            mamba_hidden: [1, T, D] from forward()

        Returns:
            [1, N_days, D] daily summaries
        """
        T = mamba_hidden.size(1)
        fpd = self.frames_per_day
        n_days = T // fpd

        if n_days == 0:
            return mamba_hidden.mean(dim=1, keepdim=True)

        # Trim to exact multiple of frames_per_day
        trimmed = mamba_hidden[:, :n_days * fpd, :]  # [1, n_days*fpd, D]
        # Reshape and pool each day
        daily = trimmed.view(1, n_days, fpd, -1)  # [1, n_days, fpd, D]
        daily_summaries = daily.mean(dim=2)  # [1, n_days, D]

        return daily_summaries


# ---------------------------------------------------------------------------
# StockMambaL2: Daily summaries → Mamba-2 → VIX +30d
# ---------------------------------------------------------------------------
class StockMambaL2(nn.Module):
    """
    Level 2: Daily embeddings → Mamba-2 → VIX +30d.

    Input: daily summary embeddings [B, N_days, d_model]
    Output: VIX +30d prediction
    """

    def __init__(self, config):
        super().__init__()
        m2_cfg = config.mamba2
        m1_cfg = config.mamba1

        # Project from L1 d_model to L2 d_model if different
        if m1_cfg.d_model != m2_cfg.d_model:
            self.proj = nn.Linear(m1_cfg.d_model, m2_cfg.d_model)
        else:
            self.proj = nn.Identity()

        # Level 2: Mamba-2
        use_m2 = Mamba2 is not None
        self.mamba = MambaStack(m2_cfg, use_mamba2=use_m2)

        # Pooling + head
        self.pooling = SequencePooling(m2_cfg.d_model, pooling_type='attention')
        self.vix_head = VIXHead(m2_cfg.d_model, dropout=m2_cfg.dropout)

    def forward(self, daily_summaries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            daily_summaries: [B, N_days, d_model_l1]

        Returns:
            dict with 'vix_pred' [B]
        """
        x = self.proj(daily_summaries)
        h = self.mamba(x)
        h_pool = self.pooling(h)
        vix_pred = self.vix_head(h_pool)

        return {'vix_pred': vix_pred}


# ---------------------------------------------------------------------------
# StockMambaFull: End-to-end L0 → L1 → L2
# ---------------------------------------------------------------------------
class StockMambaFull(nn.Module):
    """
    Combined model for joint training (future use).
    Currently trains L1 and L2 separately.
    """

    def __init__(self, config):
        super().__init__()
        self.l1 = StockMambaL1(config)
        self.l2 = StockMambaL2(config)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # L1 forward
        l1_out = self.l1(frames, frame_mask, ticker_ids)
        # Get daily summaries for L2
        daily = self.l1.get_daily_summaries(l1_out['mamba_hidden'])
        # L2 forward
        l2_out = self.l2(daily)

        return {
            'vix_pred_l1': l1_out['vix_pred'],
            'vix_pred_l2': l2_out['vix_pred'],
        }


def build_model(config, level: int = 1) -> nn.Module:
    """Factory function to build model by level.

    Args:
        config: Config dataclass
        level: 1 for Mamba-1, 2 for Mamba-2, 0 for full

    Returns:
        nn.Module
    """
    if level == 1:
        model = StockMambaL1(config)
    elif level == 2:
        model = StockMambaL2(config)
    else:
        model = StockMambaFull(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Built StockMambaL{level} with {num_params:,} parameters")

    return model
