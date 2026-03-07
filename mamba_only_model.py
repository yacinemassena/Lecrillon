"""
Mamba-Only VIX Prediction Model.

Simplified architecture: 1s stock bars → Linear → Mamba → Pool → VIX prediction.
No Transformer encoder. Mamba's selective scan handles encoding + temporal dynamics.

Dual-head architecture:
    Bars [B, T, num_features] → BarProjection → [B, T, d_model]
                               → MambaStack (4L) → [B, T, d_model]
                               → AttentionPool → [B, d_model]
                               → VIXHead (regression) → [B] scalar (VIX change)
                               → VIXBucketHead (classification) → [B, 15] logits
"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mamba import
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba
except ImportError:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except ImportError:
        raise ImportError(
            "mamba_ssm not found. Install from custom_packages/mamba_blackwell "
            "or run setupenv.sh in WSL"
        )


# ---------------------------------------------------------------------------
# Bar Projection (replaces Transformer encoder)
# ---------------------------------------------------------------------------
class BarProjection(nn.Module):
    """Project raw bar features to model dimension.
    
    Simple Linear + LayerNorm + GELU replaces the entire Transformer encoder.
    Mamba downstream will handle temporal patterns via selective scan.
    """

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, num_features] → [B, T, d_model]"""
        return self.proj(x)


# ---------------------------------------------------------------------------
# Mamba Stack
# ---------------------------------------------------------------------------
class MambaStack(nn.Module):
    """Stack of Mamba layers with residual + norm + dropout."""

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
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
# Sequence Pooling
# ---------------------------------------------------------------------------
class SequencePooling(nn.Module):
    """Pooling strategies for sequence embeddings."""

    def __init__(self, d_model: int, pooling_type: str = 'attention'):
        super().__init__()
        self.pooling_type = pooling_type.lower()
        if self.pooling_type == 'attention':
            self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, D]"""
        if self.pooling_type == 'last':
            return x[:, -1, :]
        elif self.pooling_type == 'mean':
            return x.mean(dim=1)
        elif self.pooling_type == 'attention':
            scores = self.attn(x)  # [B, T, 1]
            weights = torch.softmax(scores, dim=1)
            return (weights * x).sum(dim=1)
        else:
            return x[:, -1, :]


# ---------------------------------------------------------------------------
# VIX Prediction Head
# ---------------------------------------------------------------------------
class VIXHead(nn.Module):
    """MLP regression head → single VIX change scalar (in VIX points)."""

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

        # Initialize final layer near zero (daily VIX change ≈ 0)
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → [B]"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# VIX Bucket Classification Head
# ---------------------------------------------------------------------------
class VIXBucketHead(nn.Module):
    """MLP classification head → bucket logits for VIX daily change.
    
    Acts as regularization on the regression head by enforcing
    decision boundaries ("is this a +1 day or a +2 day?").
    """

    def __init__(self, d_model: int, num_buckets: int = 15,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_buckets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → [B, num_buckets]"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class MambaOnlyVIX(nn.Module):
    """
    Mamba-only VIX prediction with dual heads.

    1s stock bars → Linear projection → Mamba (selective scan) → Pool
        → Regression head  → VIX daily change (continuous, VIX points)
        → Classification head → bucket logits (15 classes)

    The two heads share the same backbone and apply complementary gradient
    signals: regression pushes for precision, classification regularizes
    by enforcing decision boundaries.
    """

    def __init__(
        self,
        num_features: int = 15,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        pooling: str = 'attention',
        head_hidden: int = 128,
        num_buckets: int = 15,
    ):
        super().__init__()

        self.bar_proj = BarProjection(num_features, d_model, dropout)
        self.mamba = MambaStack(
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.pool = SequencePooling(d_model, pooling)
        self.head = VIXHead(d_model, head_hidden, dropout)
        self.bucket_head = VIXBucketHead(d_model, num_buckets, head_hidden, dropout)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MambaOnlyVIX: {num_params:,} parameters "
                     f"(d_model={d_model}, layers={n_layers}, d_state={d_state})")

    def forward(
        self,
        bars: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bars: [B, T, num_features] - Raw 1s bar features
            bar_mask: [B, T] - Optional valid bar mask (1=valid, 0=pad)

        Returns:
            dict with:
                'vix_pred': [B] - predicted VIX daily change (VIX points)
                'bucket_logits': [B, num_buckets] - classification logits
        """
        # Project bars to model dim
        x = self.bar_proj(bars)  # [B, T, d_model]

        # Zero out padded positions if mask provided
        if bar_mask is not None:
            x = x * bar_mask.unsqueeze(-1)

        # Mamba selective scan
        h = self.mamba(x)  # [B, T, d_model]

        # Pool sequence
        h_pool = self.pool(h)  # [B, d_model]

        # Dual heads from shared representation
        vix_pred = self.head(h_pool)           # [B] regression
        bucket_logits = self.bucket_head(h_pool)  # [B, num_buckets] classification

        return {
            'vix_pred': vix_pred,
            'bucket_logits': bucket_logits,
        }
