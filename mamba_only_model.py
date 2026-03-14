"""
Parallel Mamba VIX Prediction Model.

Architecture with parallel streams and periodic fusion:

    Stock Bars [B, T, 29]  → StreamEncoder → StreamMamba → [B, T, d_model]
    Option Bars [B, T, 38] → StreamEncoder → StreamMamba → [B, T, d_model]  (optional)
    News Embs [B, N, 3072] → StreamEncoder → StreamMamba → [B, N, d_model]  (optional)
                                    ↓
                           FusionGate (cross-attention + gating every checkpoint_interval)
                                    ↓
                           Pool → MultiHorizonVIXHead → [B, 4] VIX change at +1d/+7d/+15d/+30d
"""

import logging
import math
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option features (all 49 features from opt_trade_2min)
# ---------------------------------------------------------------------------
OPTION_FEATURES = [
    'call_volume', 'put_volume', 'call_trade_count', 'put_trade_count',
    'put_call_ratio_volume', 'put_call_ratio_count',
    'call_premium_total', 'put_premium_total',
    'near_volume', 'mid_volume', 'far_volume',
    'near_pc_ratio', 'far_pc_ratio', 'term_skew',
    'otm_put_volume', 'atm_volume', 'otm_call_volume',
    'skew_proxy', 'atm_concentration', 'deep_otm_put_volume',
    'total_large_trade_count', 'call_large_count', 'put_large_count',
    'net_large_flow', 'large_premium_total', 'sweep_intensity',
    'max_volume_surprise', 'avg_volume_surprise',
    'uoa_call_count', 'uoa_put_count',
    'total_volume', 'total_trade_count',
    'unique_contracts', 'unique_strikes', 'unique_expiries',
    'pc_ratio_vs_20d', 'call_volume_vs_20d', 'put_volume_vs_20d',
    # New 2-min features
    'net_premium_flow', 'premium_imbalance', 'large_trade_pct', 'put_large_pct',
    'uoa_total', 'uoa_put_bias', 'deep_otm_put_pct',
    'near_far_ratio', 'sweep_to_trade_ratio',
]
NUM_OPTION_FEATURES = len(OPTION_FEATURES)  # 47 (49 cols - underlying - bar_timestamp)

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

        # Initialize final layer weights near zero (daily VIX change ≈ 0)
        # Keep bias at PyTorch default to break symmetry between heads
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → [B]"""
        return self.net(x).squeeze(-1)


# Multi-horizon prediction targets
HORIZONS = [1, 7, 15, 30]  # +1d, +7d, +15d, +30d
NUM_HORIZONS = len(HORIZONS)


class MultiHorizonVIXHead(nn.Module):
    """Multi-horizon VIX prediction: shared trunk + per-horizon heads.
    
    Predicts VIX change at +1d, +7d, +15d, +30d horizons.
    Architecture: shared MLP trunk → 4 separate linear output heads.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1, 
                 num_horizons: int = NUM_HORIZONS):
        super().__init__()
        self.num_horizons = num_horizons
        
        # Shared trunk (learns common representation)
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Per-horizon output heads
        self.horizon_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_horizons)
        ])
        
        # Initialize output heads weights near zero (VIX change ≈ 0)
        # Keep bias at PyTorch default to break symmetry between heads
        with torch.no_grad():
            for head in self.horizon_heads:
                head.weight.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model] pooled representation
        Returns:
            [B, num_horizons] predictions for each horizon
        """
        h = self.trunk(x)  # [B, hidden_dim]
        
        # Stack predictions from each horizon head
        preds = torch.cat([head(h) for head in self.horizon_heads], dim=-1)  # [B, 4]
        return preds


# ---------------------------------------------------------------------------
# Parallel Mamba Streams Architecture
# ---------------------------------------------------------------------------

class StreamEncoder(nn.Module):
    """Encoder for a single data stream (no CLS token, just projection)."""
    
    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1,
                 normalize_input: bool = False):
        super().__init__()
        self.d_model = d_model
        self.normalize_input = normalize_input
        
        if normalize_input:
            self.input_norm = nn.LayerNorm(num_features)
        
        self.proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, num_features] → [B, T, d_model]"""
        if self.normalize_input:
            x = self.input_norm(x)
        return self.proj(x)


class StreamMamba(nn.Module):
    """Mamba stack for a single stream with state management for checkpoints."""
    
    def __init__(
        self,
        n_layers: int = 2,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
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
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        film_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Process sequence through Mamba layers with optional FiLM modulation.
        
        Args:
            x: [B, T, d_model] - input sequence
            mask: [B, T] - optional mask (1=valid, 0=pad)
            film_params: list of (gamma, beta) per layer, each [B, T, d_model]
        
        Returns:
            [B, T, d_model] - processed sequence
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        for i, (layer, norm, drop) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            residual = x
            x = layer(x)
            x = norm(x)
            # FiLM modulation: x = gamma * x + beta (time-varying)
            if film_params is not None and i < len(film_params):
                gamma, beta = film_params[i]
                T = x.shape[1]
                x = gamma[:, :T, :] * x + beta[:, :T, :]
            x = drop(x)
            x = x + residual
        
        return x
    
    def get_final_state(self, x: torch.Tensor) -> torch.Tensor:
        """Get final hidden state after processing sequence. [B, T, D] → [B, D]"""
        return x[:, -1, :]


class FiLMGenerator(nn.Module):
    """
    Time-aware FiLM (Feature-wise Linear Modulation) for macro conditioning.
    
    Combines a slow-moving macro vector with time-of-day sinusoidal encoding
    to produce per-position, per-layer (gamma, beta) modulation parameters.
    
    Architecture:
        macro_vec [B, macro_dim] → MLP → macro_hidden [B, d_model]
        time_enc [B, T, d_model]  (sinusoidal from bar timestamps)
        combined [B, T, d_model] = macro_hidden.unsqueeze(1) + time_enc
        per-layer MLP → (gamma [B, T, d_model], beta [B, T, d_model])
    
    Identity init: gamma=1.0, beta=0.0 so model starts as no-op.
    """
    
    def __init__(self, macro_dim: int, d_model: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Project macro vector to d_model
        self.macro_proj = nn.Sequential(
            nn.LayerNorm(macro_dim),
            nn.Linear(macro_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        
        # Per-layer FiLM heads: combined → (gamma, beta)
        self.film_heads = nn.ModuleList()
        for _ in range(n_layers):
            head = nn.Linear(d_model, d_model * 2)
            # Identity init: gamma=1.0, beta=0.0
            with torch.no_grad():
                head.weight.zero_()
                head.bias[:d_model].fill_(1.0)   # gamma init = 1.0
                head.bias[d_model:].fill_(0.0)    # beta init = 0.0
            self.film_heads.append(head)
        
        # For logging gamma/beta statistics
        self._last_gamma_stats = {}
        self._last_beta_stats = {}
    
    def _sinusoidal_time_encoding(self, timestamps: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        Generate sinusoidal encoding from Unix-second timestamps.
        
        Encodes time-of-day (0-86400 seconds) using sinusoidal functions
        at various frequencies to capture intraday temporal structure.
        
        Args:
            timestamps: [B, T] int64 Unix seconds
            d_model: encoding dimension
            
        Returns:
            [B, T, d_model] sinusoidal encoding
        """
        # Extract time-of-day in seconds (0 to 86400)
        time_of_day = (timestamps % 86400).float()  # [B, T]
        
        # Create frequency bands
        half_d = d_model // 2
        freqs = torch.exp(
            torch.arange(half_d, device=timestamps.device, dtype=torch.float32)
            * -(math.log(10000.0) / half_d)
        )
        
        # [B, T, half_d]
        angles = time_of_day.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        
        # [B, T, d_model]
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        
        return encoding
    
    def forward(
        self,
        macro_context: torch.Tensor,
        bar_timestamps: torch.Tensor,
        seq_len: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate per-layer, per-position FiLM parameters.
        
        Args:
            macro_context: [B, macro_dim] - macro features (T-1 shifted)
            bar_timestamps: [B, T] - Unix second timestamps for bars
            seq_len: maximum sequence length to generate for
            
        Returns:
            List of (gamma, beta) tuples per layer, each [B, T, d_model]
        """
        B = macro_context.shape[0]
        device = macro_context.device
        
        # Project macro to hidden: [B, d_model]
        macro_hidden = self.macro_proj(macro_context)
        
        # Time-of-day encoding: [B, T, d_model]
        T = min(bar_timestamps.shape[1], seq_len)
        time_enc = self._sinusoidal_time_encoding(bar_timestamps[:, :T], self.d_model)
        
        # Combine: broadcast macro over time + add time encoding
        # [B, T, d_model]
        combined = macro_hidden.unsqueeze(1) + time_enc
        
        # Generate per-layer (gamma, beta)
        film_params = []
        for i, head in enumerate(self.film_heads):
            gb = head(combined)  # [B, T, d_model*2]
            gamma = gb[:, :, :self.d_model]   # [B, T, d_model]
            beta = gb[:, :, self.d_model:]     # [B, T, d_model]
            film_params.append((gamma, beta))
            
            # Store stats for logging (detached)
            self._last_gamma_stats[i] = {
                'mean': gamma.detach().mean().item(),
                'std': gamma.detach().std().item(),
            }
            self._last_beta_stats[i] = {
                'mean': beta.detach().mean().item(),
                'std': beta.detach().std().item(),
            }
        
        return film_params
    
    def get_film_stats(self) -> Dict[str, float]:
        """Return gamma/beta statistics from last forward pass for logging."""
        stats = {}
        for i in range(self.n_layers):
            if i in self._last_gamma_stats:
                stats[f'film_gamma_L{i}_mean'] = self._last_gamma_stats[i]['mean']
                stats[f'film_gamma_L{i}_std'] = self._last_gamma_stats[i]['std']
                stats[f'film_beta_L{i}_mean'] = self._last_beta_stats[i]['mean']
                stats[f'film_beta_L{i}_std'] = self._last_beta_stats[i]['std']
        return stats


class FusionGate(nn.Module):
    """Learned gating mechanism to blend fused state with original state."""
    
    def __init__(self, d_model: int, num_sources: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cross-attention: stock attends to news+options
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Gating mechanism: learned blend between original and fused
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        stock_state: torch.Tensor,
        news_state: Optional[torch.Tensor] = None,
        options_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse stock state with news and options states via cross-attention + gating.
        
        Args:
            stock_state: [B, d_model] - stock Mamba final state
            news_state: [B, d_model] - news Mamba final state (optional)
            options_state: [B, d_model] - options Mamba final state (optional)
        
        Returns:
            [B, d_model] - gated fusion of all states
        """
        B = stock_state.shape[0]
        
        # Collect auxiliary states
        aux_states = []
        if news_state is not None:
            aux_states.append(news_state)
        if options_state is not None:
            aux_states.append(options_state)
        
        if not aux_states:
            return stock_state
        
        # Stack auxiliary states: [B, num_aux, d_model]
        aux = torch.stack(aux_states, dim=1)
        
        # Cross-attention: stock queries, aux provides keys/values
        q = self.q_proj(stock_state).unsqueeze(1)  # [B, 1, d_model]
        k = self.k_proj(aux)  # [B, num_aux, d_model]
        v = self.v_proj(aux)  # [B, num_aux, d_model]
        
        # Multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, self.d_model)
        out = self.out_proj(out)
        
        # Gating: blend original stock state with attended auxiliary info
        gate_input = torch.cat([stock_state, out], dim=-1)
        gate_weight = self.gate(gate_input)
        
        # Fused = gate * attended + (1-gate) * original
        fused = gate_weight * out + (1 - gate_weight) * stock_state
        fused = self.norm(fused)
        
        return fused


class ParallelMambaVIX(nn.Module):
    """
    Parallel Mamba streams with periodic fusion checkpoints.
    
    Architecture:
        Three parallel Mamba stacks (stock, news, options) process independently.
        Every `checkpoint_interval` bars (e.g., 300 = 5 minutes):
            - Pool each stream to get CLS tokens
            - Cross-attention fusion with learned gating
            - Stock state updated with fused context
        
        Final: Pool all streams → Fuse → VIXHead → prediction
    """
    
    def __init__(
        self,
        num_features: int = 45,  # Stock features from Stock_Data_2min (47 - ticker - bar_timestamp)
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        checkpoint_interval: int = 300,  # 5 minutes
        use_news: bool = False,
        news_dim: int = 3072,
        news_n_layers: int = 2,  # Fewer layers for sparse news sequences
        use_options: bool = False,
        option_features: int = NUM_OPTION_FEATURES,
        num_fusion_heads: int = 4,
        head_hidden: int = 128,
        use_macro: bool = False,
        macro_dim: int = 21,  # FED/FRED macro features from datasets/macro/macro_daily.parquet
        use_gdelt: bool = False,
        gdelt_dim: int = 391,  # 384 MiniLM embedding + 7 stats
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.checkpoint_interval = checkpoint_interval
        self.use_news = use_news
        self.use_options = use_options
        self.use_macro = use_macro
        self.use_gdelt = use_gdelt
        self.gdelt_dim = gdelt_dim
        
        # Stock stream (always active)
        self.stock_encoder = StreamEncoder(num_features, d_model, dropout)
        self.stock_mamba = StreamMamba(
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # News stream (optional) - fewer layers for sparse sequences
        if use_news or use_gdelt:
            # Token type embedding: 0=GDELT (world state), 1=Benzinga (article event)
            self.news_type_embedding = nn.Embedding(2, d_model)
            # Initialize small so it doesn't dominate at start
            nn.init.normal_(self.news_type_embedding.weight, mean=0.0, std=0.02)
            
            # News Mamba processes combined Benzinga + GDELT sequence (fewer layers)
            self.news_mamba = StreamMamba(
                n_layers=news_n_layers,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            logger.info(f"News Mamba: {news_n_layers} layers (vs {n_layers} for stock)")
        
        if use_news:
            self.news_encoder = StreamEncoder(news_dim, d_model, dropout, normalize_input=True)
        
        # GDELT encoder (optional) - separate normalization for embedding vs stats
        if use_gdelt:
            self.gdelt_embed_dim = 384
            self.gdelt_stats_dim = 7
            # Separate normalization for embedding and stats
            self.gdelt_embed_norm = nn.LayerNorm(self.gdelt_embed_dim)
            self.gdelt_stats_norm = nn.LayerNorm(self.gdelt_stats_dim)
            # Combined projection
            self.gdelt_encoder = nn.Sequential(
                nn.Linear(gdelt_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            logger.info(f"GDELT stream enabled: {gdelt_dim} → {d_model} (into news Mamba)")
        
        # Options stream (optional) - same architecture as stock
        if use_options:
            self.options_encoder = StreamEncoder(option_features, d_model, dropout)
            self.options_mamba = StreamMamba(
                n_layers=n_layers,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            logger.info(f"Options stream enabled: {option_features} → {d_model}, {n_layers} layers")
        
        # Macro FiLM conditioning (optional) - time-aware modulation
        if use_macro:
            self.film_generator = FiLMGenerator(
                macro_dim=macro_dim,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
            )
            logger.info(f"Macro FiLM enabled: {macro_dim} features, {n_layers} layers, time-aware")
        
        # Fusion gate for checkpoint blending
        self.fusion_gate = FusionGate(d_model, num_heads=num_fusion_heads, dropout=dropout)
        
        # Auxiliary prediction heads (for per-stream losses) - single horizon for aux
        self.stock_aux_head = VIXHead(d_model, head_hidden, dropout)
        if use_news:
            self.news_aux_head = VIXHead(d_model, head_hidden, dropout)
        if use_options:
            self.options_aux_head = VIXHead(d_model, head_hidden, dropout)
        
        # Final prediction head (from fused representation) - multi-horizon
        self.final_head = MultiHorizonVIXHead(d_model, head_hidden, dropout)
        
        # Pooling for final states
        self.pool = SequencePooling(d_model, 'attention')
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Note: GDELT feeds into news Mamba, not a separate stream
        streams = ['stock'] + (['news+gdelt'] if (use_news or use_gdelt) else []) + (['options'] if use_options else [])
        conditioning = ['macro_film'] if use_macro else []
        logger.info(f"ParallelMambaVIX: {num_params:,} parameters "
                    f"(d_model={d_model}, n_layers={n_layers}, checkpoint={checkpoint_interval}, "
                    f"streams={streams}, conditioning={conditioning})")
    
    def _merge_news_sources(
        self,
        benzinga_encoded: Optional[torch.Tensor],
        benzinga_ts: Optional[torch.Tensor],
        benzinga_mask: Optional[torch.Tensor],
        gdelt_encoded: Optional[torch.Tensor],
        gdelt_ts: Optional[torch.Tensor],
        gdelt_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge Benzinga and GDELT encoded tokens into a single sequence sorted by timestamp.
        
        Args:
            benzinga_encoded: [B, N1, d_model] - encoded Benzinga news (or None)
            benzinga_ts: [B, N1] - Benzinga timestamps (or None)
            benzinga_mask: [B, N1] - Benzinga validity mask (or None)
            gdelt_encoded: [B, N2, d_model] - encoded GDELT tokens (or None)
            gdelt_ts: [B, N2] - GDELT timestamps (or None)
            gdelt_mask: [B, N2] - GDELT validity mask (or None)
            device: torch device
        
        Returns:
            combined_encoded: [B, N1+N2, d_model] - merged and sorted by timestamp
            combined_ts: [B, N1+N2] - merged timestamps
            combined_mask: [B, N1+N2] - merged validity mask
        """
        # Handle case where only one source exists
        if benzinga_encoded is None and gdelt_encoded is None:
            return None, None, None
        
        if benzinga_encoded is None:
            return gdelt_encoded, gdelt_ts, gdelt_mask
        
        if gdelt_encoded is None:
            return benzinga_encoded, benzinga_ts, benzinga_mask
        
        B = benzinga_encoded.shape[0]
        N1 = benzinga_encoded.shape[1]
        N2 = gdelt_encoded.shape[1]
        N_total = N1 + N2
        
        # Concatenate along sequence dimension
        combined_encoded = torch.cat([benzinga_encoded, gdelt_encoded], dim=1)  # [B, N1+N2, d_model]
        combined_ts = torch.cat([benzinga_ts, gdelt_ts], dim=1)  # [B, N1+N2]
        
        # Build combined mask
        if benzinga_mask is not None and gdelt_mask is not None:
            combined_mask = torch.cat([benzinga_mask, gdelt_mask], dim=1)  # [B, N1+N2]
        elif benzinga_mask is not None:
            combined_mask = torch.cat([benzinga_mask, torch.ones(B, N2, device=device)], dim=1)
        elif gdelt_mask is not None:
            combined_mask = torch.cat([torch.ones(B, N1, device=device), gdelt_mask], dim=1)
        else:
            combined_mask = torch.ones(B, N_total, device=device)
        
        # Sort by timestamp for each batch item
        sorted_indices = torch.argsort(combined_ts, dim=1)  # [B, N_total]
        
        # Gather to reorder
        # Expand indices for gathering from [B, N, d_model]
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        combined_encoded = torch.gather(combined_encoded, 1, sorted_indices_expanded)
        combined_ts = torch.gather(combined_ts, 1, sorted_indices)
        combined_mask = torch.gather(combined_mask, 1, sorted_indices)
        
        return combined_encoded, combined_ts, combined_mask
    
    def forward(
        self,
        bars: torch.Tensor,
        bar_mask: Optional[torch.Tensor] = None,
        options: Optional[torch.Tensor] = None,
        options_mask: Optional[torch.Tensor] = None,
        news_embs: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
        news_indices: Optional[torch.Tensor] = None,
        news_timestamps: Optional[torch.Tensor] = None,
        gdelt_embs: Optional[torch.Tensor] = None,
        gdelt_mask: Optional[torch.Tensor] = None,
        gdelt_timestamps: Optional[torch.Tensor] = None,
        macro_context: Optional[torch.Tensor] = None,
        bar_timestamps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with parallel streams, periodic fusion, and optional FiLM.
        
        Args:
            bars: [B, T, num_features] - stock bar features
            bar_mask: [B, T] - stock validity mask
            options: [B, T, option_features] - options features (aligned with bars)
            options_mask: [B, T] - options validity mask
            news_embs: [B, N1, news_dim] - Benzinga news embeddings
            news_mask: [B, N1] - Benzinga news validity mask
            news_indices: [B, N1] - bar index for each news article
            news_timestamps: [B, N1] - Unix timestamps for news articles
            gdelt_embs: [B, N2, gdelt_dim] - GDELT world state embeddings (391-dim)
            gdelt_mask: [B, N2] - GDELT validity mask
            gdelt_timestamps: [B, N2] - Unix timestamps for GDELT buckets
            macro_context: [B, macro_dim] - macro conditioning features (T-1)
            bar_timestamps: [B, T] - Unix second timestamps for FiLM time encoding
        
        Returns:
            dict with:
                'vix_pred': [B, 4] - final VIX prediction at +1d, +7d, +15d, +30d
                'stock_pred': [B] - stock-only prediction (aux loss, +1d only)
                'news_pred': [B] - news-only prediction (aux loss, +1d only, if enabled)
                'options_pred': [B] - options-only prediction (aux loss, +1d only, if enabled)
        """
        B, T, _ = bars.shape
        device = bars.device
        
        # Generate FiLM parameters if macro conditioning is enabled
        film_params = None
        if self.use_macro and macro_context is not None and bar_timestamps is not None:
            film_params = self.film_generator(macro_context, bar_timestamps, T)
        
        # Encode all streams
        stock_encoded = self.stock_encoder(bars)  # [B, T, d_model]
        
        # Encode Benzinga news if present
        benzinga_encoded = None
        benzinga_ts = None
        if self.use_news and news_embs is not None and news_embs.shape[1] > 0:
            benzinga_encoded = self.news_encoder(news_embs)  # [B, N1, d_model]
            # Add Benzinga type embedding (type=1)
            type_emb = self.news_type_embedding(torch.ones(benzinga_encoded.shape[:2], dtype=torch.long, device=device))
            benzinga_encoded = benzinga_encoded + type_emb
            benzinga_ts = news_timestamps  # [B, N1]
        
        # Encode GDELT if present
        gdelt_encoded = None
        gdelt_ts = None
        if self.use_gdelt and gdelt_embs is not None and gdelt_embs.shape[1] > 0:
            # Separate normalize embedding and stats
            gdelt_embed = self.gdelt_embed_norm(gdelt_embs[..., :self.gdelt_embed_dim])
            gdelt_stats = self.gdelt_stats_norm(gdelt_embs[..., self.gdelt_embed_dim:])
            gdelt_combined = torch.cat([gdelt_embed, gdelt_stats], dim=-1)
            gdelt_encoded = self.gdelt_encoder(gdelt_combined)  # [B, N2, d_model]
            # Add GDELT type embedding (type=0)
            type_emb = self.news_type_embedding(torch.zeros(gdelt_encoded.shape[:2], dtype=torch.long, device=device))
            gdelt_encoded = gdelt_encoded + type_emb
            gdelt_ts = gdelt_timestamps  # [B, N2]
        
        # Merge Benzinga and GDELT into combined news sequence, sorted by timestamp
        news_encoded = None
        combined_news_ts = None
        combined_news_mask = None
        if benzinga_encoded is not None or gdelt_encoded is not None:
            news_encoded, combined_news_ts, combined_news_mask = self._merge_news_sources(
                benzinga_encoded, benzinga_ts, news_mask if self.use_news else None,
                gdelt_encoded, gdelt_ts, gdelt_mask if self.use_gdelt else None,
                device,
            )
        
        options_encoded = None
        if self.use_options and options is not None:
            options_encoded = self.options_encoder(options)  # [B, T, d_model]
        
        # Process in checkpoint segments
        num_checkpoints = (T + self.checkpoint_interval - 1) // self.checkpoint_interval
        
        # Accumulators for processed segments
        stock_outputs = []
        news_outputs = []
        options_outputs = []
        
        # Running state for stock (carries fused context forward)
        stock_running_state = None
        
        for cp in range(num_checkpoints):
            start_idx = cp * self.checkpoint_interval
            end_idx = min((cp + 1) * self.checkpoint_interval, T)
            
            # Slice FiLM params for this segment
            segment_film = None
            if film_params is not None:
                segment_film = [
                    (g[:, start_idx:end_idx, :], b[:, start_idx:end_idx, :])
                    for g, b in film_params
                ]
            
            # --- Stock segment ---
            stock_segment = stock_encoded[:, start_idx:end_idx, :]
            if bar_mask is not None:
                stock_mask_segment = bar_mask[:, start_idx:end_idx]
            else:
                stock_mask_segment = None
            
            # If we have running state from previous checkpoint, prepend it
            if stock_running_state is not None:
                # Modulate first position with fused context
                stock_segment = stock_segment.clone()
                stock_segment[:, 0, :] = stock_segment[:, 0, :] + stock_running_state
            
            stock_out = self.stock_mamba(stock_segment, stock_mask_segment, segment_film)
            stock_outputs.append(stock_out)
            stock_state = self.stock_mamba.get_final_state(stock_out)
            
            # --- News segment (combined Benzinga + GDELT in this checkpoint window) ---
            news_state = None
            if news_encoded is not None and combined_news_ts is not None and bar_timestamps is not None:
                # Find news tokens in this checkpoint window by timestamp
                # Get bar timestamps for this segment to determine window boundaries
                segment_bar_ts = bar_timestamps[:, start_idx:end_idx]  # [B, segment_len]
                segment_start_ts = segment_bar_ts[:, 0:1]  # [B, 1]
                segment_end_ts = segment_bar_ts[:, -1:]    # [B, 1]
                
                # News is in window if its timestamp falls within segment bar timestamps
                news_in_window = (combined_news_ts >= segment_start_ts) & (combined_news_ts <= segment_end_ts)
                if combined_news_mask is not None:
                    news_in_window = news_in_window & (combined_news_mask > 0)
                
                # Process news for each batch item
                # Note: news doesn't get FiLM (it's not time-aligned like bars)
                news_states_batch = []
                for b in range(B):
                    valid_news = news_in_window[b]
                    if valid_news.sum() > 0:
                        news_segment = news_encoded[b, valid_news, :].unsqueeze(0)  # [1, n, d]
                        news_out = self.news_mamba(news_segment)
                        news_states_batch.append(self.news_mamba.get_final_state(news_out).squeeze(0))
                    else:
                        # No news in this window - use zero state
                        news_states_batch.append(torch.zeros(self.d_model, device=device))
                
                news_state = torch.stack(news_states_batch, dim=0)  # [B, d_model]
                news_outputs.append(news_state)
            
            # --- Options segment ---
            options_state = None
            if options_encoded is not None:
                options_segment = options_encoded[:, start_idx:end_idx, :]
                if options_mask is not None:
                    options_mask_segment = options_mask[:, start_idx:end_idx]
                else:
                    options_mask_segment = None
                
                # Options get same FiLM as stock (time-aligned)
                options_out = self.options_mamba(options_segment, options_mask_segment, segment_film)
                options_outputs.append(options_out)
                options_state = self.options_mamba.get_final_state(options_out)
            
            # --- Fusion at checkpoint ---
            stock_running_state = self.fusion_gate(
                stock_state,
                news_state=news_state,
                options_state=options_state,
            )
        
        # --- Final predictions ---
        # Concatenate all outputs and pool
        stock_all = torch.cat(stock_outputs, dim=1)  # [B, T, d_model]
        stock_pooled = self.pool(stock_all)
        
        results = {}
        
        # Stock auxiliary prediction
        results['stock_pred'] = self.stock_aux_head(stock_pooled)
        
        # News auxiliary prediction (if news or gdelt enabled)
        news_pooled = None
        if (self.use_news or self.use_gdelt) and news_outputs:
            news_all = torch.stack(news_outputs, dim=1)  # [B, num_checkpoints, d_model]
            news_pooled = news_all.mean(dim=1)  # Simple mean over checkpoints
            if self.use_news:
                results['news_pred'] = self.news_aux_head(news_pooled)
        
        # Options auxiliary prediction (if enabled)
        if self.use_options and options_outputs:
            options_all = torch.cat(options_outputs, dim=1)  # [B, T, d_model]
            options_pooled = self.pool(options_all)
            results['options_pred'] = self.options_aux_head(options_pooled)
        
        # Final fused prediction
        # Fuse final states from all streams
        final_news_state = news_pooled if ((self.use_news or self.use_gdelt) and news_outputs) else None
        final_options_state = options_pooled if (self.use_options and options_outputs) else None
        
        final_fused = self.fusion_gate(stock_pooled, final_news_state, final_options_state)
        results['vix_pred'] = self.final_head(final_fused)
        
        return results
