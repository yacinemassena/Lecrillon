"""
Parallel Mamba VIX Prediction Model.

Architecture with parallel streams and periodic fusion:

    Stock Bars [B, T, F]   → StreamEncoder → StreamMamba → [B, T, d_model]  (F = num_features, auto-detected)
    Option Bars [B, T, 48] → StreamEncoder → StreamMamba → [B, T, d_model]  (optional)
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
from loader.bar_mamba_dataset import NUM_OPTION_FEATURES

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
        fundamentals_state: Optional[torch.Tensor] = None,
        vix_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse stock state with news, options, fundamentals, and VIX states via cross-attention + gating.
        
        Args:
            stock_state: [B, d_model] - stock Mamba final state
            news_state: [B, d_model] - news Mamba final state (optional)
            options_state: [B, d_model] - options Mamba final state (optional)
            fundamentals_state: [B, d_model] - fundamentals projected state (optional)
            vix_state: [B, d_model] - VIX Mamba final state (optional, extended hours)
        
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
        if fundamentals_state is not None:
            aux_states.append(fundamentals_state)
        if vix_state is not None:
            aux_states.append(vix_state)
        
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


# ---------------------------------------------------------------------------
# Economic Calendar Encoder
# ---------------------------------------------------------------------------
class EconEncoder(nn.Module):
    """Encode economic calendar events into d_model representations.
    
    Each event token = event_embedding(32) + currency_embedding(8) + numeric(13) = 53 dims
    → MLP → d_model
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_event_types: int = 413,  # 412 + 1 for padding idx 0
        num_currencies: int = 5,     # 4 + 1 for padding idx 0
        event_embed_dim: int = 32,
        currency_embed_dim: int = 8,
        num_numeric_features: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.event_embedding = nn.Embedding(num_event_types, event_embed_dim, padding_idx=0)
        self.currency_embedding = nn.Embedding(num_currencies, currency_embed_dim, padding_idx=0)
        
        input_dim = event_embed_dim + currency_embed_dim + num_numeric_features  # 53
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        
        # Initialize embeddings small so they don't dominate at start
        nn.init.normal_(self.event_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.currency_embedding.weight, mean=0.0, std=0.02)
        # Keep padding as zero
        with torch.no_grad():
            self.event_embedding.weight[0].zero_()
            self.currency_embedding.weight[0].zero_()
        
        logger.info(f"EconEncoder: {num_event_types-1} event types, {num_currencies-1} currencies, "
                    f"{input_dim} → {d_model}")
    
    def forward(
        self,
        event_ids: torch.Tensor,      # [B, N] int
        currency_ids: torch.Tensor,    # [B, N] int
        numeric_features: torch.Tensor, # [B, N, 13] float
    ) -> torch.Tensor:
        """Encode econ events → [B, N, d_model]."""
        event_emb = self.event_embedding(event_ids)      # [B, N, 32]
        currency_emb = self.currency_embedding(currency_ids)  # [B, N, 8]
        combined = torch.cat([event_emb, currency_emb, numeric_features], dim=-1)  # [B, N, 53]
        return self.mlp(combined)  # [B, N, d_model]



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
        num_features: int,  # Stock features - auto-detected from dataset
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
        use_econ: bool = False,
        econ_num_event_types: int = 413,
        econ_num_currencies: int = 5,
        use_fundamentals: bool = False,
        fundamentals_dim: int = 130,
        use_vix_features: bool = False,
        vix_features_dim: int = 25,  # 25 VIX features (OHLC, VVIX, MAs, RV, technicals)
        vix_n_layers: int = 2,  # Lightweight VIX Mamba
        vix_d_model: int = 64,  # Smaller d_model for VIX (25 features vs stock's 50)
        vix_d_state: int = 16,  # Smaller state for VIX
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.checkpoint_interval = checkpoint_interval
        self.use_news = use_news
        self.use_options = use_options
        self.use_macro = use_macro
        self.use_gdelt = use_gdelt
        self.use_econ = use_econ
        self.use_fundamentals = use_fundamentals
        self.use_vix_features = use_vix_features
        self.vix_d_model = vix_d_model
        self.gdelt_dim = gdelt_dim
        
        # Fundamentals projection (cross-attention state)
        if use_fundamentals:
            self.fundamentals_proj = nn.Linear(fundamentals_dim, d_model)
        
        # VIX stream (optional) - extended hours, lightweight
        # ~540 bars/day (18h) vs stock's 195 bars (6.5h market hours)
        # Uses smaller d_model/d_state since only 21 features
        if use_vix_features:
            self.vix_encoder = StreamEncoder(vix_features_dim, vix_d_model, dropout)
            self.vix_mamba = StreamMamba(
                n_layers=vix_n_layers,
                d_model=vix_d_model,
                d_state=vix_d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            # Project VIX output to main d_model for fusion
            self.vix_proj = nn.Linear(vix_d_model, d_model)
            logger.info(f"VIX Mamba: {vix_n_layers} layers, d={vix_d_model}, state={vix_d_state} (extended hours)")
        
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
        if use_news or use_gdelt or use_econ:
            # Token type embedding: 0=GDELT, 1=Benzinga, 2=Econ calendar
            self.news_type_embedding = nn.Embedding(3, d_model)
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
        
        # Econ calendar encoder (optional) - feeds into news Mamba
        if use_econ:
            self.econ_encoder = EconEncoder(
                d_model=d_model,
                num_event_types=econ_num_event_types,
                num_currencies=econ_num_currencies,
                dropout=dropout,
            )
            logger.info(f"Econ calendar enabled: {econ_num_event_types-1} events → {d_model} (into news Mamba)")
        
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
        # Note: GDELT and Econ feed into news Mamba, not separate streams
        news_parts = [x for x in ['news' if use_news else None, 'gdelt' if use_gdelt else None, 'econ' if use_econ else None] if x]
        streams = ['stock'] + (['+'.join(news_parts)] if news_parts else []) + (['options'] if use_options else []) + (['vix_ext'] if use_vix_features else [])
        conditioning = (['macro_film'] if use_macro else []) + (['fundamentals'] if use_fundamentals else [])
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
        news_timestamps: Optional[torch.Tensor] = None,
        gdelt_embs: Optional[torch.Tensor] = None,
        gdelt_mask: Optional[torch.Tensor] = None,
        gdelt_timestamps: Optional[torch.Tensor] = None,
        macro_context: Optional[torch.Tensor] = None,
        bar_timestamps: Optional[torch.Tensor] = None,
        econ_event_ids: Optional[torch.Tensor] = None,
        econ_currency_ids: Optional[torch.Tensor] = None,
        econ_numeric: Optional[torch.Tensor] = None,
        econ_mask: Optional[torch.Tensor] = None,
        econ_timestamps: Optional[torch.Tensor] = None,
        fundamentals_context: Optional[torch.Tensor] = None,
        vix_features: Optional[torch.Tensor] = None,
        vix_mask: Optional[torch.Tensor] = None,
        vix_timestamps: Optional[torch.Tensor] = None,
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
            news_timestamps: [B, N1] - Unix timestamps for news articles (seconds)
            gdelt_embs: [B, N2, gdelt_dim] - GDELT world state embeddings (391-dim)
            gdelt_mask: [B, N2] - GDELT validity mask
            gdelt_timestamps: [B, N2] - Unix timestamps for GDELT buckets (seconds)
            macro_context: [B, macro_dim] - macro conditioning features (T-1)
            bar_timestamps: [B, T] - Unix timestamps for bars (seconds, for alignment)
            econ_event_ids: [B, N3] - econ calendar event type IDs
            econ_currency_ids: [B, N3] - econ calendar currency IDs
            econ_numeric: [B, N3, 13] - econ calendar numeric features
            econ_mask: [B, N3] - econ calendar validity mask
            econ_timestamps: [B, N3] - Unix timestamps for econ events (seconds)
            fundamentals_context: [B, fundamentals_dim] - sector fundamentals state
            vix_features: [B, V, vix_dim] - VIX features (extended hours, ~540 bars/day)
            vix_mask: [B, V] - VIX validity mask
            vix_timestamps: [B, V] - Unix timestamps for VIX bars (seconds)
        
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
        
        # Encode econ calendar if present
        econ_encoded = None
        econ_ts = None
        if self.use_econ and econ_event_ids is not None and econ_event_ids.shape[1] > 0:
            econ_encoded = self.econ_encoder(econ_event_ids, econ_currency_ids, econ_numeric)  # [B, N3, d_model]
            # Add econ type embedding (type=2)
            type_emb = self.news_type_embedding(torch.full(econ_encoded.shape[:2], 2, dtype=torch.long, device=device))
            econ_encoded = econ_encoded + type_emb
            econ_ts = econ_timestamps  # [B, N3]
        
        # Merge Benzinga, GDELT, and Econ into combined news sequence, sorted by timestamp
        news_encoded = None
        combined_news_ts = None
        combined_news_mask = None
        
        # First merge Benzinga + GDELT
        merged_encoded = None
        merged_ts = None
        merged_mask = None
        if benzinga_encoded is not None or gdelt_encoded is not None:
            merged_encoded, merged_ts, merged_mask = self._merge_news_sources(
                benzinga_encoded, benzinga_ts, news_mask if self.use_news else None,
                gdelt_encoded, gdelt_ts, gdelt_mask if self.use_gdelt else None,
                device,
            )
        
        # Then merge (Benzinga+GDELT) + Econ
        if merged_encoded is not None or econ_encoded is not None:
            news_encoded, combined_news_ts, combined_news_mask = self._merge_news_sources(
                merged_encoded, merged_ts, merged_mask,
                econ_encoded, econ_ts, econ_mask if self.use_econ else None,
                device,
            )
        
        options_encoded = None
        if self.use_options and options is not None:
            options_encoded = self.options_encoder(options)  # [B, T, d_model]
        
        # Encode VIX features if present (extended hours sequence)
        vix_encoded = None
        if self.use_vix_features and vix_features is not None and vix_features.shape[1] > 0:
            vix_encoded = self.vix_encoder(vix_features)  # [B, V, d_model]
        
        # Process in checkpoint segments
        num_checkpoints = (T + self.checkpoint_interval - 1) // self.checkpoint_interval
        
        # Accumulators for processed segments
        stock_outputs = []
        news_outputs = []
        options_outputs = []
        vix_outputs = []
        
        # Running state for stock (carries fused context forward)
        stock_running_state = None
        
        # Pre-compute fundamentals projection once (constant across checkpoints)
        fund_state = None
        if self.use_fundamentals and fundamentals_context is not None:
            fund_state = self.fundamentals_proj(fundamentals_context)  # [B, d_model]
        
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
            
            # --- VIX segment (extended hours, timestamp-filtered) ---
            vix_state = None
            if vix_encoded is not None and vix_timestamps is not None and bar_timestamps is not None:
                # Find VIX bars in this checkpoint window by timestamp
                segment_bar_ts = bar_timestamps[:, start_idx:end_idx]  # [B, segment_len]
                segment_start_ts = segment_bar_ts[:, 0:1]  # [B, 1]
                segment_end_ts = segment_bar_ts[:, -1:]    # [B, 1]
                
                # For first checkpoint: include ALL pre-market VIX (overnight accumulation)
                if cp == 0:
                    # Use earliest bar timestamp as end bound, include everything before
                    vix_in_window = vix_timestamps <= segment_end_ts
                else:
                    # VIX is in window if timestamp falls within segment bar timestamps
                    vix_in_window = (vix_timestamps >= segment_start_ts) & (vix_timestamps <= segment_end_ts)
                
                if vix_mask is not None:
                    vix_in_window = vix_in_window & (vix_mask > 0)
                
                # Process VIX for each batch item
                vix_states_batch = []
                for b in range(B):
                    valid_vix = vix_in_window[b]
                    if valid_vix.sum() > 0:
                        vix_segment = vix_encoded[b, valid_vix, :].unsqueeze(0)  # [1, n, vix_d_model]
                        vix_out = self.vix_mamba(vix_segment)
                        vix_final = self.vix_mamba.get_final_state(vix_out).squeeze(0)  # [vix_d_model]
                        vix_states_batch.append(self.vix_proj(vix_final))  # Project to d_model
                    else:
                        # No VIX in this window - use zero state
                        vix_states_batch.append(torch.zeros(self.d_model, device=device))
                
                vix_state = torch.stack(vix_states_batch, dim=0)  # [B, d_model]
                vix_outputs.append(vix_state)
            
            # --- Fusion at checkpoint ---
            stock_running_state = self.fusion_gate(
                stock_state,
                news_state=news_state,
                options_state=options_state,
                fundamentals_state=fund_state,
                vix_state=vix_state,
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
        options_pooled = None
        if self.use_options and options_outputs:
            options_all = torch.cat(options_outputs, dim=1)  # [B, T, d_model]
            options_pooled = self.pool(options_all)
            results['options_pred'] = self.options_aux_head(options_pooled)
        
        # VIX pooled state (if enabled)
        vix_pooled = None
        if self.use_vix_features and vix_outputs:
            vix_all = torch.stack(vix_outputs, dim=1)  # [B, num_checkpoints, d_model]
            vix_pooled = vix_all.mean(dim=1)  # Mean over checkpoints
        
        # Final fused prediction
        # Fuse final states from all streams
        final_news_state = news_pooled if ((self.use_news or self.use_gdelt) and news_outputs) else None
        final_options_state = options_pooled if (self.use_options and options_outputs) else None
        final_vix_state = vix_pooled if (self.use_vix_features and vix_outputs) else None
        
        final_fused = self.fusion_gate(
            stock_pooled, 
            final_news_state, 
            final_options_state, 
            fund_state,
            final_vix_state,
        )
        results['vix_pred'] = self.final_head(final_fused)
        
        return results
