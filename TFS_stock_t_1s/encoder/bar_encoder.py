"""
Bar Encoder for 1-Second Stock Data.

Unlike tick data which uses TCN for chunked sequences, bar data uses:
- Transformer encoder for bar sequences within each frame
- Mean pooling across bars to get frame embedding
- Ticker embeddings added to bar features

Architecture:
    Bars [N_bars, num_features] → Linear → Transformer → Pool → Frame Embedding [D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BarEncoder(nn.Module):
    """
    Encodes a sequence of bars within a frame using Transformer.
    
    Args:
        num_features: Number of input bar features (e.g., 15)
        hidden_dim: Model dimension (e.g., 256)
        num_layers: Number of transformer layers (e.g., 4)
        num_heads: Number of attention heads (e.g., 4)
        dropout: Dropout rate
        num_tickers: Number of tickers for embedding (0 = disabled)
        ticker_embed_dim: Dimension of ticker embeddings
    """
    
    def __init__(
        self,
        num_features: int = 15,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_tickers: int = 0,
        ticker_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_tickers = num_tickers
        self.ticker_embed_dim = ticker_embed_dim
        
        # Input projection - always based on num_features only
        # Ticker embeddings are added separately if provided
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        if num_tickers > 0:
            self.ticker_embedding = nn.Embedding(num_tickers, ticker_embed_dim)
            self.ticker_proj = nn.Linear(ticker_embed_dim, hidden_dim)
        else:
            self.ticker_embedding = None
            self.ticker_proj = None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.out_dim = hidden_dim
    
    def forward(
        self,
        bars: torch.Tensor,
        mask: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode bars within frames.
        
        Args:
            bars: [B, N_bars, num_features] - Bar features
            mask: [B, N_bars] - Valid bar mask (1 = valid, 0 = padding)
            ticker_ids: [B, N_bars] - Ticker IDs (optional)
        
        Returns:
            frame_embeddings: [B, hidden_dim] - One embedding per frame
        """
        B, N, F = bars.shape
        
        # Project to hidden dim
        x = self.input_proj(bars)  # [B, N, hidden_dim]
        
        # Add ticker embeddings if available
        if self.ticker_embedding is not None and self.ticker_proj is not None and ticker_ids is not None:
            ticker_emb = self.ticker_embedding(ticker_ids)  # [B, N, ticker_embed_dim]
            ticker_proj = self.ticker_proj(ticker_emb)  # [B, N, hidden_dim]
            x = x + ticker_proj  # Add to hidden representation
        
        # Create attention mask (True = masked/ignored)
        attn_mask = (mask == 0)  # [B, N]
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, N, hidden_dim]
        
        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        x_masked = x * mask_expanded
        x_sum = x_masked.sum(dim=1)  # [B, hidden_dim]
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        x_pooled = x_sum / mask_sum  # [B, hidden_dim]
        
        # Output projection
        out = self.out_proj(x_pooled)  # [B, hidden_dim]
        
        return out


class BarFrameEncoder(nn.Module):
    """
    Wraps BarEncoder to process batches of frames and output frame embeddings.
    Similar to ChunkedFrameEncoder but for bar data.
    """
    
    def __init__(
        self,
        bar_encoder: BarEncoder,
        d_model: int = 256,
    ):
        super().__init__()
        self.bar_encoder = bar_encoder
        self.d_model = d_model
        
        # Project to output dimension if needed
        if bar_encoder.out_dim != d_model:
            self.out_proj = nn.Linear(bar_encoder.out_dim, d_model)
        else:
            self.out_proj = nn.Identity()
    
    def forward(
        self,
        bars: torch.Tensor,
        mask: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode all frames in batch.
        
        Args:
            bars: [N_frames, N_bars, num_features]
            mask: [N_frames, N_bars]
            ticker_ids: [N_frames, N_bars] (optional)
        
        Returns:
            frame_embeddings: [N_frames, d_model]
        """
        frame_emb = self.bar_encoder(bars, mask, ticker_ids)
        return self.out_proj(frame_emb)
