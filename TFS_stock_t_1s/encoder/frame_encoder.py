"""
Frame Encoder for Tick-to-Vector compression.
Converts raw tick sequences (15s windows) into fixed-size latent vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TemporalConvBlock(nn.Module):
    """Single TCN block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        # Use GroupNorm(1, C) which is equivalent to LayerNorm over channels
        # and works directly on [B, C, L] without transposing
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x: [B, C, L]
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return F.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network for tick sequence encoding.
    """
    
    def __init__(self, in_features, hidden_dim, num_layers=12, kernel_size=3, dropout=0.1, checkpoint_every=0, num_tickers=0, ticker_embed_dim=16):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.checkpoint_every = checkpoint_every
        
        # Ticker Embedding
        if num_tickers > 0:
            self.ticker_embed = nn.Embedding(num_tickers, ticker_embed_dim)
            # Input features increase by embedding dimension
            self.use_ticker_emb = True
            input_channels = in_features + ticker_embed_dim
        else:
            self.use_ticker_emb = False
            input_channels = in_features
        
        layers = []
        channels = [input_channels] + [hidden_dim] * num_layers
        
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalConvBlock(
                    channels[i], channels[i + 1],
                    kernel_size, dilation, dropout
                )
            )
        
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = hidden_dim
        
    def forward(self, x, ticker_ids=None, use_checkpoint=False):
        # x: [B, L, F] where B is batch, L is sequence length, F is features
        
        # Handle Ticker Embedding
        if self.use_ticker_emb and ticker_ids is not None:
            # ticker_ids: [B]
            emb = self.ticker_embed(ticker_ids) # [B, E]
            emb = emb.unsqueeze(1) # [B, 1, E]
            # Repeat across sequence length
            emb = emb.expand(-1, x.size(1), -1) # [B, L, E]
            
            # Concatenate: [B, L, F] + [B, L, E] -> [B, L, F+E]
            x = torch.cat([x, emb], dim=-1)
            
        # Convert to [B, Channels, L] for conv1d
        x = x.transpose(1, 2)
        
        # Apply TCN
        for i, layer in enumerate(self.layers):
            if use_checkpoint and self.training and self.checkpoint_every > 0 and (i % self.checkpoint_every == 0):
                try:
                    x = checkpoint(layer, x, use_reentrant=False)
                except TypeError:
                    x = checkpoint(layer, x)
            else:
                x = layer(x)
        
        # Global pooling
        x = self.pool(x)  # [B, hidden_dim, 1]
        x = x.squeeze(-1)  # [B, hidden_dim]
        
        return x


class FrameEncoder(nn.Module):
    """
    Unified interface for frame encoding.
    Supports TCN architecture.
    """
    
    def __init__(self, kind, in_features, hidden_dim=512, num_layers=12, dropout=0.1, checkpoint_every=0, num_tickers=0, ticker_embed_dim=16):
        super().__init__()
        self.kind = kind
        
        if kind == 'tcn':
            self.encoder = TCNEncoder(
                in_features, hidden_dim, num_layers, dropout=dropout, checkpoint_every=checkpoint_every,
                num_tickers=num_tickers, ticker_embed_dim=ticker_embed_dim
            )
        else:
            raise ValueError(f"Unknown encoder kind: {kind}")
        
        self.out_dim = hidden_dim
    
    def forward(self, ticks, ticker_ids=None, use_checkpoint=False):
        """
        Args:
            ticks: [B, L, F] where B=batch, L=ticks per frame, F=features
            ticker_ids: [B] optional ticker IDs
            use_checkpoint: bool - whether to use gradient checkpointing
        
        Returns:
            frame_vec: [B, hidden_dim]
        """
        return self.encoder(ticks, ticker_ids=ticker_ids, use_checkpoint=use_checkpoint)
