"""
VIX Mamba Model for tick-based financial prediction.
Processes raw tick data through chunked frame encoding and Mamba layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.checkpoint import checkpoint

from encoder.frame_encoder import TCNEncoder
from encoder.chunked_encoder import ChunkedFrameEncoder

logger = logging.getLogger(__name__)


class SequencePooling(nn.Module):
    """Pooling strategies for sequence embeddings."""
    def __init__(self, d_model, pooling_type='last'):
        super().__init__()
        self.pooling_type = pooling_type.lower()
        if self.pooling_type == 'attention':
            self.attn = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [B, T, D]
        if self.pooling_type == 'last':
            return x[:, -1, :]
        elif self.pooling_type == 'mean':
            return x.mean(dim=1)
        elif self.pooling_type == 'max':
            return x.max(dim=1)[0]
        elif self.pooling_type == 'attention':
            # scores: [B, T, 1]
            scores = self.attn(x)
            weights = torch.softmax(scores, dim=1)
            return (weights * x).sum(dim=1)
        else:
            return x[:, -1, :]


class PredictionHead(nn.Module):
    """Configurable MLP head for final prediction."""
    
    def __init__(self, d_model, output_dim, config):
        super().__init__()
        
        # Use dataclass config model attributes
        m = config.model
        layers_count = m.head_layers
        dropout_rate = m.head_dropout
        act_name = m.head_activation.lower()
        
        # Map activation string to layer
        if act_name == 'swish' or act_name == 'silu':
            act_fn = nn.SiLU
        elif act_name == 'relu':
            act_fn = nn.ReLU
        else:
            act_fn = nn.GELU
            
        layers = []
        curr_dim = d_model
        
        # Hidden layers
        for _ in range(layers_count - 1):
            layers.append(nn.Linear(curr_dim, curr_dim))
            if m.head_layer_norm:
                layers.append(nn.LayerNorm(curr_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(curr_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class VIXMambaModel(nn.Module):
    """
    VIX Prediction model with Chunk-Encode-Pool strategy.
    
    Architecture:
        Chunks [SumK, L, F] -> Encode -> Pool -> [T, D]
        Scalars [T, S] ------------------------^
        Concat -> Project -> Mamba -> Prediction
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Get model config dataclass
        m = config.model
        t = config.train
        
        # Shortcuts for sub-configs
        tcn_cfg = m.tcn
        mamba_cfg = m.mamba_short # Use short-term mamba config for this model
        rv_cfg = m.rv
        ticker_cfg = m.tickers
        
        # Checkpointing config
        self.ckpt_mode = t.checkpoint_mode
        self.ckpt_every_n = t.checkpoint_every_n_layers
        self.ckpt_encoder = t.checkpoint_encoder
        
        # Log policy selection
        if self.ckpt_mode != 'off':
            logger.info(f"Checkpointing: mode={self.ckpt_mode}, n={self.ckpt_every_n}")
        
        # Resolve TCN checkpointing
        # If encoder checkpointing is requested, use every_n, otherwise 0 (disabled)
        tcn_ckpt = self.ckpt_every_n if t.checkpoint_encoder else 0
        
        dim_in = m.dim_in # This should be 3 now (price, size, dt)
        dim_out = m.dim_out
        
        # Use d_model from mamba config
        d_model = mamba_cfg.d_model

        # Frame encoder configuration
        encoder_hidden = tcn_cfg.dim_out # Was frame_encoder_dim
        encoder_layers = tcn_cfg.layers  # Was frame_encoder_layers
        
        # Base Frame Encoder (TCN)
        self.base_encoder = TCNEncoder(
            in_features=dim_in,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=tcn_cfg.dropout,
            checkpoint_every=tcn_ckpt,
            num_tickers=len(ticker_cfg.tickers) if tcn_cfg.use_ticker_emb else 0,
            ticker_embed_dim=tcn_cfg.ticker_embed_dim
        )
        
        # Chunked Wrapper (handles pooling & scalars)
        self.chunked_encoder = ChunkedFrameEncoder(
            frame_encoder=self.base_encoder,
            d_model=d_model,
            num_scalars=3,  # n_ticks, notional, vol
            stream_chunks=m.encoder_streaming,
            stream_chunk_size=m.encoder_stream_chunk_size
        )
        
        # RV Prediction Head
        self.predict_rv = rv_cfg.predict_rv
        if self.predict_rv:
            # Simple MLP for RV
            self.rv_head = nn.Sequential(
                nn.Linear(d_model, rv_cfg.head_hidden_dim),
                nn.ReLU(),
                nn.Linear(rv_cfg.head_hidden_dim, 1)
            )
        
        # Mamba layers
        try:
            from mamba_ssm import Mamba
            
            self.mamba_layers = nn.ModuleList()
            self.mamba_norms = nn.ModuleList()
            self.mamba_dropouts = nn.ModuleList()
            
            for _ in range(mamba_cfg.n_layers):
                self.mamba_layers.append(
                    Mamba(
                        d_model=d_model,
                        d_state=mamba_cfg.d_state,
                        d_conv=mamba_cfg.d_conv,
                        expand=mamba_cfg.expand,
                    )
                )
                self.mamba_norms.append(nn.LayerNorm(d_model))
                if mamba_cfg.dropout > 0:
                    self.mamba_dropouts.append(nn.Dropout(mamba_cfg.dropout))
                else:
                    self.mamba_dropouts.append(nn.Identity())
            
        except ImportError:
            raise ImportError("mamba_ssm not installed")
        
        # Pooling strategy
        pooling_type = m.pooling_type
        self.pooling = SequencePooling(d_model, pooling_type)
        
        # Prediction head
        self.head = PredictionHead(d_model=d_model, output_dim=dim_out, config=config)
        
    def _mamba_block(self, x, layer_idx):
        """Single Mamba block - separable for checkpointing."""
        h = self.mamba_layers[layer_idx](x)
        h = self.mamba_norms[layer_idx](h)
        h = self.mamba_dropouts[layer_idx](h)
        return h
        
    def forward(self, batch):
        """
        Args:
            batch: dict containing:
                - 'chunks': [SumK, chunk_len, F]
                - 'frame_id': [SumK]
                - 'weights': [SumK]
                - 'frame_scalars': [TotalFrames, S]
                - 'ticker_ids': [TotalFrames] (Optional)
                - 'num_frames': int
                - 'target': [B, 1]
        
        Returns:
            dict with 'logits' and optionally 'rv_pred'
        """
        # 1. Encode & Pool Chunks -> [TotalFrames, d_model]
        # Pass checkpoint flag to encoder if enabled
        use_encoder_ckpt = (self.ckpt_encoder and self.training and self.ckpt_mode != 'off')
        
        x = self.chunked_encoder(
            chunks=batch['chunks'],
            frame_id=batch['frame_id'],
            weights=batch['weights'],
            frame_scalars=batch['frame_scalars'],
            num_frames=batch['num_frames'],
            ticker_ids=batch.get('ticker_ids', None),
            use_checkpoint=use_encoder_ckpt
        ) # [TotalFrames, d_model]
        
        outputs = {}
        
        # 2. RV Prediction (Auxiliary Task)
        # Predict RV from the per-frame embedding BEFORE sequence modeling
        if self.predict_rv:
            rv_pred = self.rv_head(x) # [TotalFrames, 1]
            outputs['rv_pred'] = rv_pred.squeeze(-1) # [TotalFrames]
        
        # 3. Reshape to [B, T, d_model]
        # We assume fixed T per sequence
        B = len(batch['target'])
        if B == 0: 
            # Handle empty batch edge case
            outputs['logits'] = torch.zeros(0, 1, device=x.device)
            return outputs
        
        # Safety assert: num_frames must be divisible by batch size
        assert batch['num_frames'] % B == 0, \
            f"num_frames={batch['num_frames']} not divisible by batch_size={B}"
        T = batch['num_frames'] // B
        
        x = x.view(B, T, -1)
        
        # 4. Mamba Sequence Modeling
        # h = self.price_mamba(x)  # [B, T, d_model]
        h = x
        for i in range(len(self.mamba_layers)):
            should_checkpoint = False
            if self.training and self.ckpt_mode != 'off':
                if self.ckpt_mode == 'all':
                    should_checkpoint = True
                elif self.ckpt_mode == 'every_n':
                    if (i % self.ckpt_every_n == 0):
                        should_checkpoint = True
            
            if should_checkpoint:
                try:
                    h = checkpoint(self._mamba_block, h, i, use_reentrant=False)
                except TypeError:
                    # Fallback for older PyTorch versions that don't support use_reentrant
                    h = checkpoint(self._mamba_block, h, i)
            else:
                h = self._mamba_block(h, i)
        
        # 5. Pooling
        h_pool = self.pooling(h) # [B, d_model]
        
        # 6. Prediction
        y_pred = self.head(h_pool)
        outputs['logits'] = y_pred
        
        return outputs
