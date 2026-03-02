"""
Realized Volatility Prediction Head for TCN Pretraining.
Simple MLP that predicts forward RV from TCN embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RVPredictionHead(nn.Module):
    """
    MLP head for predicting Realized Volatility from TCN embeddings.
    
    Architecture:
        TCN output (512) -> Linear -> GELU -> Linear -> GELU -> Linear -> RV (1)
    """
    
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        # Final projection to scalar
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] TCN embeddings (pooled over frames)
        
        Returns:
            rv_pred: [B] predicted RV values
        """
        return self.mlp(x).squeeze(-1)


class TCNForRVPretraining(nn.Module):
    """
    Complete model for TCN pretraining on RV prediction.
    Combines TCN encoder + RV prediction head.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        chunked_encoder: nn.Module,
        rv_head: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.chunked_encoder = chunked_encoder
        self.rv_head = rv_head
    
    def forward(
        self,
        chunks: torch.Tensor,
        frame_id: torch.Tensor,
        weights: torch.Tensor,
        frame_scalars: torch.Tensor,
        num_frames: int,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for RV prediction.
        
        Args:
            chunks: [TotalK, chunk_len, F] all chunks
            frame_id: [TotalK] frame index for each chunk
            weights: [TotalK] importance weights
            frame_scalars: [TotalFrames, S] per-frame scalars
            num_frames: total number of frames
            use_checkpoint: whether to use gradient checkpointing
        
        Returns:
            rv_pred: [B] predicted RV for each sample in batch
        """
        # Encode frames
        frame_vecs = self.chunked_encoder(
            chunks=chunks,
            frame_id=frame_id,
            weights=weights,
            frame_scalars=frame_scalars,
            num_frames=num_frames,
            use_checkpoint=use_checkpoint,
        )  # [TotalFrames, d_model]
        
        # Pool frames to get sample-level embedding
        # For pretraining, we use mean pooling over all frames
        sample_emb = frame_vecs.mean(dim=0, keepdim=True)  # [1, d_model]
        
        # Predict RV
        rv_pred = self.rv_head(sample_emb)  # [1]
        
        return rv_pred
    
    def forward_batch(
        self,
        batch: dict,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for a collated batch.
        
        Args:
            batch: dict with keys from SPYRVDataset.collate_fn
            use_checkpoint: whether to use gradient checkpointing
        
        Returns:
            rv_pred: [B] predicted RV for each sample
        """
        # For batched samples, we need to track which frames belong to which sample
        # This is more complex - for now, process samples individually
        # TODO: Optimize with proper batch handling
        
        frame_vecs = self.chunked_encoder(
            chunks=batch['chunks'],
            frame_id=batch['frame_id'],
            weights=batch['weights'],
            frame_scalars=batch['frame_scalars'],
            num_frames=batch['num_frames'],
            use_checkpoint=use_checkpoint,
        )  # [TotalFrames, d_model]
        
        # Global mean pooling over all frames
        sample_emb = frame_vecs.mean(dim=0, keepdim=True)  # [1, d_model]
        
        # Predict RV
        rv_pred = self.rv_head(sample_emb)  # [1]
        
        return rv_pred
    
    def save_encoder(self, path: str):
        """Save only the encoder weights (for transfer to full model)."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'chunked_encoder_state_dict': self.chunked_encoder.state_dict(),
        }, path)
    
    @staticmethod
    def load_encoder(encoder: nn.Module, chunked_encoder: nn.Module, path: str):
        """Load pretrained encoder weights."""
        checkpoint = torch.load(path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        chunked_encoder.load_state_dict(checkpoint['chunked_encoder_state_dict'])
        return encoder, chunked_encoder


class RVLoss(nn.Module):
    """
    Loss function for RV prediction.
    Uses log-transformed targets for numerical stability.
    """
    
    def __init__(self, loss_type: str = 'huber', delta: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: [B] predicted RV
            target: [B] actual RV
        
        Returns:
            loss: scalar
        """
        # Log transform targets for numerical stability
        # RV values can span orders of magnitude
        log_pred = pred  # Assume model outputs log-scale
        log_target = torch.log1p(target)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(log_pred, log_target)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(log_pred, log_target, delta=self.delta)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(log_pred, log_target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
