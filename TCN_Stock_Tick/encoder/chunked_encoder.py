"""
Chunked Frame Encoder.
Wraps FrameEncoder to handle variable-length tick sequences via chunking and pooling.
Includes scalar feature concatenation (n_ticks, notional, vol).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .frame_encoder import FrameEncoder


class ChunkedFrameEncoder(nn.Module):
    """
    Encodes a sequence of frames where each frame consists of multiple chunks.
    
    Process:
    1. Encode all chunks in parallel -> [TotalK, D]
    2. Pool chunks per frame (weighted mean) -> [TotalFrames, D]
    3. Concatenate frame scalars -> [TotalFrames, D + S]
    4. Project to output dim -> [TotalFrames, d_model]
    """
    
    def __init__(self, frame_encoder: nn.Module, d_model: int, num_scalars: int = 3, 
                 stream_chunks: bool = False, stream_chunk_size: int = 2048):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.d_model = d_model
        self.stream_chunks = stream_chunks
        self.stream_chunk_size = stream_chunk_size
        
        # Input dim to projection = encoder_dim + num_scalars
        # Robustly determine encoder output dim
        enc_dim = getattr(frame_encoder, "out_dim", None)
        if enc_dim is None:
            enc_dim = getattr(frame_encoder, "hidden_dim", None)
        if enc_dim is None:
            raise ValueError("frame_encoder must expose out_dim or hidden_dim")
            
        self.proj_in_dim = enc_dim + num_scalars
        
        self.out_proj = nn.Sequential(
            nn.Linear(self.proj_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
    def forward(self, chunks, frame_id, weights, frame_scalars, num_frames, ticker_ids=None, use_checkpoint=False):
        """
        Args:
            chunks: [TotalK, chunk_len, F] - All chunks from all frames flattened
            frame_id: [TotalK] - Index of the frame each chunk belongs to
            weights: [TotalK] - Importance weight for each chunk
            frame_scalars: [TotalFrames, S] - Per-frame scalar features (n_ticks, notional, vol)
            num_frames: int - Total number of frames in the batch
            ticker_ids: [TotalFrames] - Optional ticker ID for each frame
            use_checkpoint: bool - Whether to use gradient checkpointing
            
        Returns:
            frame_vecs: [TotalFrames, d_model]
        """
        # Determine target device from model parameters
        target_device = self.out_proj[0].weight.device

        # If input is on CPU (Host Streaming), we must handle device movement
        # If input is already on GPU, .to(target_device) is a cheap no-op if same device
        
        frame_id = frame_id.long()
        
        # Prepare output tensors for pooling
        # We need to sum embeddings and weights per frame
        # Determine embedding dim from the encoder without running it yet if possible,
        # but since we need to run it, let's setup buffers after first batch or pre-allocate if we trust config.
        # We know self.proj_in_dim includes scalars, so encoder dim is self.proj_in_dim - num_scalars
        enc_dim = self.proj_in_dim - frame_scalars.shape[1]
        
        # sum_emb must be on the target computation device (GPU)
        sum_emb = torch.zeros(
            num_frames, enc_dim, 
            device=target_device, dtype=chunks.dtype
        )
        sum_w = torch.zeros(
            num_frames, 1, 
            device=target_device, dtype=chunks.dtype
        )

        # Helper for encoding a slice
        def encode_slice(x, t_ids=None):
            if use_checkpoint and x.requires_grad:
                # Checkpoint wrapper needs to handle args. 
                # Note: checkpoint passes *args to the function.
                return torch.utils.checkpoint.checkpoint(
                    self.frame_encoder, x, t_ids, use_reentrant=False
                )
            else:
                return self.frame_encoder(x, ticker_ids=t_ids)

        TotalK = chunks.size(0)
        
        if (not self.stream_chunks) or (TotalK <= self.stream_chunk_size):
            # 1. Non-streaming path (Encode all at once)
            # Ensure inputs are on target device
            chunks_dev = chunks.to(target_device)
            weights_dev = weights.to(target_device)
            frame_id_dev = frame_id.to(target_device)
            
            # Resolve Ticker IDs for chunks
            chunk_t_ids = None
            if ticker_ids is not None:
                ticker_ids_dev = ticker_ids.to(target_device)
                chunk_t_ids = ticker_ids_dev[frame_id_dev]
            
            chunk_emb = encode_slice(chunks_dev, t_ids=chunk_t_ids)
            
            # Weighted Pooling
            w = weights_dev.unsqueeze(-1)
            weighted_emb = chunk_emb * w
            
            sum_emb.index_add_(0, frame_id_dev, weighted_emb)
            sum_w.index_add_(0, frame_id_dev, w)
            
        else:
            # 2. Streaming path
            # Move ticker_ids to device once if present
            ticker_ids_dev = None
            if ticker_ids is not None:
                ticker_ids_dev = ticker_ids.to(target_device)

            for s in range(0, TotalK, self.stream_chunk_size):
                e = min(s + self.stream_chunk_size, TotalK)
                
                # Move slice to target device (Host -> GPU transfer happens here if needed)
                chunk_slice = chunks[s:e].to(target_device)
                fid = frame_id[s:e].to(target_device)
                w_slice = weights[s:e].to(target_device).unsqueeze(-1)
                
                # Resolve Ticker IDs for this slice
                slice_t_ids = None
                if ticker_ids_dev is not None:
                    slice_t_ids = ticker_ids_dev[fid]
                
                # Encode micro-batch
                emb_slice = encode_slice(chunk_slice, t_ids=slice_t_ids)
                
                # Accumulate
                weighted_slice = emb_slice * w_slice
                sum_emb.index_add_(0, fid, weighted_slice)
                sum_w.index_add_(0, fid, w_slice)
                
                # Free intermediate tensors
                del chunk_slice, emb_slice, weighted_slice, w_slice, fid
                if s % (self.stream_chunk_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        # Normalize by sum of weights (avoid div by zero)
        pooled_emb = sum_emb / (sum_w + 1e-8)  # [TotalFrames, D]
        
        # 3. Concatenate Scalars
        # Log-transform scalars to compress dynamic range
        # Scalars should also be on target device
        frame_scalars = frame_scalars.to(target_device)
        scalars_log = torch.log1p(frame_scalars)  # [TotalFrames, S]
        
        # Concatenate: [TotalFrames, D] + [TotalFrames, S] -> [TotalFrames, D+S]
        combined = torch.cat([pooled_emb, scalars_log], dim=-1)
        
        # 4. Project to d_model
        out = self.out_proj(combined)  # [TotalFrames, d_model]
        
        return out
