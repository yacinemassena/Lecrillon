"""
Unit tests for FrameEncoder module.
"""

import sys
from pathlib import Path
# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from encoder.frame_encoder import FrameEncoder, TCNEncoder


def test_tcn_encoder_shape():
    """Test TCN encoder output shape."""
    B, L, F = 8, 100, 5
    hidden_dim = 512
    
    encoder = TCNEncoder(
        in_features=F,
        hidden_dim=hidden_dim,
        num_layers=4
    )
    
    x = torch.randn(B, L, F)
    out = encoder(x)
    
    assert out.shape == (B, hidden_dim), f"Expected {(B, hidden_dim)}, got {out.shape}"


def test_frame_encoder_tcn():
    """Test FrameEncoder with TCN backend."""
    encoder = FrameEncoder(
        kind='tcn',
        in_features=5,
        hidden_dim=256,
        num_layers=3
    )
    
    x = torch.randn(4, 80, 5)
    out = encoder(x)
    
    assert out.shape == (4, 256)
    assert encoder.out_dim == 256


def test_frame_encoder_gradient_flow():
    """Test that gradients flow through the encoder."""
    encoder = FrameEncoder(
        kind='tcn',
        in_features=3,
        hidden_dim=128
    )
    
    x = torch.randn(2, 50, 3, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow to input"
    for param in encoder.parameters():
        assert param.grad is not None, "All parameters should have gradients"


def test_frame_encoder_batch_independence():
    """Test that batch elements are processed independently."""
    encoder = FrameEncoder(kind='tcn', in_features=4, hidden_dim=64)
    encoder.eval()  # Disable dropout for determinism
    
    # Process batch
    x_batch = torch.randn(8, 60, 4)
    out_batch = encoder(x_batch)
    
    # Process individually
    out_individual = torch.stack([encoder(x_batch[i:i+1]) for i in range(8)])
    
    assert torch.allclose(out_batch, out_individual.squeeze(1), atol=1e-5), \
        "Batch processing should give same results as individual processing"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
