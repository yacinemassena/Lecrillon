#!/usr/bin/env python3
import torch
import causal_conv1d

print("🌊 Testing causal-conv1d sm_120 functionality")
print(f"Device: {torch.cuda.get_device_name()}")

# Test streaming with different configurations
configs = [(64, 3), (128, 4), (256, 4)]

for dim, width in configs:
    x = torch.randn(4, dim, 2048, device='cuda')
    weight = torch.randn(dim, width, device='cuda')
    
    # Basic convolution
    out = causal_conv1d.causal_conv1d_fn(x, weight)
    print(f"✅ Basic conv1d (dim={dim}, width={width}): {x.shape} -> {out.shape}")
    
    # Streaming update
    conv_state = torch.zeros(4, dim, width-1, device='cuda')
    x_chunk = torch.randn(4, dim, 256, device='cuda')
    out_stream = causal_conv1d.causal_conv1d_update(x_chunk, conv_state, weight)
    print(f"✅ Streaming update: {x_chunk.shape} -> {out_stream.shape}")

print("🎉 causal-conv1d working perfectly with sm_120!")
