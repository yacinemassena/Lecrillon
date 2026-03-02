#!/usr/bin/env python3
"""Profile training bottlenecks."""
import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, GPU_PROFILES
from loader.vix_stock_dataset import MambaL1Dataset
from mamba_model import StockMambaL1

# Setup
config = Config()
profile = GPU_PROFILES['rtx5080']
config.train.max_frames_per_batch = profile['max_frames_per_batch']
config.train.grad_accum_steps = profile['grad_accum_steps']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Dataset
print("\n1. Creating dataset...")
t0 = time.time()
dataset = MambaL1Dataset(config, split='train', level=1)
print(f"   Dataset created in {time.time()-t0:.1f}s")
print(f"   Samples: {len(dataset.anchor_dates)}")

# Get one sample
print("\n2. Loading one sample from dataset...")
t0 = time.time()
sample = dataset[0]
print(f"   Sample loaded in {time.time()-t0:.1f}s")
print(f"   Frames shape: {sample.frames.shape}")
print(f"   Frame mask shape: {sample.frame_mask.shape}")

# Move to GPU
print("\n3. Moving to GPU...")
t0 = time.time()
frames = sample.frames.to(device)
frame_mask = sample.frame_mask.to(device)
print(f"   Moved to GPU in {time.time()-t0:.1f}s")

# Build model
print("\n4. Building model...")
t0 = time.time()
model = StockMambaL1(config).to(device)
print(f"   Model built in {time.time()-t0:.1f}s")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass (chunked)
print("\n5. Forward pass (chunk_size=128, no checkpoint)...")
model.eval()
with torch.no_grad():
    t0 = time.time()
    outputs = model(frames.unsqueeze(0), frame_mask.unsqueeze(0), None, chunk_size=128)
    torch.cuda.synchronize()
    print(f"   Forward pass in {time.time()-t0:.1f}s")
    print(f"   Prediction: {outputs['vix_pred'].item():.2f}")

# Forward pass (no chunking)
print("\n6. Forward pass (no chunking)...")
with torch.no_grad():
    t0 = time.time()
    outputs = model(frames.unsqueeze(0), frame_mask.unsqueeze(0), None, chunk_size=10000)
    torch.cuda.synchronize()
    print(f"   Forward pass in {time.time()-t0:.1f}s")
    print(f"   Prediction: {outputs['vix_pred'].item():.2f}")

# Training step
print("\n7. Training step (forward + backward)...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
target = torch.tensor([sample.vix_target], device=device)

t0 = time.time()
outputs = model(frames.unsqueeze(0), frame_mask.unsqueeze(0), None, chunk_size=128)
loss = torch.nn.functional.mse_loss(outputs['vix_pred'], target)
loss.backward()
optimizer.step()
torch.cuda.synchronize()
print(f"   Training step in {time.time()-t0:.1f}s")
print(f"   Loss: {loss.item():.4f}")

print("\n✅ Profiling complete")
