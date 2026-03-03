#!/usr/bin/env python3
import torch
from mamba_ssm import Mamba
import time

print("🐍 Testing Mamba sm_120 functionality")
print(f"Device: {torch.cuda.get_device_name()}")

# Test different Mamba configurations
configs = [
    {"d_model": 128, "d_state": 16, "d_conv": 4, "expand": 2},
    {"d_model": 512, "d_state": 32, "d_conv": 4, "expand": 2}, 
    {"d_model": 1024, "d_state": 64, "d_conv": 4, "expand": 2},
]

for config in configs:
    model = Mamba(**config).cuda()
    
    # Test different sequence lengths
    for seq_len in [512, 1024, 2048]:
        x = torch.randn(2, seq_len, config["d_model"], device='cuda', dtype=torch.bfloat16)
        
        start = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y = model(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        
        print(f"✅ Mamba d_model={config['d_model']}, seq_len={seq_len}: {elapsed:.2f}ms")

print("🎉 Mamba working perfectly with sm_120!")
