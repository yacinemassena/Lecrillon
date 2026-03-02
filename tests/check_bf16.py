import torch
import torch.nn as nn
import time

def benchmark_realistic():
    """Benchmark that reflects actual training workloads."""
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Large matmul (like attention/FFN in transformers)
    print("\n1. Large Matmul (4096x4096 @ 4096x4096)")
    for dtype, name in [(torch.float32, 'FP32'), (torch.bfloat16, 'BF16'), (torch.float16, 'FP16')]:
        a = torch.randn(4096, 4096, device=device, dtype=dtype)
        b = torch.randn(4096, 4096, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(50):
            c = a @ b
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        
        print(f"  {name}: {elapsed:.1f}ms (50 iters)")
        results.append((name, 'large_matmul', elapsed))
    
    # Test 2: Batched matmul (like batch of sequences)
    print("\n2. Batched Matmul [32, 2048, 512] @ [32, 512, 512]")
    for dtype, name in [(torch.float32, 'FP32'), (torch.bfloat16, 'BF16'), (torch.float16, 'FP16')]:
        a = torch.randn(32, 2048, 512, device=device, dtype=dtype)
        b = torch.randn(32, 512, 512, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            c = torch.bmm(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(50):
            c = torch.bmm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        
        print(f"  {name}: {elapsed:.1f}ms (50 iters)")
        results.append((name, 'batched_matmul', elapsed))
    
    # Test 3: Conv1d (like your TCN)
    print("\n3. Conv1d [64, 384, 2048] kernel=3")
    for dtype, name in [(torch.float32, 'FP32'), (torch.bfloat16, 'BF16'), (torch.float16, 'FP16')]:
        conv = nn.Conv1d(384, 384, 3, padding=1).to(device, dtype=dtype)
        x = torch.randn(64, 384, 2048, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            y = conv(x)
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(50):
            y = conv(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        
        print(f"  {name}: {elapsed:.1f}ms (50 iters)")
        results.append((name, 'conv1d', elapsed))
    
    # Test 4: Linear layers (MLP forward)
    print("\n4. Linear 512->512 batch=4096 (FFN-like)")
    for dtype, name in [(torch.float32, 'FP32'), (torch.bfloat16, 'BF16'), (torch.float16, 'FP16')]:
        linear = nn.Linear(512, 512).to(device, dtype=dtype)
        x = torch.randn(4096, 512, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            y = linear(x)
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(100):
            y = linear(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        
        print(f"  {name}: {elapsed:.1f}ms (100 iters)")
        results.append((name, 'linear', elapsed))
    
    # Test 5: Full forward+backward (most realistic)
    print("\n5. MLP Forward+Backward [batch=256, 512->2048->512]")
    for dtype, name in [(torch.float32, 'FP32'), (torch.bfloat16, 'BF16'), (torch.float16, 'FP16')]:
        model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        ).to(device, dtype=dtype)
        
        x = torch.randn(256, 512, device=device, dtype=dtype, requires_grad=True)
        target = torch.randn(256, 512, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            y = model(x)
            loss = (y - target).pow(2).mean()
            loss.backward()
        torch.cuda.synchronize()
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(50):
            y = model(x)
            loss = (y - target).pow(2).mean()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        
        print(f"  {name}: {elapsed:.1f}ms (50 iters)")
        results.append((name, 'fwd_bwd', elapsed))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Speedup vs FP32")
    print("=" * 60)
    
    fp32_times = {r[1]: r[2] for r in results if r[0] == 'FP32'}
    for dtype_name in ['BF16', 'FP16']:
        print(f"\n{dtype_name}:")
        for r in results:
            if r[0] == dtype_name:
                speedup = fp32_times[r[1]] / r[2]
                print(f"  {r[1]}: {speedup:.2f}x")


if __name__ == '__main__':
    benchmark_realistic()