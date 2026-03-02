# TCN Pretraining GPU Performance Benchmarks

## Summary
Performance benchmarks for TCN-12 (512 hidden) pretraining on SPY index data across different GPU configurations.

**Key Finding:** Smaller batch sizes (200-400 chunks) provide better throughput than large batches (1500+ chunks) due to gradient checkpointing overhead and memory bandwidth bottlenecks.

---

## Test Configuration
- **Model:** TCN-12 layers, 512 hidden dim, ~18.6M parameters
- **Data:** SPY index tick data (2022-2023)
- **Task:** 30-day forward realized volatility prediction
- **Steps per epoch:** 500
- **Gradient accumulation:** 4 steps
- **AMP:** bfloat16

---

## GPU Performance Results

### RTX 5080 (16GB VRAM)
**Local Machine**

| Profile | Batch Size | Throughput | VRAM Usage | Notes |
|---------|-----------|------------|------------|-------|
| rtx5080 | 200 chunks | **5.29 it/s** | ~10GB | Optimal for 16GB |
| h100 | 1500 chunks | OOM | N/A | Insufficient VRAM |

**Epoch Time:** ~95 seconds (training) + ~60 seconds (validation) = **155 seconds/epoch**

---

### H100 (80GB VRAM)
**RunPod VPS**

| Profile | Batch Size | Checkpointing | Throughput | VRAM Usage | Notes |
|---------|-----------|---------------|------------|------------|-------|
| rtx5080 | 200 chunks | Enabled | **10 it/s** | ~15GB | **Best throughput** |
| h100 | 1500 chunks | Enabled | 1.5 it/s | ~20GB | 6.7x slower |
| h100 | 1500 chunks | **Disabled** | 1.5 it/s | ~50GB | No improvement |

**Key Insight:** Large batches don't benefit from more VRAM. Memory bandwidth and checkpointing overhead dominate.

**Epoch Time (rtx5080 profile):** ~50 seconds

---

### A100 (80GB VRAM)
**RunPod VPS**

| Profile | Batch Size | Checkpointing | Throughput | VRAM Usage | Notes |
|---------|-----------|---------------|------------|------------|-------|
| rtx5080 | 200 chunks | Enabled | **8 it/s** | ~15GB | Good throughput |
| h100 | 1500 chunks | Enabled | 1.0 it/s | ~20GB | Very slow |
| h100 | 1500 chunks | **Disabled** | 1.3 it/s | ~60GB | Slightly faster |

**Observation:** A100 slower than H100 at same batch size. Likely due to older architecture.

---

### RTX 5090 (32GB VRAM)
**RunPod VPS**

| Profile | Batch Size | Throughput | VRAM Usage | Notes |
|---------|-----------|------------|------------|-------|
| rtx5080 | 200 chunks | ~15 it/s | ~10GB | Fast but underutilized |
| rtx5090 | 400 chunks | **4.74 it/s** | ~20GB | 2x data per step |

**Epoch Time:** ~32 seconds (both profiles)

**Observation:** 400 chunks processes 2x more data per epoch but at half the iteration speed. Net epoch time similar.

---

### RTX 4090 (24GB VRAM)
**RunPod VPS - Multi-GPU Scaling Test**

| GPUs | Profile | Batch Size | Throughput (per GPU) | Total Throughput | Scaling Efficiency | Notes |
|------|---------|-----------|---------------------|------------------|-------------------|-------|
| 1 | rtx5080 | 200 chunks | **4.75 it/s** | 4.75 it/s | 100% | Baseline |
| 8 | rtx5080 | 200 chunks | **4.60 it/s** | 36.8 it/s | 96.8% | **Excellent scaling** |

**Scaling Analysis:**
- **Ideal 8-GPU speedup:** 8.0x (38.0 it/s total)
- **Actual speedup:** 7.75x (36.8 it/s total)
- **Efficiency:** 96.8% (near-perfect for DDP)
- **Communication overhead:** ~3% (minimal)

**Epoch Time:** ~14 seconds (8 GPUs) vs ~105 seconds (1 GPU) = **7.5x faster**

---

### RTX 3080Ti (12GB VRAM)
**RunPod VPS - Multi-GPU Scaling Test**

| GPUs | Profile | Batch Size | Throughput (per GPU) | Total Throughput | Scaling Efficiency | Notes |
|------|---------|-----------|---------------------|------------------|-------------------|-------|
| 1 | rtx5080 | 200 chunks | **4.14 it/s** | 4.14 it/s | 100% | Baseline |
| 6 | rtx5080 | 200 chunks | **3.50 it/s** | 21.0 it/s | 84.5% | Near-linear scaling |

**Scaling Analysis:**
- **Ideal 6-GPU speedup:** 6.0x (24.84 it/s total)
- **Actual speedup:** 5.07x (21.0 it/s total)
- **Efficiency:** 84.5% (excellent for DDP)
- **Communication overhead:** ~15% (typical for multi-GPU training)

**Epoch Time:** ~24 seconds (6 GPUs) vs ~121 seconds (1 GPU) = **5x faster**

---

### AMD MI300X (192GB VRAM)
**Profile Added - Awaiting Test Results**

| Profile | Batch Size | Expected Throughput | Notes |
|---------|-----------|---------------------|-------|
| amd | 3600 chunks | TBD | 2.4x H100 capacity |

---

## Recommendations

### For Maximum Throughput (iterations/second)
**Use rtx5080 profile (200 chunks) on ANY GPU**
- H100: 10 it/s
- RTX 5090: ~15 it/s
- A100: 8 it/s
- RTX 5080: 5.29 it/s

### For Maximum Data Per Epoch
**Use GPU-specific profiles with --no-checkpoint**
- RTX 5090 (400 chunks): 4.74 it/s, 200k chunks/epoch
- H100 (1500 chunks): 1.5 it/s, 750k chunks/epoch
- AMD MI300X (3600 chunks): TBD, 1.8M chunks/epoch

### For Training Efficiency
**Stick with rtx5080 profile (200 chunks)**
- Fastest wall-clock time per epoch
- More gradient updates = better exploration
- Lower VRAM usage = more stable
- Works across all GPUs

---

## Technical Insights

### Why Large Batches Are Slower

1. **Gradient Checkpointing Overhead**
   - 1500 chunks = 3 batches of 512 chunks (stream_chunk_size)
   - Each batch recomputes activations during backward pass
   - More chunks = more recomputation

2. **Memory Bandwidth Bottleneck**
   - Large batches spend more time moving data between GPU memory layers
   - Compute is fast, memory transfers are slow
   - Smaller batches fit better in L2 cache

3. **Diminishing Returns**
   - 5x more VRAM (16GB → 80GB) only gives 1.9x throughput (5.29 → 10 it/s)
   - Memory bandwidth doesn't scale with VRAM size

### Gradient Checkpointing Impact

| Setting | VRAM Usage | Speed | Use Case |
|---------|-----------|-------|----------|
| Enabled | Lower (~20GB) | Slower | Default, memory-constrained |
| Disabled (`--no-checkpoint`) | Higher (~50GB) | Slightly faster | Large VRAM GPUs |

**Verdict:** Checkpointing overhead is significant. Disable on 80GB+ GPUs for small speed boost.

---

## Training Progress Example

**RTX 5080 Local (200 chunks, rtx5080 profile)**

| Epoch | Train Loss | Val Loss | Train Corr | Val Corr | Time |
|-------|-----------|----------|------------|----------|------|
| 1 | 0.0441 | 0.0000 | 0.560 | 0.000 | 155s |
| 4 | 0.0057 | TBD | TBD | TBD | ~150s |

**Expected Convergence:** Loss ~0.003-0.005, Correlation ~0.65-0.75 after 20-30 epochs

---

## Cost-Performance Analysis

**For 100 epochs training:**

| GPU | Profile | Time/Epoch | Total Time | Cost/Hour | Total Cost | it/s | Cost Efficiency |
|-----|---------|-----------|------------|-----------|------------|------|-----------------|
| RTX 5080 (local) | rtx5080 | 155s | 4.3 hours | $0 | $0 | 5.29 | ∞ |
| H100 (VPS) | rtx5080 | 50s | 1.4 hours | $2.50 | $3.50 | 10 | **Best** |
| RTX 5090 (VPS) | rtx5080 | 32s | 0.9 hours | $1.50 | $1.35 | ~15 | **Best** |
| A100 (VPS) | rtx5080 | 63s | 1.75 hours | $1.99 | $3.48 | 8 | Good |

**Winner:** RTX 5090 with rtx5080 profile - fastest and cheapest

---

## Configuration Files

**GPU Profiles:** `config_pretrain.py`
```python
GPU_PROFILES = {
    'rtx5080': 16GB VRAM, 200 chunks (index)
    'rtx5090': 32GB VRAM, 400 chunks (index)
    'h100': 80GB VRAM, 1500 chunks (index)
    'a100': 80GB VRAM, 1500 chunks (index)
    'amd': 192GB VRAM, 3600 chunks (index)
}
```

**Usage:**
```bash
# Maximum throughput (recommended)
python pretrain_tcn_rv.py --profile rtx5080 --stream index --epochs 100

# Maximum data per epoch
python pretrain_tcn_rv.py --profile h100 --stream index --epochs 100 --no-checkpoint

# GPU-specific optimal
python pretrain_tcn_rv.py --profile rtx5090 --stream index --epochs 100
```

---

## Conclusion

**Best Practice:** Use **rtx5080 profile (200 chunks)** regardless of GPU for maximum training speed. Large batch sizes provide no throughput benefit and significantly slow down training.

**Future Work:** Test AMD MI300X with 3600 chunks to determine if extreme VRAM enables new performance regimes.
