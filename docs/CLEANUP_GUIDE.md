# Repository Cleanup Guide

This guide organizes the root directory files into categories: **Keep**, **Move**, and **Delete**.

---

## ✅ KEEP in Root (Essential Files)

### Core Training Files
- `vix_mamba_model.py` - Main VIX prediction model
- `mamba_only_model.py` - Mamba-only variant
- `smoke_mamba_only.py` - Smoke test script
- `config.py` - Configuration dataclasses
- `train.py` - Main training script (if exists)

### Data Scripts
- `download_data.py` - Download stock data from R2
- `download_vix.py` - Download VIX data from R2
- `upload_vix.py` - Upload VIX data to R2 (if exists)

### Setup Scripts
- `setupenv.sh` - WSL/RTX 5080 environment setup
- `setupvps.sh` - VPS/RunPod environment setup
- `pack_env.sh` - Package venv for R2 upload

### Docker Files
- `Dockerfile` - Docker image definition
- `.dockerignore` - Docker build exclusions
- `build_docker.sh` - Docker build and push script
- `install_docker_wsl.sh` - Docker installation for WSL

### Documentation
- `README.md` - Main project README (if exists)
- `CHANGELOG.md` - Version history
- `architecture.md` - Architecture documentation
- `README_DOCKER.md` - Docker deployment guide

### Git Files
- `.git/` - Git repository
- `.gitignore` - Git exclusions
- `.gitattributes` - Git line ending config

### Core Directories
- `encoder/` - TCN encoder modules
- `loader/` - Dataset loaders
- `loss/` - Custom loss functions
- `custom_packages/` - Custom mamba_blackwell and causal-conv1d-sm120
- `tests/` - Test files

---

## 📁 MOVE to `docs/` or `scripts/`

### Move to `docs/`
- `BUILD_UPLOAD_INSTRUCTIONS.md` → `docs/BUILD_UPLOAD_INSTRUCTIONS.md`
- `runmamba.txt` → `docs/runmamba.txt` (or rename to `docs/RUNPOD_WORKFLOW.md`)
- `runvps.txt` → `docs/runvps.txt` (or merge into `docs/RUNPOD_WORKFLOW.md`)

### Move to `scripts/` (if they exist)
- `build_wheels.sh` → `scripts/build_wheels.sh`

---

## 🗑️ DELETE (Safe to Remove)

### Large Build Artifacts (CRITICAL - 5.6GB!)
- `mamba_venv_linux_12_0.tar.zst` (2.8GB) - **Already uploaded to R2, safe to delete**
- `mamba_venv_linux_cpu.tar.zst` (2.7GB) - **Already uploaded to R2, safe to delete**

### Build Artifacts
- `custom_wheels/` - Pre-built wheels (if already uploaded to R2)
- `__pycache__/` - Python cache (auto-regenerated)
- `.pytest_cache/` - Pytest cache (auto-regenerated)

### Empty Directories (Git doesn't track them anyway)
- `checkpoints/` - Empty, will be created when training
- `datasets/` - Empty, will be populated by download scripts
- `results/` - Empty, will be created during training
- `venv/` - Virtual environment (recreated by setup scripts)

### Deprecated/Unused
- `blackwell-sm120-requirements.txt` - Not used by any setup script
- `TCN_Stock_Tick/` - If this is archived/deprecated code

---

## 🎯 Recommended Actions

### 1. Create `docs/` directory
```bash
mkdir -p docs
git mv BUILD_UPLOAD_INSTRUCTIONS.md docs/
git mv runmamba.txt docs/RUNPOD_WORKFLOW.txt
git mv runvps.txt docs/DOCKER_WORKFLOW.txt
```

### 2. Delete large tarballs (already in R2)
```bash
rm mamba_venv_linux_12_0.tar.zst
rm mamba_venv_linux_cpu.tar.zst
# This frees up 5.6GB!
```

### 3. Clean build artifacts
```bash
rm -rf __pycache__/
rm -rf .pytest_cache/
# These will be auto-regenerated
```

### 4. Update .gitignore
Add these if not already present:
```
# Build artifacts
*.tar.zst
__pycache__/
.pytest_cache/
custom_wheels/

# Empty runtime directories
checkpoints/
datasets/
results/
venv/
venv_windows/
```

---

## 📊 After Cleanup

Your root directory will look like:

```
d:\Mamba v2/
├── docs/                          # Documentation
│   ├── BUILD_UPLOAD_INSTRUCTIONS.md
│   ├── RUNPOD_WORKFLOW.txt
│   └── DOCKER_WORKFLOW.txt
├── encoder/                       # Core modules
├── loader/
├── loss/
├── custom_packages/               # Custom CUDA extensions
├── tests/
├── tools/
├── TCN_Stock_Tick/               # (if keeping)
├── vix_mamba_model.py            # Core files
├── mamba_only_model.py
├── smoke_mamba_only.py
├── config.py
├── download_data.py
├── download_vix.py
├── setupenv.sh
├── setupvps.sh
├── pack_env.sh
├── Dockerfile
├── build_docker.sh
├── install_docker_wsl.sh
├── CHANGELOG.md
├── architecture.md
├── README_DOCKER.md
├── .gitignore
└── .gitattributes
```

**Space saved:** ~5.6GB (from deleting .tar.zst files)
**Organization:** Much cleaner with docs in `docs/`
