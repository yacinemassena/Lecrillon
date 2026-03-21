import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    t = torch.zeros(1, device="cuda")
    print("CUDA works!")
    try:
        from mamba_ssm import Mamba
        print("mamba_ssm installed!")
    except ImportError:
        print("mamba_ssm NOT installed")
