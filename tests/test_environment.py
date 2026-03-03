import importlib
import torch

def main():
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    assert torch.cuda.is_available(), "CUDA not available"

    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

    # causal conv
    causal = importlib.import_module("causal_conv1d")
    print("causal_conv1d:", getattr(causal, "__file__", "unknown"))

    # mamba module: accept either name
    mamba_mod = None
    for name in ("mamba_ssm", "mamba_blackwell"):
        try:
            mamba_mod = importlib.import_module(name)
            print(f"Mamba module '{name}':", mamba_mod.__file__)
            break
        except Exception as e:
            print(f"Import {name} failed:", e)
    assert mamba_mod is not None, "No Mamba module importable"

    # Try to import Mamba class from whichever module provides it
    Mamba = None
    try:
        from mamba_ssm import Mamba as Mamba
    except Exception:
        try:
            from mamba_blackwell import Mamba as Mamba
        except Exception as e:
            raise RuntimeError("Could not import Mamba class from either module") from e

    x = torch.randn(2, 64, 128, device=dev)
    m = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to(dev)
    y = m(x)
    assert y.shape == x.shape
    print("✅ Mamba forward OK")

    print("✅ Environment OK")

if __name__ == "__main__":
    main()
