"""
Pure-PyTorch Mamba shim for inference without mamba_ssm CUDA kernels.

Drop-in replacement for mamba_ssm.Mamba with identical parameter names/shapes.
Uses a naive sequential selective scan (slow but correct for inference).

Usage: import this before importing mamba_only_model to inject into sys.modules.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _selective_scan_inner(
    x: torch.Tensor,    # [B, D, L] float32
    dt: torch.Tensor,   # [B, D, L] float32 (after softplus)
    A: torch.Tensor,    # [D, N] float32 (negative)
    B: torch.Tensor,    # [B, N, L] float32
    C: torch.Tensor,    # [B, N, L] float32
    D_skip: torch.Tensor,  # [D] float32
) -> torch.Tensor:
    """JIT-compiled selective scan inner loop — runs ~5-10x faster than Python."""
    B_batch, D_dim, L = x.shape
    N = A.shape[1]

    y = torch.empty_like(x)  # [B, D, L]
    h = torch.zeros(B_batch, D_dim, N, device=x.device, dtype=torch.float32)

    for t in range(L):
        dt_t = dt[:, :, t]          # [B, D]
        x_t = x[:, :, t]            # [B, D]
        B_t = B[:, :, t]            # [B, N]
        C_t = C[:, :, t]            # [B, N]

        dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))    # [B, D, N]
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)             # [B, D, N]

        h = h * dA + x_t.unsqueeze(-1) * dB                    # [B, D, N]

        y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)               # [B, D]
        y[:, :, t] = y_t + D_skip * x_t

    return y


def selective_scan_fn(x, dt, A, B, C, D, z=None, delta_bias=None, delta_softplus=True):
    """
    Pure-PyTorch selective scan with JIT-compiled inner loop.

    Args:
        x:  [B, D, L]
        dt: [B, D, L]
        A:  [D, N]  (negative values)
        B:  [B, N, L]
        C:  [B, N, L]
        D:  [D]
        z:  [B, D, L]  (gate, optional)
        delta_bias: [D] (added to dt before softplus)
        delta_softplus: bool
    Returns:
        y: [B, D, L]
    """
    dtype = x.dtype

    # Prep in float32
    x_f = x.float()
    dt_f = dt.float()
    if delta_bias is not None:
        dt_f = dt_f + delta_bias.float().unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        dt_f = F.softplus(dt_f)

    y = _selective_scan_inner(x_f, dt_f, A.float(), B.float(), C.float(), D.float())

    y = y.to(dtype)
    if z is not None:
        y = y * F.silu(z)
    return y


class Mamba(nn.Module):
    """
    Pure-PyTorch Mamba layer with identical parameter names to mamba_ssm.Mamba.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D) → (B, L, D)"""
        batch, seqlen, dim = hidden_states.shape

        # Input projection → x, z
        xz = self.in_proj(hidden_states)          # [B, L, 2*D_inner]
        xz = xz.transpose(1, 2)                   # [B, 2*D_inner, L]
        x, z = xz.chunk(2, dim=1)                 # each [B, D_inner, L]

        # Causal conv1d
        x = self.act(self.conv1d(x)[..., :seqlen])  # [B, D_inner, L]

        # Project to dt, B, C
        x_for_proj = x.transpose(1, 2).contiguous().view(batch * seqlen, self.d_inner)
        x_dbl = self.x_proj(x_for_proj)           # [B*L, dt_rank + 2*d_state]
        dt, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt projection
        dt = (self.dt_proj.weight @ dt.t())        # [D_inner, B*L]
        dt = dt.view(self.d_inner, batch, seqlen).permute(1, 0, 2)  # [B, D_inner, L]

        # Reshape B, C
        B_proj = B_proj.view(batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()  # [B, N, L]
        C_proj = C_proj.view(batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()  # [B, N, L]

        # A matrix
        A = -torch.exp(self.A_log.float())         # [D_inner, N]

        # Selective scan
        y = selective_scan_fn(
            x, dt, A, B_proj, C_proj,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )  # [B, D_inner, L]

        # Output projection
        y = y.transpose(1, 2)                      # [B, L, D_inner]
        out = self.out_proj(y)                     # [B, L, D_model]

        return out
