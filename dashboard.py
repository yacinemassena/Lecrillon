"""
Training Dashboard - Clean terminal UI for training progress.

Uses rich library for live-updating status display.
"""
from rich.console import Console
from rich.text import Text
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class TrainingState:
    """Current training state for dashboard display."""
    # Epoch info
    epoch: int = 0
    total_epochs: int = 0
    
    # Step info
    step: int = 0
    total_steps: int = 0
    
    # Metrics
    loss: float = 0.0
    val_loss: float = 0.0
    val_mae: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    
    # Timing
    data_time: float = 0.0
    fwd_time: float = 0.0
    bwd_time: float = 0.0
    step_time: float = 0.0
    
    # Cache
    cache_status: str = "idle"
    cache_files: int = 0
    cache_total: int = 0
    cache_mb: float = 0.0
    cache_hit_rate: float = 0.0
    
    # GPU
    vram_used: float = 0.0
    vram_total: float = 0.0
    gpu_util: float = 0.0
    
    # Sequence
    seq_len: int = 0
    batch_size: int = 0


class SimpleDashboard:
    """Simpler dashboard that just prints status lines."""
    
    def __init__(self):
        self.console = Console()
        self.state = TrainingState()
        self.last_print = 0
        self.last_log_message: Optional[str] = None

    def start(self):
        self.console.print("[bold cyan]🧠 Mamba VIX Training[/]")
        self.console.print("─" * 60)
    
    def stop(self):
        self.console.print("─" * 60)
        self.console.print("[bold green]✅ Training complete[/]")
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        # Rate limit prints
        now = time.time()
        if now - self.last_print < 0.5:
            return
        self.last_print = now
        
        s = self.state
        
        # Single-line status update
        if s.cache_status == "loading":
            self.console.print(
                f"[yellow]⏳ Cache[/] {s.cache_files}/{s.cache_total} files | "
                f"{s.cache_mb:.0f}MB",
                end="\r"
            )
        elif s.step > 0:
            step_str = f"{s.step}" if s.total_steps == 0 else f"{s.step}/{s.total_steps}"
            self.console.print(
                f"[green]E{s.epoch}[/] Step {step_str} | "
                f"loss={s.loss:.4f} | seq={s.seq_len:,} | "
                f"data={s.data_time:.1f}s fwd={s.fwd_time:.2f}s | "
                f"VRAM={s.vram_used:.1f}GB"
            )
    
    def log(self, message: str):
        if message == self.last_log_message:
            return
        self.last_log_message = message
        self.console.print(message)
