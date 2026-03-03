"""
Training Dashboard - Clean terminal UI for training progress.

Uses rich library for live-updating status display.
"""
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from dataclasses import dataclass, field
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


class TrainingDashboard:
    """Live-updating training dashboard."""
    
    def __init__(self):
        self.console = Console()
        self.state = TrainingState()
        self.live: Optional[Live] = None
        self.start_time = time.time()
        
    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=12),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="training", ratio=2),
            Layout(name="cache", ratio=1),
        )
        return layout
    
    def _render_header(self) -> Panel:
        """Render the header panel."""
        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        
        text = Text()
        text.append("🧠 ", style="bold")
        text.append("Mamba VIX Training", style="bold cyan")
        text.append(f"  │  Epoch {self.state.epoch}/{self.state.total_epochs}", style="white")
        text.append(f"  │  ⏱️ {int(hours):02d}:{int(mins):02d}:{int(secs):02d}", style="dim")
        
        return Panel(text, style="blue")
    
    def _render_training(self) -> Panel:
        """Render training metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim")
        table.add_column("value", style="bold")
        
        s = self.state
        step_str = f"{s.step}" if s.total_steps == 0 else f"{s.step}/{s.total_steps}"
        
        table.add_row("Step", step_str)
        table.add_row("Loss", f"{s.loss:.4f}")
        table.add_row("Val Loss", f"{s.val_loss:.4f}" if s.val_loss > 0 else "-")
        table.add_row("Val MAE", f"{s.val_mae:.4f}" if s.val_mae > 0 else "-")
        table.add_row("Grad Norm", f"{s.grad_norm:.2f}")
        table.add_row("", "")
        table.add_row("Seq Len", f"{s.seq_len:,}")
        table.add_row("Batch", f"{s.batch_size}")
        table.add_row("", "")
        table.add_row("Data", f"{s.data_time:.2f}s")
        table.add_row("Forward", f"{s.fwd_time:.2f}s")
        table.add_row("Backward", f"{s.bwd_time:.2f}s")
        
        return Panel(table, title="📊 Training", border_style="green")
    
    def _render_cache(self) -> Panel:
        """Render cache status panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim")
        table.add_column("value", style="bold")
        
        s = self.state
        
        # Status with color
        status_color = {
            "idle": "dim",
            "loading": "yellow",
            "ready": "green",
            "full": "cyan",
        }.get(s.cache_status, "white")
        
        table.add_row("Status", Text(s.cache_status.upper(), style=status_color))
        table.add_row("Files", f"{s.cache_files}/{s.cache_total}")
        table.add_row("Size", f"{s.cache_mb:.0f} MB")
        table.add_row("Hit Rate", f"{s.cache_hit_rate:.1%}")
        table.add_row("", "")
        table.add_row("VRAM", f"{s.vram_used:.1f}/{s.vram_total:.1f} GB")
        
        return Panel(table, title="💾 Cache & GPU", border_style="cyan")
    
    def _render_footer(self) -> Panel:
        """Render footer with current action."""
        s = self.state
        
        if s.cache_status == "loading":
            text = Text(f"⏳ Loading {s.cache_files}/{s.cache_total} files into cache...", style="yellow")
        elif s.step > 0:
            text = Text(f"🚀 Training step {s.step}...", style="green")
        else:
            text = Text("⏸️ Waiting...", style="dim")
        
        return Panel(text, style="dim")
    
    def _render(self) -> Layout:
        """Render the full dashboard."""
        layout = self._make_layout()
        layout["header"].update(self._render_header())
        layout["training"].update(self._render_training())
        layout["cache"].update(self._render_cache())
        layout["footer"].update(self._render_footer())
        return layout
    
    def start(self):
        """Start the live dashboard."""
        self.start_time = time.time()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self.live.start()
    
    def stop(self):
        """Stop the live dashboard."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def update(self, **kwargs):
        """Update state and refresh display."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        if self.live:
            self.live.update(self._render())
    
    def log(self, message: str):
        """Log a message below the dashboard."""
        if self.live:
            self.live.console.print(message)
        else:
            self.console.print(message)


# Simple usage without Live (for compatibility)
class SimpleDashboard:
    """Simpler dashboard that just prints status lines."""
    
    def __init__(self):
        self.console = Console()
        self.state = TrainingState()
        self.last_print = 0
        
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
        self.console.print(message)
