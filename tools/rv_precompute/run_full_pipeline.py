"""
Full RV Precompute Pipeline.

Runs both steps:
1. Extract SPY from Polygon stock trades
2. Compute daily and forward RV

Usage:
    python run_full_pipeline.py
    
Or run steps separately:
    python extract_spy.py --input "D:/polygon stock data/trades" --output "D:/Mamba v2/datasets/SPY_trades"
    python compute_rv.py --input "D:/Mamba v2/datasets/SPY_trades" --output "D:/Mamba v2/datasets/SPY_daily_rv"
"""

import subprocess
import sys
from pathlib import Path

# Configuration
POLYGON_DATA = "D:/polygon stock data/trades"
SPY_TRADES_DIR = "D:/Mamba v2/datasets/SPY_trades"
RV_OUTPUT_DIR = "D:/Mamba v2/datasets/SPY_daily_rv"
TICKER = "SPY"
RV_HORIZON = 30
WORKERS = 8


def run_step(script: str, args: list):
    """Run a Python script with arguments."""
    cmd = [sys.executable, script] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60 + "\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"Error: {script} failed with code {result.returncode}")
        sys.exit(1)


def main():
    script_dir = Path(__file__).parent
    
    print("=" * 60)
    print("SPY Realized Volatility Precompute Pipeline")
    print("=" * 60)
    print(f"Input: {POLYGON_DATA}")
    print(f"Ticker: {TICKER}")
    print(f"SPY trades output: {SPY_TRADES_DIR}")
    print(f"RV output: {RV_OUTPUT_DIR}")
    print(f"RV horizon: {RV_HORIZON} days")
    print("=" * 60)
    
    # Step 1: Extract SPY
    run_step(
        str(script_dir / "extract_spy.py"),
        [
            "--input", POLYGON_DATA,
            "--output", SPY_TRADES_DIR,
            "--ticker", TICKER,
            "--workers", str(WORKERS),
            "--skip-existing",
        ]
    )
    
    # Step 2: Compute RV
    run_step(
        str(script_dir / "compute_rv.py"),
        [
            "--input", SPY_TRADES_DIR,
            "--output", RV_OUTPUT_DIR,
            "--horizon", str(RV_HORIZON),
            "--workers", str(WORKERS),
        ]
    )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"RV file: {RV_OUTPUT_DIR}/spy_daily_rv_{RV_HORIZON}d.parquet")
    print("=" * 60)


if __name__ == '__main__':
    main()
