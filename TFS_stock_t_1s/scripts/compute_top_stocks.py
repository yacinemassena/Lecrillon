"""
Compute Top N Stocks by Daily Volume.

One-time precomputation to generate a list of the most liquid stocks.
Output: top_100_stocks.txt (one ticker per line)

Usage:
    python scripts/compute_top_stocks.py --data_path datasets/2022-2023/polygon_stock_trades --top_n 100
"""

import argparse
import logging
from pathlib import Path

# Project root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_top_stocks(data_path: str, top_n: int = 100, output_file: str = None) -> list:
    """
    Compute top N stocks by total traded volume across all days.
    
    Args:
        data_path: Path to polygon_stock_trades directory
        top_n: Number of top stocks to return
        output_file: Output file path (optional)
    
    Returns:
        List of top N ticker symbols
    """
    data_path = Path(data_path)
    files = sorted(data_path.glob('*.parquet'))
    files = [f for f in files if not f.name.startswith('._')]
    
    logger.info(f"Processing {len(files)} files from {data_path}")
    
    # Accumulate volume per ticker
    ticker_volume = defaultdict(float)
    ticker_days = defaultdict(int)
    
    for file_path in tqdm(files, desc="Computing volumes"):
        try:
            # Read only ticker and size columns for speed
            df = pd.read_parquet(file_path, columns=['ticker', 'size'])
            
            # Sum volume per ticker
            daily_vol = df.groupby('ticker')['size'].sum()
            
            for ticker, vol in daily_vol.items():
                ticker_volume[ticker] += vol
                ticker_days[ticker] += 1
                
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            continue
    
    # Sort by total volume
    sorted_tickers = sorted(ticker_volume.items(), key=lambda x: -x[1])
    top_tickers = [t[0] for t in sorted_tickers[:top_n]]
    
    logger.info(f"Top {top_n} stocks by volume:")
    for i, ticker in enumerate(top_tickers[:10]):
        vol = ticker_volume[ticker]
        days = ticker_days[ticker]
        logger.info(f"  {i+1}. {ticker}: {vol/1e9:.2f}B shares, {days} days")
    
    # Save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for ticker in top_tickers:
                f.write(f"{ticker}\n")
        
        logger.info(f"Saved top {top_n} stocks to {output_path}")
    
    return top_tickers


def load_top_stocks(file_path: str) -> set:
    """Load top stocks from precomputed file."""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute top N stocks by volume')
    parser.add_argument('--data_path', type=str, 
                        default=str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'polygon_stock_trades'),
                        help='Path to stock trades data')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top stocks')
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'top_100_stocks.txt'),
                        help='Output file path')
    
    args = parser.parse_args()
    
    compute_top_stocks(
        data_path=args.data_path,
        top_n=args.top_n,
        output_file=args.output,
    )
