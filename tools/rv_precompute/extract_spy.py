"""
Extract SPY ticker from Polygon stock trades data.

Reads daily parquet files containing all stock trades,
filters for SPY ticker only, and saves to output directory.

Usage:
    python extract_spy.py --input "D:/polygon stock data/trades" --output "D:/Mamba v2/datasets/SPY_trades"
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_file(args_tuple):
    """Process a single file - extract SPY trades."""
    input_path, output_dir, ticker = args_tuple
    
    try:
        # Read only necessary columns for speed
        df = pd.read_parquet(input_path, columns=['ticker', 'price', 'size', 'sip_timestamp'])
        
        # Filter for ticker
        df_ticker = df[df['ticker'] == ticker].copy()
        
        if len(df_ticker) == 0:
            return input_path.name, 0, "no_data"
        
        # Sort by timestamp
        df_ticker = df_ticker.sort_values('sip_timestamp').reset_index(drop=True)
        
        # Save
        output_path = output_dir / input_path.name
        df_ticker.to_parquet(output_path, index=False, compression='zstd')
        
        return input_path.name, len(df_ticker), "success"
        
    except Exception as e:
        return input_path.name, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description='Extract SPY from Polygon stock trades')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with daily parquet files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for SPY-only files')
    parser.add_argument('--ticker', type=str, default='SPY',
                        help='Ticker to extract (default: SPY)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that already exist in output')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    files = sorted(input_dir.glob('*.parquet'))
    files = [f for f in files if not f.name.startswith('._')]
    
    logger.info(f"Found {len(files)} files in {input_dir}")
    logger.info(f"Extracting ticker: {args.ticker}")
    logger.info(f"Output directory: {output_dir}")
    
    # Skip existing if requested
    if args.skip_existing:
        existing = set(f.name for f in output_dir.glob('*.parquet'))
        files = [f for f in files if f.name not in existing]
        logger.info(f"Skipping {len(existing)} existing files, processing {len(files)}")
    
    if not files:
        logger.info("No files to process")
        return
    
    # Process files in parallel
    tasks = [(f, output_dir, args.ticker) for f in files]
    
    success_count = 0
    total_ticks = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            filename, n_ticks, status = future.result()
            
            if status == "success":
                success_count += 1
                total_ticks += n_ticks
            elif status == "no_data":
                pass  # No SPY data for this day (rare)
            else:
                errors.append((filename, status))
    
    logger.info(f"Completed: {success_count} files, {total_ticks:,} total ticks")
    
    if errors:
        logger.warning(f"Errors ({len(errors)}):")
        for filename, error in errors[:10]:
            logger.warning(f"  {filename}: {error}")


if __name__ == '__main__':
    main()
