"""
Compute Realized Volatility (RV) from SPY tick data.

Computes:
1. Daily RV = sqrt(sum(log_returns^2))
2. 30-day forward RV = sqrt(sum(daily_rv^2)) over next 30 calendar days

Usage:
    python compute_rv.py --input "D:/Mamba v2/datasets/SPY_trades" --output "D:/Mamba v2/datasets/SPY_daily_rv"
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_daily_rv(file_path: Path) -> dict:
    """Compute daily RV for a single file."""
    try:
        df = pd.read_parquet(file_path)
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        
        # Get price column
        price_col = next((c for c in ['price', 'close', 'last'] if c in df.columns), None)
        if price_col is None:
            return None
        
        prices = df[price_col].values
        if len(prices) < 2:
            return None
        
        # Compute log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        log_returns = log_returns[np.isfinite(log_returns)]
        
        if len(log_returns) == 0:
            return None
        
        # Daily RV = sqrt(sum(log_returns^2))
        daily_rv = np.sqrt(np.sum(log_returns ** 2))
        
        # Parse date from filename
        date_str = file_path.stem.split('.')[0]
        date = pd.to_datetime(date_str, errors='coerce')
        
        if pd.isna(date):
            return None
        
        return {
            'date': date,
            'daily_rv': daily_rv,
            'n_ticks': len(prices),
            'n_returns': len(log_returns),
        }
        
    except Exception as e:
        logger.warning(f"Error processing {file_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute RV from SPY tick data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with SPY parquet files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for RV parquet file')
    parser.add_argument('--horizon', type=int, default=30,
                        help='Forward RV horizon in calendar days (default: 30)')
    parser.add_argument('--min-days', type=int, default=10,
                        help='Minimum trading days required for forward RV (default: 10)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    files = sorted(input_dir.glob('*.parquet'))
    files = [f for f in files if not f.name.startswith('._')]
    
    logger.info(f"Found {len(files)} files in {input_dir}")
    
    # Compute daily RV in parallel
    logger.info("Computing daily RV...")
    daily_data = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_daily_rv, f): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Daily RV"):
            result = future.result()
            if result is not None:
                daily_data.append(result)
    
    # Create DataFrame and sort by date
    rv_df = pd.DataFrame(daily_data)
    rv_df = rv_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Computed daily RV for {len(rv_df)} days")
    logger.info(f"Date range: {rv_df['date'].min()} to {rv_df['date'].max()}")
    
    # Compute forward RV
    logger.info(f"Computing {args.horizon}-day forward RV...")
    rv_df[f'rv_{args.horizon}d_forward'] = np.nan
    
    for i in tqdm(range(len(rv_df)), desc="Forward RV"):
        current_date = rv_df.loc[i, 'date']
        end_date = current_date + pd.Timedelta(days=args.horizon)
        
        # Get RVs in forward window
        mask = (rv_df['date'] > current_date) & (rv_df['date'] <= end_date)
        forward_rvs = rv_df.loc[mask, 'daily_rv'].values
        
        if len(forward_rvs) >= args.min_days:
            # Forward RV = sqrt(sum(daily_rv^2))
            rv_df.loc[i, f'rv_{args.horizon}d_forward'] = np.sqrt(np.sum(forward_rvs ** 2))
    
    # Summary stats
    forward_col = f'rv_{args.horizon}d_forward'
    n_forward = rv_df[forward_col].notna().sum()
    logger.info(f"Forward RV coverage: {n_forward} / {len(rv_df)} ({100*n_forward/len(rv_df):.1f}%)")
    
    # Save
    output_file = output_dir / f'spy_daily_rv_{args.horizon}d.parquet'
    rv_df.to_parquet(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    
    # Also save a summary CSV for inspection
    summary_file = output_dir / f'spy_daily_rv_{args.horizon}d_summary.csv'
    rv_df.head(100).to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("RV Statistics")
    print("=" * 60)
    print(f"Total days: {len(rv_df)}")
    print(f"Date range: {rv_df['date'].min().date()} to {rv_df['date'].max().date()}")
    print(f"Forward RV coverage: {n_forward} days")
    print(f"\nDaily RV stats:")
    print(rv_df['daily_rv'].describe())
    print(f"\n{args.horizon}-day Forward RV stats:")
    print(rv_df[forward_col].describe())


if __name__ == '__main__':
    main()
