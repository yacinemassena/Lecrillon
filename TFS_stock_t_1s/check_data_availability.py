"""
Check data availability for Stock → Mamba → VIX pipeline.

Checks:
- Stock_Data_1s parquet files
- VIX CSV files
- Top 100 stocks ticker list
- SPY daily RV (optional)

Generates available_streams.json with status.
"""

import json
import logging
import platform
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()


def check_local_data():
    """Check local data availability."""
    if platform.system() == 'Linux':
        # WSL: use /mnt/d/ path to Windows data
        data_root = Path('/mnt/d/Mamba v2/datasets')
    else:
        data_root = Path(r'D:\Mamba v2\datasets')

    results = {}

    # Stock 1s bars
    stock_path = data_root / 'Stock_Data_1s'
    if stock_path.exists():
        n_files = len(list(stock_path.glob('*.parquet')))
        results['stock_1s'] = {
            'available': n_files > 0,
            'path': str(stock_path),
            'files': n_files,
        }
        status = f"Found {n_files} files" if n_files > 0 else "Empty"
        logger.info(f"{'✓' if n_files > 0 else '✗'} STOCK_1S: {status} at {stock_path}")
    else:
        results['stock_1s'] = {'available': False, 'path': str(stock_path), 'files': 0}
        logger.info(f"✗ STOCK_1S: Not found at {stock_path}")

    # VIX data
    vix_path = data_root / 'VIX'
    if vix_path.exists():
        n_csv = len(list(vix_path.glob('VIX_*.csv')))
        results['vix'] = {
            'available': n_csv > 0,
            'path': str(vix_path),
            'files': n_csv,
        }
        status = f"Found {n_csv} CSV files" if n_csv > 0 else "Empty"
        logger.info(f"{'✓' if n_csv > 0 else '✗'} VIX: {status} at {vix_path}")
    else:
        results['vix'] = {'available': False, 'path': str(vix_path), 'files': 0}
        logger.info(f"✗ VIX: Not found at {vix_path}")

    # Top 100 stocks list
    tickers_file = PROJECT_ROOT / 'scripts' / 'top_100_stocks.txt'
    if tickers_file.exists():
        with open(tickers_file) as f:
            n_tickers = sum(1 for line in f if line.strip())
        results['tickers'] = {
            'available': n_tickers > 0,
            'path': str(tickers_file),
            'count': n_tickers,
        }
        logger.info(f"✓ TICKERS: {n_tickers} tickers in {tickers_file}")
    else:
        results['tickers'] = {'available': False, 'path': str(tickers_file), 'count': 0}
        logger.info(f"✗ TICKERS: Not found at {tickers_file}")

    # SPY daily RV (optional)
    rv_path = data_root / 'SPY_daily_rv'
    if rv_path.exists():
        rv_files = list(rv_path.glob('*.parquet'))
        results['rv'] = {
            'available': len(rv_files) > 0,
            'path': str(rv_path),
            'files': len(rv_files),
        }
        logger.info(f"✓ RV: Found {len(rv_files)} files at {rv_path}")
    else:
        results['rv'] = {'available': False, 'path': str(rv_path), 'files': 0}
        logger.info(f"✗ RV: Not found (optional)")

    return results


def main():
    logger.info("=" * 60)
    logger.info("Checking Data Availability for Mamba Pipeline")
    logger.info("=" * 60)

    results = check_local_data()

    # Determine training readiness
    can_train_l1 = (
        results.get('stock_1s', {}).get('available', False)
        and results.get('vix', {}).get('available', False)
    )
    can_train_l2 = can_train_l1  # Same data, different windows

    results['can_train'] = {
        'level_1': can_train_l1,
        'level_2': can_train_l2,
    }

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    if can_train_l1:
        logger.info("✓ Ready for Mamba Level 1 training (stock → next-day VIX)")
        logger.info("  Command: python train_mamba.py --profile rtx5080 --level 1")
    else:
        logger.info("✗ Missing data for training. Need: Stock_Data_1s + VIX CSVs")

    if can_train_l2:
        logger.info("✓ Ready for Mamba Level 2 training (daily summaries → VIX +30d)")
        logger.info("  Command: python train_mamba.py --profile rtx5080 --level 2")

    # Save config
    config_path = PROJECT_ROOT / 'available_streams.json'
    with open(config_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nConfig saved to: {config_path}")


if __name__ == '__main__':
    main()
