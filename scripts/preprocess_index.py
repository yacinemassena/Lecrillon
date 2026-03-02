import argparse
import logging
import time
import gzip
import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(
        description="Preprocess INDEX minute data to epoch-ns Parquet (Polars)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone of input timestamps (e.g., 'UTC', 'America/New_York')",
    )
    parser.add_argument(
        "--partition",
        choices=["none", "ticker", "date", "ticker_date"],
        default="none",
        help="Partitioning strategy",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec (default: zstd)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Ignored in Polars version (retained for compatibility)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run sanity checks on output data",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a smoke test on the first N rows",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=1000,
        help="Number of rows for smoke test",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that have already been processed",
    )
    return parser.parse_args()


def process_file(
    filepath: Path,
    output_base_dir: str,
    args: argparse.Namespace,
    input_base_dir: Optional[Path] = None
):
    # Determine output path logic early
    if args.partition == "none":
        if input_base_dir:
            try:
                rel_path = filepath.relative_to(input_base_dir)
                target_path = Path(output_base_dir) / rel_path.with_suffix('.parquet')
            except ValueError:
                target_path = Path(output_base_dir) / filepath.with_suffix('.parquet').name
        else:
            target_path = Path(output_base_dir) / filepath.with_suffix('.parquet').name
            
        if args.skip_existing and target_path.exists():
            return
            
        target_path.parent.mkdir(parents=True, exist_ok=True)
        write_target = str(target_path)
    else:
        # For partitioned dataset, we write to root dir
        write_target = output_base_dir

    logger.info(f"Processing file: {filepath}")
    
    try:
        # Load data
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            # csv
            df = pd.read_csv(filepath)
            
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Check aliases
        if 'value' in df.columns and 'price' not in df.columns:
            df.rename(columns={'value': 'price'}, inplace=True)
            
        # Verify columns
        required_cols = {'ticker', 'price'}
        if not required_cols.issubset(df.columns):
            logger.error(f"Missing columns in {filepath}. Found: {df.columns.tolist()}")
            return
            
        has_ts_ns = 'ts_ns' in df.columns
        has_timestamp = 'timestamp' in df.columns
        
        if not (has_ts_ns or has_timestamp):
             logger.error(f"Missing timestamp column in {filepath}.")
             return

        # Drop nulls
        cols_to_check = ['ticker', 'price']
        if has_ts_ns: cols_to_check.append('ts_ns')
        else: cols_to_check.append('timestamp')
        
        df.dropna(subset=cols_to_check, inplace=True)
        
        # Types
        df['ticker'] = df['ticker'].astype(str)
        df['price'] = df['price'].astype(np.float32)
        
        # Timestamp handling
        if not has_ts_ns:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            
            # Convert TZ if needed (assuming input is already UTC aware from to_datetime(utc=True)
            # or if naive, we assume args.timezone. 
            # pd.to_datetime(utc=True) makes it UTC aware.
            # If we want to strictly follow args.timezone for naive inputs:
            # But usually to_datetime handles iso8601 fine.
            # Let's align with original logic: convert to nanosecond epoch.
            
            df['ts_ns'] = df['timestamp'].astype(np.int64)
        else:
            df['ts_ns'] = df['ts_ns'].astype(np.int64)
            
        # Select final columns
        df = df[['ticker', 'ts_ns', 'price']]
        
        # Sort and deduplicate
        df.sort_values(['ticker', 'ts_ns'], inplace=True)
        df.drop_duplicates(subset=['ticker', 'ts_ns'], keep='last', inplace=True)
        
        # Smoke test limit
        if args.smoke_test:
            df = df.head(args.smoke_rows)
            
        # Write
        if args.partition == "none":
            df.to_parquet(write_target, compression=args.compression, index=False)
        else:
            partition_cols = []
            if "ticker" in args.partition:
                partition_cols.append("ticker")
            if "date" in args.partition:
                df['date'] = pd.to_datetime(df['ts_ns'], unit='ns').dt.date.astype(str)
                partition_cols.append("date")
                
            df.to_parquet(
                write_target, 
                compression=args.compression, 
                index=False, 
                partition_cols=partition_cols
            )

        if args.smoke_test:
            print("\n--- Smoke Test Output (Pandas) ---")
            if args.partition == "none":
                verify_df = pd.read_parquet(write_target)
                print(verify_df.head())
                print(verify_df.dtypes)

    except Exception as e:
        logger.error(f"Failed to process file {filepath}: {e}")
        # Cleanup partial file if single file mode
        if args.partition == "none" and 'target_path' in locals() and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise


def run_verify(output_dir: str):
    logger.info("Verifying output data (Pandas)...")
    try:
        files = list(Path(output_dir).rglob("*.parquet"))
        if not files:
             logger.warning("No parquet files found to verify.")
             return
             
        # Just check first file for speed
        df = pd.read_parquet(files[0])
        logger.info(f"Verified {files[0]}: {len(df)} rows.")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")


def run_preprocessing(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    timezone: str = "UTC",
    partition: str = "none",
    compression: str = "zstd",
    chunksize: int = 100_000,
    verify: bool = False,
    smoke_test: bool = False,
    smoke_rows: int = 1000,
    skip_existing: bool = False,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if smoke_test:
        logger.info("Running in SMOKE TEST mode")
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Collect files
    files = []
    input_base_dir = None
    
    if input_path.is_file():
        files.append(input_path)
        input_base_dir = input_path.parent
    elif input_path.is_dir():
        # Recursive search for csv, csv.gz, and parquet
        files.extend(sorted(list(input_path.rglob("*.csv"))))
        files.extend(sorted(list(input_path.rglob("*.csv.gz"))))
        files.extend(sorted(list(input_path.rglob("*.parquet"))))
        input_base_dir = input_path
    else:
        logger.error(f"Input {input_path} not found.")
        return

    # Filter out hidden/resource fork files (e.g. ._file.csv.gz)
    files = [f for f in files if not f.name.startswith("._")]

    if not files:
        logger.error("No valid data files found.")
        return

    logger.info(f"Found {len(files)} files to process.")

    start_time = time.time()
    
    # Args compatibility
    class Args:
        pass
    
    args = Args()
    args.timezone = timezone
    args.partition = partition
    args.compression = compression
    args.chunksize = chunksize
    args.smoke_test = smoke_test
    args.smoke_rows = smoke_rows
    args.skip_existing = skip_existing
    
    # Polars is multi-threaded by default, so we can just loop sequentially 
    # and let Polars handle the parallelism within each file operation.
    # For very many small files, we might want ProcessPoolExecutor, 
    # but for typical large CSVs, simple loop is fine.
    
    for i, f in enumerate(files):
        # Handle .csv.gz files: Unzip, Keep CSV, Delete GZ
        if f.name.endswith('.csv.gz'):
            try:
                logger.info(f"Unzipping {f}...")
                new_f = f.with_suffix('') # Remove .gz
                
                with gzip.open(f, 'rb') as f_in:
                    with open(new_f, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Delete original compressed file
                f.unlink()
                f = new_f
                logger.info(f"Unzipped to {f}")
                
            except Exception as e:
                logger.error(f"Failed to unzip {f}: {e}")
                continue

        process_file(f, str(output_path), args, input_base_dir)
        if smoke_test and i >= 0:
            logger.info("Smoke test: stopping after first file.")
            break

    end_time = time.time()
    logger.info(f"Processing finished in {end_time - start_time:.2f} seconds.")

    if verify:
        run_verify(str(output_path))


def main():
    args = setup_args()
    run_preprocessing(
        input_path=args.input,
        output_path=args.output,
        timezone=args.timezone,
        partition=args.partition,
        compression=args.compression,
        chunksize=args.chunksize,
        verify=args.verify,
        smoke_test=args.smoke_test,
        smoke_rows=args.smoke_rows,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
