# Index Data Preprocessing

This script preprocesses INDEX minute data (CSV) into a Parquet dataset with nanosecond-precision Unix timestamps.

## Usage

```bash
python scripts/preprocess_index.py --input <INPUT_PATH> --output <OUTPUT_PATH> [OPTIONS]
```

### Arguments

- `--input`: Path to input CSV file or directory containing CSV files (supports recursive search for `.csv` and `.csv.gz`).
- `--output`: Directory where the Parquet dataset will be written.
- `--timezone`: Timezone of the input timestamps (default: `UTC`). See "Timezone Caveat" below.
- `--partition`: Partitioning strategy (`none`, `ticker`, `date`, `ticker_date`). Default: `none`.
- `--compression`: Compression codec for Parquet files (default: `zstd`).
- `--chunksize`: Number of rows to process at a time (default: 100,000).
- `--verify`: Run sanity checks on the output data (sorting, duplicates).
- `--smoke-test`: Run a quick test on the first file/N rows without processing the entire dataset.

### Example

Process US market data (NY time) and partition by ticker:

```bash
python scripts/preprocess_index.py \
  --input datasets/INDEX \
  --output datasets/INDEX_parquet \
  --timezone "America/New_York" \
  --partition ticker
```

## Timezone Caveat

The input timestamps are parsed as **naive** datetimes (no offset). You **must** specify the correct `--timezone` corresponding to the data source.

- If your data is already in UTC, use `--timezone UTC`.
- If your data is in US Eastern Time (e.g., standard US market data), use `--timezone America/New_York`.

The script performs the following conversion:
1. Parse string to naive datetime.
2. Localize to the specified `--timezone`.
3. Convert to UTC.
4. Store as `int64` nanoseconds since epoch.

## Output Format

The output is a Parquet dataset with the following columns:

- `ticker` (string)
- `ts_ns` (int64, epoch nanoseconds)
- `price` (float32)

If partitioning is enabled, the directory structure will reflect the partition keys (e.g., `output/ticker=AAPL/part-0.parquet`).
