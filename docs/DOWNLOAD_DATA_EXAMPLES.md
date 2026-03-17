# Download Data Examples

Complete guide for downloading datasets from Cloudflare R2 storage.

## Quick Reference

```bash
# Download all data types for 2024
python download_data.py --year 2024 --data-type all

# Download specific data type for 2024
python download_data.py --year 2024 --data-type stock
python download_data.py --year 2024 --data-type vix
python download_data.py --year 2024 --data-type options
python download_data.py --year 2024 --data-type news

# Download preprocessed memmaps for fast loading (~25 GB)
python download_data.py --data-type preprocessed

# Download year range (2023-2024)
python download_data.py --start-year 2023 --end-year 2024 --data-type all

# Download everything (all years, all types)
python download_data.py --data-type all
```

## Data Types

The script supports these data type options:

- **`stock`** - 2-minute stock bar data (Stock_Data_2min)
- **`vix`** - VIX data including extended-hours features
- **`options`** - 2-minute options trade data (opt_trade_2min)
- **`news`** - Benzinga daily news embeddings (3072-dim)
- **`macro`** - Enhanced macro data (FED/FRED, 55 features)
- **`gdelt`** - GDELT world event embeddings (391-dim)
- **`econ`** - Economic calendar events
- **`fundamentals`** - Fundamentals cross-attention state (130-dim)
- **`preprocessed`** - Pre-built numpy memmaps for fast loading (~25 GB)
- **`all`** - All of the above
- **`full`** - Mirror entire datasets/ tree

## Year Filtering

**Important**: Year flags apply to ALL data types when using `--data-type all`.

### Single Year

```bash
# Download only 2024 data for all types
python download_data.py --year 2024 --data-type all

# Download only 2024 stock data
python download_data.py --year 2024 --data-type stock

# Download only 2024 options data
python download_data.py --year 2024 --data-type options

# Download only 2024 news embeddings
python download_data.py --year 2024 --data-type news
```

### Year Range

```bash
# Download 2023-2024 for all types
python download_data.py --start-year 2023 --end-year 2024 --data-type all

# Download 2020-2024 stock data only
python download_data.py --start-year 2020 --end-year 2024 --data-type stock

# Download 2014-2024 options data
python download_data.py --start-year 2014 --end-year 2024 --data-type options

# Download 2015-2024 news embeddings
python download_data.py --start-year 2015 --end-year 2024 --data-type news
```

### All Years (No Filter)

```bash
# Download all available data (no year filter)
python download_data.py --data-type all

# Download all available stock data
python download_data.py --data-type stock

# Download all available options data
python download_data.py --data-type options

# Download all available news data
python download_data.py --data-type news
```

## Stock Data Examples

```bash
# Download 2024 stock data only
python download_data.py --year 2024 --data-type stock

# Download 2023-2024 stock data
python download_data.py --start-year 2023 --end-year 2024 --data-type stock

# Download all stock data to custom directory
python download_data.py --data-type stock --stock-dir /custom/path/stock

# Force re-download 2024 stock data
python download_data.py --year 2024 --data-type stock --force
```

**Output**: Files saved to `datasets/Stock_Data_2min/YYYY-MM-DD.parquet`

## VIX Data Examples

```bash
# Download all VIX data (small files, usually download all)
python download_data.py --data-type vix

# Download VIX to custom directory
python download_data.py --data-type vix --vix-dir /custom/path/vix

# Force re-download VIX data
python download_data.py --data-type vix --force
```

**Output**: Files saved to `datasets/VIX/*.csv`

## Options Data Examples

Options data includes two subdirectories:
- `option_contract_bars_1s/` - Contract-level bars
- `option_underlying_bars_1s/` - Underlying-level bars (used by data loader)

```bash
# Download 2024 options data
python download_data.py --year 2024 --data-type options

# Download 2014-2024 options data (full history)
python download_data.py --start-year 2014 --end-year 2024 --data-type options

# Download 2023-2024 options data
python download_data.py --start-year 2023 --end-year 2024 --data-type options

# Download all options data to custom directory
python download_data.py --data-type options --options-dir /custom/path/options

# Force re-download 2024 options
python download_data.py --year 2024 --data-type options --force
```

**Output**: Files saved to:
- `datasets/opt_trade_1sec/option_contract_bars_1s/YYYY-MM-DD.parquet`
- `datasets/opt_trade_1sec/option_underlying_bars_1s/YYYY-MM-DD.parquet`

## News Data Examples

News embeddings are stored as yearly files (`YYYY_embedded.parquet`).

```bash
# Download 2024 news embeddings
python download_data.py --year 2024 --data-type news

# Download 2020-2024 news embeddings
python download_data.py --start-year 2020 --end-year 2024 --data-type news

# Download 2014-2024 news embeddings (full history)
python download_data.py --start-year 2014 --end-year 2024 --data-type news

# Download all news data to custom directory
python download_data.py --data-type news --news-dir /custom/path/news

# Force re-download 2024 news
python download_data.py --year 2024 --data-type news --force
```

**Output**: Files saved to `datasets/benzinga_embeddings/YYYY_embedded.parquet`

## Preprocessed Memmaps (Recommended)

Pre-built numpy memmap files enable **~10,000x faster** data loading (20s/sample → 1.7ms/sample).
The training script auto-detects `datasets/preprocessed/` and uses memmap loading automatically.

```bash
# Download preprocessed memmaps (~25 GB, dominated by news embeddings)
python download_data.py --data-type preprocessed

# Or with rclone (faster for large files)
python download_data_fast.py --data-type preprocessed --transfers 32

# Train — auto-detects preprocessed dir, no extra flags needed
python train.py --seq-len 15000
```

**Contents** (8 feeds, all pre-computed):
| Feed | Shape | Size |
|------|-------|------|
| Stock | [1.1M, 50] | 219 MB |
| Options | [541K, 48] | 104 MB |
| VIX | [1.5M, 25] | 148 MB |
| GDELT | [579K, 391] | 906 MB |
| News | [1.9M, 3072] | 23.7 GB |
| Econ | [57K, 10] | 2.3 MB |
| Macro | [9.3K, 55] | 2.0 MB |
| Fundamentals | [3.8K, 130] | 2.0 MB |

To rebuild memmaps from raw parquets:
```bash
python tools/preprocess_dataset.py --data-root datasets
```

## Combined Examples

### Training Setup (2014-2024 for all sources)

```bash
# Download all data from 2014-2024 (when options/news became available)
python download_data.py --start-year 2014 --end-year 2024 --data-type all
```

### Recent Data Only (2024)

```bash
# Download only 2024 data for quick testing
python download_data.py --year 2024 --data-type all
```

### Multi-Source Experiment (2023-2024)

```bash
# Download 2023-2024 for all sources
python download_data.py --start-year 2023 --end-year 2024 --data-type all
```

### Incremental Updates

```bash
# Download only 2025 data (new year update)
python download_data.py --year 2025 --data-type all

# Download only missing 2024 options data
python download_data.py --year 2024 --data-type options
```

## Advanced Options

### Custom Directories

```bash
# Specify custom paths for each data type
python download_data.py --year 2024 --data-type all \
    --stock-dir /data/stock \
    --vix-dir /data/vix \
    --options-dir /data/options \
    --news-dir /data/news
```

### Force Re-download

Use `--force` to re-download files even if they already exist locally:

```bash
# Force re-download all 2024 data
python download_data.py --year 2024 --data-type all --force

# Force re-download specific type
python download_data.py --year 2024 --data-type options --force
```

### Resume Capability

By default, the script skips files that already exist with matching sizes:

```bash
# This will skip already downloaded files
python download_data.py --start-year 2014 --end-year 2024 --data-type all

# If interrupted, just run again - it will resume where it left off
python download_data.py --start-year 2014 --end-year 2024 --data-type all
```

## Data Availability

| Source | Date Range | Path |
|--------|------------|------|
| Stock bars | 2003-present | `datasets/Stock_Data_2min/` |
| VIX | 2005-present | `datasets/VIX/` |
| Options | 2014-06+ | `datasets/opt_trade_2min/` |
| News | 2009-present | `datasets/benzinga_embeddings/` |
| GDELT | 2004-present | `datasets/GDELT/` |
| Macro | 2000-present | `datasets/MACRO/` |
| Econ calendar | 2007-present | `datasets/econ_calendar/` |
| Fundamentals | 2010-present | `datasets/fundamentals/` |
| Preprocessed | All feeds | `datasets/preprocessed/` |

**Recommendation**: For fastest training, download `preprocessed` instead of raw parquets.

## Common Workflows

### Initial Setup (Full History)

```bash
# Download everything from 2014 onwards
python download_data.py --start-year 2014 --end-year 2024 --data-type all
```

### Quick Test (Recent Data)

```bash
# Download only 2024 for quick testing
python download_data.py --year 2024 --data-type all
```

### Baseline Only (Stock + VIX)

```bash
# Download only stock and VIX for baseline experiments
python download_data.py --start-year 2014 --end-year 2024 --data-type stock
python download_data.py --data-type vix
```

### Add News Later

```bash
# Already have stock/vix, now add news
python download_data.py --start-year 2014 --end-year 2024 --data-type news
```

### Add Options Later

```bash
# Already have stock/vix/news, now add options
python download_data.py --start-year 2014 --end-year 2024 --data-type options
```

## Troubleshooting

### Check What's Downloaded

```bash
# List stock files
ls -lh datasets/Stock_Data_1s/

# List options files
ls -lh datasets/opt_trade_1sec/option_underlying_bars_1s/

# List news files
ls -lh datasets/benzinga_embeddings/
```

### Re-download Corrupted Files

```bash
# Force re-download specific year
python download_data.py --year 2024 --data-type all --force
```

### Disk Space

Approximate sizes:
- Stock data (all years): ~767 GB
- Options data (all years): ~150 GB
- News data (all years): ~80 GB
- VIX data (all years): ~5 GB
- GDELT (all years): ~30 GB
- Macro: ~10 MB
- Econ calendar: ~5 MB
- Fundamentals: ~10 MB
- **Preprocessed memmaps: ~25 GB** (recommended)

**Tip**: If you only need fast training, download just `preprocessed` (~25 GB) instead of all raw data (~1 TB+).

## Help

```bash
# View all options
python download_data.py --help
```
