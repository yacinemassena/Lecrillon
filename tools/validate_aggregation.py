#!/usr/bin/env python3
"""
Full validation audit for aggregated 2-minute bar datasets.

Checks:
1. File counts: Compare 1s vs 2min directories
2. Missing days: Find days in 1s that are missing from 2min
3. Empty columns: Detect columns with all NaN/zero values
4. Data integrity: NaN/Inf checks, volume consistency
5. Column counts: Verify expected schema
6. Folder sizes: Total size of each dataset
7. Sample statistics: Min/max/mean for key columns

Usage:
    python tools/validate_aggregation.py
    python tools/validate_aggregation.py --stock-only
    python tools/validate_aggregation.py --options-only
    python tools/validate_aggregation.py --sample 100  # Check 100 random files
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def get_folder_size(path: Path) -> Tuple[int, int]:
    """Return (total_bytes, file_count) for a directory."""
    total_size = 0
    file_count = 0
    for f in path.glob('*.parquet'):
        if not f.name.startswith('._'):
            total_size += f.stat().st_size
            file_count += 1
    return total_size, file_count


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def get_parquet_dates(directory: Path) -> Set[str]:
    """Get set of date strings from parquet filenames."""
    dates = set()
    for f in directory.glob('*.parquet'):
        if not f.name.startswith('._'):
            # Extract date from filename like 2024-01-02.parquet
            stem = f.stem
            try:
                datetime.strptime(stem, '%Y-%m-%d')
                dates.add(stem)
            except ValueError:
                pass
    return dates


def check_empty_columns(file_path: Path) -> List[str]:
    """Check for columns that are entirely NaN or zero."""
    empty_cols = []
    try:
        con = duckdb.connect()
        
        # Get column names
        cols = con.execute(f"""
            SELECT column_name 
            FROM (DESCRIBE SELECT * FROM read_parquet('{file_path.as_posix()}'))
        """).fetchall()
        col_names = [c[0] for c in cols]
        
        # Check each numeric column
        for col in col_names:
            if col in ['ticker', 'underlying', 'bar_timestamp']:
                continue
            
            result = con.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN "{col}" IS NULL OR "{col}" != "{col}" THEN 1 END) as null_count,
                    COUNT(CASE WHEN "{col}" = 0 THEN 1 END) as zero_count
                FROM read_parquet('{file_path.as_posix()}')
            """).fetchone()
            
            total, null_count, zero_count = result
            if total > 0:
                if null_count == total:
                    empty_cols.append(f"{col} (all NaN)")
                elif zero_count == total:
                    empty_cols.append(f"{col} (all zero)")
        
        con.close()
    except Exception as e:
        empty_cols.append(f"ERROR: {e}")
    
    return empty_cols


def check_file_integrity(file_path: Path, expected_cols: int, id_col: str = 'ticker') -> Dict:
    """Check a single file for data integrity issues."""
    issues = {
        'path': file_path.name,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        con = duckdb.connect()
        
        # Get basic stats
        result = con.execute(f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT "{id_col}") as ticker_count
            FROM read_parquet('{file_path.as_posix()}')
        """).fetchone()
        
        row_count, ticker_count = result
        issues['stats']['rows'] = row_count
        issues['stats']['tickers'] = ticker_count
        
        if row_count == 0:
            issues['errors'].append("Empty file")
            con.close()
            return issues
        
        # Check column count
        col_count = con.execute(f"""
            SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM read_parquet('{file_path.as_posix()}'))
        """).fetchone()[0]
        
        if col_count != expected_cols:
            issues['errors'].append(f"Column count mismatch: {col_count} vs expected {expected_cols}")
        
        # Check for NaN/Inf in numeric columns
        cols = con.execute(f"""
            SELECT column_name, column_type
            FROM (DESCRIBE SELECT * FROM read_parquet('{file_path.as_posix()}'))
        """).fetchall()
        
        numeric_cols = [c[0] for c in cols if 'DOUBLE' in c[1] or 'FLOAT' in c[1] or 'INT' in c[1]]
        
        for col in numeric_cols[:10]:  # Check first 10 numeric columns
            try:
                nan_inf = con.execute(f"""
                    SELECT COUNT(*) 
                    FROM read_parquet('{file_path.as_posix()}')
                    WHERE "{col}" != "{col}" OR "{col}" = 'inf' OR "{col}" = '-inf'
                """).fetchone()[0]
                
                if nan_inf > 0:
                    pct = 100 * nan_inf / row_count
                    if pct > 50:
                        issues['errors'].append(f"{col}: {pct:.1f}% NaN/Inf")
                    elif pct > 5:
                        issues['warnings'].append(f"{col}: {pct:.1f}% NaN/Inf")
            except:
                pass
        
        # Check bar count per ticker (should be ~195 for 2min bars)
        bar_stats = con.execute(f"""
            SELECT 
                MIN(bar_count) as min_bars,
                MAX(bar_count) as max_bars,
                AVG(bar_count) as avg_bars
            FROM (
                SELECT "{id_col}", COUNT(*) as bar_count
                FROM read_parquet('{file_path.as_posix()}')
                GROUP BY "{id_col}"
            )
        """).fetchone()
        
        min_bars, max_bars, avg_bars = bar_stats
        issues['stats']['min_bars'] = min_bars
        issues['stats']['max_bars'] = max_bars
        issues['stats']['avg_bars'] = round(avg_bars, 1) if avg_bars else 0
        
        if min_bars and min_bars < 50:
            issues['warnings'].append(f"Some tickers have very few bars: {min_bars}")
        if max_bars and max_bars > 250:
            issues['warnings'].append(f"Some tickers have too many bars: {max_bars}")
        
        con.close()
        
    except Exception as e:
        issues['errors'].append(f"Read error: {e}")
    
    return issues


def validate_volume_consistency(input_dir: Path, output_dir: Path, sample_files: List[Path], 
                                  vol_col: str = 'volume') -> List[str]:
    """Check that volume sums match between 1s and 2min files."""
    mismatches = []
    
    for out_file in tqdm(sample_files, desc="Volume check", unit="file"):
        in_file = input_dir / out_file.name
        if not in_file.exists():
            continue
        
        try:
            con = duckdb.connect()
            
            # Get volume sum from input
            in_vol = con.execute(f"""
                SELECT SUM("{vol_col}") FROM read_parquet('{in_file.as_posix()}')
            """).fetchone()[0]
            
            # Get volume sum from output
            out_vol = con.execute(f"""
                SELECT SUM("{vol_col}") FROM read_parquet('{out_file.as_posix()}')
            """).fetchone()[0]
            
            con.close()
            
            if in_vol and out_vol:
                diff_pct = abs(in_vol - out_vol) / in_vol * 100
                if diff_pct > 0.01:  # More than 0.01% difference
                    mismatches.append(f"{out_file.name}: {diff_pct:.4f}% volume mismatch")
        except Exception as e:
            mismatches.append(f"{out_file.name}: Error - {e}")
    
    return mismatches


def main():
    parser = argparse.ArgumentParser(description='Validate aggregated 2-minute bar datasets')
    parser.add_argument('--stock-only', action='store_true', help='Only validate stock data')
    parser.add_argument('--options-only', action='store_true', help='Only validate options data')
    parser.add_argument('--sample', type=int, default=50, help='Number of files to sample for detailed checks')
    parser.add_argument('--full', action='store_true', help='Check all files (slow)')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    # Define directories
    stock_1s = project_root / 'datasets' / 'Stock_Data_1s'
    stock_2min = project_root / 'datasets' / 'Stock_Data_2min'
    options_1s = project_root / 'datasets' / 'opt_trade_1sec' / 'option_underlying_bars_1s'
    options_2min = project_root / 'datasets' / 'opt_trade_2min'
    
    print("=" * 80)
    print("AGGREGATION VALIDATION AUDIT")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STOCK DATA VALIDATION
    # =========================================================================
    if not args.options_only:
        print("=" * 80)
        print("STOCK DATA (Stock_Data_1s → Stock_Data_2min)")
        print("=" * 80)
        
        # Folder sizes
        if stock_1s.exists():
            size_1s, count_1s = get_folder_size(stock_1s)
            print(f"\n📁 Stock_Data_1s: {format_size(size_1s)} ({count_1s} files)")
        else:
            print(f"\n❌ Stock_Data_1s: Directory not found")
            count_1s = 0
        
        if stock_2min.exists():
            size_2min, count_2min = get_folder_size(stock_2min)
            print(f"📁 Stock_Data_2min: {format_size(size_2min)} ({count_2min} files)")
        else:
            print(f"❌ Stock_Data_2min: Directory not found")
            count_2min = 0
        
        # Missing days
        if stock_1s.exists() and stock_2min.exists():
            dates_1s = get_parquet_dates(stock_1s)
            dates_2min = get_parquet_dates(stock_2min)
            
            missing = sorted(dates_1s - dates_2min)
            extra = sorted(dates_2min - dates_1s)
            
            print(f"\n📊 Date coverage:")
            print(f"   Input dates: {len(dates_1s)}")
            print(f"   Output dates: {len(dates_2min)}")
            
            if missing:
                print(f"\n⚠️  Missing {len(missing)} days in output:")
                for d in missing[:10]:
                    print(f"      {d}")
                if len(missing) > 10:
                    print(f"      ... and {len(missing) - 10} more")
            else:
                print(f"   ✅ All input dates present in output")
            
            if extra:
                print(f"\n⚠️  Extra {len(extra)} days in output (not in input):")
                for d in extra[:5]:
                    print(f"      {d}")
        
        # Sample file integrity checks
        if stock_2min.exists() and count_2min > 0:
            print(f"\n📋 File integrity checks (sampling {min(args.sample, count_2min)} files):")
            
            all_files = sorted([f for f in stock_2min.glob('*.parquet') if not f.name.startswith('._')])
            
            if args.full:
                sample_files = all_files
            else:
                step = max(1, len(all_files) // args.sample)
                sample_files = all_files[::step][:args.sample]
            
            error_count = 0
            warning_count = 0
            all_stats = defaultdict(list)
            
            for f in tqdm(sample_files, desc="Checking files", unit="file"):
                result = check_file_integrity(f, expected_cols=47, id_col='ticker')
                
                if result['errors']:
                    error_count += 1
                    print(f"\n   ❌ {result['path']}: {', '.join(result['errors'])}")
                
                if result['warnings']:
                    warning_count += 1
                
                for k, v in result['stats'].items():
                    all_stats[k].append(v)
            
            print(f"\n   Checked {len(sample_files)} files:")
            print(f"   ✅ {len(sample_files) - error_count} OK")
            if error_count:
                print(f"   ❌ {error_count} with errors")
            if warning_count:
                print(f"   ⚠️  {warning_count} with warnings")
            
            if all_stats['rows']:
                print(f"\n   📈 Statistics across sampled files:")
                print(f"      Rows: {min(all_stats['rows']):,} - {max(all_stats['rows']):,} (avg: {np.mean(all_stats['rows']):,.0f})")
                print(f"      Tickers: {min(all_stats['tickers']):,} - {max(all_stats['tickers']):,}")
                print(f"      Bars/ticker: {min(all_stats['avg_bars']):.0f} - {max(all_stats['avg_bars']):.0f}")
            
            # Empty column check on first file
            print(f"\n📋 Empty column check (first file):")
            empty_cols = check_empty_columns(all_files[0])
            if empty_cols:
                print(f"   ⚠️  Found {len(empty_cols)} empty columns:")
                for col in empty_cols[:10]:
                    print(f"      - {col}")
            else:
                print(f"   ✅ No empty columns detected")
            
            # Volume consistency check
            if stock_1s.exists():
                print(f"\n📋 Volume consistency check (sampling {min(10, len(sample_files))} files):")
                vol_sample = sample_files[:10]
                mismatches = validate_volume_consistency(stock_1s, stock_2min, vol_sample)
                if mismatches:
                    print(f"   ⚠️  Volume mismatches found:")
                    for m in mismatches:
                        print(f"      - {m}")
                else:
                    print(f"   ✅ Volume sums match between 1s and 2min")
    
    # =========================================================================
    # OPTIONS DATA VALIDATION
    # =========================================================================
    if not args.stock_only:
        print("\n")
        print("=" * 80)
        print("OPTIONS DATA (opt_trade_1sec → opt_trade_2min)")
        print("=" * 80)
        
        # Folder sizes
        if options_1s.exists():
            size_1s, count_1s = get_folder_size(options_1s)
            print(f"\n📁 option_underlying_bars_1s: {format_size(size_1s)} ({count_1s} files)")
        else:
            print(f"\n❌ option_underlying_bars_1s: Directory not found")
            count_1s = 0
        
        if options_2min.exists():
            size_2min, count_2min = get_folder_size(options_2min)
            print(f"📁 opt_trade_2min: {format_size(size_2min)} ({count_2min} files)")
        else:
            print(f"❌ opt_trade_2min: Directory not found")
            count_2min = 0
        
        # Missing days
        if options_1s.exists() and options_2min.exists():
            dates_1s = get_parquet_dates(options_1s)
            dates_2min = get_parquet_dates(options_2min)
            
            missing = sorted(dates_1s - dates_2min)
            extra = sorted(dates_2min - dates_1s)
            
            print(f"\n📊 Date coverage:")
            print(f"   Input dates: {len(dates_1s)}")
            print(f"   Output dates: {len(dates_2min)}")
            
            if missing:
                print(f"\n⚠️  Missing {len(missing)} days in output:")
                for d in missing[:10]:
                    print(f"      {d}")
                if len(missing) > 10:
                    print(f"      ... and {len(missing) - 10} more")
            else:
                print(f"   ✅ All input dates present in output")
            
            if extra:
                print(f"\n⚠️  Extra {len(extra)} days in output (not in input):")
                for d in extra[:5]:
                    print(f"      {d}")
        
        # Sample file integrity checks
        if options_2min.exists() and count_2min > 0:
            print(f"\n📋 File integrity checks (sampling {min(args.sample, count_2min)} files):")
            
            all_files = sorted([f for f in options_2min.glob('*.parquet') if not f.name.startswith('._')])
            
            if args.full:
                sample_files = all_files
            else:
                step = max(1, len(all_files) // args.sample)
                sample_files = all_files[::step][:args.sample]
            
            error_count = 0
            warning_count = 0
            all_stats = defaultdict(list)
            
            for f in tqdm(sample_files, desc="Checking files", unit="file"):
                result = check_file_integrity(f, expected_cols=49, id_col='underlying')
                
                if result['errors']:
                    error_count += 1
                    print(f"\n   ❌ {result['path']}: {', '.join(result['errors'])}")
                
                if result['warnings']:
                    warning_count += 1
                
                for k, v in result['stats'].items():
                    all_stats[k].append(v)
            
            print(f"\n   Checked {len(sample_files)} files:")
            print(f"   ✅ {len(sample_files) - error_count} OK")
            if error_count:
                print(f"   ❌ {error_count} with errors")
            if warning_count:
                print(f"   ⚠️  {warning_count} with warnings")
            
            if all_stats['rows']:
                print(f"\n   📈 Statistics across sampled files:")
                print(f"      Rows: {min(all_stats['rows']):,} - {max(all_stats['rows']):,} (avg: {np.mean(all_stats['rows']):,.0f})")
                print(f"      Underlyings: {min(all_stats['tickers']):,} - {max(all_stats['tickers']):,}")
                print(f"      Bars/underlying: {min(all_stats['avg_bars']):.0f} - {max(all_stats['avg_bars']):.0f}")
            
            # Empty column check on first file
            print(f"\n📋 Empty column check (first file):")
            empty_cols = check_empty_columns(all_files[0])
            if empty_cols:
                print(f"   ⚠️  Found {len(empty_cols)} empty columns:")
                for col in empty_cols[:10]:
                    print(f"      - {col}")
            else:
                print(f"   ✅ No empty columns detected")
            
            # Volume consistency check
            if options_1s.exists():
                print(f"\n📋 Volume consistency check (sampling {min(10, len(sample_files))} files):")
                vol_sample = sample_files[:10]
                mismatches = validate_volume_consistency(options_1s, options_2min, vol_sample, vol_col='total_volume')
                if mismatches:
                    print(f"   ⚠️  Volume mismatches found:")
                    for m in mismatches:
                        print(f"      - {m}")
                else:
                    print(f"   ✅ Volume sums match between 1s and 2min")
    
    print("\n")
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
