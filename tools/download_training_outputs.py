#!/usr/bin/env python3
"""
Download training outputs (checkpoints, logs, reports) from R2.

Usage:
    python tools/download_training_outputs.py --run-id run_20260318_143022
    python tools/download_training_outputs.py --run-id experiment_vix --checkpoint-only
    python tools/download_training_outputs.py --list-runs
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from r2_index.config import BUCKET_NAME, get_s3_client


def list_available_runs(s3_client) -> List[str]:
    """List all available training runs in R2."""
    prefix = "training_outputs/"
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
        
        runs = []
        for page in pages:
            for common_prefix in page.get('CommonPrefixes', []):
                run_path = common_prefix['Prefix']
                run_id = run_path.replace(prefix, '').rstrip('/')
                if run_id:
                    runs.append(run_id)
        
        return sorted(runs, reverse=True)
    except Exception as e:
        print(f"❌ Failed to list runs: {e}")
        return []


def download_file(s3_client, s3_key: str, local_path: Path, verbose: bool = True) -> bool:
    """Download a single file from R2."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file size for display
        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        size_mb = head['ContentLength'] / 1e6
        
        if verbose:
            print(f"  ⬇️  {s3_key.split('/')[-1]} ({size_mb:.1f} MB) -> {local_path}")
        
        s3_client.download_file(BUCKET_NAME, s3_key, str(local_path))
        return True
    except Exception as e:
        print(f"  ❌ Failed to download {s3_key}: {e}")
        return False


def download_training_outputs(
    run_id: str,
    output_dir: Optional[Path] = None,
    checkpoint_only: bool = False,
    logs_only: bool = False,
    reports_only: bool = False,
    verbose: bool = True,
) -> int:
    """Download training outputs from R2.
    
    Returns:
        Number of files downloaded successfully.
    """
    s3_client = get_s3_client()
    
    # Default output directory
    if output_dir is None:
        output_dir = Path.cwd() / f"downloaded_outputs_{run_id}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_prefix = f"training_outputs/{run_id}"
    
    downloaded_count = 0
    total_size = 0
    
    # List all files under this run
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=base_prefix)
        
        all_files = []
        for page in pages:
            for obj in page.get('Contents', []):
                all_files.append(obj['Key'])
        
        if not all_files:
            print(f"❌ No files found for run_id: {run_id}")
            print(f"\n💡 Use --list-runs to see available runs")
            return 0
        
        print(f"\n📦 Found {len(all_files)} file(s) for run: {run_id}")
        
        # Filter by type
        files_to_download = []
        for s3_key in all_files:
            relative_path = s3_key.replace(f"{base_prefix}/", "")
            
            if checkpoint_only and not relative_path.startswith("checkpoints/"):
                continue
            if logs_only and not relative_path.startswith("logs/"):
                continue
            if reports_only and not (relative_path.startswith("reports/") or 
                                    relative_path.startswith("results/") or 
                                    relative_path.startswith("metrics/")):
                continue
            
            files_to_download.append((s3_key, relative_path))
        
        if not files_to_download:
            print(f"⚠️  No files match the specified filters")
            return 0
        
        print(f"\n⬇️  Downloading {len(files_to_download)} file(s) to: {output_dir}")
        
        # Download files
        for s3_key, relative_path in files_to_download:
            local_path = output_dir / relative_path
            if download_file(s3_client, s3_key, local_path, verbose):
                downloaded_count += 1
                head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
                total_size += head['ContentLength']
        
        # Summary
        total_size_mb = total_size / 1e6
        print(f"\n✅ Download complete!")
        print(f"   Files downloaded: {downloaded_count}")
        print(f"   Total size: {total_size_mb:.1f} MB")
        print(f"   Output directory: {output_dir}")
        
        return downloaded_count
        
    except Exception as e:
        print(f"❌ Error downloading files: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download training outputs from R2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available runs
  python tools/download_training_outputs.py --list-runs
  
  # Download everything for a specific run
  python tools/download_training_outputs.py --run-id run_20260318_143022
  
  # Download only checkpoints
  python tools/download_training_outputs.py --run-id experiment_vix --checkpoint-only
  
  # Download to custom directory
  python tools/download_training_outputs.py --run-id run_20260318_143022 --output-dir ./my_results
        """
    )
    
    parser.add_argument('--run-id', type=str,
                        help='Run identifier to download')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (default: ./downloaded_outputs_<run_id>)')
    parser.add_argument('--checkpoint-only', action='store_true',
                        help='Download only checkpoints')
    parser.add_argument('--logs-only', action='store_true',
                        help='Download only logs')
    parser.add_argument('--reports-only', action='store_true',
                        help='Download only reports/results')
    parser.add_argument('--list-runs', action='store_true',
                        help='List all available training runs')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-file output')
    
    args = parser.parse_args()
    
    s3_client = get_s3_client()
    
    # List runs mode
    if args.list_runs:
        print("\n📋 Available training runs in R2:")
        runs = list_available_runs(s3_client)
        if runs:
            for run in runs:
                print(f"  • {run}")
            print(f"\n💡 To download a run:")
            print(f"   python tools/download_training_outputs.py --run-id {runs[0]}")
        else:
            print("  (none found)")
        sys.exit(0)
    
    # Download mode
    if not args.run_id:
        parser.error("--run-id is required (or use --list-runs to see available runs)")
    
    downloaded = download_training_outputs(
        run_id=args.run_id,
        output_dir=args.output_dir,
        checkpoint_only=args.checkpoint_only,
        logs_only=args.logs_only,
        reports_only=args.reports_only,
        verbose=not args.quiet,
    )
    
    sys.exit(0 if downloaded > 0 else 1)


if __name__ == '__main__':
    main()
