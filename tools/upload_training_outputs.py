#!/usr/bin/env python3
"""
Upload training outputs (checkpoints, logs, reports) to R2.

Usage:
    python tools/upload_training_outputs.py
    python tools/upload_training_outputs.py --run-id my_experiment_20260318
    python tools/upload_training_outputs.py --checkpoint-only
    python tools/upload_training_outputs.py --logs-only
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from r2_index.config import BUCKET_NAME, get_s3_client


def upload_file(s3_client, local_path: Path, s3_key: str, verbose: bool = True) -> bool:
    """Upload a single file to R2."""
    try:
        size_mb = local_path.stat().st_size / 1e6
        if verbose:
            print(f"  ⬆️  {local_path.name} ({size_mb:.1f} MB) -> {s3_key}")
        s3_client.upload_file(str(local_path), BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"  ❌ Failed to upload {local_path}: {e}")
        return False


def upload_training_outputs(
    run_id: Optional[str] = None,
    checkpoint_only: bool = False,
    logs_only: bool = False,
    reports_only: bool = False,
    verbose: bool = True,
) -> int:
    """Upload training outputs to R2.
    
    Returns:
        Number of files uploaded successfully.
    """
    s3_client = get_s3_client()
    
    # Auto-generate run_id if not provided
    if not run_id:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    base_prefix = f"training_outputs/{run_id}"
    
    # Find workspace root
    workspace_root = Path(__file__).parent.parent
    
    uploaded_count = 0
    total_size = 0
    
    # Upload checkpoints (recursively find in subdirectories like checkpoints/best/)
    if not logs_only and not reports_only:
        checkpoints_dir = workspace_root / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.rglob("*.pt")) + list(checkpoints_dir.rglob("*.pth"))
            if checkpoint_files:
                print(f"\n📦 Uploading {len(checkpoint_files)} checkpoint(s)...")
                for ckpt_file in checkpoint_files:
                    # Preserve subdirectory structure (e.g., best/checkpoint.pt)
                    relative_path = ckpt_file.relative_to(checkpoints_dir)
                    s3_key = f"{base_prefix}/checkpoints/{relative_path.as_posix()}"
                    if upload_file(s3_client, ckpt_file, s3_key, verbose):
                        uploaded_count += 1
                        total_size += ckpt_file.stat().st_size
            else:
                print("\n⚠️  No checkpoints found in checkpoints/")
        else:
            print("\n⚠️  checkpoints/ directory not found")
    
    # Upload logs
    if not checkpoint_only and not reports_only:
        logs_dir = workspace_root / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.txt"))
            if log_files:
                print(f"\n📝 Uploading {len(log_files)} log file(s)...")
                for log_file in log_files:
                    s3_key = f"{base_prefix}/logs/{log_file.name}"
                    if upload_file(s3_client, log_file, s3_key, verbose):
                        uploaded_count += 1
                        total_size += log_file.stat().st_size
            else:
                print("\n⚠️  No log files found in logs/")
        else:
            print("\n⚠️  logs/ directory not found")
        
        # Also upload root-level log files (like train_8000_b24.log)
        root_logs = list(workspace_root.glob("*.log")) + list(workspace_root.glob("train_*.log"))
        if root_logs:
            print(f"\n📝 Uploading {len(root_logs)} root-level log file(s)...")
            for log_file in root_logs:
                s3_key = f"{base_prefix}/logs/{log_file.name}"
                if upload_file(s3_client, log_file, s3_key, verbose):
                    uploaded_count += 1
                    total_size += log_file.stat().st_size
    
    # Upload reports/results
    if not checkpoint_only and not logs_only:
        for reports_dir_name in ["reports", "results", "metrics"]:
            reports_dir = workspace_root / reports_dir_name
            if reports_dir.exists():
                report_files = list(reports_dir.rglob("*"))
                report_files = [f for f in report_files if f.is_file()]
                if report_files:
                    print(f"\n📊 Uploading {len(report_files)} file(s) from {reports_dir_name}/...")
                    for report_file in report_files:
                        relative_path = report_file.relative_to(reports_dir)
                        s3_key = f"{base_prefix}/{reports_dir_name}/{relative_path.as_posix()}"
                        if upload_file(s3_client, report_file, s3_key, verbose):
                            uploaded_count += 1
                            total_size += report_file.stat().st_size
    
    # Summary
    total_size_mb = total_size / 1e6
    print(f"\n✅ Upload complete!")
    print(f"   Files uploaded: {uploaded_count}")
    print(f"   Total size: {total_size_mb:.1f} MB")
    print(f"   R2 prefix: {base_prefix}")
    print(f"\n💡 To download on another machine:")
    print(f"   python tools/download_training_outputs.py --run-id {run_id}")
    
    return uploaded_count


def main():
    parser = argparse.ArgumentParser(
        description="Upload training outputs to R2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload everything with auto-generated run ID
  python tools/upload_training_outputs.py
  
  # Upload with custom run ID
  python tools/upload_training_outputs.py --run-id experiment_vix_20260318
  
  # Upload only checkpoints
  python tools/upload_training_outputs.py --checkpoint-only
  
  # Upload only logs
  python tools/upload_training_outputs.py --logs-only
        """
    )
    
    parser.add_argument('--run-id', type=str,
                        help='Run identifier (default: auto-generated timestamp)')
    parser.add_argument('--checkpoint-only', action='store_true',
                        help='Upload only checkpoints')
    parser.add_argument('--logs-only', action='store_true',
                        help='Upload only logs')
    parser.add_argument('--reports-only', action='store_true',
                        help='Upload only reports/results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-file output')
    
    args = parser.parse_args()
    
    uploaded = upload_training_outputs(
        run_id=args.run_id,
        checkpoint_only=args.checkpoint_only,
        logs_only=args.logs_only,
        reports_only=args.reports_only,
        verbose=not args.quiet,
    )
    
    sys.exit(0 if uploaded > 0 else 1)


if __name__ == '__main__':
    main()
