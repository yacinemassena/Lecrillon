"""
Check data availability on VPS and generate stream-specific config.

This script:
1. Checks which datasets are available locally
2. Queries R2 to see what's available for download
3. Generates a config file with available streams only
4. Provides download commands for missing data

Usage:
    # On VPS - check what's available
    python check_data_availability.py
    
    # Check and auto-download available data from R2
    python check_data_availability.py --download
    
    # Download specific stream
    python check_data_availability.py --download --stream index
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# R2 credentials (same as in setup_vps.sh)
R2_ENDPOINT = 'https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com'
R2_ACCESS_KEY = 'fdfa18bf64b18c61bbee64fda98ca20b'
R2_SECRET_KEY = '394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8'
R2_BUCKET = 'europe'

# Expected data paths (tries multiple locations)
EXPECTED_PATHS = {
    'index': {
        'local': [
            '/TCN/datasets/2017-2025/index_data_2017_2025/values_v1',
            '/TCN/datasets/index_data_2017_2025/values_v1',
            'D:/Mamba v2/datasets/index_data_2017_2025/values_v1',
        ],
        'r2_prefix': 'datasets/index_data_2017_2025/values_v1/',
        'min_files': 100,
    },
    'stocks': {
        'local': [
            '/TCN/datasets/2017-2025/polygon_stock_trades',
            '/TCN/datasets/polygon_stock_trades',
            'D:/Mamba v2/datasets/polygon_stock_trades',
        ],
        'r2_prefix': 'datasets/polygon_stock_trades/',
        'min_files': 50,
    },
    'options': {
        'local': [
            '/TCN/datasets/2017-2025/options_trades',
            '/TCN/datasets/options_trades',
            'D:/Mamba v2/datasets/options_trades',
        ],
        'r2_prefix': 'datasets/options_trades/',
        'min_files': 50,
    },
    'rv': {
        'local': [
            '/TCN/datasets/2017-2025/SPY_daily_rv/spy_daily_rv_30d.parquet',
            '/TCN/datasets/SPY_daily_rv/spy_daily_rv_30d.parquet',
            'D:/Mamba v2/datasets/SPY_daily_rv/spy_daily_rv_30d.parquet',
        ],
        'r2_prefix': 'datasets/SPY_daily_rv/spy_daily_rv_30d.parquet',
        'min_files': 1,
    },
}


def check_local_data() -> Dict[str, Optional[str]]:
    """Check which datasets are available locally."""
    print("=" * 60)
    print("Checking Local Data Availability")
    print("=" * 60)
    
    available = {}
    
    for stream, config in EXPECTED_PATHS.items():
        found = False
        found_path = None
        
        if stream == 'rv':
            # RV is a single file
            for path in config['local']:
                if Path(path).exists():
                    found = True
                    found_path = path
                    print(f"✓ {stream.upper()}: Found at {path}")
                    break
        else:
            # Data streams are directories
            for path in config['local']:
                if Path(path).exists():
                    files = list(Path(path).glob('**/*.parquet'))
                    if len(files) >= config['min_files']:
                        found = True
                        found_path = path
                        print(f"✓ {stream.upper()}: Found {len(files)} files at {path}")
                        break
        
        if not found:
            print(f"✗ {stream.upper()}: Not found locally")
        
        available[stream] = found_path
    
    return available


def check_r2_data() -> Dict[str, int]:
    """Check what's available on R2."""
    if not HAS_BOTO3:
        print("\n⚠ boto3 not installed - cannot check R2. Install with: pip install boto3")
        return {}
    
    print("\n" + "=" * 60)
    print("Checking R2 Data Availability")
    print("=" * 60)
    
    try:
        s3 = boto3.client('s3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY
        )
        
        available = {}
        
        for stream, config in EXPECTED_PATHS.items():
            prefix = config['r2_prefix']
            
            if stream == 'rv':
                # Check if single file exists
                try:
                    s3.head_object(Bucket=R2_BUCKET, Key=prefix)
                    available[stream] = 1
                    print(f"✓ {stream.upper()}: Available on R2")
                except:
                    available[stream] = 0
                    print(f"✗ {stream.upper()}: Not found on R2")
            else:
                # Count files in directory
                paginator = s3.get_paginator('list_objects_v2')
                count = 0
                for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
                    if 'Contents' in page:
                        count += len([obj for obj in page['Contents'] if not obj['Key'].endswith('/')])
                
                available[stream] = count
                if count > 0:
                    print(f"✓ {stream.upper()}: {count} files available on R2")
                else:
                    print(f"✗ {stream.upper()}: Not found on R2")
        
        return available
    
    except Exception as e:
        print(f"\n⚠ Error checking R2: {e}")
        return {}


def download_from_r2(stream: str, target_dir: str) -> bool:
    """Download a specific stream from R2."""
    if not HAS_BOTO3:
        print("boto3 not installed - cannot download")
        return False
    
    config = EXPECTED_PATHS[stream]
    prefix = config['r2_prefix']
    
    try:
        s3 = boto3.client('s3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY
        )
        
        target_path = Path(target_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if stream == 'rv':
            # Download single file
            print(f"Downloading {stream} file...")
            s3.download_file(R2_BUCKET, prefix, str(target_path))
            print(f"  ✓ Downloaded to {target_path}")
            return True
        else:
            # Download directory
            target_path.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {stream} data...")
            paginator = s3.get_paginator('list_objects_v2')
            count = 0
            
            for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue
                    
                    # Remove prefix to get relative path
                    rel_path = key.replace(prefix, '')
                    local_file = target_path / rel_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    s3.download_file(R2_BUCKET, key, str(local_file))
                    count += 1
                    if count % 10 == 0:
                        print(f"  Downloaded {count} files...")
            
            print(f"  ✓ Downloaded {count} files to {target_path}")
            return True
    
    except Exception as e:
        print(f"  ✗ Error downloading {stream}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Check data availability for TCN training')
    parser.add_argument('--download', action='store_true',
                        help='Auto-download missing data from R2')
    parser.add_argument('--r2-only', action='store_true',
                        help='Only check R2, skip local scan')
    parser.add_argument('--stream', type=str, choices=['index', 'stocks', 'options', 'rv'],
                        help='Download specific stream from R2')
    parser.add_argument('--target-dir', type=str,
                        help='Target directory for downloads (default: /TCN/datasets/2017-2025/)')
    
    args = parser.parse_args()
    
    # Check local data
    if not args.r2_only:
        local_available = check_local_data()
    else:
        local_available = {k: None for k in EXPECTED_PATHS.keys()}
    
    # Check R2 data
    r2_available = check_r2_data()
    
    # Determine what's available
    available_streams = []
    for stream in ['index', 'stocks', 'options', 'rv']:
        if local_available.get(stream):
            available_streams.append(stream)
    
    # Generate config
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    config = {
        'available_streams': available_streams,
        'can_train': {
            'index': 'index' in available_streams and 'rv' in available_streams,
            'stocks': 'stocks' in available_streams and 'rv' in available_streams,
            'options': 'options' in available_streams and 'rv' in available_streams,
        },
        'recommended_stream': None,
    }
    
    # Recommend stream based on availability
    if config['can_train']['index']:
        config['recommended_stream'] = 'index'
    elif config['can_train']['stocks']:
        config['recommended_stream'] = 'stocks'
    elif config['can_train']['options']:
        config['recommended_stream'] = 'options'
    
    print(f"\nAvailable locally: {', '.join(available_streams) if available_streams else 'None'}")
    print(f"\nCan train:")
    for stream, can_train in config['can_train'].items():
        status = "✓" if can_train else "✗"
        print(f"  {status} {stream}")
    
    if config['recommended_stream']:
        print(f"\n✓ Recommended: Train on '{config['recommended_stream']}' stream")
        print(f"  Command: python pretrain_tcn_rv.py --stream {config['recommended_stream']} --profile h100")
    else:
        print("\n⚠ No complete datasets available for training")
        print("  Missing: RV file or at least one data stream")
    
    # Download if requested
    if args.download or args.stream:
        print("\n" + "=" * 60)
        print("Downloading Data from R2")
        print("=" * 60)
        
        # Determine target directory
        if args.target_dir:
            target_base = args.target_dir
        elif os.path.exists('/TCN'):
            target_base = '/TCN/datasets/2017-2025'
        else:
            target_base = 'D:/Mamba v2/datasets'
        
        if args.stream:
            # Download specific stream
            streams_to_download = [args.stream]
        else:
            # Download all missing streams that are available on R2
            streams_to_download = []
            for stream in ['rv', 'index', 'stocks', 'options']:
                if not local_available.get(stream) and r2_available.get(stream, 0) > 0:
                    streams_to_download.append(stream)
        
        for stream in streams_to_download:
            if stream == 'rv':
                target = f"{target_base}/SPY_daily_rv/spy_daily_rv_30d.parquet"
            elif stream == 'index':
                target = f"{target_base}/index_data_2017_2025/values_v1"
            elif stream == 'stocks':
                target = f"{target_base}/polygon_stock_trades"
            elif stream == 'options':
                target = f"{target_base}/options_trades"
            
            download_from_r2(stream, target)
        
        print("\nRe-checking local data after download...")
        local_available = check_local_data()
        available_streams = [s for s in ['index', 'stocks', 'options', 'rv'] if local_available.get(s)]
        
        # Update config
        config['available_streams'] = available_streams
        config['can_train'] = {
            'index': 'index' in available_streams and 'rv' in available_streams,
            'stocks': 'stocks' in available_streams and 'rv' in available_streams,
            'options': 'options' in available_streams and 'rv' in available_streams,
        }
        if config['can_train']['index']:
            config['recommended_stream'] = 'index'
        elif config['can_train']['stocks']:
            config['recommended_stream'] = 'stocks'
        elif config['can_train']['options']:
            config['recommended_stream'] = 'options'
    
    # Save config
    config_file = 'available_streams.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_file}")
    
    print("\n" + "=" * 60)
    
    # Exit with appropriate code
    if config['recommended_stream']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
