"""
Validate stream configuration without requiring torch installation.
Quick smoke test for stream-specific batch sizing.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_pretrain import GPU_PROFILES, STREAM_CONFIGS

def validate_config():
    """Validate stream configurations."""
    print("=" * 60)
    print("Stream Configuration Validation")
    print("=" * 60)
    
    # Check GPU profiles
    print("\nGPU Profiles:")
    for name, profile in GPU_PROFILES.items():
        print(f"  {name}: {profile.vram_gb}GB, {profile.max_chunks_per_batch} chunks")
    
    # Check stream configs
    print("\nStream Configurations:")
    for stream_name, stream_config in STREAM_CONFIGS.items():
        print(f"\n  {stream_name.upper()}:")
        print(f"    Path: {stream_config.data_path}")
        print(f"    Filter tickers: {stream_config.filter_tickers}")
        print(f"    Num tickers: {stream_config.num_tickers}")
        print(f"    16GB batch: {stream_config.max_chunks_16gb} chunks")
        print(f"    80GB batch: {stream_config.max_chunks_80gb} chunks")
        print(f"    Prefetch: {stream_config.prefetch_files} files")
        
        # Check if path exists
        path = Path(stream_config.data_path)
        exists = path.exists()
        print(f"    Path exists: {exists}")
        
        if exists:
            if stream_name == 'index':
                # Index has year subdirs
                file_count = sum(1 for year_dir in path.iterdir() 
                               if year_dir.is_dir() 
                               for f in year_dir.glob('*.parquet'))
            else:
                file_count = len(list(path.glob('*.parquet')))
            print(f"    Files found: {file_count}")
    
    # Simulate batch sizing for index on RTX 5080
    print("\n" + "=" * 60)
    print("Smoke Test: INDEX stream on RTX 5080")
    print("=" * 60)
    
    gpu_profile = GPU_PROFILES['rtx5080']
    stream_config = STREAM_CONFIGS['index']
    
    # Determine batch size
    if gpu_profile.vram_gb <= 16:
        max_chunks = stream_config.max_chunks_16gb
    else:
        max_chunks = stream_config.max_chunks_80gb
    
    print(f"GPU: {gpu_profile.name} ({gpu_profile.vram_gb}GB)")
    print(f"Stream: index")
    print(f"Max chunks/batch: {max_chunks} (stream-tuned)")
    print(f"Prefetch files: {stream_config.prefetch_files}")
    print(f"Filter tickers: False")
    
    # Estimate memory usage (rough)
    chunk_len = 256
    dim_in = 3
    bytes_per_float = 4
    chunk_size_mb = (chunk_len * dim_in * bytes_per_float) / (1024 * 1024)
    batch_size_mb = chunk_size_mb * max_chunks
    
    print(f"\nEstimated batch VRAM:")
    print(f"  Chunk size: {chunk_size_mb:.2f} MB")
    print(f"  Batch size: {batch_size_mb:.2f} MB ({batch_size_mb/1024:.2f} GB)")
    print(f"  Available VRAM: {gpu_profile.vram_gb} GB")
    print(f"  Utilization: {(batch_size_mb/1024)/gpu_profile.vram_gb*100:.1f}%")
    
    # Check data path
    index_path = Path(stream_config.data_path)
    if index_path.exists():
        print(f"\n✓ Data path exists: {index_path}")
        
        # Count files
        file_count = sum(1 for year_dir in index_path.iterdir() 
                        if year_dir.is_dir() 
                        for f in year_dir.glob('*.parquet'))
        print(f"✓ Found {file_count} parquet files")
        
        if file_count > 0:
            print("\n✓ Configuration valid - ready for training!")
        else:
            print("\n✗ No data files found")
    else:
        print(f"\n✗ Data path does not exist: {index_path}")
    
    print("=" * 60)

if __name__ == '__main__':
    validate_config()
