"""Upload Stock_Data_1s to Cloudflare R2 with real-time progress and speed."""
import boto3
from pathlib import Path
import os
import time
from datetime import datetime, timedelta

# R2 Configuration
R2_ENDPOINT = "https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com"
ACCESS_KEY = "fdfa18bf64b18c61bbee64fda98ca20b"
SECRET_KEY = "394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8"
BUCKET_NAME = "europe"

# Initialize S3 client for R2
s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='auto'
)

class UploadProgressTracker:
    def __init__(self, filename, filesize):
        self.filename = filename
        self.filesize = filesize
        self.uploaded = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.last_uploaded = 0
        
    def __call__(self, bytes_transferred):
        self.uploaded += bytes_transferred
        current_time = time.time()
        
        # Update every 0.5 seconds
        if current_time - self.last_update >= 0.5:
            elapsed = current_time - self.start_time
            percent = (self.uploaded / self.filesize) * 100
            
            # Calculate speed
            speed_bps = self.uploaded / elapsed if elapsed > 0 else 0
            speed_mbps = (speed_bps * 8) / (1024 * 1024)  # Convert to Mbps
            speed_mbs = speed_bps / (1024 * 1024)  # Convert to MB/s
            
            # Calculate ETA
            if speed_bps > 0:
                remaining_bytes = self.filesize - self.uploaded
                eta_seconds = remaining_bytes / speed_bps
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "calculating..."
            
            # Format output
            uploaded_mb = self.uploaded / (1024 * 1024)
            total_mb = self.filesize / (1024 * 1024)
            
            print(f"\r{self.filename}: {uploaded_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%) | "
                  f"Speed: {speed_mbps:.2f} Mbps ({speed_mbs:.2f} MB/s) | ETA: {eta}", 
                  end='', flush=True)
            
            self.last_update = current_time
            self.last_uploaded = self.uploaded

def upload_file(local_path, s3_key):
    """Upload a single file to R2 with progress tracking."""
    file_size = os.path.getsize(local_path)
    filename = Path(local_path).name
    
    print(f"\nUploading: {filename} ({file_size / (1024*1024):.1f} MB)")
    
    tracker = UploadProgressTracker(filename, file_size)
    
    try:
        s3.upload_file(
            str(local_path),
            BUCKET_NAME,
            s3_key,
            Callback=tracker
        )
        print()  # New line after progress
        
        # Calculate final stats
        elapsed = time.time() - tracker.start_time
        avg_speed_mbps = (file_size * 8) / (elapsed * 1024 * 1024) if elapsed > 0 else 0
        
        print(f"✓ Completed in {elapsed:.1f}s (avg: {avg_speed_mbps:.2f} Mbps)")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False

def get_uploaded_files(s3_prefix):
    """Get set of already uploaded files from R2."""
    print("Checking for already uploaded files...")
    uploaded = set()
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Extract just the filename from the full key
                    key = obj['Key']
                    if key.startswith(s3_prefix + '/'):
                        filename = key[len(s3_prefix) + 1:]
                        uploaded.add(filename)
        
        print(f"Found {len(uploaded)} files already uploaded")
    except Exception as e:
        print(f"Warning: Could not check uploaded files: {e}")
        print("Starting fresh upload...")
    
    return uploaded

def upload_directory(local_dir, s3_prefix, start_year=None, end_year=None):
    """Upload directory to R2 with progress tracking.
    
    Args:
        local_dir: Local directory path
        s3_prefix: S3 prefix for uploads
        start_year: Optional start year (inclusive) to filter files
        end_year: Optional end year (inclusive) to filter files
    """
    local_path = Path(local_dir)
    
    if not local_path.exists():
        print(f"Error: Directory not found: {local_dir}")
        return
    
    # Get already uploaded files
    uploaded_files = get_uploaded_files(s3_prefix)
    
    # Collect all files
    all_files = sorted([f for f in local_path.rglob('*') if f.is_file()])
    
    # Filter by year range if specified
    if start_year is not None or end_year is not None:
        filtered_files = []
        for file_path in all_files:
            # Extract year from filename (format: YYYY-MM-DD.parquet)
            filename = file_path.name
            try:
                file_year = int(filename[:4])
                if start_year is not None and file_year < start_year:
                    continue
                if end_year is not None and file_year > end_year:
                    continue
                filtered_files.append(file_path)
            except (ValueError, IndexError):
                # If filename doesn't match expected format, include it
                filtered_files.append(file_path)
        all_files = filtered_files
    
    # Filter out already uploaded files
    files = []
    skipped_count = 0
    skipped_size = 0
    
    for file_path in all_files:
        relative_path = str(file_path.relative_to(local_path)).replace('\\', '/')
        if relative_path in uploaded_files:
            skipped_count += 1
            skipped_size += file_path.stat().st_size
        else:
            files.append(file_path)
    
    total_files = len(files)
    total_size = sum(f.stat().st_size for f in files)
    
    print("=" * 80)
    print(f"Upload Summary")
    print("=" * 80)
    print(f"Source: {local_dir}")
    print(f"Destination: s3://{BUCKET_NAME}/{s3_prefix}/")
    
    if start_year is not None or end_year is not None:
        year_filter = ""
        if start_year is not None and end_year is not None:
            year_filter = f"{start_year}-{end_year}"
        elif start_year is not None:
            year_filter = f"{start_year}+"
        elif end_year is not None:
            year_filter = f"up to {end_year}"
        print(f"Year filter: {year_filter}")
    
    if skipped_count > 0:
        print(f"\nAlready uploaded (skipping): {skipped_count:,} files ({skipped_size / (1024**3):.2f} GB)")
        print(f"Remaining to upload: {total_files:,} files")
    else:
        print(f"\nTotal files: {total_files:,}")
    
    print(f"Total size to upload: {total_size / (1024**3):.2f} GB ({total_size / (1024**4):.2f} TB)")
    print("=" * 80)
    
    # Track overall progress
    overall_start = time.time()
    uploaded_count = 0
    failed_count = 0
    total_uploaded_bytes = 0
    
    for idx, file_path in enumerate(files, 1):
        relative_path = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
        
        print(f"\n[{idx}/{total_files}] ", end='')
        
        file_size = file_path.stat().st_size
        success = upload_file(file_path, s3_key)
        
        if success:
            uploaded_count += 1
            total_uploaded_bytes += file_size
        else:
            failed_count += 1
        
        # Show overall progress
        elapsed = time.time() - overall_start
        overall_percent = (idx / total_files) * 100
        overall_speed_mbps = (total_uploaded_bytes * 8) / (elapsed * 1024 * 1024) if elapsed > 0 else 0
        
        if total_uploaded_bytes > 0 and elapsed > 0:
            avg_speed_bps = total_uploaded_bytes / elapsed
            remaining_bytes = total_size - total_uploaded_bytes
            eta_seconds = remaining_bytes / avg_speed_bps if avg_speed_bps > 0 else 0
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
        
        print(f"\nOverall Progress: {uploaded_count}/{total_files} files ({overall_percent:.1f}%) | "
              f"Avg Speed: {overall_speed_mbps:.2f} Mbps | ETA: {eta}")
        print("-" * 80)
    
    # Final summary
    total_elapsed = time.time() - overall_start
    avg_speed_mbps = (total_uploaded_bytes * 8) / (total_elapsed * 1024 * 1024) if total_elapsed > 0 else 0
    
    print("\n" + "=" * 80)
    print("Upload Complete!")
    print("=" * 80)
    print(f"Uploaded: {uploaded_count}/{total_files} files")
    print(f"Failed: {failed_count} files")
    print(f"Total size: {total_uploaded_bytes / (1024**3):.2f} GB")
    print(f"Total time: {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"Average speed: {avg_speed_mbps:.2f} Mbps ({avg_speed_mbps/8:.2f} MB/s)")
    print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload Stock_Data_1s to Cloudflare R2')
    parser.add_argument('--start-year', type=int, help='Start year (inclusive)')
    parser.add_argument('--end-year', type=int, help='End year (inclusive)')
    parser.add_argument('--local-dir', type=str, default=r"D:\Mamba v2\datasets\Stock_Data_1s",
                        help='Local directory path')
    parser.add_argument('--s3-prefix', type=str, default="datasets/Stock_Data_1s",
                        help='S3 prefix for uploads')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Stock Data Upload to Cloudflare R2")
    print("=" * 80)
    
    upload_directory(args.local_dir, args.s3_prefix, args.start_year, args.end_year)

if __name__ == "__main__":
    main()
