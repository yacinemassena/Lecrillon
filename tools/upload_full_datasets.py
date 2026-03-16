#!/usr/bin/env python3
"""
Upload the full local datasets tree to Cloudflare R2 under europe/datasets/.

This includes `datasets/VIX/Vix_features/` along with the rest of the processed
dataset tree, unless explicitly excluded.

Usage:
    python tools/upload_full_datasets.py
    python tools/upload_full_datasets.py --year 2024
    python tools/upload_full_datasets.py --start-year 2023 --end-year 2024
    python tools/upload_full_datasets.py --force
    python tools/upload_full_datasets.py --small-file-threshold-mb 100 --small-file-workers 16
    python tools/upload_full_datasets.py --verbose
    python tools/upload_full_datasets.py --index-db-path /path/to/index.db
    python tools/upload_full_datasets.py --index-max-age-days 30
    python tools/upload_full_datasets.py --disable-index-db
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional, Tuple

from r2_index.config import BUCKET_NAME, DEFAULT_INDEX_DB_PATH, R2_DATASETS_PREFIX, get_s3_client
from r2_index.db import connect_db, get_object_count, has_matching_object, is_index_fresh, upsert_object

DEFAULT_SMALL_FILE_THRESHOLD_MB = 100
DEFAULT_SMALL_FILE_WORKERS = 8
PROGRESS_REPORT_EVERY = 1000
DEFAULT_EXCLUDED_DIRS = {
    'MACRO/sec_data',
    'MACRO/rest_data',
}


def matches_year_filter(relative_path: str,
                        year: Optional[int] = None,
                        start_year: Optional[int] = None,
                        end_year: Optional[int] = None) -> bool:
    if not year and not (start_year and end_year):
        return True

    parts = [p for p in relative_path.replace('\\', '/').split('/') if p]
    filename = parts[-1] if parts else relative_path
    candidate_year = None

    for part in reversed(parts[:-1]):
        if len(part) == 4 and part.isdigit():
            candidate_year = int(part)
            break

    if candidate_year is None:
        for token in (filename.split('-')[0], filename.split('_')[0]):
            if len(token) == 4 and token.isdigit():
                candidate_year = int(token)
                break

    if candidate_year is None:
        return True

    if year:
        return candidate_year == year
    return start_year <= candidate_year <= end_year


def should_skip_upload(s3_client, s3_key: str, local_size: int, force: bool, use_remote_check: bool) -> bool:
    if force:
        return False
    if not use_remote_check:
        return False

    try:
        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        return head['ContentLength'] == local_size
    except Exception:
        return False


def record_index_entry(index_conn, s3_key: str, size_bytes: int, source: str) -> None:
    if index_conn is None:
        return
    upsert_object(index_conn, s3_key=s3_key, size_bytes=size_bytes, source=source)


def upload_one_file(file_path: Path, relative_path: str, force: bool, use_remote_check: bool) -> Tuple[str, str, int]:
    s3_client = get_s3_client()
    s3_key = f"{R2_DATASETS_PREFIX}{relative_path}"
    local_size = file_path.stat().st_size

    if should_skip_upload(s3_client, s3_key, local_size, force, use_remote_check):
        return ('skipped', relative_path, local_size)

    s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
    return ('uploaded', relative_path, local_size)


def upload_full_dataset_tree(s3_client, local_root: Path,
                             year: Optional[int] = None,
                             start_year: Optional[int] = None,
                             end_year: Optional[int] = None,
                             force: bool = False,
                             small_file_threshold_mb: int = DEFAULT_SMALL_FILE_THRESHOLD_MB,
                             small_file_workers: int = DEFAULT_SMALL_FILE_WORKERS,
                             verbose: bool = False,
                             excluded_dirs: Optional[set] = None,
                             index_db_path: Optional[Path] = None,
                             index_max_age_days: int = 15,
                             disable_index_db: bool = False):
    local_root = local_root.resolve()
    if not local_root.exists():
        print(f"❌ Local datasets directory not found: {local_root}", flush=True)
        return 0

    if excluded_dirs is None:
        excluded_dirs = DEFAULT_EXCLUDED_DIRS

    index_conn = None
    index_lookup_ready = False
    effective_index_db_path = Path(index_db_path or DEFAULT_INDEX_DB_PATH)
    if not disable_index_db:
        try:
            index_conn = connect_db(effective_index_db_path)
            if is_index_fresh(index_conn, index_max_age_days):
                index_lookup_ready = True
                print(
                    f"   Using fresh index DB: {effective_index_db_path} ({get_object_count(index_conn)} rows)",
                    flush=True,
                )
            else:
                print(
                    f"   Index DB is missing or older than {index_max_age_days} days: {effective_index_db_path}",
                    flush=True,
                )
                print("   Falling back to R2 existence checks for objects not yet indexed.", flush=True)
        except Exception as e:
            index_conn = None
            print(f"   Failed to open index DB {effective_index_db_path}: {e}", flush=True)
            print("   Falling back to R2 existence checks.", flush=True)

    all_files = sorted(p for p in local_root.rglob('*') if p.is_file())
    upload_candidates = []
    excluded_count = 0
    index_skipped_count = 0
    for file_path in all_files:
        relative_path = file_path.relative_to(local_root).as_posix()
        parent_path = str(Path(relative_path).parent).replace('\\', '/')
        if any(parent_path == excluded_dir or parent_path.startswith(f"{excluded_dir}/")
               for excluded_dir in excluded_dirs):
            excluded_count += 1
            continue
        if matches_year_filter(relative_path, year, start_year, end_year):
            local_size = file_path.stat().st_size
            s3_key = f"{R2_DATASETS_PREFIX}{relative_path}"
            if index_lookup_ready and not force and has_matching_object(index_conn, s3_key, local_size):
                index_skipped_count += 1
                continue
            upload_candidates.append((file_path, relative_path, local_size))

    print(f"🗂️ Uploading full datasets tree to R2...", flush=True)
    print(f"   Local: {local_root}", flush=True)
    print(f"   Bucket: {BUCKET_NAME}", flush=True)
    print(f"   Prefix: {R2_DATASETS_PREFIX}", flush=True)
    print(f"   Files selected: {len(upload_candidates)}", flush=True)
    print(f"   Files excluded by directory filter: {excluded_count}", flush=True)
    print(f"   Files skipped from fresh index DB: {index_skipped_count}", flush=True)
    print(f"   Parallel uploads for files smaller than: {small_file_threshold_mb} MB", flush=True)
    print(f"   Small-file workers: {small_file_workers}", flush=True)

    uploaded_count = 0
    skipped_count = index_skipped_count
    failed_count = 0
    total_uploaded_bytes = 0
    small_file_threshold_bytes = small_file_threshold_mb * 1024 * 1024

    small_files = []
    large_files = []
    for file_path, relative_path, local_size in upload_candidates:
        item = (file_path, relative_path, local_size)
        if local_size < small_file_threshold_bytes:
            small_files.append(item)
        else:
            large_files.append(item)

    print(f"   Small files: {len(small_files)}", flush=True)
    print(f"   Large files: {len(large_files)}", flush=True)

    if small_files:
        print("⚡ Uploading small files in parallel...", flush=True)
        with ThreadPoolExecutor(max_workers=small_file_workers) as executor:
            parallel_completed = 0
            max_in_flight = max(small_file_workers * 4, small_file_workers)
            pending = {}
            small_iter = iter(small_files)

            def submit_next_batch():
                while len(pending) < max_in_flight:
                    try:
                        file_path, relative_path, local_size = next(small_iter)
                    except StopIteration:
                        break
                    future = executor.submit(upload_one_file, file_path, relative_path, force, not index_lookup_ready)
                    pending[future] = (relative_path, local_size)

            submit_next_batch()

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    relative_path, local_size = pending.pop(future)
                    size_mb = local_size / 1e6
                    parallel_completed += 1
                    try:
                        status, _, transferred_size = future.result()
                        if status == 'skipped':
                            skipped_count += 1
                            record_index_entry(index_conn, f"{R2_DATASETS_PREFIX}{relative_path}", transferred_size, 'remote_head')
                            if verbose:
                                print(f"  [skip] {relative_path} ({size_mb:.1f} MB)", flush=True)
                            if parallel_completed % PROGRESS_REPORT_EVERY == 0:
                                print(
                                    f"  Progress: {parallel_completed}/{len(small_files)} small files processed | "
                                    f"uploaded={uploaded_count} skipped={skipped_count} failed={failed_count}",
                                    flush=True,
                                )
                            continue

                        uploaded_count += 1
                        total_uploaded_bytes += transferred_size
                        record_index_entry(index_conn, f"{R2_DATASETS_PREFIX}{relative_path}", transferred_size, 'upload_run')
                        if verbose or uploaded_count <= 20 or parallel_completed % PROGRESS_REPORT_EVERY == 0:
                            print(f"  [{uploaded_count}] {relative_path} ({size_mb:.1f} MB)", flush=True)
                        if parallel_completed % PROGRESS_REPORT_EVERY == 0:
                            print(
                                f"  Progress: {parallel_completed}/{len(small_files)} small files processed | "
                                f"uploaded={uploaded_count} skipped={skipped_count} failed={failed_count}",
                                flush=True,
                            )
                    except Exception as e:
                        failed_count += 1
                        print(f"  ❌ Failed {relative_path}: {e}", flush=True)

                submit_next_batch()

            print(
                f"✅ Small-file phase complete: processed={parallel_completed} uploaded={uploaded_count} "
                f"skipped={skipped_count} failed={failed_count}",
                flush=True,
            )

    if large_files:
        print("📦 Uploading large files sequentially...", flush=True)

    large_completed = 0
    for file_path, relative_path, local_size in large_files:
        s3_key = f"{R2_DATASETS_PREFIX}{relative_path}"
        large_completed += 1

        if should_skip_upload(s3_client, s3_key, local_size, force, not index_lookup_ready):
            skipped_count += 1
            record_index_entry(index_conn, s3_key, local_size, 'remote_head')
            if verbose:
                size_mb = local_size / 1e6
                print(f"  [skip] {relative_path} ({size_mb:.1f} MB)", flush=True)
            if large_completed % 100 == 0:
                print(
                    f"  Large-file progress: {large_completed}/{len(large_files)} processed | "
                    f"uploaded={uploaded_count} skipped={skipped_count} failed={failed_count}",
                    flush=True,
                )
            continue

        size_mb = local_size / 1e6
        print(f"  [{uploaded_count + 1}] {relative_path} ({size_mb:.1f} MB)", flush=True)
        try:
            s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
            uploaded_count += 1
            total_uploaded_bytes += local_size
            record_index_entry(index_conn, s3_key, local_size, 'upload_run')
            if large_completed % 100 == 0:
                print(
                    f"  Large-file progress: {large_completed}/{len(large_files)} processed | "
                    f"uploaded={uploaded_count} skipped={skipped_count} failed={failed_count}",
                    flush=True,
                )
        except Exception as e:
            failed_count += 1
            print(f"  ❌ Failed {relative_path}: {e}", flush=True)

    print(
        f"✅ Upload complete: {uploaded_count} uploaded, {skipped_count} skipped, {failed_count} failed, "
        f"{total_uploaded_bytes / 1e9:.2f} GB transferred",
        flush=True,
    )
    return uploaded_count


def main():
    parser = argparse.ArgumentParser(
        description='Upload the full local datasets tree to Cloudflare R2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/upload_full_datasets.py
  python tools/upload_full_datasets.py --year 2025
  python tools/upload_full_datasets.py --year 2024
  python tools/upload_full_datasets.py --start-year 2023 --end-year 2024
  python tools/upload_full_datasets.py --force
  python tools/upload_full_datasets.py --small-file-threshold-mb 100 --small-file-workers 16
  python tools/upload_full_datasets.py --verbose
  python tools/upload_full_datasets.py --index-db-path /path/to/index.db
  python tools/upload_full_datasets.py --index-max-age-days 30
  python tools/upload_full_datasets.py --disable-index-db

Notes:
  - This uploads processed VIX feature files from datasets/VIX/Vix_features/
    automatically as part of the full datasets tree.
  - Use --year to upload only one year's dated parquet files, including VIX features.
        """
    )

    parser.add_argument('--year', type=int, help='Specific year to upload where detectable')
    parser.add_argument('--start-year', type=int, help='Start year for range upload')
    parser.add_argument('--end-year', type=int, help='End year for range upload')
    parser.add_argument('--force', action='store_true', help='Force re-upload even if remote size matches')
    parser.add_argument('--local-dir', type=Path, default=Path('datasets'), help='Local datasets directory')
    parser.add_argument('--small-file-threshold-mb', type=int, default=DEFAULT_SMALL_FILE_THRESHOLD_MB,
                        help='Upload files smaller than this threshold in parallel (default: 100)')
    parser.add_argument('--small-file-workers', type=int, default=DEFAULT_SMALL_FILE_WORKERS,
                        help='Thread count for small-file parallel uploads (default: 8)')
    parser.add_argument('--verbose', action='store_true', help='Print every uploaded or skipped file')
    parser.add_argument('--exclude', type=str, action='append', default=[],
                        help='Additional directories to exclude (relative to datasets/). Can be repeated.')
    parser.add_argument('--no-default-excludes', action='store_true',
                        help='Disable default excludes (MACRO/sec_data, MACRO/rest_data)')
    parser.add_argument('--index-db-path', type=Path, default=DEFAULT_INDEX_DB_PATH,
                        help='SQLite index DB path used to avoid R2 existence checks')
    parser.add_argument('--index-max-age-days', type=int, default=15,
                        help='Maximum age in days for using the local index DB for skip decisions')
    parser.add_argument('--disable-index-db', action='store_true',
                        help='Disable local index DB usage and use direct R2 checks only')

    args = parser.parse_args()

    if args.year and (args.start_year or args.end_year):
        parser.error('Cannot specify both --year and --start-year/--end-year')
    if (args.start_year and not args.end_year) or (args.end_year and not args.start_year):
        parser.error('Must specify both --start-year and --end-year for range upload')
    if args.start_year and args.end_year and args.start_year > args.end_year:
        parser.error('--start-year must be <= --end-year')

    # Build excluded dirs set
    if args.no_default_excludes:
        excluded_dirs = set(args.exclude)
    else:
        excluded_dirs = DEFAULT_EXCLUDED_DIRS.copy()
        excluded_dirs.update(args.exclude)

    s3 = get_s3_client()
    upload_full_dataset_tree(
        s3_client=s3,
        local_root=args.local_dir,
        year=args.year,
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
        small_file_threshold_mb=args.small_file_threshold_mb,
        small_file_workers=args.small_file_workers,
        verbose=args.verbose,
        excluded_dirs=excluded_dirs,
        index_db_path=args.index_db_path,
        index_max_age_days=args.index_max_age_days,
        disable_index_db=args.disable_index_db,
    )


if __name__ == '__main__':
    main()
