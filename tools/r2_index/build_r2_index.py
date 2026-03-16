#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from .config import BUCKET_NAME, DEFAULT_INDEX_DB_PATH, R2_DATASETS_PREFIX, get_s3_client
    from .db import bulk_upsert_objects, clear_objects, connect_db, get_object_count, set_metadata, utc_now_iso
except ImportError:
    from config import BUCKET_NAME, DEFAULT_INDEX_DB_PATH, R2_DATASETS_PREFIX, get_s3_client
    from db import bulk_upsert_objects, clear_objects, connect_db, get_object_count, set_metadata, utc_now_iso


def main():
    parser = argparse.ArgumentParser(description='Build a local SQLite index of R2 dataset objects')
    parser.add_argument('--db-path', type=Path, default=DEFAULT_INDEX_DB_PATH, help='SQLite DB path')
    parser.add_argument('--prefix', type=str, default=R2_DATASETS_PREFIX, help='R2 prefix to index')
    parser.add_argument('--clear', action='store_true', help='Clear existing indexed objects before rebuilding')
    parser.add_argument('--batch-size', type=int, default=1000, help='SQLite upsert batch size')
    args = parser.parse_args()

    s3 = get_s3_client()
    conn = connect_db(args.db_path)

    if args.clear:
        clear_objects(conn)

    paginator = s3.get_paginator('list_objects_v2')
    indexed_at = utc_now_iso()
    total_seen = 0
    total_written = 0
    batch = []

    print(f'Indexing bucket={BUCKET_NAME} prefix={args.prefix} into {args.db_path}', flush=True)

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=args.prefix):
        for obj in page.get('Contents', []):
            batch.append((
                obj['Key'],
                int(obj['Size']),
                obj.get('ETag'),
                obj.get('LastModified').isoformat() if obj.get('LastModified') else None,
                indexed_at,
                'r2_list',
            ))
            total_seen += 1
            if len(batch) >= args.batch_size:
                total_written += bulk_upsert_objects(conn, batch)
                batch.clear()
                if total_seen % 10000 == 0:
                    print(f'  Indexed {total_seen} objects...', flush=True)

    if batch:
        total_written += bulk_upsert_objects(conn, batch)

    set_metadata(conn, 'last_full_refresh_at', indexed_at)
    set_metadata(conn, 'bucket_name', BUCKET_NAME)
    set_metadata(conn, 'prefix', args.prefix)

    print(f'Indexed {total_written} objects. DB now contains {get_object_count(conn)} rows.', flush=True)


if __name__ == '__main__':
    main()
