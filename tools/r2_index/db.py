from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from .config import DEFAULT_INDEX_DB_PATH
except ImportError:
    from config import DEFAULT_INDEX_DB_PATH


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_INDEX_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    initialize_db(conn)
    return conn


def initialize_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS objects (
            s3_key TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            etag TEXT,
            last_modified TEXT,
            indexed_at TEXT NOT NULL,
            source TEXT
        )
        '''
    )
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        '''
    )
    conn.commit()


def set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        'INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
        (key, value),
    )
    conn.commit()


def get_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
    return row[0] if row else None


def clear_objects(conn: sqlite3.Connection) -> None:
    conn.execute('DELETE FROM objects')
    conn.commit()


def upsert_object(
    conn: sqlite3.Connection,
    s3_key: str,
    size_bytes: int,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    indexed_at: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    timestamp = indexed_at or utc_now_iso()
    conn.execute(
        '''
        INSERT INTO objects(s3_key, size_bytes, etag, last_modified, indexed_at, source)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(s3_key) DO UPDATE SET
            size_bytes = excluded.size_bytes,
            etag = excluded.etag,
            last_modified = excluded.last_modified,
            indexed_at = excluded.indexed_at,
            source = excluded.source
        ''',
        (s3_key, size_bytes, etag, last_modified, timestamp, source),
    )
    conn.commit()


def bulk_upsert_objects(conn: sqlite3.Connection, rows: Iterable[tuple[str, int, Optional[str], Optional[str], str, Optional[str]]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        '''
        INSERT INTO objects(s3_key, size_bytes, etag, last_modified, indexed_at, source)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(s3_key) DO UPDATE SET
            size_bytes = excluded.size_bytes,
            etag = excluded.etag,
            last_modified = excluded.last_modified,
            indexed_at = excluded.indexed_at,
            source = excluded.source
        ''',
        rows,
    )
    conn.commit()
    return len(rows)


def has_matching_object(conn: sqlite3.Connection, s3_key: str, size_bytes: int) -> bool:
    row = conn.execute(
        'SELECT 1 FROM objects WHERE s3_key = ? AND size_bytes = ? LIMIT 1',
        (s3_key, size_bytes),
    ).fetchone()
    return row is not None


def get_object_count(conn: sqlite3.Connection) -> int:
    row = conn.execute('SELECT COUNT(*) FROM objects').fetchone()
    return int(row[0]) if row else 0


def get_last_full_refresh(conn: sqlite3.Connection) -> Optional[datetime]:
    value = get_metadata(conn, 'last_full_refresh_at')
    if not value:
        return None
    return datetime.fromisoformat(value)


def is_index_fresh(conn: sqlite3.Connection, max_age_days: int) -> bool:
    last_refresh = get_last_full_refresh(conn)
    if last_refresh is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    if last_refresh.tzinfo is None:
        last_refresh = last_refresh.replace(tzinfo=timezone.utc)
    return last_refresh >= cutoff
