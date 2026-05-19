"""Shared store for email-consuming plugins.

Tracks which RFC 2822 Message-IDs have already been processed per plugin,
so plugins can leave emails untouched in the inbox (no delete, no Seen flag)
and still avoid re-processing the same email on every IMAP scan.

Schema is intentionally tiny — one table, plugin-scoped composite key.
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Iterable

_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "email_dedupe.db"
)
_lock = threading.Lock()
_logger = logging.getLogger("email_dedupe")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_emails (
            plugin_name  TEXT NOT NULL,
            message_id   TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            PRIMARY KEY (plugin_name, message_id)
        )
        """
    )
    return conn


def is_processed(plugin: str, message_id: str) -> bool:
    if not message_id:
        return False
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM processed_emails WHERE plugin_name=? AND message_id=?",
                (plugin, message_id),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


def mark_processed(plugin: str, message_id: str) -> None:
    if not message_id:
        return
    ts = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO processed_emails "
                "(plugin_name, message_id, processed_at) VALUES (?, ?, ?)",
                (plugin, message_id, ts),
            )
            conn.commit()
        finally:
            conn.close()


def bulk_mark_processed(plugin: str, message_ids: Iterable[str]) -> int:
    """Mark many message IDs at once. Returns count of newly-inserted rows."""
    ts = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with _lock:
        conn = _get_conn()
        try:
            for mid in message_ids:
                if not mid:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO processed_emails "
                    "(plugin_name, message_id, processed_at) VALUES (?, ?, ?)",
                    (plugin, mid, ts),
                )
                inserted += cur.rowcount
            conn.commit()
        finally:
            conn.close()
    return inserted


def has_any(plugin: str) -> bool:
    """True if this plugin has ever marked an email processed."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM processed_emails WHERE plugin_name=? LIMIT 1",
                (plugin,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


def count(plugin: str) -> int:
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM processed_emails WHERE plugin_name=?",
                (plugin,),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
