"""Persistent chat history and memory system.

Records all Discord messages to SQLite for long-term memory.
Provides memory context (user profiles + server events) for prompt injection.
"""

import asyncio
import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple

from config import PROJECT_DIR, MEMORY_CHANNEL_ALLOWLIST

DB_PATH = os.path.join(PROJECT_DIR, "chat_history.db")

# Module-level connection (lazy-initialized, one per process)
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Get or create the module-level DB connection."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection):
    """Create tables and indexes if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL UNIQUE,
            guild_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            author_id TEXT NOT NULL,
            author_name TEXT NOT NULL,
            content TEXT NOT NULL,
            reply_to_message_id TEXT,
            has_attachments INTEGER DEFAULT 0,
            attachment_info TEXT,
            created_at TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_channel_time
            ON messages(guild_id, channel_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_author
            ON messages(guild_id, author_id, created_at);

        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            UNIQUE(guild_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS server_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id TEXT NOT NULL,
            event_summary TEXT NOT NULL,
            participants TEXT,
            channel_id TEXT,
            occurred_at TEXT NOT NULL,
            source_message_ids TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_guild_time
            ON server_events(guild_id, occurred_at);

        CREATE TABLE IF NOT EXISTS channel_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            channel_name TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            UNIQUE(guild_id, channel_id)
        );

        CREATE TABLE IF NOT EXISTS summarizer_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id TEXT NOT NULL,
            last_processed_message_id TEXT NOT NULL,
            last_processed_at TEXT NOT NULL,
            last_run_at TEXT NOT NULL,
            UNIQUE(guild_id)
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _record_message_sync(
    message_id: str,
    guild_id: str,
    channel_id: str,
    author_id: str,
    author_name: str,
    content: str,
    reply_to_message_id: Optional[str],
    has_attachments: bool,
    attachment_info: Optional[str],
    created_at: str,
):
    """Insert a message into the DB (synchronous, called via to_thread)."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO messages
               (message_id, guild_id, channel_id, author_id, author_name,
                content, reply_to_message_id, has_attachments, attachment_info,
                created_at, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message_id, guild_id, channel_id, author_id, author_name,
                content, reply_to_message_id,
                1 if has_attachments else 0,
                attachment_info,
                created_at,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"[chat_history] Error recording message: {e}")


async def record_message(message) -> None:
    """Record a discord.Message to the chat history DB.

    Safe to call on every message — uses INSERT OR IGNORE so duplicates are skipped.
    Respects MEMORY_CHANNEL_ALLOWLIST: if set, only records messages from those channels.
    """
    if MEMORY_CHANNEL_ALLOWLIST and str(message.channel.id) not in MEMORY_CHANNEL_ALLOWLIST:
        return

    # Build attachment info if present
    attachment_info = None
    if message.attachments:
        attachment_info = json.dumps([
            {"filename": a.filename, "url": a.url, "content_type": a.content_type}
            for a in message.attachments
        ])

    reply_to = None
    if message.reference and message.reference.message_id:
        reply_to = str(message.reference.message_id)

    await asyncio.to_thread(
        _record_message_sync,
        message_id=str(message.id),
        guild_id=str(message.guild.id),
        channel_id=str(message.channel.id),
        author_id=str(message.author.id),
        author_name=message.author.display_name,
        content=message.content or "",
        reply_to_message_id=reply_to,
        has_attachments=bool(message.attachments),
        attachment_info=attachment_info,
        created_at=message.created_at.isoformat(),
    )


async def record_bot_response(
    guild_id: int,
    channel_id: int,
    bot_user_id: int,
    bot_name: str,
    content: str,
    message_id: int,
    reply_to_message_id: Optional[int] = None,
) -> None:
    """Record the bot's own response to the chat history."""
    await asyncio.to_thread(
        _record_message_sync,
        message_id=str(message_id),
        guild_id=str(guild_id),
        channel_id=str(channel_id),
        author_id=str(bot_user_id),
        author_name=bot_name,
        content=content,
        reply_to_message_id=str(reply_to_message_id) if reply_to_message_id else None,
        has_attachments=False,
        attachment_info=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Querying (for the summarizer)
# ---------------------------------------------------------------------------

def get_messages_since(
    guild_id: str,
    since_message_id: Optional[str] = None,
    limit: int = 500,
) -> List[dict]:
    """Get messages since a given message ID, across all channels."""
    conn = _get_conn()
    if since_message_id:
        # Get the rowid of the watermark message, then fetch everything after
        row = conn.execute(
            "SELECT id FROM messages WHERE message_id = ?", (since_message_id,)
        ).fetchone()
        if row:
            cursor = conn.execute(
                """SELECT message_id, guild_id, channel_id, author_id, author_name,
                          content, reply_to_message_id, created_at
                   FROM messages
                   WHERE guild_id = ? AND id > ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (guild_id, row["id"], limit),
            )
        else:
            # Watermark message not found — fetch latest N
            cursor = conn.execute(
                """SELECT message_id, guild_id, channel_id, author_id, author_name,
                          content, reply_to_message_id, created_at
                   FROM messages
                   WHERE guild_id = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (guild_id, limit),
            )
            return [dict(r) for r in cursor.fetchall()][::-1]  # reverse to chronological
    else:
        cursor = conn.execute(
            """SELECT message_id, guild_id, channel_id, author_id, author_name,
                      content, reply_to_message_id, created_at
               FROM messages
               WHERE guild_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (guild_id, limit),
        )
        return [dict(r) for r in cursor.fetchall()][::-1]

    return [dict(r) for r in cursor.fetchall()]


def get_latest_message_time(guild_id: str) -> Optional[str]:
    """Get the created_at timestamp of the most recent message in the guild."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT created_at FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 1",
        (guild_id,),
    ).fetchone()
    return row["created_at"] if row else None


def get_latest_message_id(guild_id: str) -> Optional[str]:
    """Get the message_id of the most recent message in the guild."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT message_id FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 1",
        (guild_id,),
    ).fetchone()
    return row["message_id"] if row else None


def has_new_messages_since(guild_id: str, since_message_id: str) -> bool:
    """Check if there are any messages newer than the given message ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id FROM messages WHERE message_id = ?", (since_message_id,)
    ).fetchone()
    if not row:
        return True  # watermark not found, assume there are new messages
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE guild_id = ? AND id > ?",
        (guild_id, row["id"]),
    ).fetchone()
    return count["cnt"] > 0


# ---------------------------------------------------------------------------
# Summarizer state
# ---------------------------------------------------------------------------

def get_summarizer_state(guild_id: str) -> Optional[dict]:
    """Get the summarizer watermark for a guild."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM summarizer_state WHERE guild_id = ?", (guild_id,)
    ).fetchone()
    return dict(row) if row else None


def update_summarizer_state(guild_id: str, last_message_id: str):
    """Update the summarizer watermark."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO summarizer_state (guild_id, last_processed_message_id, last_processed_at, last_run_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(guild_id) DO UPDATE SET
               last_processed_message_id = excluded.last_processed_message_id,
               last_processed_at = excluded.last_processed_at,
               last_run_at = excluded.last_run_at""",
        (guild_id, last_message_id, now, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# User profiles & events (read/write for summarizer)
# ---------------------------------------------------------------------------

def get_user_profiles(guild_id: str) -> List[dict]:
    """Get all user profiles for a guild."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM user_profiles WHERE guild_id = ? ORDER BY user_name",
        (guild_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def upsert_user_profile(
    guild_id: str,
    user_id: str,
    user_name: str,
    profile: str,
):
    """Insert or update a user profile."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO user_profiles
               (guild_id, user_id, user_name, profile, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(guild_id, user_id) DO UPDATE SET
               user_name = excluded.user_name,
               profile = excluded.profile,
               updated_at = excluded.updated_at""",
        (guild_id, user_id, user_name, profile, now),
    )
    conn.commit()


def insert_server_event(
    guild_id: str,
    event_summary: str,
    participants: Optional[List[str]] = None,
    channel_id: Optional[str] = None,
    occurred_at: Optional[str] = None,
    source_message_ids: Optional[List[str]] = None,
):
    """Insert a server event."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO server_events
               (guild_id, event_summary, participants, channel_id, occurred_at, source_message_ids, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            guild_id,
            event_summary,
            json.dumps(participants) if participants else None,
            channel_id,
            occurred_at or now,
            json.dumps(source_message_ids) if source_message_ids else None,
            now,
        ),
    )
    conn.commit()


def get_recent_events(guild_id: str, limit: int = 10) -> List[dict]:
    """Get the most recent server events."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT event_summary, participants, channel_id, occurred_at
           FROM server_events
           WHERE guild_id = ?
           ORDER BY occurred_at DESC
           LIMIT ?""",
        (guild_id, limit),
    ).fetchall()
    return [dict(r) for r in rows][::-1]  # chronological order


# ---------------------------------------------------------------------------
# Channel summaries (read/write for summarizer)
# ---------------------------------------------------------------------------

def get_channel_summaries(guild_id: str) -> List[dict]:
    """Get all channel summaries for a guild."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM channel_summaries WHERE guild_id = ? ORDER BY channel_name",
        (guild_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def upsert_channel_summary(
    guild_id: str,
    channel_id: str,
    channel_name: str,
    summary: str,
):
    """Insert or update a channel summary."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO channel_summaries
               (guild_id, channel_id, channel_name, summary, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(guild_id, channel_id) DO UPDATE SET
               channel_name = excluded.channel_name,
               summary = excluded.summary,
               updated_at = excluded.updated_at""",
        (guild_id, channel_id, channel_name, summary, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Memory context for prompt injection
# ---------------------------------------------------------------------------

def get_memory_context(guild_id: str, channel_id: str = None, max_events: int = 10) -> str:
    """Build a memory context string for injection into the LLM prompt.

    Returns a compact text block with user profiles, channel summaries,
    and recent events — or empty string if no memory data exists yet.
    If channel_id is provided, the current channel's summary is highlighted.
    """
    profiles = get_user_profiles(guild_id)
    channels = get_channel_summaries(guild_id)
    events = get_recent_events(guild_id, limit=max_events)

    if not profiles and not events and not channels:
        return ""

    parts = []

    if profiles:
        lines = ["[Memory — User Profiles]"]
        for p in profiles:
            lines.append(f"\n### {p['user_name']} (id={p['user_id']})")
            lines.append(p["profile"])
        parts.append("\n".join(lines))

    if channels:
        lines = ["[Memory — Channel Summaries]"]
        for c in channels:
            current = " (CURRENT CHANNEL)" if channel_id and c["channel_id"] == channel_id else ""
            lines.append(f"\n### #{c['channel_name']}{current}")
            lines.append(c["summary"])
        parts.append("\n".join(lines))

    if events:
        lines = ["[Memory — Recent Server Events]"]
        for e in events:
            date = e["occurred_at"][:10] if e.get("occurred_at") else "unknown"
            lines.append(f"- {date}: {e['event_summary']}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
