"""Persistent chat history and memory system.

Records all Discord messages to SQLite for long-term memory.
Provides memory context (user profiles + server events) for prompt injection.
"""

import aiohttp
import asyncio
import base64
import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple

from config import PROJECT_DIR, MEMORY_CHANNEL_ALLOWLIST, IMAGE_RECOGNITION_MODEL, OLLAMA_API_URL, VISION_MODEL_CTX

ALIASES_PATH = os.path.join(PROJECT_DIR, "user_aliases.json")


def get_user_aliases() -> dict:
    """Load user ID -> alias info from user_aliases.json.

    Returns dict of user_id -> {"real": "...", "preferred": "..."}.
    Handles both old format (str) and new format (dict).
    """
    try:
        with open(ALIASES_PATH) as f:
            data = json.load(f)
        aliases = data.get("aliases", {})
        # Normalize old format (plain string) to new format
        for uid, val in aliases.items():
            if isinstance(val, str):
                aliases[uid] = {"real": val, "preferred": val}
        return aliases
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

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
            image_summary TEXT,
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
            friendliness_score REAL DEFAULT 0.0,
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

        CREATE TABLE IF NOT EXISTS channel_names (
            guild_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            channel_name TEXT NOT NULL,
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
        cursor = conn.execute(
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
        if cursor.rowcount == 0:
            print(f"[chat_history] Message {message_id} already exists (INSERT OR IGNORE)")
        conn.commit()
    except sqlite3.Error as e:
        print(f"[chat_history] Error recording message {message_id}: {e}")


def _update_image_summary_sync(message_id: str, image_summary: str):
    """Update the image_summary column for a message."""
    conn = _get_conn()
    conn.execute(
        "UPDATE messages SET image_summary = ? WHERE message_id = ?",
        (image_summary, message_id),
    )
    conn.commit()


def _update_channel_name_sync(guild_id: str, channel_id: str, channel_name: str):
    """Update the channel_names lookup table."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO channel_names (guild_id, channel_id, channel_name, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(guild_id, channel_id) DO UPDATE SET
               channel_name = excluded.channel_name,
               updated_at = excluded.updated_at""",
        (guild_id, channel_id, channel_name, now),
    )
    conn.commit()


def get_channel_name(guild_id: str, channel_id: str) -> Optional[str]:
    """Look up a channel name from the cache."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT channel_name FROM channel_names WHERE guild_id = ? AND channel_id = ?",
        (guild_id, channel_id),
    ).fetchone()
    return row["channel_name"] if row else None


async def record_message(message) -> None:
    """Record a discord.Message to the chat history DB.

    Safe to call on every message — uses INSERT OR IGNORE so duplicates are skipped.
    Respects MEMORY_CHANNEL_ALLOWLIST: if set, only records messages from those channels.
    """
    channel_id_str = str(message.channel.id)
    if MEMORY_CHANNEL_ALLOWLIST and channel_id_str not in MEMORY_CHANNEL_ALLOWLIST:
        print(f"[chat_history] Skipping message {message.id} — channel {channel_id_str} not in allowlist")
        return

    # Build attachment info if present
    attachment_info = None
    has_attachments = bool(message.attachments)
    if message.attachments:
        attachment_info = json.dumps([
            {"filename": a.filename, "url": a.url, "content_type": a.content_type}
            for a in message.attachments
        ])

    reply_to = None
    if message.reference and message.reference.message_id:
        reply_to = str(message.reference.message_id)

    content = message.content or ""
    print(f"[chat_history] Recording message {message.id} from {message.author.display_name} "
          f"in {channel_id_str} — content={content[:80]!r} attachments={has_attachments}")

    await asyncio.to_thread(
        _record_message_sync,
        message_id=str(message.id),
        guild_id=str(message.guild.id),
        channel_id=channel_id_str,
        author_id=str(message.author.id),
        author_name=message.author.display_name,
        content=content,
        reply_to_message_id=reply_to,
        has_attachments=has_attachments,
        attachment_info=attachment_info,
        created_at=message.created_at.isoformat(),
    )

    # Cache the channel name from Discord
    channel_name = getattr(message.channel, "name", None)
    if channel_name:
        await asyncio.to_thread(
            _update_channel_name_sync,
            str(message.guild.id), str(message.channel.id), channel_name,
        )

    # Summarize images in the background (non-blocking)
    image_attachments = [
        a for a in message.attachments
        if a.content_type and a.content_type.startswith("image/")
    ]
    if image_attachments:
        print(f"[chat_history] Queuing image summarization for message {message.id} "
              f"({len(image_attachments)} image(s))")
        asyncio.create_task(_summarize_images(str(message.id), image_attachments))


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


async def _summarize_images(message_id: str, attachments) -> None:
    """Download image attachments and summarize them via the vision model.

    Runs as a background task — updates the message's image_summary column
    when complete. Failures are silently ignored (the message still gets
    recorded, just without an image summary).
    """
    try:
        images_b64 = []
        async with aiohttp.ClientSession() as session:
            for att in attachments:
                async with session.get(att.url) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        images_b64.append(base64.b64encode(data).decode("utf-8"))

        if not images_b64:
            return

        prompt = (
            "Briefly describe what is in this image. Focus on the key content — "
            "if it's a screenshot, describe what it shows. If it's a meme, describe "
            "the meme and its meaning. If it's a photo, describe the subject. "
            "Keep it to 1-3 sentences."
        )

        payload = {
            "model": IMAGE_RECOGNITION_MODEL,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"num_ctx": VISION_MODEL_CTX},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
                summary = data.get("response", "").strip()

        if summary:
            await asyncio.to_thread(_update_image_summary_sync, message_id, summary)
            print(f"[chat_history] Image summary for {message_id}: {summary[:100]}")

    except Exception as e:
        print(f"[chat_history] Image summarization failed for {message_id}: {e}")


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
                          content, reply_to_message_id, image_summary, created_at
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
                          content, reply_to_message_id, image_summary, created_at
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
                      content, reply_to_message_id, image_summary, created_at
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
    friendliness_score: Optional[float] = None,
):
    """Insert or update a user profile.

    If friendliness_score is None, the existing score is preserved on update.
    """
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    if friendliness_score is not None:
        conn.execute(
            """INSERT INTO user_profiles
                   (guild_id, user_id, user_name, profile, friendliness_score, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                   user_name = excluded.user_name,
                   profile = excluded.profile,
                   friendliness_score = excluded.friendliness_score,
                   updated_at = excluded.updated_at""",
            (guild_id, user_id, user_name, profile, friendliness_score, now),
        )
    else:
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


def get_user_friendliness(guild_id: str, user_id: str) -> float:
    """Get the friendliness score for a user. Returns 0.0 if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT friendliness_score FROM user_profiles WHERE guild_id = ? AND user_id = ?",
        (guild_id, user_id),
    ).fetchone()
    return float(row["friendliness_score"] or 0.0) if row else 0.0


def update_friendliness_score(guild_id: str, user_id: str, adjustment: float):
    """Apply a friendliness adjustment, clamped to [-10.0, 10.0]."""
    conn = _get_conn()
    current = get_user_friendliness(guild_id, user_id)
    new_score = max(-10.0, min(10.0, current + adjustment))
    conn.execute(
        "UPDATE user_profiles SET friendliness_score = ? WHERE guild_id = ? AND user_id = ?",
        (new_score, guild_id, user_id),
    )
    conn.commit()
    return new_score


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

def _friendliness_label(score: float) -> str:
    """Convert a friendliness score to a human-readable relationship label."""
    if score <= -7:
        return "hostile"
    elif score <= -3:
        return "unfriendly"
    elif score <= -1:
        return "cool"
    elif score <= 1:
        return "neutral"
    elif score <= 3:
        return "friendly"
    elif score <= 7:
        return "good friend"
    else:
        return "close friend"


def get_memory_context(
    guild_id: str,
    channel_id: str = None,
    active_user_ids: List[str] = None,
    max_events: int = 10,
) -> str:
    """Build a memory context string for injection into the LLM prompt.

    Only includes profiles for users in active_user_ids (from recent conversation).
    Only includes the current channel's summary (not all channels).
    Always includes server-wide events.

    Args:
        guild_id: The guild to get memory for.
        channel_id: Current channel — only this channel's summary is included.
        active_user_ids: User IDs from the recent conversation context.
            If None, includes all profiles (backward compat).
        max_events: Max recent server events to include.
    """
    all_profiles = get_user_profiles(guild_id)
    channels = get_channel_summaries(guild_id)
    events = get_recent_events(guild_id, limit=max_events)
    aliases = get_user_aliases()

    # Filter profiles to active users only
    if active_user_ids is not None:
        active_set = set(active_user_ids)
        profiles = [p for p in all_profiles if p["user_id"] in active_set]
    else:
        profiles = all_profiles

    # Filter to current channel only
    current_channel = None
    if channel_id:
        for c in channels:
            if c["channel_id"] == channel_id:
                current_channel = c
                break

    if not profiles and not events and not current_channel:
        return ""

    parts = []

    if profiles:
        lines = ["[Memory — User Profiles]"]
        for p in profiles:
            alias = aliases.get(p["user_id"], {})
            preferred = alias.get("preferred", "")
            real = alias.get("real", "")
            if preferred and real:
                name_label = f"{preferred} (real name: {real})"
            elif real:
                name_label = f"{p['user_name']} (real name: {real})"
            else:
                name_label = p["user_name"]
            score = float(p.get("friendliness_score") or 0.0)
            label = _friendliness_label(score)
            lines.append(f"\n### {name_label} (id={p['user_id']}, relationship: {score:.1f}/10 — {label})")
            lines.append(p["profile"])
        parts.append("\n".join(lines))

    if current_channel:
        lines = ["[Memory — Current Channel]"]
        lines.append(f"\n### #{current_channel['channel_name']}")
        lines.append(current_channel["summary"])
        parts.append("\n".join(lines))

    if events:
        lines = ["[Memory — Recent Server Events]"]
        for e in events:
            date = e["occurred_at"][:10] if e.get("occurred_at") else "unknown"
            lines.append(f"- {date}: {e['event_summary']}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
