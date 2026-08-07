#!/usr/bin/env python3
"""Report custom emoji usage for a guild from the persistent counter table.

Counts live in the ``emoji_usage`` table in chat_history.db and are updated as
messages and reactions happen, so this is a table lookup rather than a scan of
every message ever recorded.

Two caveats worth relaying to users:
  * ``source=message`` counts can be rebuilt at any time with --backfill.
  * ``source=reaction`` counts start from when the reaction handlers shipped —
    Discord exposes no reaction history to replay, so old reactions are absent.

Examples:
  emoji_stats.py --guild-id 363154169294618625
  emoji_stats.py --guild-id 363154169294618625 --source reaction --limit 10
  emoji_stats.py --guild-id 363154169294618625 --unused
  emoji_stats.py --guild-id 363154169294618625 --backfill
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient
from discord._permissions import require_permission
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_DIR)


def fetch_guild_emojis(guild_id):
    """The guild's current custom emoji set, keyed by id (same call as list_emojis.py)."""
    client = DiscordClient()
    emojis = client.get(f"/guilds/{guild_id}/emojis")
    return {
        e["id"]: {
            "id": e["id"],
            "name": e["name"],
            "animated": bool(e.get("animated")),
        }
        for e in emojis
    }


def do_backfill(guild_id):
    """Recount source='message' rows from the messages table.

    Idempotent: the guild's existing message-sourced rows are dropped first,
    so running this twice does not double the counts. Reaction rows are left
    alone — they cannot be reconstructed.
    """
    from chat_history import CUSTOM_EMOJI_RE, _get_conn

    conn = _get_conn()
    conn.execute(
        "DELETE FROM emoji_usage WHERE guild_id = ? AND source = 'message'",
        (str(guild_id),),
    )

    # (emoji_id) -> [name, animated, count, last_used_at]
    tally = {}
    scanned = 0
    cursor = conn.execute(
        "SELECT content, created_at FROM messages WHERE guild_id = ? ORDER BY id ASC",
        (str(guild_id),),
    )
    for row in cursor:
        scanned += 1
        content = row["content"] or ""
        if "<" not in content:
            continue
        for animated_flag, name, emoji_id in CUSTOM_EMOJI_RE.findall(content):
            entry = tally.setdefault(emoji_id, [name, 0, 0, None])
            entry[0] = name
            entry[1] = 1 if animated_flag == "a" else 0
            entry[2] += 1
            created = row["created_at"]
            if created and (entry[3] is None or created > entry[3]):
                entry[3] = created

    conn.executemany(
        """INSERT INTO emoji_usage
               (guild_id, emoji_id, emoji_name, animated, source, count, last_used_at)
           VALUES (?, ?, ?, ?, 'message', ?, ?)""",
        [
            (str(guild_id), emoji_id, name, animated, count, last_used)
            for emoji_id, (name, animated, count, last_used) in tally.items()
        ],
    )
    conn.commit()

    return {
        "messages_scanned": scanned,
        "emoji_rows_written": len(tally),
        "total_uses_counted": sum(v[2] for v in tally.values()),
    }


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('discord')
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument(
        "--source", choices=["message", "reaction", "all"], default="all",
        help="Count messages, reactions, or both summed (default: all)",
    )
    parser.add_argument(
        "--unused", action="store_true",
        help="List the guild's custom emojis with a zero count instead of usage rows",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max rows to return (default 0 = no limit)",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Recount source='message' rows from the stored message history (idempotent)",
    )
    args = parser.parse_args()

    require_permission("VIEW_CHANNEL", guild_id=args.guild_id)

    from chat_history import get_emoji_usage

    if args.backfill:
        summary = do_backfill(args.guild_id)
        output({"guild_id": args.guild_id, "backfilled": True, **summary})

    source = None if args.source == "all" else args.source
    rows = get_emoji_usage(args.guild_id, source=source)

    if args.unused:
        guild_emojis = fetch_guild_emojis(args.guild_id)
        counted = {r["emoji_id"] for r in rows if (r["count"] or 0) > 0}
        unused = [
            {
                "name": e["name"],
                "id": e["id"],
                "animated": e["animated"],
                "count": 0,
                "last_used_at": None,
            }
            for eid, e in guild_emojis.items()
            if eid not in counted
        ]
        unused.sort(key=lambda e: (e["name"] or "").lower())
        unused_total = len(unused)
        if args.limit > 0:
            unused = unused[:args.limit]
        output({
            "guild_id": args.guild_id,
            "source": args.source,
            "guild_emoji_count": len(guild_emojis),
            "unused_count": unused_total,
            "returned": len(unused),
            "emojis": unused,
        })

    formatted = [
        {
            "name": r["emoji_name"],
            "id": r["emoji_id"],
            "animated": bool(r["animated"]),
            "count": r["count"] or 0,
            "last_used_at": r["last_used_at"],
        }
        for r in rows
    ]
    if args.limit > 0:
        formatted = formatted[:args.limit]

    output({
        "guild_id": args.guild_id,
        "source": args.source,
        "count": len(formatted),
        "emojis": formatted,
    })


if __name__ == "__main__":
    main()
