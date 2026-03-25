#!/usr/bin/env python3
"""Fetch recent messages from a Discord channel.

Returns messages in chronological order (oldest first).

Examples:
  # Last 10 messages (default)
  get_channel_history.py --channel-id 123456789

  # Last 50 messages
  get_channel_history.py --channel-id 123456789 --limit 50

  # Messages before a specific message ID
  get_channel_history.py --channel-id 123456789 --before 987654321

  # Messages from a specific user
  get_channel_history.py --channel-id 123456789 --user-id 118567805678256128
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient


def _format_message(msg):
    """Extract key fields from a Discord message object."""
    return {
        "id": msg["id"],
        "author": msg["author"]["username"],
        "author_id": msg["author"]["id"],
        "content": msg["content"],
        "timestamp": msg["timestamp"],
        "attachments": [a["url"] for a in msg.get("attachments", [])],
        "reply_to": msg.get("referenced_message", {}).get("id") if msg.get("referenced_message") else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--channel-id", required=True,
                        help="Discord channel ID")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of messages to fetch (max 100, default 10)")
    parser.add_argument("--before", default=None,
                        help="Get messages before this message ID")
    parser.add_argument("--after", default=None,
                        help="Get messages after this message ID")
    parser.add_argument("--user-id", default=None,
                        help="Filter to messages from this user ID (client-side filter)")
    args = parser.parse_args()

    if args.limit > 100:
        error("Discord API limits to 100 messages per request")

    client = DiscordClient()

    params = {"limit": args.limit}
    if args.before:
        params["before"] = args.before
    if args.after:
        params["after"] = args.after

    messages = client.get(f"/channels/{args.channel_id}/messages", params)

    # Discord returns newest-first; reverse for chronological order
    messages.reverse()

    formatted = [_format_message(m) for m in messages]

    # Client-side user filter
    if args.user_id:
        formatted = [m for m in formatted if m["author_id"] == args.user_id]

    output({"channel_id": args.channel_id, "count": len(formatted), "messages": formatted})


if __name__ == "__main__":
    main()
