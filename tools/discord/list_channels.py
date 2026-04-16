#!/usr/bin/env python3
"""List channels in a Discord guild.

Returns text, voice, category, and other channel types. Use --type to filter.

Channel type IDs: 0=text, 2=voice, 4=category, 5=announcement, 13=stage, 15=forum

Examples:
  # All channels
  list_channels.py --guild-id 363154169294618625

  # Text channels only
  list_channels.py --guild-id 363154169294618625 --type 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient

CHANNEL_TYPE_NAMES = {
    0: "text",
    2: "voice",
    4: "category",
    5: "announcement",
    13: "stage",
    15: "forum",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument("--type", type=int, default=None,
                        help="Filter by channel type (0=text, 2=voice, 4=category)")
    args = parser.parse_args()

    client = DiscordClient()
    channels = client.get(f"/guilds/{args.guild_id}/channels")

    if args.type is not None:
        channels = [c for c in channels if c["type"] == args.type]

    # Sort by position within their category
    channels.sort(key=lambda c: (c.get("parent_id") or "", c.get("position", 0)))

    formatted = [{
        "id": c["id"],
        "name": c["name"],
        "type": CHANNEL_TYPE_NAMES.get(c["type"], c["type"]),
        "parent_id": c.get("parent_id"),
        "position": c.get("position"),
        "topic": c.get("topic"),
    } for c in channels]

    output({"guild_id": args.guild_id, "count": len(formatted), "channels": formatted})


if __name__ == "__main__":
    main()
