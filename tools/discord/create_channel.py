#!/usr/bin/env python3
"""Create a new channel in a Discord guild.

Channel types: 0=text, 2=voice, 4=category

Examples:
  # Text channel
  create_channel.py --guild-id 363154169294618625 --name "new-chat"

  # Voice channel
  create_channel.py --guild-id 363154169294618625 --name "Voice Room" --type 2

  # Text channel under a category
  create_channel.py --guild-id 363154169294618625 --name "project-x" --parent-id 456789012345

  # With a topic
  create_channel.py --guild-id 363154169294618625 --name "announcements" --topic "Important updates only"
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


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('discord')
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument("--name", required=True, help="Channel name")
    parser.add_argument("--type", type=int, default=0,
                        help="Channel type (0=text, 2=voice, 4=category). Default 0")
    parser.add_argument("--parent-id", default=None, help="Parent category channel ID")
    parser.add_argument("--topic", default=None, help="Channel topic (text channels only)")
    args = parser.parse_args()

    require_permission("MANAGE_CHANNELS", guild_id=args.guild_id)

    client = DiscordClient()

    payload = {"name": args.name, "type": args.type}
    if args.parent_id:
        payload["parent_id"] = args.parent_id
    if args.topic:
        payload["topic"] = args.topic

    result = client.post(f"/guilds/{args.guild_id}/channels", payload)

    output({
        "success": True,
        "action": "create_channel",
        "channel_id": result["id"],
        "name": result["name"],
        "type": result["type"],
        "guild_id": args.guild_id,
    })


if __name__ == "__main__":
    main()
