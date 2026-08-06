#!/usr/bin/env python3
"""Rename an existing Discord channel.

Examples:
  rename_channel.py --channel-id 456789012345 --name "new-channel-name"
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
    parser.add_argument("--channel-id", required=True, help="ID of the channel to rename")
    parser.add_argument("--name", required=True, help="New channel name")
    args = parser.parse_args()

    require_permission("MANAGE_CHANNELS", channel_id=args.channel_id)

    client = DiscordClient()

    result = client.patch(f"/channels/{args.channel_id}", {"name": args.name})

    output({
        "success": True,
        "action": "rename_channel",
        "channel_id": result["id"],
        "name": result["name"],
        "type": result["type"],
    })


if __name__ == "__main__":
    main()
