#!/usr/bin/env python3
"""Pin or unpin a message in a Discord channel.

Examples:
  # Pin a message
  pin_message.py --channel-id 123456789 --message-id 987654321

  # Unpin a message
  pin_message.py --channel-id 123456789 --message-id 987654321 --unpin
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
    parser.add_argument("--channel-id", required=True, help="Channel containing the message")
    parser.add_argument("--message-id", required=True, help="Message to pin/unpin")
    parser.add_argument("--unpin", action="store_true", help="Unpin instead of pin")
    args = parser.parse_args()

    require_permission("MANAGE_MESSAGES", channel_id=args.channel_id)

    client = DiscordClient()

    endpoint = f"/channels/{args.channel_id}/pins/{args.message_id}"
    if args.unpin:
        client.delete(endpoint)
        action = "unpin"
    else:
        client.put(endpoint)
        action = "pin"

    output({
        "success": True,
        "action": action,
        "channel_id": args.channel_id,
        "message_id": args.message_id,
    })


if __name__ == "__main__":
    main()
