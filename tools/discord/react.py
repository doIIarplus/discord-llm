#!/usr/bin/env python3
"""Add a reaction to a Discord message.

For Unicode emoji, pass the emoji directly. For custom emoji, use name:id format.

Examples:
  # Unicode emoji
  react.py --channel-id 123456789 --message-id 987654321 --emoji "👍"

  # Custom guild emoji
  react.py --channel-id 123456789 --message-id 987654321 --emoji "custom_emoji:123456789"
"""

import argparse
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--channel-id", required=True, help="Channel containing the message")
    parser.add_argument("--message-id", required=True, help="Message to react to")
    parser.add_argument("--emoji", required=True,
                        help="Emoji to react with (Unicode char or name:id for custom)")
    args = parser.parse_args()

    client = DiscordClient()

    # URL-encode the emoji for the path
    encoded_emoji = urllib.parse.quote(args.emoji)
    client.put(
        f"/channels/{args.channel_id}/messages/{args.message_id}/reactions/{encoded_emoji}/@me"
    )

    output({
        "success": True,
        "channel_id": args.channel_id,
        "message_id": args.message_id,
        "emoji": args.emoji,
    })


if __name__ == "__main__":
    main()
