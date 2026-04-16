#!/usr/bin/env python3
"""Edit a message previously sent by the bot.

Only works on messages authored by the bot itself.

Examples:
  edit_message.py --channel-id 123456789 --message-id 987654321 --content "Updated text"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--channel-id", required=True, help="Channel containing the message")
    parser.add_argument("--message-id", required=True, help="Message ID to edit")
    parser.add_argument("--content", required=True, help="New message content")
    args = parser.parse_args()

    client = DiscordClient()
    result = client.patch(f"/channels/{args.channel_id}/messages/{args.message_id}", {
        "content": args.content,
    })

    output({
        "success": True,
        "action": "edit_message",
        "message_id": result["id"],
        "channel_id": result["channel_id"],
        "content": result["content"],
    })


if __name__ == "__main__":
    main()
