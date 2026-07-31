#!/usr/bin/env python3
"""Delete a message from a Discord channel.

The bot can delete its own messages freely. Deleting other users' messages
requires the Manage Messages permission.

Examples:
  delete_message.py --channel-id 123456789 --message-id 987654321
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient
from discord._permissions import require_permission


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--channel-id", required=True, help="Channel containing the message")
    parser.add_argument("--message-id", required=True, help="Message ID to delete")
    args = parser.parse_args()

    client = DiscordClient()

    # DM (1) and group DM (3) channels have no guild to resolve permissions
    # against. In those channels the bot can only delete its own messages,
    # which needs no Manage Messages permission, so skip the check entirely.
    channel = client.get(f"/channels/{args.channel_id}")
    if channel.get("type") not in (1, 3):
        require_permission("MANAGE_MESSAGES", channel_id=args.channel_id)

    client.delete(f"/channels/{args.channel_id}/messages/{args.message_id}")

    output({
        "success": True,
        "action": "delete_message",
        "channel_id": args.channel_id,
        "message_id": args.message_id,
    })


if __name__ == "__main__":
    main()
