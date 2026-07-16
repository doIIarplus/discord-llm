#!/usr/bin/env python3
"""Create a thread in a Discord channel.

Can create a thread from an existing message or a standalone thread in a channel.

Examples:
  # Thread from a message
  create_thread.py --channel-id 123456789 --message-id 987654321 --name "Discussion"

  # Standalone thread (no parent message)
  create_thread.py --channel-id 123456789 --name "General topic" --content "Opening message"

  # Auto-archive after 1 hour (default 24h; options: 60, 1440, 4320, 10080)
  create_thread.py --channel-id 123456789 --name "Quick chat" --content "Hi" --auto-archive 60
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
    parser.add_argument("--channel-id", required=True, help="Parent channel ID")
    parser.add_argument("--name", required=True, help="Thread name")
    parser.add_argument("--message-id", default=None,
                        help="Message ID to create thread from (omit for standalone thread)")
    parser.add_argument("--content", default=None,
                        help="Opening message content (required for standalone threads)")
    parser.add_argument("--auto-archive", type=int, default=1440,
                        help="Auto-archive after N minutes (60, 1440, 4320, 10080). Default 1440")
    args = parser.parse_args()

    if not args.message_id and not args.content:
        error("Standalone threads require --content for the opening message")

    require_permission("CREATE_PUBLIC_THREADS", channel_id=args.channel_id)

    client = DiscordClient()

    if args.message_id:
        # Create thread from existing message
        result = client.post(
            f"/channels/{args.channel_id}/messages/{args.message_id}/threads",
            {"name": args.name, "auto_archive_duration": args.auto_archive},
        )
    else:
        # Create standalone thread
        result = client.post(
            f"/channels/{args.channel_id}/threads",
            {
                "name": args.name,
                "auto_archive_duration": args.auto_archive,
                "type": 11,  # PUBLIC_THREAD
                "message": {"content": args.content},
            },
        )

    output({
        "success": True,
        "action": "create_thread",
        "thread_id": result["id"],
        "thread_name": result["name"],
        "parent_channel_id": args.channel_id,
    })


if __name__ == "__main__":
    main()
