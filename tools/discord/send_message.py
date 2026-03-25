#!/usr/bin/env python3
"""Send a message to a Discord channel as the bot.

Unlike webhooks, this sends from the bot's own account (avatar, name, permissions).
Supports Discord markdown, mentions (<@USER_ID>), and embeds.

Examples:
  # Simple message
  send_message.py --channel-id 123456789 --content "Hello from the bot!"

  # Mention a user
  send_message.py --channel-id 123456789 --content "<@118567805678256128> reminder: buy cream puffs"

  # Reply to a specific message
  send_message.py --channel-id 123456789 --content "Here's your answer" --reply-to 987654321
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
    parser.add_argument("--channel-id", required=True,
                        help="Discord channel ID to send to")
    parser.add_argument("--content", required=True,
                        help="Message content (supports Discord markdown and <@USER_ID> mentions)")
    parser.add_argument("--reply-to", default=None,
                        help="Message ID to reply to")
    args = parser.parse_args()

    client = DiscordClient()

    payload = {"content": args.content}
    if args.reply_to:
        payload["message_reference"] = {"message_id": args.reply_to}

    result = client.post(f"/channels/{args.channel_id}/messages", payload)
    output({
        "sent": True,
        "message_id": result["id"],
        "channel_id": result["channel_id"],
        "content": result["content"],
    })


if __name__ == "__main__":
    main()
