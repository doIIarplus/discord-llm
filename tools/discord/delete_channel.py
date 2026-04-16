#!/usr/bin/env python3
"""Delete a Discord channel.

This is irreversible. Requires Manage Channels permission.

Examples:
  delete_channel.py --channel-id 123456789
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
    parser.add_argument("--channel-id", required=True, help="Channel ID to delete")
    args = parser.parse_args()

    client = DiscordClient()
    result = client.delete(f"/channels/{args.channel_id}")

    output({
        "success": True,
        "action": "delete_channel",
        "channel_id": args.channel_id,
        "name": result.get("name") if result else None,
    })


if __name__ == "__main__":
    main()
