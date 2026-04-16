#!/usr/bin/env python3
"""Set or clear a member's nickname in a Discord guild.

Requires Manage Nicknames permission. Use --clear to remove the nickname.

Examples:
  set_nickname.py --guild-id 363154169294618625 --user-id 118567805678256128 --nickname "Cool Guy"
  set_nickname.py --guild-id 363154169294618625 --user-id 118567805678256128 --clear
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
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument("--user-id", required=True, help="User ID to set nickname for")
    parser.add_argument("--nickname", default=None, help="New nickname")
    parser.add_argument("--clear", action="store_true", help="Remove nickname")
    args = parser.parse_args()

    if not args.nickname and not args.clear:
        error("Provide --nickname or --clear")

    client = DiscordClient()
    nick = None if args.clear else args.nickname
    client.patch(f"/guilds/{args.guild_id}/members/{args.user_id}", {"nick": nick})

    output({
        "success": True,
        "action": "clear_nickname" if args.clear else "set_nickname",
        "guild_id": args.guild_id,
        "user_id": args.user_id,
        "nickname": nick,
    })


if __name__ == "__main__":
    main()
