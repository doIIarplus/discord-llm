#!/usr/bin/env python3
"""Get information about a Discord user.

If --guild-id is provided, returns guild-specific info (nickname, roles, join date).
Otherwise returns basic user info.

Examples:
  # Basic user info
  get_user.py --user-id 118567805678256128

  # Guild member info (includes nickname, roles, join date)
  get_user.py --user-id 118567805678256128 --guild-id 363154169294618625
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
    parser.add_argument("--user-id", required=True, help="Discord user ID")
    parser.add_argument("--guild-id", default=None,
                        help="Guild ID for member-specific info (nickname, roles, join date)")
    args = parser.parse_args()

    require_permission("VIEW_CHANNEL", guild_id=args.guild_id)

    client = DiscordClient()

    if args.guild_id:
        member = client.get(f"/guilds/{args.guild_id}/members/{args.user_id}")
        user = member["user"]
        output({
            "user_id": user["id"],
            "username": user["username"],
            "display_name": user.get("global_name"),
            "nickname": member.get("nick"),
            "roles": member.get("roles", []),
            "joined_at": member.get("joined_at"),
            "avatar": user.get("avatar"),
            "bot": user.get("bot", False),
        })
    else:
        user = client.get(f"/users/{args.user_id}")
        output({
            "user_id": user["id"],
            "username": user["username"],
            "display_name": user.get("global_name"),
            "avatar": user.get("avatar"),
            "bot": user.get("bot", False),
        })


if __name__ == "__main__":
    main()
