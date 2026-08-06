#!/usr/bin/env python3
"""Remove a role from a Discord guild member.

Examples:
  remove_role.py --guild-id 363154169294618625 --user-id 118567805678256128 --role-id 456789012345
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
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument("--user-id", required=True, help="User ID to remove role from")
    parser.add_argument("--role-id", required=True, help="Role ID to remove")
    args = parser.parse_args()

    require_permission("MANAGE_ROLES", guild_id=args.guild_id)

    client = DiscordClient()
    client.delete(f"/guilds/{args.guild_id}/members/{args.user_id}/roles/{args.role_id}")

    output({
        "success": True,
        "action": "remove_role",
        "guild_id": args.guild_id,
        "user_id": args.user_id,
        "role_id": args.role_id,
    })


if __name__ == "__main__":
    main()
