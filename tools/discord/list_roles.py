#!/usr/bin/env python3
"""List all roles in a Discord guild.

Returns roles sorted by position (highest first), with IDs, names,
colors, and permissions info.

Examples:
  list_roles.py --guild-id 363154169294618625
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
    args = parser.parse_args()

    client = DiscordClient()
    roles = client.get(f"/guilds/{args.guild_id}/roles")

    # Sort by position descending (highest role first)
    roles.sort(key=lambda r: r.get("position", 0), reverse=True)

    formatted = []
    for r in roles:
        formatted.append({
            "id": r["id"],
            "name": r["name"],
            "color": f"#{r['color']:06x}" if r.get("color") else None,
            "position": r.get("position", 0),
            "mentionable": r.get("mentionable", False),
            "managed": r.get("managed", False),
            "member_count": r.get("member_count"),
        })

    output({"guild_id": args.guild_id, "count": len(formatted), "roles": formatted})


if __name__ == "__main__":
    main()
