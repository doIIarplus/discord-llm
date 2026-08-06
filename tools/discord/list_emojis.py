#!/usr/bin/env python3
"""List all custom emojis in a Discord guild.

Returns each emoji's name, ID, animated flag, and the mention form used to
post it in a message (``<:name:id>``, or ``<a:name:id>`` when animated).

Examples:
  list_emojis.py --guild-id 363154169294618625
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
    args = parser.parse_args()

    require_permission("VIEW_CHANNEL", guild_id=args.guild_id)

    client = DiscordClient()
    emojis = client.get(f"/guilds/{args.guild_id}/emojis")

    formatted = []
    for e in emojis:
        animated = bool(e.get("animated"))
        prefix = "a" if animated else ""
        formatted.append({
            "id": e["id"],
            "name": e["name"],
            "animated": animated,
            "mention": f"<{prefix}:{e['name']}:{e['id']}>",
            "managed": e.get("managed", False),
            "available": e.get("available", True),
        })

    formatted.sort(key=lambda e: (e["name"] or "").lower())

    output({"guild_id": args.guild_id, "count": len(formatted), "emojis": formatted})


if __name__ == "__main__":
    main()
