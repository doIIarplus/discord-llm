#!/usr/bin/env python3
"""Delete a custom emoji from a Discord guild.

Requires the Manage Expressions permission (a.k.a. Manage Emojis and Stickers).
Identify the emoji either by ID or by name. The full mention form is accepted
for --emoji-id and the numeric ID is parsed out of it.

Deletion is IRREVERSIBLE and deletes exactly one emoji per invocation.

Examples:
  delete_emoji.py --guild-id 363154169294618625 --emoji-id 752012209659314226
  delete_emoji.py --guild-id 363154169294618625 --emoji-id "<:bruh:752012209659314226>"
  delete_emoji.py --guild-id 363154169294618625 --name bruh
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient
from discord._permissions import require_permission
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

# <:name:id> or <a:name:id>
_MENTION_RE = re.compile(r"^<a?:[^:]+:(\d+)>$")


def parse_emoji_id(value):
    """Return the numeric emoji ID from a raw ID or a full mention form."""
    value = value.strip()
    m = _MENTION_RE.match(value)
    if m:
        return m.group(1)
    if value.isdigit():
        return value
    error(
        f"Could not parse an emoji ID from '{value}'. Pass a numeric ID or a "
        f"mention like '<:bruh:752012209659314226>'."
    )


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('discord')
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emoji-id", help="Emoji ID, or full mention form '<:name:id>'")
    group.add_argument("--name", help="Emoji name (must be unambiguous in the guild)")
    args = parser.parse_args()

    require_permission("MANAGE_EXPRESSIONS", guild_id=args.guild_id)

    client = DiscordClient()
    emojis = client.get(f"/guilds/{args.guild_id}/emojis")

    if args.emoji_id:
        emoji_id = parse_emoji_id(args.emoji_id)
        matches = [e for e in emojis if str(e["id"]) == emoji_id]
        if not matches:
            error(
                f"No custom emoji with ID {emoji_id} in guild {args.guild_id}.",
                details={"guild_id": args.guild_id, "emoji_id": emoji_id},
            )
    else:
        matches = [e for e in emojis if e.get("name") == args.name]
        if not matches:
            error(
                f"No custom emoji named '{args.name}' in guild {args.guild_id}.",
                details={"guild_id": args.guild_id, "name": args.name},
            )
        if len(matches) > 1:
            # Ambiguous by name — refuse rather than guess which one to destroy.
            error(
                f"{len(matches)} emojis are named '{args.name}'. Re-run with "
                f"--emoji-id to pick one.",
                details={"candidates": [{"id": e["id"], "name": e["name"]} for e in matches]},
            )

    emoji = matches[0]
    client.delete(f"/guilds/{args.guild_id}/emojis/{emoji['id']}")

    output({
        "success": True,
        "action": "delete_emoji",
        "guild_id": args.guild_id,
        "id": emoji["id"],
        "name": emoji["name"],
        "animated": bool(emoji.get("animated")),
    })


if __name__ == "__main__":
    main()
