#!/usr/bin/env python3
"""Timeout (mute) a member in a Discord guild.

Sets a communication_disabled_until timestamp. The member cannot send messages,
react, or join voice channels until the timeout expires. Use --remove to clear.

Max timeout is 28 days (Discord limitation).

Examples:
  # Timeout for 10 minutes
  timeout_user.py --guild-id 363154169294618625 --user-id 118567805678256128 --duration 10m

  # Timeout for 1 hour
  timeout_user.py --guild-id 363154169294618625 --user-id 118567805678256128 --duration 1h

  # Timeout for 7 days
  timeout_user.py --guild-id 363154169294618625 --user-id 118567805678256128 --duration 7d

  # Remove timeout
  timeout_user.py --guild-id 363154169294618625 --user-id 118567805678256128 --remove
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient
from discord._permissions import require_permission
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

MAX_TIMEOUT = timedelta(days=28)


def parse_duration(s):
    """Parse a duration string like '10m', '1h', '7d' into a timedelta."""
    match = re.fullmatch(r"(\d+)\s*([mhd])", s.strip().lower())
    if not match:
        error("Invalid duration format. Use e.g. '10m', '1h', '7d'")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('discord')
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument("--user-id", required=True, help="User ID to timeout")
    parser.add_argument("--duration", default=None,
                        help="Timeout duration (e.g. 10m, 1h, 7d). Max 28d")
    parser.add_argument("--remove", action="store_true", help="Remove existing timeout")
    args = parser.parse_args()

    if not args.duration and not args.remove:
        error("Provide --duration or --remove")

    require_permission("MODERATE_MEMBERS", guild_id=args.guild_id)

    client = DiscordClient()

    if args.remove:
        client.patch(f"/guilds/{args.guild_id}/members/{args.user_id}", {
            "communication_disabled_until": None,
        })
        output({
            "success": True,
            "action": "remove_timeout",
            "guild_id": args.guild_id,
            "user_id": args.user_id,
        })
    else:
        delta = parse_duration(args.duration)
        if delta > MAX_TIMEOUT:
            error(f"Timeout cannot exceed 28 days (requested {args.duration})")

        until = datetime.now(timezone.utc) + delta
        iso = until.isoformat()

        client.patch(f"/guilds/{args.guild_id}/members/{args.user_id}", {
            "communication_disabled_until": iso,
        })
        output({
            "success": True,
            "action": "timeout_user",
            "guild_id": args.guild_id,
            "user_id": args.user_id,
            "until": iso,
            "duration": args.duration,
        })


if __name__ == "__main__":
    main()
