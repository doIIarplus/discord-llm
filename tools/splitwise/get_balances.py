#!/usr/bin/env python3
"""Show current Splitwise balances with all friends (non-zero only)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output
from _auth import require_owner
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('splitwise')
    require_owner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Include friends with zero balance")
    args = parser.parse_args()

    client = SplitwiseClient()
    result = client.get("/get_friends")
    friends = result.get("friends", [])

    balances = []
    for f in friends:
        for b in f.get("balance", []):
            amount = float(b.get("amount", 0))
            if not args.all and amount == 0:
                continue
            balances.append({
                "friend_id": f.get("id"),
                "name": f"{f.get('first_name', '')} {f.get('last_name', '')}".strip(),
                "amount": b.get("amount"),
                "currency": b.get("currency_code"),
            })

    output({"balances": balances})


if __name__ == "__main__":
    main()
