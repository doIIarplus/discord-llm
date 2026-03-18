#!/usr/bin/env python3
"""Show current Splitwise balances with all friends (non-zero only)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output


def main():
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
