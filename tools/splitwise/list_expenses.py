#!/usr/bin/env python3
"""List recent Splitwise expenses, optionally filtered by friend or group."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20,
                        help="Max expenses to return (default: 20)")
    parser.add_argument("--friend-id", type=int,
                        help="Filter to expenses involving this friend")
    parser.add_argument("--group-id", type=int,
                        help="Filter to expenses in this group")
    parser.add_argument("--dated-after",
                        help="Only expenses after this date (YYYY-MM-DD)")
    parser.add_argument("--dated-before",
                        help="Only expenses before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    params = {"limit": args.limit}
    if args.friend_id:
        params["friend_id"] = args.friend_id
    if args.group_id:
        params["group_id"] = args.group_id
    if args.dated_after:
        params["dated_after"] = args.dated_after
    if args.dated_before:
        params["dated_before"] = args.dated_before

    client = SplitwiseClient()
    # Build query string
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    result = client.get(f"/get_expenses?{qs}")
    expenses = result.get("expenses", [])

    output({
        "expenses": [
            {
                "id": e.get("id"),
                "description": e.get("description"),
                "cost": e.get("cost"),
                "currency_code": e.get("currency_code"),
                "date": e.get("date"),
                "created_by": e.get("created_by", {}).get("id"),
                "deleted_at": e.get("deleted_at"),
                "users": [
                    {
                        "user_id": u.get("user", {}).get("id"),
                        "name": f"{u.get('user', {}).get('first_name', '')} {u.get('user', {}).get('last_name', '')}".strip(),
                        "paid_share": u.get("paid_share"),
                        "owed_share": u.get("owed_share"),
                        "net_balance": u.get("net_balance"),
                    }
                    for u in e.get("users", [])
                ],
            }
            for e in expenses
            if not e.get("deleted_at")  # skip deleted expenses
        ]
    })


if __name__ == "__main__":
    main()
