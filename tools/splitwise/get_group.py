#!/usr/bin/env python3
"""Get details for a specific Splitwise group including members and balances."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_id", type=int, help="Splitwise group ID")
    args = parser.parse_args()

    client = SplitwiseClient()
    result = client.get(f"/get_group/{args.group_id}")
    group = result.get("group", {})

    output({
        "id": group.get("id"),
        "name": group.get("name"),
        "members": [
            {
                "id": m.get("id"),
                "first_name": m.get("first_name"),
                "last_name": m.get("last_name"),
                "email": m.get("email"),
                "balance": m.get("balance", []),
            }
            for m in group.get("members", [])
        ],
        "simplified_debts": group.get("simplified_debts", []),
    })


if __name__ == "__main__":
    main()
