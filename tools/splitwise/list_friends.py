#!/usr/bin/env python3
"""List all Splitwise friends with their IDs, names, emails, and balances."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _client import SplitwiseClient
from _common import output


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    client = SplitwiseClient()
    result = client.get("/get_friends")
    friends = result.get("friends", [])

    output({
        "friends": [
            {
                "id": f.get("id"),
                "first_name": f.get("first_name"),
                "last_name": f.get("last_name"),
                "email": f.get("email"),
                "balance": f.get("balance", []),
            }
            for f in friends
        ]
    })


if __name__ == "__main__":
    main()
