#!/usr/bin/env python3
"""List all Splitwise groups the authenticated user belongs to."""

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
    result = client.get("/get_groups")
    groups = result.get("groups", [])

    output({
        "groups": [
            {
                "id": g.get("id"),
                "name": g.get("name"),
                "members": [
                    {
                        "id": m.get("id"),
                        "first_name": m.get("first_name"),
                        "last_name": m.get("last_name"),
                    }
                    for m in g.get("members", [])
                ],
            }
            for g in groups
            if g.get("id") != 0  # skip the non-group expenses pseudo-group
        ]
    })


if __name__ == "__main__":
    main()
