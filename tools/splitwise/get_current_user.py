#!/usr/bin/env python3
"""Get the authenticated Splitwise user's info (ID, name, email)."""

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
    result = client.get("/get_current_user")
    user = result.get("user", {})

    output({
        "id": user.get("id"),
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "email": user.get("email"),
    })


if __name__ == "__main__":
    main()
