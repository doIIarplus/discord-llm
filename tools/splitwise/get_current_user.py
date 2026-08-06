#!/usr/bin/env python3
"""Get the authenticated Splitwise user's info (ID, name, email)."""

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
