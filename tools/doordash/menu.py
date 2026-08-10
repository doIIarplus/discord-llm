#!/usr/bin/env python3
"""Show a restaurant's menu.

Takes a --store-id from search.py (stores[].store_id) or order.py history.
The response's menu_id plus items[].item_id are what cart.py add needs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from _common import output
from _dd import add_intent_arg, require_owner, run_dd
from _guild_access import require_integration


def main():
    require_integration("doordash")
    require_owner()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store-id", required=True, help="Store ID (numeric, e.g. 928163)")
    add_intent_arg(parser)
    args = parser.parse_args()

    output(run_dd(["menu", "--store-id", args.store_id], args.intent))


if __name__ == "__main__":
    main()
