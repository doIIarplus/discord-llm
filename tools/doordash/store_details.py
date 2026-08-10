#!/usr/bin/env python3
"""Show details for one DoorDash store: name, address, image, business metadata.

Read-only. Maps to `dd-cli store-details --store-id ID`.

This is the ONLY way to get a store's street address. search.py and
`order.py history` return store_id + store_name but no address, so this is what
disambiguates chain locations — "which Starbucks?", "the one on Main Street",
or confirming a specific branch before ordering. Surface `printable_address`
from the response for any address-shaped question about a store.

For a real delivery ETA use `order.py preview` on a cart at this store instead;
the store record's delivery_time is not address-aware.
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
    parser.add_argument(
        "--store-id",
        required=True,
        help="Store ID (numeric), from search.py, find_nearby_stores.py, or "
             "order.py history",
    )
    add_intent_arg(parser)
    args = parser.parse_args()

    output(run_dd(["store-details", "--store-id", args.store_id], args.intent))


if __name__ == "__main__":
    main()
