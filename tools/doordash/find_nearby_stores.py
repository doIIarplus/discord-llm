#!/usr/bin/env python3
"""Discover nearby NON-restaurant stores: grocery, alcohol, convenience, etc.

Maps to `dd-cli find-nearby-stores`. Read-only.

This is the entry point for everything that is not a restaurant — restaurants
go through search.py. The returned stores[].store_id is what find_items.py
needs to search a store's catalog.

Verticals (--vertical):
  grocery      (default) grocery + DashMart; may extend past traditional grocers
  alcohol      alcohol merchants only
  convenience  convenience stores
  pets         pet stores
  retail       general retail (home goods, beauty, baby, electronics, office,
               sporting, home improvement, books, jewelry, auto). Excludes
               convenience, regulated verticals, and prepared food.
  nv           every non-restaurant merchant type

Location: the search radius is a fixed 16 miles and cannot be changed. With
--lat/--lng omitted it falls back to the user's default saved delivery address,
which is usually what "near me" means — pass both together only to search
somewhere else (address.py lists the saved addresses and their coordinates).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from _common import error, output
from _dd import add_intent_arg, require_owner, run_dd
from _guild_access import require_integration

VERTICALS = ["grocery", "alcohol", "convenience", "pets", "retail", "nv"]


def main():
    require_integration("doordash")
    require_owner()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vertical",
        choices=VERTICALS,
        default="grocery",
        help="Merchant-type filter (default: grocery). Use 'nv' for every "
             "non-restaurant type. For restaurants use search.py instead.",
    )
    parser.add_argument(
        "--max", type=int, default=10, help="Max stores to return (default 10)"
    )
    parser.add_argument(
        "--lat",
        type=float,
        help="Latitude override. Pass with --lng. Omit both to use the default "
             "saved delivery address.",
    )
    parser.add_argument(
        "--lng",
        type=float,
        help="Longitude override. Pass with --lat. Omit both to use the default "
             "saved delivery address.",
    )
    add_intent_arg(parser)
    args = parser.parse_args()

    # dd-cli wants both or neither; one alone silently falls back to the saved
    # address, which would quietly search the wrong place.
    if (args.lat is None) != (args.lng is None):
        error("--lat and --lng must be passed together (or both omitted to use "
              "the default saved delivery address)")

    dd_args = ["find-nearby-stores", "--vertical", args.vertical, "--max", args.max]
    if args.lat is not None:
        dd_args += ["--lat", args.lat, "--lng", args.lng]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
