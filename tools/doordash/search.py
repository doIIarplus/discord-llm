#!/usr/bin/env python3
"""Find nearby restaurants on DoorDash.

Entry point for the ordering flow. Returns stores[].store_id, which feeds
menu.py (restaurants) or find_items.py (retail/grocery).

Location: dd-cli falls back to env DD_LAT/DD_LNG and finally a Cupertino
default, which is probably wrong. Prefer resolving real coordinates first —
`address.py --intent ...` lists saved addresses; use the is_default entry's
lat/lng and pass them here.
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
    parser.add_argument("query", help="Search text, e.g. 'salad' or 'sushi near me'")
    parser.add_argument("--limit", type=int, help="Max restaurants to return (dd-cli default: 5)")
    parser.add_argument("--lat", type=float, help="Latitude (default: env DD_LAT)")
    parser.add_argument("--lng", type=float, help="Longitude (default: env DD_LNG)")
    add_intent_arg(parser)
    args = parser.parse_args()

    dd_args = ["search", "--query", args.query]
    if args.limit is not None:
        dd_args += ["--limit", args.limit]
    if args.lat is not None:
        dd_args += ["--lat", args.lat]
    if args.lng is not None:
        dd_args += ["--lng", args.lng]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
