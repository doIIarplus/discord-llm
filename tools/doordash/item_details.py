#!/usr/bin/env python3
"""Show full item details: pricing, description, customization options.

Two verticals, selected with --kind (dd-cli splits them into two commands):

  --kind restaurant  -> dd-cli restaurant-item-details (also needs --menu-id)
  --kind retail      -> dd-cli item-details (grocery / retail)

Use this before cart.py add to learn which customizations exist. For
restaurants, pass only the selected extras[].options[].option_id values as
nested_options — extra_id is display grouping, not a selection.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from _common import error, output
from _dd import add_intent_arg, require_owner, run_dd
from _guild_access import require_integration


def main():
    require_integration("doordash")
    require_owner()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kind",
        choices=["restaurant", "retail"],
        required=True,
        help="restaurant menu item vs. retail/grocery item",
    )
    parser.add_argument("--store-id", required=True, help="Store ID (numeric)")
    parser.add_argument(
        "--item-id",
        required=True,
        help="Menu item ID. Strip any `i_` prefix from menu output (i_232... -> 232...)",
    )
    parser.add_argument(
        "--menu-id", help="Menu ID from menu.py. Required when --kind restaurant"
    )
    add_intent_arg(parser)
    args = parser.parse_args()

    if args.kind == "restaurant":
        if not args.menu_id:
            error("--menu-id is required when --kind restaurant (get it from menu.py)")
        dd_args = [
            "restaurant-item-details",
            "--store-id", args.store_id,
            "--menu-id", args.menu_id,
            "--item-id", args.item_id,
        ]
    else:
        dd_args = ["item-details", "--store-id", args.store_id, "--item-id", args.item_id]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
