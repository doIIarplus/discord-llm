#!/usr/bin/env python3
"""Manage the DoorDash cart: add items, show contents, clear it.

Subcommands (mapping to dd-cli):

  add    -> cart add-items    Add items to a cart, creating one if needed.
  show   -> cart show         Show a cart's contents (no pricing — use
                              order.py preview for the total).
  clear  -> cart delete       Empty a cart and abandon it.
  list   -> cart list         List the consumer's open carts.

`add` without --cart-uuid APPENDS to an existing open cart at that store if one
exists, so run `cart.py list` first if a surprise cart would matter.

Examples:

  cart.py add --store-id 928163 --menu-id 1657275 \\
    --items-json '[{"item_id":"25192228142","item_name":"House Salad","quantity":1}]' \\
    --intent "Summary: Help the user order lunch"

  cart.py show --cart-uuid <uuid> --intent "..."
"""

import argparse
import json
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
    sub = parser.add_subparsers(dest="action", required=True)

    p_add = sub.add_parser("add", help="Add items to a cart (dd-cli cart add-items)")
    p_add.add_argument("--store-id", required=True, help="Store ID (numeric)")
    p_add.add_argument(
        "--menu-id",
        required=True,
        help="Menu ID for THIS store, from menu.py (restaurants) or "
             "item_details.py --kind retail (grocery). Must match the store id.",
    )
    p_add.add_argument(
        "--items-json",
        required=True,
        help='JSON array of items, e.g. \'[{"item_id":"abc","item_name":"X",'
             '"quantity":2}]\'. Each entry needs item_id + item_name + quantity; '
             "may include nested_options[] for customizations.",
    )
    p_add.add_argument("--cart-uuid", help="Add to this existing cart instead of resolving one")
    p_add.add_argument(
        "--fulfillment", choices=["delivery", "pickup"],
        help="Fulfillment mode at cart creation (default delivery)",
    )
    add_intent_arg(p_add)

    p_show = sub.add_parser("show", help="Show cart contents (dd-cli cart show)")
    p_show.add_argument("--cart-uuid", required=True, help="Cart UUID from `cart.py add`")
    add_intent_arg(p_show)

    p_clear = sub.add_parser("clear", help="Empty and abandon a cart (dd-cli cart delete)")
    p_clear.add_argument("--cart-uuid", required=True, help="Cart UUID from `cart.py add`")
    add_intent_arg(p_clear)

    p_list = sub.add_parser("list", help="List open carts (dd-cli cart list)")
    p_list.add_argument("--store-id", help="Only carts at this store")
    add_intent_arg(p_list)

    args = parser.parse_args()

    if args.action == "add":
        # Fail here rather than shipping malformed JSON to dd-cli, so the model
        # gets an error it can actually fix.
        try:
            parsed = json.loads(args.items_json)
        except (json.JSONDecodeError, ValueError) as e:
            error(f"--items-json is not valid JSON: {e}")
        if not isinstance(parsed, list) or not parsed:
            error("--items-json must be a non-empty JSON array of item objects")

        dd_args = [
            "cart", "add-items",
            "--store-id", args.store_id,
            "--menu-id", args.menu_id,
            "--items-json", args.items_json,
        ]
        if args.cart_uuid:
            dd_args += ["--cart-uuid", args.cart_uuid]
        if args.fulfillment:
            dd_args += ["--fulfillment", args.fulfillment]
    elif args.action == "show":
        dd_args = ["cart", "show", "--cart-uuid", args.cart_uuid]
    elif args.action == "clear":
        dd_args = ["cart", "delete", "--cart-uuid", args.cart_uuid]
    else:
        dd_args = ["cart", "list"]
        if args.store_id:
            dd_args += ["--store-id", args.store_id]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
