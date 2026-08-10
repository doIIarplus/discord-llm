#!/usr/bin/env python3
"""Manage the DoorDash cart: add items, show contents, clear it.

Subcommands (mapping to dd-cli):

  add    -> cart add-items    Add items to a cart, creating one if needed.
  show   -> cart show         Show a cart's contents (no pricing — use
                              order.py preview for the total).
  remove -> cart remove-item  Drop one line from a cart, keeping the cart.
  clear  -> cart delete       Empty a cart and abandon it.
  list   -> cart list         List the consumer's open carts.

`add` without --cart-uuid APPENDS to an existing open cart at that store if one
exists, so run `cart.py list` first if a surprise cart would matter.

--items-json schema (dd-cli `cart add-items`)
---------------------------------------------

  [
    {
      "item_id": "i_21941681157",
      "item_name": "Avocado & Quinoa Superfood Ensalada",
      "quantity": 1,
      "nested_options": [
        {"id": "o_40817472508", "name": "Chipotle Vinaigrette (On the Side)", "quantity": 1,
         "options": [ {"id": "o_...", "name": "...", "quantity": 1} ]}
      ]
    }
  ]

  * The key inside a nested_options entry is `id`, NOT `option_id`. Using
    `option_id` makes DoorDash reject the request with an "option is nested at
    the wrong level" error.
  * A top-level "options" key on the ITEM is ignored — modifiers must go under
    `nested_options`.
  * Deeper combo/nested choices recurse via an "options": [...] array INSIDE an
    option entry, using the same {"id", "name", "quantity"} shape.
  * Prefixed ids are correct (`i_` for items, `o_` for options). Pass them
    verbatim as returned by menu.py / item_details.py — do not strip prefixes.
  * Items with required modifiers (size, dressing, etc.) fail to add unless the
    required option ids are supplied. Get them from
    `item_details.py --kind restaurant --store-id ID --menu-id ID --item-id ID`.

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

    p_add = sub.add_parser(
        "add",
        help="Add items to a cart (dd-cli cart add-items)",
        description=(
            "Add items to a cart, creating one if needed. Without --cart-uuid this "
            "APPENDS to an existing open cart at that store.\n\n"
            "--items-json shape:\n"
            "  [\n"
            "    {\n"
            '      "item_id": "i_21941681157",\n'
            '      "item_name": "Avocado & Quinoa Superfood Ensalada",\n'
            '      "quantity": 1,\n'
            '      "nested_options": [\n'
            '        {"id": "o_40817472508", "name": "Chipotle Vinaigrette (On the Side)", '
            '"quantity": 1,\n'
            '         "options": [ {"id": "o_...", "name": "...", "quantity": 1} ]}\n'
            "      ]\n"
            "    }\n"
            "  ]\n\n"
            "  - Inside nested_options the key is `id`, NOT `option_id`. `option_id` is\n"
            "    rejected by DoorDash with 'option is nested at the wrong level'.\n"
            "  - A top-level \"options\" key on the item is IGNORED; modifiers belong\n"
            "    under nested_options.\n"
            "  - Deeper combo choices recurse via \"options\": [...] inside an option\n"
            "    entry, same {\"id\", \"name\", \"quantity\"} shape.\n"
            "  - Prefixed ids (i_ items, o_ options) are correct; pass verbatim from\n"
            "    menu.py / item_details.py.\n"
            "  - Items with required modifiers (size, dressing) fail to add unless the\n"
            "    required option ids are supplied; get them from\n"
            "    item_details.py --kind restaurant --store-id ID --menu-id ID --item-id ID."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help='JSON array of items, e.g. \'[{"item_id":"i_abc","item_name":"X",'
             '"quantity":2}]\'. Each entry needs item_id + item_name + quantity. '
             'Customizations go in nested_options[], whose entries use the key '
             '"id" (NOT "option_id" — that gets rejected as "option is nested at '
             'the wrong level") plus "name"/"quantity", and recurse via an inner '
             '"options":[...] array of the same shape. A top-level "options" key '
             'on the item is ignored. Pass i_/o_ prefixed ids verbatim from '
             "menu.py / item_details.py; required modifiers must be included or "
             "the add fails. See `cart.py add --help` for the full schema.",
    )
    p_add.add_argument("--cart-uuid", help="Add to this existing cart instead of resolving one")
    p_add.add_argument(
        "--fulfillment", choices=["delivery", "pickup"],
        help="Fulfillment mode at cart creation (default delivery)",
    )
    p_add.add_argument(
        "--group-cart",
        action="store_true",
        help="Create a shareable GROUP cart instead of a personal one. The "
             "response's cart.group_cart_url is the link to share (null for "
             "personal carts). With --cart-uuid pointing at someone else's "
             "group cart it joins that cart as a participant instead.",
    )
    p_add.add_argument(
        "--spend-limit-cents",
        type=int,
        help="Per-participant spending limit in CENTS for a new host-pays-all "
             "group cart (2500 = $25.00). Omit for unlimited. Requires "
             "--group-cart and cannot be used with --cart-uuid.",
    )
    add_intent_arg(p_add)

    p_show = sub.add_parser("show", help="Show cart contents (dd-cli cart show)")
    p_show.add_argument("--cart-uuid", required=True, help="Cart UUID from `cart.py add`")
    add_intent_arg(p_show)

    p_remove = sub.add_parser(
        "remove",
        help="Remove one line item from a cart (dd-cli cart remove-item)",
        description=(
            "Remove a single line from a cart, keeping the cart itself usable "
            "(cart_uuid stays valid). Use this to fix a wrong cart instead of "
            "clearing it and rebuilding from scratch.\n\n"
            "--cart-item-id is the cart LINE id — `cart.py show` items[].id — "
            "NOT the menu item_id used by --items-json. They are different ids, "
            "so `cart.py show` has to be called first to get it.\n\n"
            "To change a quantity rather than drop the item, `cart.py add` is "
            "cleaner: adds are additive, so add (target - current)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_remove.add_argument("--cart-uuid", required=True, help="Cart UUID from `cart.py add`")
    p_remove.add_argument(
        "--cart-item-id",
        required=True,
        help="Cart-LINE id from `cart.py show` items[].id — NOT the menu "
             "item_id. Run `cart.py show` first to get it.",
    )
    add_intent_arg(p_remove)

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

        # dd-cli rejects these combinations, but its error is opaque — catch
        # them here so the model gets something it can act on.
        if args.spend_limit_cents is not None:
            if not args.group_cart:
                error("--spend-limit-cents requires --group-cart: a spend limit "
                      "only exists on a host-pays-all group cart")
            if args.spend_limit_cents < 1:
                error("--spend-limit-cents must be a positive number of CENTS "
                      "(2500 = $25.00); omit it entirely for unlimited spending")
        if args.cart_uuid and args.spend_limit_cents is not None:
            error("--spend-limit-cents cannot be combined with --cart-uuid: the "
                  "limit is set when a new group cart is created, not on an "
                  "existing one")

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
        if args.group_cart:
            dd_args.append("--group-cart")
        if args.spend_limit_cents is not None:
            dd_args += ["--spend-limit-cents", args.spend_limit_cents]
    elif args.action == "show":
        dd_args = ["cart", "show", "--cart-uuid", args.cart_uuid]
    elif args.action == "remove":
        dd_args = [
            "cart", "remove-item",
            "--cart-uuid", args.cart_uuid,
            "--cart-item-id", args.cart_item_id,
        ]
    elif args.action == "clear":
        dd_args = ["cart", "delete", "--cart-uuid", args.cart_uuid]
    else:
        dd_args = ["cart", "list"]
        if args.store_id:
            dd_args += ["--store-id", args.store_id]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
