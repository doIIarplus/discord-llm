#!/usr/bin/env python3
"""List, apply, and remove DoorDash promotions.

Subcommands (mapping to dd-cli):

  list   -> promo list    Campaign promos eligible at a store. Read-only.
  apply  -> promo apply   Put a promo code on a cart. Reversible (`remove`).
  remove -> promo remove  Take a previously-applied promo back off a cart.

`list` is scoped to BOTH the signed-in consumer and the store, so results don't
transfer between stores and shouldn't be cached. An empty list is a normal
answer — it means nothing is eligible right now, not that something failed.
It is also a per-store catalog, not cart state: it does not say which promos are
currently discounting a given cart.

Two kinds of codes:
  * User-typed / referral codes — pass only --promo-code.
  * Campaign promos from `list` — pass all four values from the same row
    (--promo-code + --campaign-id + --ad-group-id + --ad-id). Remove takes
    whatever apply was given.

Reading an apply/remove response: `success: false` means it did not apply. The
message is often generic — the usual cause is the cart's subtotal_cents being
under the minimum stated in the promo's title/description (e.g. "20% off $15+"),
in which case adding items and retrying the same code works. Either way, run
`order.py preview` afterwards for the real currency-formatted total; the promo
response only carries cents.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from _common import output
from _dd import add_intent_arg, require_owner, run_dd
from _guild_access import require_integration

_CAMPAIGN_HELP = (
    "Campaign promos returned by `promo.py list` need all three of "
    "--campaign-id / --ad-group-id / --ad-id alongside --promo-code. Leave them "
    "off for user-typed or referral codes."
)


def add_promo_args(parser):
    """Attach the cart + code arguments shared by `apply` and `remove`."""
    parser.add_argument("--cart-uuid", required=True, help="Cart UUID from cart.py add")
    parser.add_argument(
        "--promo-code",
        required=True,
        help="The promo code string. For campaign promos this is the `code` "
             "field on the `promo.py list` row.",
    )
    parser.add_argument("--campaign-id", help=_CAMPAIGN_HELP)
    parser.add_argument("--ad-group-id", help=_CAMPAIGN_HELP)
    parser.add_argument("--ad-id", help=_CAMPAIGN_HELP)
    add_intent_arg(parser)
    return parser


def build_promo_args(action, args) -> list:
    dd_args = [
        "promo", action,
        "--cart-uuid", args.cart_uuid,
        "--promo-code", args.promo_code,
    ]
    if args.campaign_id:
        dd_args += ["--campaign-id", args.campaign_id]
    if args.ad_group_id:
        dd_args += ["--ad-group-id", args.ad_group_id]
    if args.ad_id:
        dd_args += ["--ad-id", args.ad_id]
    return dd_args


def main():
    require_integration("doordash")
    require_owner()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser(
        "list", help="Promos eligible at a store — read-only (dd-cli promo list)"
    )
    p_list.add_argument(
        "--store-id",
        required=True,
        help="Store ID from search.py, find_nearby_stores.py, cart.py list, or "
             "order.py history",
    )
    add_intent_arg(p_list)

    p_apply = sub.add_parser(
        "apply",
        help="Apply a promo code to a cart (dd-cli promo apply)",
        description="Apply a promo code to a cart. Reversible with `promo.py "
                    "remove`, so run it without asking first.\n\n" + _CAMPAIGN_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_promo_args(p_apply)

    p_remove = sub.add_parser(
        "remove",
        help="Remove an applied promo code from a cart (dd-cli promo remove)",
        description="Remove a promo already on a cart. Pass the same flag values "
                    "that were used to apply it.\n\n" + _CAMPAIGN_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_promo_args(p_remove)

    args = parser.parse_args()

    if args.action == "list":
        dd_args = ["promo", "list", "--store-id", args.store_id]
    else:
        dd_args = build_promo_args(args.action, args)

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
