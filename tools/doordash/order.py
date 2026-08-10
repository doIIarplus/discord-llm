#!/usr/bin/env python3
"""Preview, place, and review DoorDash orders.

Subcommands (mapping to dd-cli):

  preview      -> order preview   Price a cart. Read-only, no charge. This is
                                  where the real total (fees, tax, delivery)
                                  comes from — cart.py show does NOT include
                                  pricing.
  place        -> order submit    SUBMIT THE ORDER. Spends real money,
                                  irreversible.
  reorder      -> order reorder   Build a NEW cart from a past order. Does not
                                  charge. Returns a cart_uuid for `preview`.
  checkout-url -> order checkout-url
                                  Browser checkout link for a cart. Read-only.
  history      -> order history   Recent past orders. Top-level items only —
                                  this does NOT include modifiers/customizations.
  receipt      -> order receipt   Full itemized receipt for one past order,
                                  including the options/modifiers history omits.
  status       -> order status    Whether a submitted order actually went through.

`place` refuses to run without --confirm. Only pass --confirm after showing the
user the actual items and the preview total and getting an explicit yes — not
because they said "order me food" earlier in the conversation.

Quote-affecting flags (--scheduled-time, --fulfillment, --priority,
--no-apply-credits) exist on BOTH `preview` and `place`. Whatever is passed to
`preview` must be passed identically to `place`, or the amount charged will not
match the total the user approved.

Note --tip-cents is CENTS: 500 is $5.00, 5 is $0.05.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from _common import error, output
from _dd import add_intent_arg, require_owner, run_dd
from _guild_access import require_integration

# Flags that change what the cart actually costs. dd-cli accepts them on both
# `order preview` and `order submit`, and a quote is only valid for the exact
# combination it was computed with — so whatever preview was given has to be
# repeated verbatim on place.
_QUOTE_FLAG_NOTE = (
    "Quote-affecting: if this is passed to `preview`, it MUST be passed "
    "identically to `place`, or the charged total won't match the quote the "
    "user approved."
)


def add_quote_flags(parser):
    """Attach the pricing flags shared by `preview` and `place`."""
    parser.add_argument(
        "--scheduled-time",
        help="ISO 8601 scheduled time with a UTC suffix, e.g. "
             "2026-04-21T18:00:00Z. Omit for ASAP. " + _QUOTE_FLAG_NOTE,
    )
    parser.add_argument(
        "--fulfillment",
        choices=["delivery", "pickup"],
        help="Fulfillment mode for this quote. " + _QUOTE_FLAG_NOTE,
    )
    parser.add_argument(
        "--priority",
        action="store_true",
        help="Request Priority (express) delivery — a faster PAID upgrade. "
             "Delivery only: invalid with --fulfillment pickup and with "
             "--scheduled-time (Priority is ASAP). Not offered on every cart, "
             "so confirm quote.delivery_availability.delivery_options[] has a "
             "PRIORITY entry before promising it. " + _QUOTE_FLAG_NOTE,
    )
    parser.add_argument(
        "--no-apply-credits",
        action="store_true",
        help="Opt out of applying the user's DoorDash credits. Credits apply "
             "by default; pass this ONLY on an explicit request not to use "
             "them (all-or-nothing, no partial amounts). " + _QUOTE_FLAG_NOTE,
    )
    return parser


def build_quote_flags(args) -> list:
    """Turn the shared pricing flags into dd-cli args, rejecting bad combos."""
    if args.priority and args.fulfillment == "pickup":
        error("--priority is delivery only and cannot be combined with "
              "--fulfillment pickup")
    if args.priority and args.scheduled_time:
        error("--priority is an ASAP-only upgrade and cannot be combined with "
              "--scheduled-time")

    dd_args = []
    if args.scheduled_time:
        dd_args += ["--scheduled-time", args.scheduled_time]
    if args.fulfillment:
        dd_args += ["--fulfillment", args.fulfillment]
    if args.priority:
        dd_args.append("--priority")
    if args.no_apply_credits:
        dd_args.append("--no-apply-credits")
    return dd_args


def main():
    require_integration("doordash")
    require_owner()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_prev = sub.add_parser("preview", help="Price a cart without charging (dd-cli order preview)")
    p_prev.add_argument(
        "--cart-uuid", required=True, help="Cart UUID from cart.py add or `order.py reorder`"
    )
    add_quote_flags(p_prev)
    add_intent_arg(p_prev)

    p_place = sub.add_parser(
        "place",
        help="SUBMIT the order — spends money, irreversible (dd-cli order submit)",
    )
    p_place.add_argument(
        "--cart-uuid", required=True, help="Cart UUID from cart.py add or `order.py reorder`"
    )
    p_place.add_argument(
        "--confirm",
        action="store_true",
        help="REQUIRED. Asserts the user has seen the actual items and the "
             "preview total and explicitly approved this purchase.",
    )
    p_place.add_argument(
        "--tip-cents", type=int, help="Dasher tip in CENTS (500 = $5.00). Default 0."
    )
    add_quote_flags(p_place)
    add_intent_arg(p_place)

    p_reorder = sub.add_parser(
        "reorder",
        help="Build a NEW cart from a past order — no charge (dd-cli order reorder)",
        description=(
            "Create a new cart from a past order's items, including all their "
            "modifiers. Nothing is charged — the returned cart_uuid goes into "
            "`order.py preview` next.\n\n"
            "This is the fastest correct way to repeat an order: `order.py "
            "history` omits modifiers entirely, so rebuilding a cart by hand "
            "from it silently loses customizations.\n\n"
            "Caveats:\n"
            "  - Not every order is reorderable. Check the response for\n"
            "    `success: false` plus `fail_reason` rather than assuming a\n"
            "    cart came back.\n"
            "  - The new cart INHERITS THE ORIGINAL ORDER'S FULFILLMENT MODE,\n"
            "    so reordering a past pickup order silently produces a pickup\n"
            "    cart. Verify with `order.py preview` before showing a total.\n"
            "  - Out-of-stock or substituted items can drop silently; diff the\n"
            "    cart against the original order if that would matter.\n"
            "  - Only one open cart per store exists, and reorder always makes\n"
            "    a fresh one — `cart.py list --store-id ID` first if a\n"
            "    forgotten cart at that store would be a surprise."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_reorder.add_argument(
        "--order-uuid",
        required=True,
        help="Order UUID of the past order, from `order.py history` orders[].order_uuid",
    )
    add_intent_arg(p_reorder)

    p_url = sub.add_parser(
        "checkout-url",
        help="Browser checkout link for a cart — read-only (dd-cli order checkout-url)",
        description=(
            "Get a browser checkout URL for a cart. Read-only and safe to run — "
            "it charges nothing and the cart stays editable.\n\n"
            "This is the fallback for checkout edits the CLI cannot express: "
            "changing the payment method, changing the delivery address, "
            "tweaking quantities without rebuilding the cart, or tipping in the "
            "browser flow. `order.py place` remains the normal finalizer — "
            "don't hand over a URL by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_url.add_argument(
        "--cart-uuid", required=True, help="Cart UUID from cart.py add or `order.py reorder`"
    )
    add_intent_arg(p_url)

    p_hist = sub.add_parser("history", help="Recent order history (dd-cli order history)")
    p_hist.add_argument("--max", type=int, help="Max orders to return, 1-100 (default 50)")
    p_hist.add_argument("--days", type=int, help="Window in days, 0-365 (default 90)")
    add_intent_arg(p_hist)

    p_rcpt = sub.add_parser(
        "receipt",
        help="Full itemized receipt for a past order, including modifiers "
             "(dd-cli order receipt)",
    )
    p_rcpt.add_argument(
        "--order-uuid",
        required=True,
        help="Order UUID as returned by `order.py history` or `order.py place`",
    )
    add_intent_arg(p_rcpt)

    p_stat = sub.add_parser("status", help="Check a submitted order went through")
    p_stat.add_argument("--order-uuid", required=True, help="Order UUID from `place`")
    add_intent_arg(p_stat)

    args = parser.parse_args()

    if args.action == "preview":
        dd_args = ["order", "preview", "--cart-uuid", args.cart_uuid]
        dd_args += build_quote_flags(args)

    elif args.action == "place":
        if not args.confirm:
            error(
                "refusing to place the order: --confirm was not passed. "
                "order.py place spends real money and cannot be undone. Show the "
                "user the actual items and the total from `order.py preview`, get "
                "an explicit yes, then re-run with --confirm.",
                details={"cart_uuid": args.cart_uuid},
            )
        # -y suppresses dd-cli's own interactive prompt. It has to be passed:
        # this runs as a captured subprocess with no tty, so an interactive
        # confirmation would just hang until the 120s timeout. Our --confirm
        # gate above is the real check.
        dd_args = ["order", "submit", "--cart-uuid", args.cart_uuid, "--yes"]
        if args.tip_cents is not None:
            dd_args += ["--tip-cents", args.tip_cents]
        dd_args += build_quote_flags(args)

    elif args.action == "reorder":
        dd_args = ["order", "reorder", "--order-uuid", args.order_uuid]

    elif args.action == "checkout-url":
        dd_args = ["order", "checkout-url", "--cart-uuid", args.cart_uuid]

    elif args.action == "history":
        dd_args = ["order", "history"]
        if args.max is not None:
            dd_args += ["--max", args.max]
        if args.days is not None:
            dd_args += ["--days", args.days]

    elif args.action == "receipt":
        dd_args = ["order", "receipt", "--order-uuid", args.order_uuid]

    else:
        dd_args = ["order", "status", "--order-uuid", args.order_uuid]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
