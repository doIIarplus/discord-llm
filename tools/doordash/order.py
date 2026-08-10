#!/usr/bin/env python3
"""Preview, place, and review DoorDash orders.

Subcommands (mapping to dd-cli):

  preview  -> order preview   Price a cart. Read-only, no charge. This is where
                              the real total (fees, tax, delivery) comes from —
                              cart.py show does NOT include pricing.
  place    -> order submit    SUBMIT THE ORDER. Spends real money, irreversible.
  history  -> order history   Recent past orders.
  status   -> order status    Whether a submitted order actually went through.

`place` refuses to run without --confirm. Only pass --confirm after showing the
user the actual items and the preview total and getting an explicit yes — not
because they said "order me food" earlier in the conversation.

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


def main():
    require_integration("doordash")
    require_owner()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_prev = sub.add_parser("preview", help="Price a cart without charging (dd-cli order preview)")
    p_prev.add_argument("--cart-uuid", required=True, help="Cart UUID from cart.py add")
    p_prev.add_argument(
        "--fulfillment", choices=["delivery", "pickup"],
        help="Set the cart's mode before pricing. If passed here, pass the SAME "
             "value to `place`, or the charged total won't match the quote.",
    )
    p_prev.add_argument(
        "--scheduled-time",
        help="ISO 8601 UTC delivery time, e.g. 2026-04-21T18:00:00Z. Omit for ASAP.",
    )
    add_intent_arg(p_prev)

    p_place = sub.add_parser(
        "place",
        help="SUBMIT the order — spends money, irreversible (dd-cli order submit)",
    )
    p_place.add_argument("--cart-uuid", required=True, help="Cart UUID from cart.py add")
    p_place.add_argument(
        "--confirm",
        action="store_true",
        help="REQUIRED. Asserts the user has seen the actual items and the "
             "preview total and explicitly approved this purchase.",
    )
    p_place.add_argument(
        "--tip-cents", type=int, help="Dasher tip in CENTS (500 = $5.00). Default 0."
    )
    p_place.add_argument(
        "--fulfillment", choices=["delivery", "pickup"],
        help="Must match what was passed to preview, if anything was.",
    )
    p_place.add_argument(
        "--scheduled-time",
        help="ISO 8601 UTC delivery time, e.g. 2026-04-21T18:00:00Z. Omit for ASAP.",
    )
    add_intent_arg(p_place)

    p_hist = sub.add_parser("history", help="Recent order history (dd-cli order history)")
    p_hist.add_argument("--max", type=int, help="Max orders to return, 1-100 (default 50)")
    p_hist.add_argument("--days", type=int, help="Window in days, 0-365 (default 90)")
    add_intent_arg(p_hist)

    p_stat = sub.add_parser("status", help="Check a submitted order went through")
    p_stat.add_argument("--order-uuid", required=True, help="Order UUID from `place`")
    add_intent_arg(p_stat)

    args = parser.parse_args()

    if args.action == "preview":
        dd_args = ["order", "preview", "--cart-uuid", args.cart_uuid]
        if args.fulfillment:
            dd_args += ["--fulfillment", args.fulfillment]
        if args.scheduled_time:
            dd_args += ["--scheduled-time", args.scheduled_time]

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
        if args.fulfillment:
            dd_args += ["--fulfillment", args.fulfillment]
        if args.scheduled_time:
            dd_args += ["--scheduled-time", args.scheduled_time]

    elif args.action == "history":
        dd_args = ["order", "history"]
        if args.max is not None:
            dd_args += ["--max", args.max]
        if args.days is not None:
            dd_args += ["--days", args.days]

    else:
        dd_args = ["order", "status", "--order-uuid", args.order_uuid]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
