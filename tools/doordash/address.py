#!/usr/bin/env python3
"""List the consumer's saved delivery addresses, or change the default.

Without --set this lists addresses (dd-cli address list). The entry with
is_default true is the "near me" location — its lat/lng are what search.py
should be given, since dd-cli's own fallback is a Cupertino default. The
default can appear anywhere in the list, so scan all of it.

With --set ADDRESS_ID it changes the default delivery address
(dd-cli address set), using addresses[].address_id from the list.
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
        "--set",
        dest="address_id",
        metavar="ADDRESS_ID",
        help="Set this address as the default (addresses[].address_id from the list)",
    )
    add_intent_arg(parser)
    args = parser.parse_args()

    if args.address_id:
        # --yes: no tty in a captured subprocess, so dd-cli's interactive
        # confirmation would hang. Switching the default address is reversible
        # (just set it back), so it isn't gated like order.py place.
        dd_args = ["address", "set", "--address-id", args.address_id, "--yes"]
    else:
        dd_args = ["address", "list"]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
