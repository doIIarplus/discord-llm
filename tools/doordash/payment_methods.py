#!/usr/bin/env python3
"""List the consumer's saved DoorDash payment methods (cards on file).

Read-only. Useful for confirming a card exists before order.py place.
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
    add_intent_arg(parser)
    args = parser.parse_args()

    output(run_dd(["payment-method", "list"], args.intent))


if __name__ == "__main__":
    main()
