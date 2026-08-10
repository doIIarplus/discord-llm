#!/usr/bin/env python3
"""Search items within a retail or grocery store by name or keyword.

Retail/grocery only — restaurant stores return success:true with empty results;
use menu.py for those. Cheaper than pulling the whole catalog when you already
know roughly what you want. Pass QUERY more than once to resolve several items
in one call.

Results feed cart.py add (results[query][].item_id).
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
    parser.add_argument("--store-id", required=True, help="Store ID (numeric, e.g. 928163)")
    parser.add_argument(
        "query", nargs="+", help="Item name(s) to look up, e.g. 'milk' 'eggs'"
    )
    add_intent_arg(parser)
    args = parser.parse_args()

    dd_args = ["find-items", "--store-id", args.store_id]
    for q in args.query:
        dd_args += ["--query", q]

    output(run_dd(dd_args, args.intent))


if __name__ == "__main__":
    main()
