#!/usr/bin/env python3
"""List all scheduled tasks."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('scheduler')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Include disabled tasks")
    parser.parse_args()

    if not os.path.exists(TASKS_FILE):
        output({"tasks": []})
        return

    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    if not parser.parse_args().all:
        tasks = [t for t in tasks if t.get("enabled", True)]

    output({"tasks": tasks})


if __name__ == "__main__":
    main()
