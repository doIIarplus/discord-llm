#!/usr/bin/env python3
"""Delete a scheduled task by ID."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('scheduler')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="The task ID to delete")
    args = parser.parse_args()

    if not os.path.exists(TASKS_FILE):
        error(f"No tasks file found")

    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    original_count = len(tasks)
    tasks = [t for t in tasks if t.get("task_id") != args.task_id]

    if len(tasks) == original_count:
        error(f"Task '{args.task_id}' not found")

    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2, default=str)

    output({"deleted": True, "task_id": args.task_id})


if __name__ == "__main__":
    main()
