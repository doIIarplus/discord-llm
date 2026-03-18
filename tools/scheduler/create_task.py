#!/usr/bin/env python3
"""Create a recurring scheduled task.

Stores a CLI command with a cron schedule in tasks.json.
The run_due.py script (called by system cron) executes tasks when due.

Examples:
  # Monthly Splitwise bill on the 5th at 9 AM
  create_task.py --name "Phone plan" --schedule "0 9 5 * *" \
    --command "python tools/splitwise/create_expense.py --amount 50 --description 'Phone plan' --split-with 12345 67890"

  # Daily reminder at 8 AM
  create_task.py --name "Daily standup" --schedule "0 8 * * *" \
    --command "echo 'Time for standup'"
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")


def _load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE) as f:
        return json.load(f)


def _save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2, default=str)


def _next_run(cron_expr):
    """Compute the next run time from a cron expression.

    Uses croniter if available, otherwise returns a placeholder.
    """
    try:
        from croniter import croniter
        cron = croniter(cron_expr, datetime.now(timezone.utc))
        return cron.get_next(datetime).isoformat()
    except ImportError:
        return "croniter not installed — install with: pip install croniter"
    except (ValueError, KeyError) as e:
        error(f"Invalid cron expression '{cron_expr}': {e}")


def _validate_cron(cron_expr):
    """Validate cron expression format (5 fields)."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        error(
            f"Cron expression must have 5 fields (minute hour day month weekday), "
            f"got {len(parts)}: '{cron_expr}'"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True,
                        help="Human-readable task name")
    parser.add_argument("--schedule", required=True,
                        help="Cron expression (5 fields: min hour day month weekday)")
    parser.add_argument("--command", required=True,
                        help="Full CLI command to execute when due")
    parser.add_argument("--description", default="",
                        help="Optional longer description")
    args = parser.parse_args()

    _validate_cron(args.schedule)

    tasks = _load_tasks()

    task = {
        "task_id": uuid.uuid4().hex[:12],
        "name": args.name,
        "description": args.description,
        "schedule": args.schedule,
        "command": args.command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
        "next_run": _next_run(args.schedule),
        "enabled": True,
    }

    tasks.append(task)
    _save_tasks(tasks)

    output(task)


if __name__ == "__main__":
    main()
