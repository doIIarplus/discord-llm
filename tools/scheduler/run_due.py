#!/usr/bin/env python3
"""Execute scheduled tasks that are due.

Checks tasks.json for tasks whose next_run is in the past, executes their
command via subprocess, and updates last_run/next_run timestamps.

Intended to be called by system cron every minute:
  * * * * * cd /home/dollarplus/projects/discord_llm_bot && /home/dollarplus/projects/discord_llm_bot/venv/bin/python tools/scheduler/run_due.py >> /tmp/scheduler.log 2>&1
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _next_run(cron_expr):
    """Compute the next run time from now."""
    try:
        from croniter import croniter
        cron = croniter(cron_expr, datetime.now(timezone.utc))
        return cron.get_next(datetime).isoformat()
    except ImportError:
        return None


def _is_due(task):
    """Check if a task's next_run is in the past."""
    next_run = task.get("next_run")
    if not next_run or not task.get("enabled", True):
        return False
    try:
        next_dt = datetime.fromisoformat(next_run)
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= next_dt
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    if not os.path.exists(TASKS_FILE):
        output({"executed": [], "message": "No tasks file"})
        return

    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    executed = []
    now = datetime.now(timezone.utc)

    for task in tasks:
        if not _is_due(task):
            continue

        if args.dry_run:
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "command": task["command"],
                "would_run": True,
            })
            continue

        # Execute the command
        try:
            result = subprocess.run(
                task["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=PROJECT_DIR,
            )
            success = result.returncode == 0
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "success": success,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
            })
        except subprocess.TimeoutExpired:
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "success": False,
                "error": "Command timed out (120s)",
            })
        except Exception as e:
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "success": False,
                "error": str(e),
            })

        # Update timestamps
        task["last_run"] = now.isoformat()
        next_run = _next_run(task["schedule"])
        if next_run:
            task["next_run"] = next_run

    # Remove one-shot tasks that have been executed
    if not args.dry_run and executed:
        executed_ids = {e["task_id"] for e in executed}
        tasks = [
            t for t in tasks
            if not (t.get("once") and t["task_id"] in executed_ids)
        ]
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2, default=str)

    output({"executed": executed})


if __name__ == "__main__":
    main()
