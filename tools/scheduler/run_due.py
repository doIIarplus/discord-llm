#!/usr/bin/env python3
"""Execute scheduled tasks that are due.

Checks tasks.json for tasks whose next_run is in the past, executes their
command via subprocess, and updates last_run/next_run timestamps.

Intended to be called by system cron every minute:
  * * * * * cd /home/dollarplus/projects/discord_llm_bot && /home/dollarplus/projects/discord_llm_bot/venv/bin/python tools/scheduler/run_due.py 2>&1

Logs are written to scheduler.log in the project root.
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output

TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LOG_FILE = os.path.join(PROJECT_DIR, "scheduler.log")

# File logger — always appends, with timestamps
_logger = logging.getLogger("scheduler")
_logger.setLevel(logging.DEBUG)
_handler = logging.FileHandler(LOG_FILE)
_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger.addHandler(_handler)


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

    # Prevent concurrent execution (cron can overlap if a task takes >60s)
    lock_file = TASKS_FILE + ".lock"
    lock_fd = open(lock_file, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _logger.warning("Another instance is already running, skipping")
        lock_fd.close()
        output({"executed": [], "message": "Locked by another instance"})
        return

    if not os.path.exists(TASKS_FILE):
        _logger.debug("No tasks file found")
        lock_fd.close()
        output({"executed": [], "message": "No tasks file"})
        return

    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    due_count = sum(1 for t in tasks if _is_due(t))
    if due_count:
        _logger.info(f"Checking {len(tasks)} tasks, {due_count} due")

    executed = []
    now = datetime.now(timezone.utc)

    for task in tasks:
        if not _is_due(task):
            continue

        task_label = f"[{task['task_id']}] {task['name']}"

        if args.dry_run:
            _logger.info(f"DRY RUN: {task_label} would run: {task['command']}")
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "command": task["command"],
                "would_run": True,
            })
            continue

        # Resolve bare "python " to the current venv interpreter so tasks
        # work under cron (which has a minimal PATH without the venv).
        command = task["command"]
        if command.startswith("python "):
            command = sys.executable + command[6:]

        _logger.info(f"RUNNING: {task_label} — {command}")

        # Execute the command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=PROJECT_DIR,
            )
            success = result.returncode == 0
            if success:
                _logger.info(f"SUCCESS: {task_label} (exit 0)")
                if result.stdout.strip():
                    _logger.debug(f"  stdout: {result.stdout.strip()[:500]}")
            else:
                _logger.error(f"FAILED: {task_label} (exit {result.returncode})")
                if result.stdout.strip():
                    _logger.error(f"  stdout: {result.stdout.strip()[:500]}")
                if result.stderr.strip():
                    _logger.error(f"  stderr: {result.stderr.strip()[:500]}")
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "success": success,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
            })
        except subprocess.TimeoutExpired:
            _logger.error(f"TIMEOUT: {task_label} (120s limit)")
            executed.append({
                "task_id": task["task_id"],
                "name": task["name"],
                "success": False,
                "error": "Command timed out (120s)",
            })
        except Exception as e:
            _logger.error(f"EXCEPTION: {task_label} — {e}")
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
        once_removed = [
            t["name"] for t in tasks
            if t.get("once") and t["task_id"] in executed_ids
        ]
        tasks = [
            t for t in tasks
            if not (t.get("once") and t["task_id"] in executed_ids)
        ]
        if once_removed:
            _logger.info(f"Removed one-shot tasks: {', '.join(once_removed)}")
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2, default=str)

    lock_fd.close()
    output({"executed": executed})


if __name__ == "__main__":
    main()
