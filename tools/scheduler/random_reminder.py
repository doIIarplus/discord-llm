#!/usr/bin/env python3
"""Send a first-amendment rights reminder to Yang and reschedule itself randomly."""

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone

CHANNEL_ID = "1171545202486431745"
TASK_NAME = "yang-rights-reminder"
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PYTHON = sys.executable

MESSAGES = [
    "<@134429572405002240> hey don't let anyone violate your first amendment rights",
    "<@134429572405002240> reminder: your first amendment rights cannot be taken from you",
    "<@134429572405002240> remember bro, no one can legally violate your first amendment rights",
    "<@134429572405002240> psa: your first amendment rights are protected, don't let anyone tell u otherwise",
    "<@134429572405002240> just a reminder that ur first amendment rights exist and cannot be violated",
    "<@134429572405002240> hey quick reminder — first amendment rights are yours and nobody can take them",
    "<@134429572405002240> don't forget: the first amendment protects ur rights, stand on that",
]


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"[random_reminder] WARNING: command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
    return result


def delete_existing_tasks():
    """Find and delete any tasks named yang-rights-reminder or yang-first-amendment-reminder."""
    result = run([PYTHON, os.path.join(TOOLS_DIR, "scheduler/list_tasks.py")])
    if result.returncode != 0:
        return
    try:
        tasks = json.loads(result.stdout)
        if isinstance(tasks, dict) and "tasks" in tasks:
            tasks = tasks["tasks"]
        stale_names = {TASK_NAME, "yang-first-amendment-reminder"}
        for task in tasks:
            if task.get("name") in stale_names:
                task_id = task["task_id"]
                print(f"[random_reminder] deleting stale task {task_id} ({task['name']})")
                run([PYTHON, os.path.join(TOOLS_DIR, "scheduler/delete_task.py"), task_id])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[random_reminder] WARNING: could not parse task list: {e}", file=sys.stderr)


def send_message():
    msg = random.choice(MESSAGES)
    print(f"[random_reminder] sending: {msg}")
    run([
        PYTHON,
        os.path.join(TOOLS_DIR, "discord/send_message.py"),
        "--channel-id", CHANNEL_ID,
        "--content", msg,
    ])


def pick_next_datetime():
    """Pick a random future datetime using the specified distribution."""
    now = datetime.now(timezone.utc)

    # Day offset: gauss(7, 4) clamped to [2, 18]
    day_offset = int(random.gauss(7, 4))
    day_offset = max(2, min(18, day_offset))

    # Hour: 70% daytime (10-22), 15% early morning (0-5), 15% late night (22-24)
    roll = random.random()
    if roll < 0.70:
        hour = random.randint(10, 22)
    elif roll < 0.85:
        hour = random.randint(0, 5)
    else:
        hour = random.randint(22, 23)

    minute = random.randint(0, 59)

    target = now + timedelta(days=day_offset)
    target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target


def schedule_next(target: datetime):
    cron = f"{target.minute} {target.hour} {target.day} {target.month} *"
    print(f"[random_reminder] scheduling next run at {target.isoformat()} (cron: {cron})")
    result = run([
        PYTHON,
        os.path.join(TOOLS_DIR, "scheduler/create_task.py"),
        "--name", TASK_NAME,
        "--schedule", cron,
        "--once",
        "--description", "randomized first amendment rights reminder for Yang",
        "--command", "python tools/scheduler/random_reminder.py",
    ])
    if result.returncode == 0:
        print(f"[random_reminder] next reminder scheduled for {target.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        print(f"[random_reminder] ERROR: failed to schedule next run", file=sys.stderr)


def main():
    delete_existing_tasks()
    send_message()
    next_dt = pick_next_datetime()
    schedule_next(next_dt)


if __name__ == "__main__":
    main()
