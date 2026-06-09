#!/usr/bin/env python3
"""Send a first-amendment rights reminder to Yang and reschedule itself randomly."""

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env"))

CHANNEL_ID = "1171545202486431745"
TASK_NAME = "yang-rights-reminder"
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PYTHON = sys.executable

MESSAGES = [
    "<@134429572405002240> hey. protect your first amendment rights.",
    "<@134429572405002240> don't let them take your first amendment rights.",
    "<@134429572405002240> your first amendment rights. don't forget.",
    "<@134429572405002240> first amendment. yours. protect it.",
    "<@134429572405002240> they can't take your first amendment rights. remember that.",
    "<@134429572405002240> hey. first amendment rights. protect them.",
    "<@134429572405002240> just a reminder. first amendment. it's yours.",
]


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"[random_reminder] WARNING: command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
    return result


def delete_existing_tasks():
    """Delete any existing yang-rights-reminder tasks so we can replace with the new one."""
    result = run([PYTHON, os.path.join(TOOLS_DIR, "scheduler/list_tasks.py")])
    if result.returncode != 0:
        return
    try:
        data = json.loads(result.stdout)
        tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
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
    """Bimodal distribution: 70% spooky window (10pm–4am), 30% any time in next 3–10 days."""
    now = datetime.now(timezone.utc)

    if random.random() < 0.70:
        # Spooky window: 10pm–4am (22:00–27:59 wrapping to next day)
        # Represent as minutes past 22:00; window = 6 hours = 360 minutes
        # Day offset: pick a random day 1–10 days out, then snap to the spooky window
        day_offset = random.randint(1, 10)
        base = now + timedelta(days=day_offset)
        spooky_minute_offset = random.randint(0, 359)  # 0–359 minutes past 22:00
        hour = (22 + spooky_minute_offset // 60) % 24
        minute = spooky_minute_offset % 60
        # If we wrapped past midnight, add an extra day
        extra_day = 1 if (22 + spooky_minute_offset // 60) >= 24 else 0
        target = base + timedelta(days=extra_day)
        target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
    else:
        # Any time in the next 3–10 days
        day_offset = random.randint(3, 10)
        hour = random.randint(0, 23)
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
