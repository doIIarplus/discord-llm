#!/usr/bin/env python3
"""Self-rescheduling randomized reminder for Yang.

Sends a message to Yang, then schedules the next firing using a log-normal
delay distribution (mean ~96h, sigma 1.2) clamped to [6h, 14d].
"""

import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHANNEL_ID = "1171545202486431745"
YANG_ID = "134429572405002240"
MESSAGE = f"<@{YANG_ID}> don't let anyone violate your first amendment rights"
TASK_NAME = "random-rights-reminder"

# Log-normal params: target mean ~96h. For log-normal, mean = exp(mu + sigma^2/2).
# With sigma=1.2: mu = ln(96) - (1.2^2)/2 = 4.564 - 0.72 = 3.844
LN_MU = math.log(96) - (1.2 ** 2) / 2
LN_SIGMA = 1.2
CLAMP_MIN_H = 6
CLAMP_MAX_H = 14 * 24  # 14 days


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def delete_stale_tasks():
    result = run([sys.executable, "tools/scheduler/list_tasks.py", "--all"], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        return
    try:
        data = json.loads(result.stdout)
        tasks = data.get("tasks", [])
    except (json.JSONDecodeError, KeyError):
        return

    keywords = {"random-rights-reminder", "rights", "first-amendment", "first_amendment"}
    for task in tasks:
        name = task.get("name", "").lower()
        if any(kw in name for kw in keywords):
            task_id = task["task_id"]
            r = run([sys.executable, "tools/scheduler/delete_task.py", task_id], cwd=PROJECT_ROOT)
            if r.returncode == 0:
                print(f"Deleted stale task: {task['name']} ({task_id})")
            else:
                print(f"Failed to delete task {task_id}: {r.stderr.strip()}", file=sys.stderr)


def send_message():
    result = run(
        [sys.executable, "tools/discord/send_message.py",
         "--channel-id", CHANNEL_ID,
         "--content", MESSAGE],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"send_message failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    print(f"Sent: {MESSAGE}")


def compute_next_cron():
    delay_hours = random.lognormvariate(LN_MU, LN_SIGMA)
    delay_hours = max(CLAMP_MIN_H, min(CLAMP_MAX_H, delay_hours))
    next_dt = datetime.now(timezone.utc) + timedelta(hours=delay_hours)
    cron = f"{next_dt.minute} {next_dt.hour} {next_dt.day} {next_dt.month} *"
    return cron, next_dt, delay_hours


def schedule_next(cron, next_dt, delay_hours):
    cmd = (
        f"python tools/discord/random_reminder.py"
    )
    result = run(
        [sys.executable, "tools/scheduler/create_task.py",
         "--name", TASK_NAME,
         "--schedule", cron,
         "--command", cmd,
         "--description", "Self-rescheduling randomized first-amendment reminder for Yang",
         "--once"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"create_task failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Next reminder scheduled: {next_dt.strftime('%Y-%m-%d %H:%M UTC')} "
        f"(in {delay_hours:.1f}h) — cron: {cron}"
    )


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('discord')
    delete_stale_tasks()
    send_message()
    cron, next_dt, delay_hours = compute_next_cron()
    schedule_next(cron, next_dt, delay_hours)


if __name__ == "__main__":
    main()
