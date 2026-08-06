#!/usr/bin/env python3
"""Check on the background coding job in this channel.

Owner-only: restricted to the bot owner's Discord account.

Use this when the user asks "how's it going" / "is it done". With no --job-id it
reports the channel's active job, falling back to the most recent one.

Statuses: running · awaiting_push (finished, branch ready, waiting on the user) ·
done · failed · cancelled.

The live progress message already updates itself, so don't call this on a timer —
only when the user actually asks.

Examples:
  status.py
  status.py --job-id 4f2a1b9c0d11
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _client import call, emit, guard  # noqa: E402


def main():
    guard()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-id", help="Defaults to this channel's current/latest job")
    args = p.parse_args()

    emit(call("/agent/status", {"job_id": args.job_id or ""}))


if __name__ == "__main__":
    main()
