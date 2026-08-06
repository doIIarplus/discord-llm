#!/usr/bin/env python3
"""Stop the background coding job in this channel.

Owner-only: restricted to the bot owner's Discord account.

Kills the agent process and removes its worktree. Any commits it already made
survive on the branch; uncommitted edits in the worktree are lost. Use when the
user says stop / cancel / never mind.

Examples:
  cancel.py
  cancel.py --job-id 4f2a1b9c0d11
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
    p.add_argument("--job-id", help="Defaults to this channel's running job")
    args = p.parse_args()

    emit(call("/agent/cancel", {"job_id": args.job_id or ""}))


if __name__ == "__main__":
    main()
