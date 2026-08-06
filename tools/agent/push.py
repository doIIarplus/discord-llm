#!/usr/bin/env python3
"""Publish a finished job's branch to GitHub.

Owner-only: restricted to the bot owner's Discord account.

Call this ONLY when the user explicitly asks — "push it", "ship it", "yes push".
A finished job sits in `awaiting_push` with its work committed to a branch; this
is the step that makes it public, and it is the one irreversible part of the
flow. Never call it on your own initiative.

Pushes the task branch (not the default branch), so nothing lands on main until
the user merges. Never force-pushes.

Examples:
  push.py
  push.py --job-id 4f2a1b9c0d11
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
    p.add_argument("--job-id", help="Defaults to this channel's latest job")
    args = p.parse_args()

    result = call("/agent/push", {"job_id": args.job_id or ""})
    branch = result.get("pushed_branch") or result.get("branch")
    repo = result.get("repo")
    if branch and repo:
        result["url"] = f"https://github.com/{repo}/compare/{branch}?expand=1"
        result["note"] = ("Branch pushed. It is NOT merged — share the compare "
                          "link if the user wants to open a PR.")
    emit(result)


if __name__ == "__main__":
    main()
