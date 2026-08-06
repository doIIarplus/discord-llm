#!/usr/bin/env python3
"""Hand a real coding job to a background agent, and return immediately.

Owner-only: restricted to the bot owner's Discord account.

Use this when the user asks for actual work on a repo — add a feature, fix a
bug, refactor, write tests. Do NOT use it to answer questions about code: for
"how does X work", just read the files and answer in the chat turn.

What happens: the bot creates an isolated git worktree off the repo's default
branch, runs a full Claude Code agent in it (with subagents available), and
streams progress into a message in this channel. The user's own checkout is
never touched.

You get a job_id back straight away — the work has NOT finished. Say what you
kicked off and stop; the progress message updates itself. Don't poll status in a
loop, and don't claim it's done.

Nothing is pushed. The job ends on a branch and waits for the user to ask.

Examples:
  start_task.py --repo kooner47/crees --task "add retry with backoff to the api client"
  start_task.py --repo doIIarplus/nova-tab --task "fix the off-by-one in tab reordering"
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
    p.add_argument("--repo", required=True,
                   help="owner/repo — must already be cloned locally")
    p.add_argument("--task", required=True,
                   help="What to do, in full. The agent sees only this, not the "
                        "chat history — so include the relevant context.")
    args = p.parse_args()

    info = call("/agent/start", {"repo": args.repo, "task": args.task})
    info["note"] = ("Job started in the background. Progress is being posted to "
                    "this channel automatically — tell the user what you kicked "
                    "off and stop. It is NOT done yet.")
    emit(info)


if __name__ == "__main__":
    main()
