#!/usr/bin/env python3
"""Show what you've changed in a cloned repo, before publishing it.

Owner-only: restricted to the bot owner's Discord account.

Since commit_push.py pushes straight to the default branch, run this first when
the change is non-trivial — it's the last chance to catch an edit that went
somewhere unintended.

Examples:
  status.py --repo-path github_workspace/owner/name
  status.py --repo-path github_workspace/owner/name --diff
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import default_branch, git, guard, resolve_repo_path  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import output  # noqa: E402

_MAX_DIFF = 20000


def main():
    guard()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-path", required=True, help="Path printed by clone.py")
    p.add_argument("--diff", action="store_true",
                   help="Include the actual diff, not just the file list")
    args = p.parse_args()

    path = resolve_repo_path(args.repo_path)

    porcelain = git(["status", "--porcelain"], cwd=path).stdout.strip()
    changed = []
    for line in porcelain.splitlines():
        if len(line) > 3:
            changed.append({"status": line[:2].strip(), "file": line[3:]})

    result = {
        "path": path,
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path).stdout.strip(),
        "default_branch": default_branch(path),
        "head": git(["rev-parse", "--short", "HEAD"], cwd=path).stdout.strip(),
        "changed_files": changed,
        "change_count": len(changed),
        "diffstat": git(["diff", "--stat", "HEAD"], cwd=path, check=False).stdout.strip(),
    }

    if args.diff:
        d = git(["diff", "HEAD"], cwd=path, check=False).stdout
        result["diff_truncated"] = len(d) > _MAX_DIFF
        result["diff"] = d[:_MAX_DIFF]

    output(result)


if __name__ == "__main__":
    main()
