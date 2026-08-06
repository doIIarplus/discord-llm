#!/usr/bin/env python3
"""Get a local working copy of a GitHub repo — reusing one if it already exists.

Owner-only: restricted to the bot owner's Discord account.

Checks the user's project directory and the tools' workspace for an existing
checkout of the repo FIRST, and only clones when there isn't one. So asking for
doIIarplus/discord-llm hands back ~/projects/discord_llm_bot rather than making
a second copy that would drift from the one they actually work in.

Read the response before editing:
  - `reused: true` means this is the user's REAL working directory. If
    `uncommitted_changes` is non-zero, that's THEIR work in progress — never
    sweep it into your commit. Use commit_push.py --add with explicit paths.
  - `reused: false` means a throwaway clone in the workspace; safe to change freely.

Examples:
  clone.py --repo doIIarplus/discord-llm
  clone.py --repo owner/name --ref some-branch
  clone.py --repo owner/name --fresh     # ignore local copies, clone into workspace
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import (  # noqa: E402
    LOCAL_ROOTS, WORKSPACE, default_branch, find_local_checkouts, git, guard,
    normalize_repo, repo_dir,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import output  # noqa: E402


def describe(path, repo, reused, cloned_now):
    dirty = git(["status", "--porcelain"], cwd=path, check=False).stdout.strip()
    n_dirty = len(dirty.splitlines()) if dirty else 0
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path).stdout.strip()
    log = git(["log", "-3", "--pretty=%h %s"], cwd=path, check=False).stdout.strip()

    result = {
        "repo": repo,
        "path": path,
        "reused": reused,
        "cloned_now": cloned_now,
        "branch": branch,
        "default_branch": default_branch(path),
        "head": git(["rev-parse", "--short", "HEAD"], cwd=path).stdout.strip(),
        "uncommitted_changes": n_dirty,
        "recent_commits": log.splitlines(),
    }
    if reused:
        result["warning"] = (
            "This is an EXISTING local checkout, not a scratch clone — most "
            "likely the user's real working copy."
        )
        if n_dirty:
            result["warning"] += (
                f" It has {n_dirty} uncommitted change(s) that were already there. "
                "Do NOT commit them as part of your work: pass explicit paths to "
                "commit_push.py --add."
            )
            result["preexisting_changes"] = dirty.splitlines()[:20]
    return result


def main():
    guard()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="owner/repo (or a github.com URL)")
    p.add_argument("--ref", help="branch to check out (default: the repo's default branch)")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore existing local copies; clone into the workspace")
    p.add_argument("--reset", action="store_true",
                   help="DISCARD local changes and hard-reset to the remote state. "
                        "Refused on a reused checkout unless --force-reset is given.")
    p.add_argument("--force-reset", action="store_true",
                   help="Allow --reset to destroy uncommitted work in the user's own checkout")
    args = p.parse_args()

    repo = normalize_repo(args.repo)

    # 1. Reuse an existing checkout unless told otherwise.
    existing = [] if args.fresh else find_local_checkouts(repo)
    if existing:
        path = existing[0]
        git(["fetch", "origin", "--prune"], cwd=path, check=False)

        if args.reset:
            dirty = git(["status", "--porcelain"], cwd=path, check=False).stdout.strip()
            if dirty and not args.force_reset:
                from _common import error
                error(
                    f"refusing to --reset {path}: it's the user's own checkout and "
                    f"has {len(dirty.splitlines())} uncommitted change(s), which a "
                    "reset would destroy. Re-run with --force-reset only if the user "
                    "explicitly asked to discard that work, or use --fresh to clone "
                    "a separate copy instead.",
                    details={"changes": dirty.splitlines()[:20]})
            branch = args.ref or default_branch(path)
            git(["checkout", "-B", branch, f"origin/{branch}"], cwd=path)
            git(["reset", "--hard", f"origin/{branch}"], cwd=path)
            git(["clean", "-fd"], cwd=path)
        elif args.ref:
            git(["checkout", args.ref], cwd=path, check=False)

        result = describe(path, repo, reused=True, cloned_now=False)
        if len(existing) > 1:
            result["other_local_copies"] = existing[1:]
        result["searched_roots"] = LOCAL_ROOTS
        output(result)

    # 2. Otherwise clone into the workspace.
    path = repo_dir(repo)
    os.makedirs(WORKSPACE, exist_ok=True)
    fresh_clone = not os.path.isdir(os.path.join(path, ".git"))
    if fresh_clone:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        git(["clone", f"git@github.com:{repo}.git", path])
    else:
        git(["fetch", "origin", "--prune"], cwd=path, check=False)

    branch = args.ref or default_branch(path)
    if args.reset:
        git(["checkout", "-B", branch, f"origin/{branch}"], cwd=path)
        git(["reset", "--hard", f"origin/{branch}"], cwd=path)
        git(["clean", "-fd"], cwd=path)
    elif not fresh_clone:
        dirty = bool(git(["status", "--porcelain"], cwd=path, check=False).stdout.strip())
        if not dirty:
            git(["checkout", branch], cwd=path, check=False)
            git(["merge", "--ff-only", f"origin/{branch}"], cwd=path, check=False)
    elif args.ref:
        git(["checkout", args.ref], cwd=path)

    output(describe(path, repo, reused=False, cloned_now=fresh_clone))


if __name__ == "__main__":
    main()
