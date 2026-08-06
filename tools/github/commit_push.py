#!/usr/bin/env python3
"""Commit your changes and push them to the repo's default branch.

Owner-only: restricted to the bot owner's Discord account.

This publishes DIRECTLY to the default branch (main/master) — there is no PR and
no review step, so the change is live on GitHub the moment this succeeds. Run
tools/github/status.py --diff first for anything non-trivial.

Stages all changes by default; pass --add to stage specific paths instead.
Refuses to push when nothing changed, and never force-pushes: if the remote has
moved on, it stops and tells you rather than overwriting someone else's commits.

Examples:
  commit_push.py --repo-path github_workspace/owner/name -m "fix typo in README"
  commit_push.py --repo-path ... -m "add tests" --add tests/ --dry-run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import WORKSPACE, default_branch, git, guard, resolve_repo_path  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import error, output  # noqa: E402

TRAILER = "Co-Authored-By: jaspt (Discord bot) <noreply@anthropic.com>"


def main():
    guard()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-path", required=True, help="Path printed by clone.py")
    p.add_argument("-m", "--message", required=True, help="Commit message")
    p.add_argument("--add", nargs="*", default=None,
                   help="Specific paths to stage (default: everything changed)")
    p.add_argument("--dry-run", action="store_true",
                   help="Stage and show what WOULD be committed, then stop")
    p.add_argument("--no-trailer", action="store_true",
                   help="Omit the bot co-author trailer")
    p.add_argument("--commit-everything", action="store_true",
                   help="In the user's OWN checkout, allow staging all changes "
                        "including work that was already there. Only with their "
                        "explicit say-so.")
    args = p.parse_args()

    path = resolve_repo_path(args.repo_path)

    if not git(["status", "--porcelain"], cwd=path).stdout.strip():
        error("nothing to commit — no files changed in this checkout",
              details={"path": path})

    # Guard: `git add -A` in the user's real working directory would sweep up
    # whatever they had in progress. Inside the throwaway workspace everything
    # present is ours, so blanket staging is fine there.
    in_workspace = path.startswith(os.path.realpath(WORKSPACE) + os.sep)
    if not args.add and not in_workspace and not args.commit_everything:
        changed = git(["status", "--porcelain"], cwd=path).stdout.strip().splitlines()
        error(
            f"{path} is an existing local checkout, not a scratch clone, and "
            f"has {len(changed)} changed file(s) — some may be the user's own "
            "work in progress. Refusing to stage everything blindly. Pass --add "
            "with the specific paths you changed, or --commit-everything if the "
            "user confirmed all of it should go in.",
            details={"changed_files": changed[:30]})

    # Stage. Pathspecs are passed after '--' so a leading dash can't be read
    # as a git option.
    if args.add:
        git(["add", "--"] + list(args.add), cwd=path)
    else:
        git(["add", "-A"], cwd=path)

    staged = git(["diff", "--cached", "--name-only"], cwd=path).stdout.strip()
    if not staged:
        error("nothing staged — the --add paths matched no changes",
              details={"add": args.add})
    staged_files = staged.splitlines()
    diffstat = git(["diff", "--cached", "--stat"], cwd=path).stdout.strip()

    branch = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path).stdout.strip()
    target = default_branch(path)

    if args.dry_run:
        output({
            "dry_run": True, "path": path, "branch": branch,
            "would_push_to": target,
            "staged_files": staged_files, "diffstat": diffstat,
            "note": "Re-run without --dry-run to commit and push.",
        })

    message = args.message if args.no_trailer else f"{args.message}\n\n{TRAILER}"
    git(["commit", "-m", message], cwd=path)
    sha = git(["rev-parse", "--short", "HEAD"], cwd=path).stdout.strip()

    # Push without --force. If the remote moved on, this fails and we say so
    # rather than clobbering commits that arrived since the clone.
    push = git(["push", "origin", f"HEAD:{target}"], cwd=path, check=False)
    if push.returncode != 0:
        stderr = (push.stderr or "").strip()
        hint = ""
        if "non-fast-forward" in stderr or "rejected" in stderr:
            hint = (" The remote has commits this clone doesn't. The commit is "
                    "made locally but NOT pushed — re-clone with --reset and "
                    "redo the edit, or pull and merge manually.")
        error(f"push to {target} failed.{hint}",
              details={"stderr": stderr[:2000], "local_commit": sha})

    remote = git(["remote", "get-url", "origin"], cwd=path).stdout.strip()
    slug = remote.replace("git@github.com:", "").replace(
        "https://github.com/", "").removesuffix(".git")

    output({
        "pushed": True,
        "repo": slug,
        "branch": target,
        "commit": sha,
        "url": f"https://github.com/{slug}/commit/{sha}",
        "files": staged_files,
        "diffstat": diffstat,
        "note": f"Live on {target} now — there was no review step.",
    })


if __name__ == "__main__":
    main()
