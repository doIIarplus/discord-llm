#!/usr/bin/env python3
"""List the GitHub repos an account owns, and say which are already cloned locally.

Owner-only: restricted to the bot owner's Discord account.

Use this when the user refers to a repo vaguely ("my bot repo", "that astro
site") — list first, match the name, then act. Saves guessing at slugs.

Visibility: without GITHUB_TOKEN set, only PUBLIC repos are returned, because
the API is queried unauthenticated. Set GITHUB_TOKEN in .env to include private
repos. The output says which mode was used, so don't tell the user "you have N
repos" without checking `includes_private`.

Each entry carries `local_path` when a checkout already exists on this machine —
pass that straight to status.py / commit_push.py instead of cloning again.

Examples:
  list_repos.py
  list_repos.py --owner someone-else
  list_repos.py --match bot --sort updated --limit 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import (  # noqa: E402
    GITHUB_TOKEN, LOCAL_ROOTS, api_get, find_local_checkouts, guard, git,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import output  # noqa: E402

DEFAULT_OWNER = os.environ.get("GITHUB_DEFAULT_OWNER", "doIIarplus")


def main():
    guard()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", default=DEFAULT_OWNER,
                   help=f"GitHub user/org (default: {DEFAULT_OWNER})")
    p.add_argument("--match", help="Only repos whose name/description contains this")
    p.add_argument("--sort", default="updated",
                   choices=["updated", "created", "pushed", "full_name"])
    p.add_argument("--limit", type=int, default=50, help="Max repos to return")
    p.add_argument("--local-only", action="store_true",
                   help="Only repos already cloned on this machine")
    args = p.parse_args()

    # Authenticated: /user/repos includes private. Unauthenticated: public only.
    if GITHUB_TOKEN and args.owner == DEFAULT_OWNER:
        url = f"https://api.github.com/user/repos?per_page=100&sort={args.sort}&affiliation=owner"
    else:
        url = f"https://api.github.com/users/{args.owner}/repos?per_page=100&sort={args.sort}"

    data = api_get(url)
    if not isinstance(data, list):
        data = []

    # One filesystem scan, reused for every repo, instead of walking per repo.
    local_by_slug = {}
    for root in LOCAL_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            d = os.path.join(root, entry)
            if not os.path.isdir(os.path.join(d, ".git")):
                continue
            from _gh import remote_slug
            slug = remote_slug(d)
            if slug:
                local_by_slug.setdefault(slug, os.path.realpath(d))

    repos = []
    for r in data:
        name, full = r.get("name", ""), r.get("full_name", "")
        desc = r.get("description") or ""
        if args.match and args.match.lower() not in f"{name} {desc}".lower():
            continue
        local = local_by_slug.get(full)
        if args.local_only and not local:
            continue
        entry = {
            "repo": full,
            "private": bool(r.get("private")),
            "description": desc[:160],
            "language": r.get("language"),
            "default_branch": r.get("default_branch"),
            "updated_at": r.get("pushed_at") or r.get("updated_at"),
            "local_path": local,
        }
        if local:
            dirty = git(["status", "--porcelain"], cwd=local, check=False).stdout.strip()
            entry["local_uncommitted_changes"] = len(dirty.splitlines()) if dirty else 0
        repos.append(entry)

    repos = repos[:args.limit]
    output({
        "owner": args.owner,
        "count": len(repos),
        "includes_private": bool(GITHUB_TOKEN),
        "visibility_note": (
            "Authenticated — private repos included."
            if GITHUB_TOKEN else
            "Unauthenticated — PUBLIC repos only. Private repos are NOT listed; "
            "set GITHUB_TOKEN in .env to see them."
        ),
        "already_cloned": sum(1 for r in repos if r["local_path"]),
        "repos": repos,
        "note": "Repos with local_path are already on disk — pass that path to "
                "status.py / commit_push.py rather than cloning again.",
    })


if __name__ == "__main__":
    main()
