"""Isolated git worktrees for agent tasks.

Why worktrees rather than editing the checkout directly: the user's working copy
usually has uncommitted work in it (their editor is open on it). An agent doing
`git add -A` there would sweep that up, and two tasks at once would trample each
other. A worktree is a real branch in a separate directory sharing the same
object store — cheap to create, disposable, and completely inert with respect to
the main checkout.

Repo resolution reuses tools/github/_gh.py (find_local_checkouts / normalize_repo)
so "the repo the user already has" means the same thing everywhere.
"""

import os
import re
import subprocess
import sys
import unicodedata

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "tools"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "tools", "github"))

BRANCH_PREFIX = os.environ.get("AGENT_BRANCH_PREFIX", "jaspt")
WORKTREE_DIRNAME = ".worktrees"
_GIT_TIMEOUT = 300


class WorkspaceError(RuntimeError):
    """Raised when a worktree can't be prepared — surfaced to the user as-is."""


def _git(args, cwd=None, check=True):
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new")
    p = subprocess.run(["git"] + args, cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=_GIT_TIMEOUT)
    if check and p.returncode != 0:
        raise WorkspaceError(
            f"git {' '.join(args[:2])} failed: {(p.stderr or p.stdout).strip()[:500]}")
    return p


def slugify(text: str, max_len: int = 40) -> str:
    """A branch-safe slug from a free-text task description."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    text = re.sub(r"-{2,}", "-", text)[:max_len].strip("-")
    return text or "task"


def resolve_repo(repo: str):
    """Canonical slug + the local checkout to base a worktree on."""
    from _gh import find_local_checkouts, normalize_repo_value

    slug = normalize_repo_value(repo)
    checkouts = find_local_checkouts(slug)
    if not checkouts:
        raise WorkspaceError(
            f"no local checkout of {slug} found. Clone it first with "
            "tools/github/clone.py, then start the task.")
    return slug, checkouts[0]


def default_branch(checkout: str) -> str:
    p = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
             cwd=checkout, check=False)
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().rsplit("/", 1)[-1]
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=checkout).stdout.strip()


def create(repo: str, task: str, job_id: str) -> dict:
    """Make a worktree for a task. Returns {repo, checkout, path, branch, base}."""
    slug, checkout = resolve_repo(repo)

    # A repo mid-rebase/merge can't take a new worktree; say so plainly rather
    # than leaving a confusing git error.
    gitdir = _git(["rev-parse", "--git-dir"], cwd=checkout).stdout.strip()
    gitdir = os.path.join(checkout, gitdir) if not os.path.isabs(gitdir) else gitdir
    for marker in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD"):
        if os.path.exists(os.path.join(gitdir, marker)):
            raise WorkspaceError(
                f"{checkout} is in the middle of a {marker.split('-')[0]} — "
                "finish or abort that first, then retry.")

    base = default_branch(checkout)
    _git(["fetch", "origin", "--prune"], cwd=checkout, check=False)

    branch = f"{BRANCH_PREFIX}/{slugify(task)}-{job_id[:6]}"
    path = os.path.join(checkout, WORKTREE_DIRNAME, branch.replace("/", "_"))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Base off origin/<default> when we have it, so the agent starts from the
    # published state rather than whatever the user has checked out locally.
    start = f"origin/{base}"
    if _git(["rev-parse", "--verify", "--quiet", start], cwd=checkout,
            check=False).returncode != 0:
        start = "HEAD"

    _git(["worktree", "add", "-b", branch, path, start], cwd=checkout)
    return {"repo": slug, "checkout": checkout, "path": path,
            "branch": branch, "base": base, "start": start}


def summarize(worktree: str, base_ref: str) -> dict:
    """What the agent actually changed: commits, diffstat, files."""
    commits = _git(["log", f"{base_ref}..HEAD", "--pretty=%h %s"],
                   cwd=worktree, check=False).stdout.strip()
    diffstat = _git(["diff", "--stat", base_ref], cwd=worktree, check=False).stdout.strip()
    files = _git(["diff", "--name-only", base_ref], cwd=worktree, check=False).stdout.strip()
    uncommitted = _git(["status", "--porcelain"], cwd=worktree, check=False).stdout.strip()
    return {
        "commits": commits.splitlines() if commits else [],
        "diffstat": diffstat,
        "files": files.splitlines() if files else [],
        "uncommitted": uncommitted.splitlines() if uncommitted else [],
    }


def commit_all(worktree: str, message: str) -> str:
    """Commit whatever the agent left uncommitted. Returns the short sha or ''."""
    if not _git(["status", "--porcelain"], cwd=worktree, check=False).stdout.strip():
        return ""
    _git(["add", "-A"], cwd=worktree)
    _git(["commit", "-m", message], cwd=worktree)
    return _git(["rev-parse", "--short", "HEAD"], cwd=worktree).stdout.strip()


def has_commits(worktree: str, base_ref: str) -> bool:
    out = _git(["rev-list", "--count", f"{base_ref}..HEAD"],
               cwd=worktree, check=False).stdout.strip()
    return out.isdigit() and int(out) > 0


def remove(checkout: str, worktree: str, branch: str, keep_branch: bool) -> None:
    """Tear down a worktree. The branch survives when it has work worth keeping."""
    _git(["worktree", "remove", "--force", worktree], cwd=checkout, check=False)
    _git(["worktree", "prune"], cwd=checkout, check=False)
    if not keep_branch and branch:
        _git(["branch", "-D", branch], cwd=checkout, check=False)


def push_branch(worktree: str, branch: str) -> str:
    """Publish a task branch. Never force-pushes."""
    _git(["push", "-u", "origin", f"{branch}:{branch}"], cwd=worktree)
    return branch


def prune_stale(checkout: str) -> None:
    """Drop worktree records whose directories vanished (e.g. a killed job)."""
    _git(["worktree", "prune"], cwd=checkout, check=False)
