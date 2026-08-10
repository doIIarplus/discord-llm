"""Sandboxed git workspace for agent tasks.

Everything happens under ~/git_projects — the agent never touches the user's own
checkouts in ~/projects. That was the earlier design and it had two problems: a
worktree created inside a repo shows up as untracked `.worktrees/` in the user's
`git status`, and any blanket `git add` risked sweeping up work in progress from
their editor. A separate sandbox removes both hazards by construction.

Layout:
    ~/git_projects/<owner>/<repo>/                     working clone, kept fresh
    ~/git_projects/.worktrees/<owner>__<repo>/<branch>/  one per task

Repos are cloned on demand, so "work on X" just works without a setup step.
Worktrees are used within the sandbox because they're cheap (shared object
store) and let several tasks on one repo run without colliding.
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
# The sandbox. Deliberately outside the bot's project and outside ~/projects.
WORKSPACE_ROOT = os.path.realpath(
    os.environ.get("AGENT_WORKSPACE", os.path.expanduser("~/git_projects")))
WORKTREE_ROOT = os.path.join(WORKSPACE_ROOT, ".worktrees")
_GIT_TIMEOUT = 600


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


def sandbox_path(slug: str) -> str:
    """Where a repo lives in the sandbox, confined to WORKSPACE_ROOT."""
    path = os.path.realpath(os.path.join(WORKSPACE_ROOT, *slug.split("/")))
    if not path.startswith(WORKSPACE_ROOT + os.sep):
        raise WorkspaceError(f"refusing to operate outside the sandbox: {path}")
    return path


def resolve_repo(repo: str, clone_if_missing: bool = True):
    """Canonical slug + its sandbox clone, cloning on demand.

    Never returns a path in ~/projects — the agent works only in the sandbox, so
    the user's own checkouts and any work in progress there stay untouched.
    """
    from _gh import normalize_repo_value

    try:
        slug = normalize_repo_value(repo)
    except ValueError as e:
        raise WorkspaceError(str(e))

    path = sandbox_path(slug)
    if os.path.isdir(os.path.join(path, ".git")):
        # Keep it current, but never destroy anything: fetch only.
        _git(["fetch", "origin", "--prune"], cwd=path, check=False)
        return slug, path

    if not clone_if_missing:
        raise WorkspaceError(f"{slug} is not cloned in the sandbox yet")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        _git(["clone", f"git@github.com:{slug}.git", path])
    except WorkspaceError as e:
        raise WorkspaceError(
            f"could not clone {slug} into the sandbox — check the repo exists "
            f"and the machine's SSH key can read it. ({e})")
    return slug, path


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
    path = os.path.join(WORKTREE_ROOT, slug.replace("/", "__"),
                        branch.replace("/", "_"))
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


class PushDenied(WorkspaceError):
    """Push rejected because the key can't write to that repo (needs a fork)."""


_DENIED_MARKERS = ("permission denied", "403", "access rights",
                   "repository not found", "denied to")


def push_branch(worktree: str, branch: str) -> dict:
    """Publish a task branch. Never force-pushes.

    Distinguishes "you can't write here" from other failures, because the first
    is a fork situation the user can act on and the second is a real error.
    """
    p = _git(["push", "-u", "origin", f"{branch}:{branch}"],
             cwd=worktree, check=False)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        low = err.lower()
        if any(m in low for m in _DENIED_MARKERS):
            raise PushDenied(
                "no write access to this repo, so the branch stays local. "
                "The work is committed and safe — to publish it you'd need a "
                "fork (set GITHUB_TOKEN to enable that) or push access.")
        raise WorkspaceError(f"push failed: {err[:400]}")
    sha = _git(["rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    return {"branch": branch, "sha": sha, "short_sha": sha[:8]}


def delete_branch(checkout: str, branch: str, remote: bool = True) -> None:
    """Remove a task branch locally and (optionally) on the remote."""
    if remote:
        _git(["push", "origin", "--delete", branch], cwd=checkout, check=False)
    _git(["branch", "-D", branch], cwd=checkout, check=False)


def prune_stale(checkout: str) -> None:
    """Drop worktree records whose directories vanished (e.g. a killed job)."""
    _git(["worktree", "prune"], cwd=checkout, check=False)
