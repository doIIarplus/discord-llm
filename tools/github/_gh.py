"""Shared plumbing for the GitHub repo tools: owner gate, workspace, git runner.

Access control
--------------
These tools clone and PUSH to real repositories using the machine's SSH key, so
they are restricted to a single Discord user (the bot owner). The requester is
read from ``DISCORD_REQUESTING_USER_ID``, which bot.py injects into the tool
subprocess env — not from anything the model passes as an argument. Fails closed
when the requester can't be verified. Same mechanism as
tools/splitwise/_auth.py and tools/discord/_permissions.py.

Known limit: the guild these run in also has unrestricted Bash, so a user who
cannot invoke these tools could still ask the bot to run `git push` directly.
This gate stops the realistic failure mode (the model acting on another user's
request through the sanctioned path); it is not a sandbox.

Workspace
---------
Repos are cloned under ``github_workspace/<owner>/<repo>`` in the project dir,
reused across turns so a follow-up request doesn't re-clone. That directory is
gitignored.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import error  # noqa: E402
from _guild_access import require_integration  # noqa: E402

# This file is tools/github/_gh.py, so the project root is three levels up.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE = os.path.join(PROJECT_DIR, "github_workspace")

# Where to look for checkouts that already exist, so we reuse them instead of
# cloning a second copy of a repo the user already has locally. Colon-separated
# override via GITHUB_LOCAL_ROOTS. These roots also define what counts as a
# legal --repo-path: anything outside them is refused.
LOCAL_ROOTS = [
    os.path.realpath(p) for p in (
        os.environ.get("GITHUB_LOCAL_ROOTS")
        or f"{os.path.expanduser('~/projects')}:{WORKSPACE}"
    ).split(":") if p.strip()
]
# How deep under each root to look for a .git directory.
_SCAN_DEPTH = 3

# GitHub API token. Entirely optional: without it, only PUBLIC repos are
# visible. With it, private repos the token can see are included too.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""

# The only Discord user permitted to use these tools. Overridable for other
# deployments, but defaults to the bot owner (dollarplus).
OWNER_DISCORD_ID = os.environ.get("GITHUB_OWNER_DISCORD_ID", "118567805678256128")

# owner/repo, optionally as a full URL. Deliberately strict: no shell
# metacharacters, no path traversal, no ssh host override.
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

_GIT_TIMEOUT = 300


def require_owner():
    """Abort unless the requesting Discord user is the bot owner."""
    requester = os.environ.get("DISCORD_REQUESTING_USER_ID") or None
    if not requester:
        error(
            "GitHub tools are restricted to the bot owner, but the requesting "
            "user's identity could not be verified, so access was denied."
        )
    if str(requester) != str(OWNER_DISCORD_ID):
        error(
            "Access denied: the GitHub tools can push to real repositories and "
            "are restricted to the bot owner "
            f"(requesting discord_id={requester}). Relay this to the user "
            "plainly — do not retry or attempt a workaround.",
            details={"requesting_user_id": requester},
        )


def guard():
    """Every GitHub tool starts here: guild allowlist, then owner check."""
    require_integration("github")
    require_owner()


def normalize_repo_value(raw: str) -> str:
    """Canonicalize owner/repo, RAISING ValueError on bad input.

    The pure form of normalize_repo(). Callers running inside the bot process
    must use this one — normalize_repo() exits the interpreter, which is correct
    for a CLI tool and fatal for the bot.
    """
    if not raw or not raw.strip():
        raise ValueError("repo is required (owner/repo)")
    val = raw.strip()
    val = re.sub(r"^https?://github\.com/", "", val)
    val = re.sub(r"^git@github\.com:", "", val)
    val = re.sub(r"\.git$", "", val).strip("/")
    if not _REPO_RE.match(val):
        raise ValueError(f"invalid repo {raw!r}: expected 'owner/repo' (GitHub only)")
    return val


def normalize_repo(raw: str) -> str:
    """CLI-facing wrapper: same validation, but exits with a JSON error.

    Rejects anything that isn't a plain GitHub slug so a model-supplied value
    can't turn into an arbitrary ssh destination or a path traversal.
    """
    try:
        return normalize_repo_value(raw)
    except ValueError as e:
        error(str(e), details={"raw": raw})


def repo_dir(repo: str) -> str:
    """Where a fresh clone of owner/repo would go (inside WORKSPACE)."""
    path = os.path.realpath(os.path.join(WORKSPACE, *repo.split("/")))
    if not path.startswith(os.path.realpath(WORKSPACE) + os.sep):
        error(f"refusing to operate outside the workspace: {path}")
    return path


def _under_a_root(path: str) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in LOCAL_ROOTS)


def remote_slug(path: str):
    """The canonical owner/repo of a checkout's origin, or None."""
    p = git(["remote", "get-url", "origin"], cwd=path, check=False)
    if p.returncode != 0:
        return None
    url = p.stdout.strip()
    url = re.sub(r"^git@github\.com:", "", url)
    url = re.sub(r"^https?://(?:[^@]+@)?github\.com/", "", url)
    url = re.sub(r"\.git$", "", url).strip("/")
    return url if _REPO_RE.match(url) else None


def find_local_checkouts(repo: str):
    """Every existing checkout of `repo` found under LOCAL_ROOTS.

    This is what stops the tools cloning a second copy of something the user
    already has (e.g. ~/projects/discord_llm_bot is doIIarplus/discord-llm).
    """
    found = []
    for root in LOCAL_ROOTS:
        if not os.path.isdir(root):
            continue
        root_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.count(os.sep) - root_depth >= _SCAN_DEPTH:
                dirnames[:] = []
                continue
            # Don't descend into heavy or irrelevant trees.
            dirnames[:] = [d for d in dirnames
                           if d not in {"node_modules", "venv", ".venv", "__pycache__"}]
            if not os.path.isdir(os.path.join(dirpath, ".git")):
                continue
            dirnames[:] = []          # found a repo — don't recurse into it
            if remote_slug(dirpath) == repo:
                found.append(os.path.realpath(dirpath))
    return found


def resolve_repo_path(raw: str) -> str:
    """Validate a --repo-path: must be a git checkout under a known root.

    Roots are the user's project dir plus the tools' own workspace, so the
    model can work on repos already cloned locally — but not on arbitrary
    filesystem paths.
    """
    if not raw:
        error("--repo-path is required")
    path = os.path.realpath(raw)
    if not _under_a_root(path):
        error(
            f"refusing to operate on {raw!r}: GitHub tools only touch git "
            f"checkouts under {', '.join(LOCAL_ROOTS)}. Run "
            "tools/github/clone.py first and use the path it prints."
        )
    if not os.path.isdir(os.path.join(path, ".git")):
        error(f"not a git checkout: {path}")
    return path


def git(args, cwd=None, check=True, timeout=_GIT_TIMEOUT):
    """Run a git command, returning CompletedProcess. Never uses a shell."""
    env = dict(os.environ)
    # Non-interactive: fail fast instead of hanging on a credential or host-key
    # prompt inside a Discord turn.
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new")
    try:
        p = subprocess.run(
            ["git"] + args, cwd=cwd, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        error(f"git {' '.join(args[:2])} timed out after {timeout}s")
    if check and p.returncode != 0:
        error(
            f"git {' '.join(args[:2])} failed",
            details={"stderr": (p.stderr or "").strip()[:2000],
                     "stdout": (p.stdout or "").strip()[:1000]},
        )
    return p


def api_get(url: str):
    """GET the GitHub API. Uses GITHUB_TOKEN when set (needed for private repos)."""
    import json as _json
    import urllib.error
    import urllib.request

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "jaspt-github-tools",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return _json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = _json.loads(e.read().decode()).get("message", "")
        except Exception:
            pass
        if e.code == 403 and "rate limit" in detail.lower():
            error("GitHub API rate limit hit. Unauthenticated requests are capped "
                  "at 60/hour — set GITHUB_TOKEN in .env to raise it.",
                  details={"status": e.code, "message": detail})
        if e.code == 404:
            error(f"GitHub API 404 for {url} — user/org not found, or the repo is "
                  "private and no GITHUB_TOKEN is set.",
                  details={"status": e.code, "message": detail})
        error(f"GitHub API request failed ({e.code})",
              details={"url": url, "message": detail})
    except urllib.error.URLError as e:
        error(f"could not reach the GitHub API: {e.reason}")


def default_branch(path: str) -> str:
    """The remote's default branch (what 'push directly' targets)."""
    p = git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], cwd=path, check=False)
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().rsplit("/", 1)[-1]
    # Fall back to whatever is currently checked out.
    return git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path).stdout.strip()
