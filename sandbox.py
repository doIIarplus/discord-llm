"""Path sandboxing utilities for the Discord LLM Bot.

All file operations in the bot must go through these helpers to ensure
no file access escapes the project directory.
"""

import os

# Resolved project root — every file operation is confined here
PROJECT_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))


class SandboxViolation(Exception):
    """Raised when a file operation attempts to escape the project directory."""


def safe_path(path: str) -> str:
    """Resolve *path* and verify it lives under PROJECT_DIR.

    Accepts both relative and absolute paths.  Symlinks are resolved so
    ``../../etc/passwd`` style tricks are caught.

    Returns the resolved absolute path on success.
    Raises SandboxViolation if the path escapes the project directory.
    """
    resolved = os.path.realpath(os.path.join(PROJECT_DIR, path))
    if not resolved.startswith(PROJECT_DIR + os.sep) and resolved != PROJECT_DIR:
        raise SandboxViolation(
            f"Access denied: {path!r} resolves to {resolved!r} which is outside "
            f"the project directory {PROJECT_DIR!r}"
        )
    return resolved


def safe_open(path: str, mode: str = "r", **kwargs):
    """Drop-in replacement for open() that enforces the sandbox."""
    return open(safe_path(path), mode, **kwargs)
