"""Shared helpers for the DoorDash CLI tools.

These tools are thin subprocess wrappers around the already-installed `dd-cli`
binary (v0.2.2). Nothing here talks to DoorDash directly — no API calls, no
vendored client. If dd-cli isn't installed, the tools fail loudly rather than
falling back to anything.

Two things are enforced here rather than left to the prompt:

* `require_owner()` — DoorDash orders spend dollarplus's money, so the tools are
  restricted to that Discord user, checked against the trusted
  DISCORD_REQUESTING_USER_ID the bot injects (same pattern as
  tools/splitwise/_auth.py). Fails closed.
* `--intent` — every tool-backed dd-cli command in v0.2.2 requires it, so
  `run_dd` always appends it and every wrapper exposes it as a required arg.

Authentication note: dd-cli normally stores credentials in the OS keychain,
which does not exist under WSL. The headless path is the DD_CLI_ACCESS_TOKEN env
var, obtained by running `dd-cli export-token` on a desktop machine. run_dd
passes it through from the environment / .env when set, and translates the two
known failure modes (no keychain, no waitlist access) into structured errors the
model can relay verbatim.
"""

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import error  # noqa: E402

# Load .env so DD_CLI_ACCESS_TOKEN is available on standalone invocation, the
# same way tools/_common.py does it.
try:
    from dotenv import load_dotenv

    load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    )
except ImportError:
    pass

# The Discord user allowed to use DoorDash tools (dollarplus). Overridable via
# env for other deployments.
OWNER_DISCORD_ID = os.environ.get("DOORDASH_OWNER_DISCORD_ID", "118567805678256128")

DD_CLI_BIN = shutil.which("dd-cli") or os.path.expanduser("~/.local/bin/dd-cli")

# dd-cli exits with these on the two failure modes we can actually explain to a
# user. Anything else is passed through verbatim.
_KEYCHAIN_MARKER = "keychain unavailable"
_ACCESS_MARKERS = ("waitlist", "not have access", "no access", "access request")


def require_owner():
    """Abort unless the requesting Discord user is the DoorDash account owner."""
    requester = os.environ.get("DISCORD_REQUESTING_USER_ID") or None
    if not requester:
        error(
            "DoorDash tools are restricted to the account owner, but the "
            "requesting user's identity could not be verified, so access was "
            "denied."
        )
    if str(requester) != str(OWNER_DISCORD_ID):
        error(
            "Access denied: DoorDash tools are tied to dollarplus's personal "
            f"DoorDash account and can only be used by that user (requesting "
            f"discord_id={requester}).",
            details={"requesting_user_id": requester},
        )


def _require_binary():
    if not (os.path.isfile(DD_CLI_BIN) and os.access(DD_CLI_BIN, os.X_OK)):
        json.dump({"error": "dd-cli not installed"}, sys.stderr, indent=2)
        print(file=sys.stderr)
        sys.exit(1)


def _translate_failure(stderr: str, returncode: int):
    """Turn a known dd-cli failure into a structured error, or pass it through."""
    low = (stderr or "").lower()
    if _KEYCHAIN_MARKER in low:
        error(
            "not_authenticated",
            details={
                "message": (
                    "dd-cli has no keychain in WSL. Set DD_CLI_ACCESS_TOKEN "
                    "(get it by running `dd-cli export-token` on a desktop "
                    "machine)."
                )
            },
        )
    if any(marker in low for marker in _ACCESS_MARKERS):
        error(
            "no_access",
            details={
                "message": "account not off the DoorDash CLI waitlist yet"
            },
        )
    print(stderr.rstrip() or f"dd-cli exited with status {returncode}", file=sys.stderr)
    sys.exit(1)


def run_dd(args: list, intent: str, json_output: bool = True) -> dict:
    """Run dd-cli with `args`, always appending the required --intent.

    Returns the parsed JSON result, or {"output": <stdout>} when dd-cli emitted
    something that isn't JSON. Exits 1 (with a structured error on stderr) on
    any failure.
    """
    _require_binary()

    cmd = [DD_CLI_BIN]
    if json_output:
        cmd.append("--json-output")
    cmd += [str(a) for a in args]
    cmd += ["--intent", intent]

    env = os.environ.copy()
    token = os.environ.get("DD_CLI_ACCESS_TOKEN")
    if token:
        env["DD_CLI_ACCESS_TOKEN"] = token

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env
        )
    except subprocess.TimeoutExpired:
        error("dd-cli timed out after 120s", details={"command": cmd[1:]})
    except OSError as e:
        error(f"could not run dd-cli: {e}", details={"binary": DD_CLI_BIN})

    if proc.returncode != 0:
        _translate_failure(proc.stderr, proc.returncode)

    # A zero exit with a keychain/access message on stderr shouldn't happen, but
    # dd-cli's gate has been inconsistent enough that it's worth catching.
    low = (proc.stderr or "").lower()
    if _KEYCHAIN_MARKER in low or any(m in low for m in _ACCESS_MARKERS):
        _translate_failure(proc.stderr, proc.returncode)

    stdout = proc.stdout or ""
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return {"output": stdout}


INTENT_HELP = (
    "REQUIRED. The goal behind this workflow in plain language — who it is for "
    'and why, e.g. "Summary: Help the user order lunch". Not a restatement of '
    "the command."
)


def add_intent_arg(parser):
    """Attach the required --intent argument every dd-cli command needs."""
    parser.add_argument("--intent", required=True, help=INTENT_HELP)
    return parser
