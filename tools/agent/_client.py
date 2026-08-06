"""Shared client for the agent job endpoints on the bot's local service.

Stdlib only — same shape as tools/images/_client.py, and it talks to the same
loopback service using the same token file.

Access: these tools start a process that edits and can publish code, so they
carry the same gates as the GitHub tools — the guild allowlist plus an owner
check on the trusted DISCORD_REQUESTING_USER_ID.
"""

import json
import os
import sys
import urllib.error
import urllib.request

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, "github"))

TOKEN_FILE = os.environ.get(
    "IMAGE_SERVICE_TOKEN_FILE",
    os.path.join(TOOLS_DIR, "images", ".image_service"),
)


def die(message, details=None, code=1):
    err = {"error": message}
    if details is not None:
        err["details"] = details
    json.dump(err, sys.stderr, indent=2, default=str)
    print(file=sys.stderr)
    sys.exit(code)


def guard():
    """Guild allowlist, then owner check. Both enforced in code."""
    from _guild_access import require_integration
    require_integration("agent")
    from _gh import require_owner
    require_owner()


def _service():
    try:
        with open(TOKEN_FILE) as f:
            lines = [l.strip() for l in f.read().splitlines() if l.strip()]
    except FileNotFoundError:
        die("the bot's local service isn't running (no token file). Agent jobs "
            "run inside the bot process, so the bot has to be up.")
    if len(lines) < 3:
        die(f"malformed service token file: {TOKEN_FILE}")
    return f"http://{lines[0]}:{lines[1]}", lines[2]


def call(endpoint, payload):
    """POST to the service, injecting the trusted Discord context."""
    base, token = _service()
    payload = dict(payload)
    payload.setdefault("guild_id", os.environ.get("DISCORD_REQUESTING_GUILD_ID"))
    payload.setdefault("channel_id", os.environ.get("DISCORD_REQUESTING_CHANNEL_ID"))
    payload.setdefault("requester_id", os.environ.get("DISCORD_REQUESTING_USER_ID"))

    req = urllib.request.Request(
        f"{base}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", e.reason)
        except Exception:
            detail = e.reason
        die(f"{detail}", details={"status": e.code})
    except urllib.error.URLError as e:
        die(f"could not reach the bot service at {base}: {e.reason}")


def emit(data):
    json.dump(data, sys.stdout, indent=2, default=str)
    print()
    sys.exit(0)
