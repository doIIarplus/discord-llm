"""Shared client for talking to the bot's local image service.

These tools are deliberately dependency-free (stdlib urllib only) and must NOT
import torch/diffusers — the whole point is to reuse the model already resident
in the bot process rather than loading a second copy per subprocess.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# tools/images/.image_service — written by the bot at startup, mode 0600.
# IMAGE_SERVICE_TOKEN_FILE overrides it (must match the bot's config) so a test
# harness can point at its own service without disturbing a live bot's.
TOKEN_FILE = os.environ.get(
    "IMAGE_SERVICE_TOKEN_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".image_service"),
)


def _read_service() -> tuple:
    """Return (base_url, token). Exits with a clear error if the bot isn't up."""
    try:
        with open(TOKEN_FILE) as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except FileNotFoundError:
        die(
            "image service is not running (no token file). These tools require "
            "the bot process to be up — it hosts the resident Flux pipeline."
        )
    if len(lines) < 3:
        die(f"malformed image service token file: {TOKEN_FILE}")
    host, port, token = lines[0], lines[1], lines[2]
    return f"http://{host}:{port}", token


def die(message: str, code: int = 1):
    """Print an error to stderr and exit — the tools/ convention."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def post(endpoint: str, payload: dict) -> dict:
    """POST JSON to the image service and return the decoded response."""
    base_url, token = _read_service()

    # Forwarded by the bot so the service can attribute images to this turn and
    # attach them to the reply. Absent -> the image is still produced, but the
    # bot won't auto-attach it.
    request_id = os.environ.get("IMAGE_REQUEST_ID", "")
    if request_id:
        payload = {**payload, "request_id": request_id}

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        # Generous timeout: a cold Flux load plus generation can take a while.
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", e.reason)
        except Exception:
            detail = e.reason
        die(f"image service returned {e.code}: {detail}")
    except urllib.error.URLError as e:
        die(
            f"could not reach image service at {base_url}: {e.reason}. "
            "Is the bot running?"
        )


def emit(result: dict) -> None:
    """Print the tool's JSON result to stdout (tools/ convention)."""
    print(json.dumps(result, indent=2))


# Dimension presets, mirroring tools/flux/generate.py so the two CLIs agree.
PRESETS = {
    "square":       (1024, 1024),
    "square-hq":    (1536, 1536),
    "portrait":     (832,  1216),
    "portrait-hq":  (1024, 1536),
    "landscape":    (1216, 832),
    "landscape-hq": (1536, 1024),
}


def resolve_dims(preset: str, width, height) -> tuple:
    """Preset dims, with explicit --width/--height taking precedence."""
    w, h = PRESETS[preset]
    return (width or w, height or h)
