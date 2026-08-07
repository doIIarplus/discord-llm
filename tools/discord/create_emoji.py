#!/usr/bin/env python3
"""Upload a custom emoji to a Discord guild.

Requires the Manage Expressions permission (a.k.a. Manage Emojis and Stickers).
The image source is either a URL (downloaded here) or a local file inside the
project directory. Discord accepts png/jpeg/gif up to 256KB; a webp source is
converted to png automatically.

Optionally restrict the emoji to specific roles with --roles.

Examples:
  create_emoji.py --guild-id 363154169294618625 --name bruh --url https://example.com/bruh.png
  create_emoji.py --guild-id 363154169294618625 --name chart --file api_out/chart.png
  create_emoji.py --guild-id 363154169294618625 --name vip --url https://example.com/a.gif --roles 1234 5678
"""

import argparse
import base64
import io
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error
from discord._client import DiscordClient
from discord._permissions import require_permission
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration

# sandbox.py lives at the project root, one level above tools/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sandbox import safe_path, SandboxViolation

_NAME_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")

# Discord's hard cap on emoji image size.
MAX_BYTES = 256 * 1024

# Browser-ish UA: Discord's CDN (and plenty of other hosts) 403 urllib's default.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def detect_mime(data):
    """Return the image mime type from the bytes' magic number, or None.

    Sniffed from content rather than the file extension — the extension is
    model-supplied and a mislabeled upload just gets rejected by Discord.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def fetch_url(url):
    """Download image bytes from a URL."""
    if not url.lower().startswith(("http://", "https://")):
        error(f"--url must be an http(s) URL, got '{url}'")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            # Read a little past the cap so oversize sources fail loudly below.
            return resp.read(MAX_BYTES * 8 + 1)
    except urllib.error.HTTPError as e:
        error(f"Could not download image: HTTP {e.code}", details={"url": url})
    except Exception as e:
        error(f"Could not download image: {e}", details={"url": url})


def read_file(path):
    """Read image bytes from a local path confined to the project directory."""
    try:
        resolved = safe_path(path)
    except SandboxViolation as e:
        error(str(e))
    if not os.path.isfile(resolved):
        error(f"File not found: {path}", details={"resolved": resolved})
    with open(resolved, "rb") as f:
        return f.read()


def to_uploadable(data):
    """Return (mime, data) in a format Discord accepts, converting webp to png."""
    mime = detect_mime(data)
    if mime is None:
        error(
            "Unrecognized image format. Supported: png, jpeg, gif, webp "
            "(webp is converted to png)."
        )
    if mime == "image/webp":
        try:
            from PIL import Image
        except ImportError:
            error("Pillow is required to convert webp images to png.")
        try:
            with Image.open(io.BytesIO(data)) as img:
                buf = io.BytesIO()
                img.convert("RGBA").save(buf, format="PNG")
                data = buf.getvalue()
        except Exception as e:
            error(f"Failed to convert webp to png: {e}")
        mime = "image/png"
    return mime, data


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('discord')
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--guild-id", required=True, help="Discord guild (server) ID")
    parser.add_argument("--name", required=True,
                        help="Emoji name: 2-32 chars of letters, digits, underscore")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Download the emoji image from this URL")
    source.add_argument("--file", help="Read the emoji image from this path (inside the project dir)")
    parser.add_argument("--roles", nargs="+", metavar="ID", default=None,
                        help="Restrict the emoji to these role IDs")
    args = parser.parse_args()

    require_permission("MANAGE_EXPRESSIONS", guild_id=args.guild_id)

    if not _NAME_RE.match(args.name):
        error(
            f"Invalid emoji name '{args.name}'. Must be 2-32 characters of "
            f"letters, digits, or underscores."
        )

    data = fetch_url(args.url) if args.url else read_file(args.file)
    if not data:
        error("Image source is empty.")

    mime, data = to_uploadable(data)

    if len(data) > MAX_BYTES:
        error(
            f"Image is {len(data) // 1024}KB, over Discord's 256KB emoji limit. "
            f"Resize it (emojis render at 128x128) and try again.",
            details={"bytes": len(data), "max_bytes": MAX_BYTES},
        )

    b64 = base64.b64encode(data).decode("ascii")
    body = {
        "name": args.name,
        "image": f"data:{mime};base64,{b64}",
        "roles": args.roles or [],
    }

    client = DiscordClient()
    emoji = client.post(f"/guilds/{args.guild_id}/emojis", body)

    animated = bool(emoji.get("animated"))
    prefix = "a" if animated else ""

    output({
        "success": True,
        "action": "create_emoji",
        "guild_id": args.guild_id,
        "id": emoji["id"],
        "name": emoji["name"],
        "animated": animated,
        "mention": f"<{prefix}:{emoji['name']}:{emoji['id']}>",
        "bytes": len(data),
        "mime": mime,
    })


if __name__ == "__main__":
    main()
