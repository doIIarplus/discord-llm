#!/usr/bin/env python3
"""Send a message to a Discord channel via webhook.

Webhooks are per-channel URLs that let you POST messages without a bot connection.
Create one in Discord: Channel Settings > Integrations > Webhooks > New Webhook.

Store webhook URLs in .env as DISCORD_WEBHOOK_<NAME>=https://discord.com/api/webhooks/...
or pass directly with --webhook-url.

Examples:
  # Using a named webhook from .env (DISCORD_WEBHOOK_GENERAL)
  send_webhook.py --webhook general --content "Hello from the scheduler!"

  # Using a direct URL
  send_webhook.py --webhook-url "https://discord.com/api/webhooks/123/abc" --content "Reminder: standup time"

  # Mention a user (use their Discord user ID)
  send_webhook.py --webhook general --content "<@118567805678256128> hey, monthly reminder"

  # Custom display name
  send_webhook.py --webhook general --content "Server rebooting in 5m" --username "System Alert"
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error


def _resolve_webhook_url(name=None, url=None):
    """Resolve a webhook URL from a name (env var lookup) or direct URL."""
    if url:
        return url
    if name:
        env_key = f"DISCORD_WEBHOOK_{name.upper()}"
        resolved = os.environ.get(env_key, "")
        if not resolved:
            error(
                f"Webhook '{name}' not found. Set {env_key} in .env or environment.",
                details={"env_var": env_key},
            )
        return resolved
    error("Must specify either --webhook NAME or --webhook-url URL")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--webhook", metavar="NAME",
                        help="Named webhook (looks up DISCORD_WEBHOOK_<NAME> from env)")
    parser.add_argument("--webhook-url", metavar="URL",
                        help="Direct webhook URL")
    parser.add_argument("--content", required=True,
                        help="Message content (supports Discord markdown and <@USER_ID> mentions)")
    parser.add_argument("--username", default=None,
                        help="Override the webhook's display name")
    args = parser.parse_args()

    webhook_url = _resolve_webhook_url(name=args.webhook, url=args.webhook_url)

    payload = {"content": args.content}
    if args.username:
        payload["username"] = args.username

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        headers={"Content-Type": "application/json"},
        data=data,
    )

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        # Discord returns 204 No Content on success
        if resp.status in (200, 204):
            output({"sent": True, "content": args.content})
        else:
            error(f"Unexpected status: {resp.status}", details={"body": resp.read().decode()})
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        error(f"Discord webhook error: {e.code}", details={"body": body})
    except Exception as e:
        error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
