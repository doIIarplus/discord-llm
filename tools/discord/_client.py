"""Discord REST API client for CLI tools.

Reads DISCORD_BOT_TOKEN from environment. All methods are synchronous
(using urllib) since CLI tools run as short-lived processes.
No external dependencies beyond stdlib.
"""

import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import error

BASE_URL = "https://discord.com/api/v10"


class DiscordClient:
    def __init__(self):
        self.token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not self.token:
            error("DISCORD_BOT_TOKEN not set in environment")

    def _headers(self):
        return {
            "Authorization": f"Bot {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (https://discord.com, 1.0)",
        }

    def get(self, endpoint, params=None):
        """GET request to Discord API. Returns parsed JSON."""
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        req = urllib.request.Request(url, headers=self._headers())
        return self._do(req, endpoint)

    def post(self, endpoint, data=None):
        """POST JSON to Discord API. Returns parsed JSON."""
        url = f"{BASE_URL}{endpoint}"
        body = json.dumps(data or {}).encode("utf-8")
        req = urllib.request.Request(url, headers=self._headers(), data=body, method="POST")
        return self._do(req, endpoint)

    def put(self, endpoint, data=None):
        """PUT to Discord API. Returns parsed JSON or None for 204."""
        url = f"{BASE_URL}{endpoint}"
        body = json.dumps(data or {}).encode("utf-8") if data else b""
        req = urllib.request.Request(url, headers=self._headers(), data=body, method="PUT")
        return self._do(req, endpoint)

    def delete(self, endpoint):
        """DELETE request to Discord API."""
        url = f"{BASE_URL}{endpoint}"
        req = urllib.request.Request(url, headers=self._headers(), method="DELETE")
        return self._do(req, endpoint)

    def _do(self, req, endpoint):
        """Execute request, handle errors."""
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            error(
                f"Discord API error: {e.code}",
                details={"endpoint": endpoint, "body": body},
            )
        except Exception as e:
            error(f"Request failed: {e}")
