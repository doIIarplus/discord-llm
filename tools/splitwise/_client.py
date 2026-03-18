"""Splitwise API client for CLI tools.

Reads SPLITWISE_API_KEY from environment. All methods are synchronous
(using requests) since CLI tools run as short-lived processes.
"""

import os
import sys

import requests

# Allow imports from parent tools/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import error

BASE_URL = "https://secure.splitwise.com/api/v3.0"


class SplitwiseClient:
    def __init__(self):
        self.api_key = os.environ.get("SPLITWISE_API_KEY", "")
        if not self.api_key:
            error("SPLITWISE_API_KEY not set in environment")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _auth_header(self):
        """Auth-only header (no Content-Type) for form-encoded POSTs."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def get(self, endpoint):
        """GET request to the Splitwise API. Returns parsed JSON."""
        url = f"{BASE_URL}{endpoint}"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            error(
                f"Splitwise API error: {resp.status_code}",
                details={"endpoint": endpoint, "body": resp.text},
            )
        return resp.json()

    def post(self, endpoint, data):
        """POST JSON to the Splitwise API. Returns parsed JSON."""
        url = f"{BASE_URL}{endpoint}"
        resp = requests.post(url, headers=self._headers(), json=data, timeout=30)
        if not resp.ok:
            error(
                f"Splitwise API error: {resp.status_code}",
                details={"endpoint": endpoint, "body": resp.text},
            )
        return resp.json()

    def post_form(self, endpoint, data):
        """POST form-encoded data to the Splitwise API. Returns parsed JSON.

        Used by create_expense which requires form encoding with the
        users__N__field_name format.
        """
        url = f"{BASE_URL}{endpoint}"
        resp = requests.post(url, headers=self._auth_header(), data=data, timeout=30)
        if not resp.ok:
            error(
                f"Splitwise API error: {resp.status_code}",
                details={"endpoint": endpoint, "body": resp.text},
            )
        return resp.json()
