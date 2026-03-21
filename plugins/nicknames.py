"""Splitwise friend lookup plugin — resolve short names to Splitwise friends."""

import asyncio
import json
import subprocess
import sys

from plugin_base import BasePlugin
from sandbox import safe_path


def _fetch_friends() -> list[dict]:
    """Call the list_friends CLI tool and return the friends list."""
    script = safe_path("tools/splitwise/list_friends.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"list_friends failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    return data.get("friends", [])


def resolve_friend(query: str) -> list[dict]:
    """Substring match query against Splitwise friends list.

    Returns list of matching friend dicts (id, first_name, last_name, balance).
    Case-insensitive match against "first_name last_name".
    """
    friends = _fetch_friends()
    query_lower = query.lower()
    results = []
    for f in friends:
        full_name = f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
        if query_lower in full_name.lower():
            results.append(f)
    return results


class NicknamesPlugin(BasePlugin):
    name = "nicknames"
    version = "2.0.0"
    description = "Splitwise friend lookup — resolve short names to Splitwise friends"

    async def on_load(self):
        self.register_slash_command(
            name="whois",
            description="Search Splitwise friends by name (substring match)",
            callback=self._whois,
        )

    async def _whois(self, interaction, query: str):
        await interaction.response.defer()
        try:
            matches = await asyncio.to_thread(resolve_friend, query)
        except Exception as e:
            await interaction.followup.send(f"error looking up friends: {e}")
            return

        if not matches:
            await interaction.followup.send(f"no Splitwise friends matching **{query}**")
            return

        lines = []
        for f in matches:
            name = f"{f.get('first_name', '')} {f.get('last_name', '')}".strip()
            friend_id = f.get("id", "?")
            balances = f.get("balance", [])
            if balances:
                bal_parts = [f"{b.get('amount', '0')} {b.get('currency_code', '')}" for b in balances]
                bal_str = ", ".join(bal_parts)
            else:
                bal_str = "settled up"
            lines.append(f"**{name}** (ID: {friend_id}) — {bal_str}")

        await interaction.followup.send("\n".join(lines))
