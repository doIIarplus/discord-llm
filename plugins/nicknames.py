"""Nickname/alias plugin — maps short names to full names per server."""

import json
import os

from discord import app_commands

from plugin_base import BasePlugin
from sandbox import safe_path

DATA_FILE = safe_path("plugins/data/nicknames.json")


def _load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_data(data: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def resolve_nickname(guild_id: int | str, query: str) -> list[tuple[str, str]]:
    """Substring match query against all nicknames for a guild.

    Returns list of (short_name, full_name) tuples that match.
    Case-insensitive — matches against both the short name and the full name.
    """
    data = _load_data()
    guild_key = str(guild_id)
    nicknames = data.get(guild_key, {})
    query_lower = query.lower()
    results = []
    for short, full in nicknames.items():
        if query_lower in short.lower() or query_lower in full.lower():
            results.append((short, full))
    return results


class NicknamesPlugin(BasePlugin):
    name = "nicknames"
    version = "1.0.0"
    description = "Nickname/alias system — map short names to full names"

    async def on_load(self):
        self.register_slash_command(
            name="set_nickname",
            description="Map a short name to a full name",
            callback=self._set_nickname,
        )
        self.register_slash_command(
            name="remove_nickname",
            description="Remove a nickname mapping",
            callback=self._remove_nickname,
        )
        self.register_slash_command(
            name="list_nicknames",
            description="Show all nickname mappings for this server",
            callback=self._list_nicknames,
        )
        self.register_slash_command(
            name="whois",
            description="Substring search against all registered nicknames",
            callback=self._whois,
        )

    async def _set_nickname(self, interaction, short_name: str, full_name: str):
        guild_id = str(interaction.guild_id)
        data = _load_data()
        if guild_id not in data:
            data[guild_id] = {}
        data[guild_id][short_name.lower()] = full_name
        _save_data(data)
        await interaction.response.send_message(
            f"**{short_name.lower()}** → {full_name}"
        )

    async def _remove_nickname(self, interaction, short_name: str):
        guild_id = str(interaction.guild_id)
        data = _load_data()
        nicknames = data.get(guild_id, {})
        key = short_name.lower()
        if key not in nicknames:
            await interaction.response.send_message(
                f"no nickname found for **{short_name}**"
            )
            return
        full = nicknames.pop(key)
        _save_data(data)
        await interaction.response.send_message(
            f"removed **{key}** (was → {full})"
        )

    async def _list_nicknames(self, interaction):
        guild_id = str(interaction.guild_id)
        data = _load_data()
        nicknames = data.get(guild_id, {})
        if not nicknames:
            await interaction.response.send_message("no nicknames set for this server")
            return
        lines = [f"**{short}** → {full}" for short, full in sorted(nicknames.items())]
        await interaction.response.send_message("\n".join(lines))

    async def _whois(self, interaction, query: str):
        matches = resolve_nickname(interaction.guild_id, query)
        if not matches:
            await interaction.response.send_message(
                f"no matches for **{query}**"
            )
            return
        lines = [f"**{short}** → {full}" for short, full in matches]
        await interaction.response.send_message("\n".join(lines))
