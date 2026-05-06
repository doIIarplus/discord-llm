"""User blocklist plugin — silence specific users completely.

Wraps the bot's on_message handler so blocked users are skipped before any
recording, context-building, or trigger logic runs. Also filters retrieval
from chat_history so any pre-existing messages from blocked users never
appear in the LLM prompt.
"""

import json
from typing import Set

import discord

import chat_history
from plugin_base import BasePlugin
from sandbox import safe_path

BLOCKLIST_PATH = "blocklist.json"
DEFAULT_BLOCKED = {"354326385822531584"}  # KeerXKeer


def _load() -> Set[str]:
    try:
        with open(safe_path(BLOCKLIST_PATH)) as f:
            data = json.load(f)
        return {str(uid) for uid in data.get("blocked_user_ids", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _save(ids: Set[str]) -> None:
    with open(safe_path(BLOCKLIST_PATH), "w") as f:
        json.dump({"blocked_user_ids": sorted(ids)}, f, indent=2)


class BlocklistPlugin(BasePlugin):
    name = "blocklist"
    version = "1.0.0"
    description = "Persistent blocklist of user IDs the bot ignores entirely"

    def __init__(self, bot_context):
        super().__init__(bot_context)
        self._blocked: Set[str] = set()
        self._original_on_message = None
        self._original_get_recent = None

    async def on_load(self):
        existing = _load()
        # Seed defaults on first run, and ensure the seeded ID stays present
        # if the file existed but was missing it.
        merged = existing | DEFAULT_BLOCKED
        if merged != existing:
            _save(merged)
        self._blocked = merged

        client = self.ctx.discord_client
        plugin = self

        # Wrap on_message so blocked users are skipped before record_message
        # or any other processing runs.
        original = client.on_message

        async def patched_on_message(message):
            if message.author and str(message.author.id) in plugin._blocked:
                plugin.logger.info(
                    "Skipping message from blocked user %s (%s)",
                    getattr(message.author, "display_name", "?"),
                    message.author.id,
                )
                return
            await original(message)

        self._original_on_message = original
        client.on_message = patched_on_message

        # Filter blocked users out of historical context retrieval too,
        # so messages recorded before a block still get scrubbed.
        self._original_get_recent = chat_history.get_recent_channel_messages

        def filtered_get_recent(*args, **kwargs):
            rows = plugin._original_get_recent(*args, **kwargs)
            return [r for r in rows if str(r.get("author_id", "")) not in plugin._blocked]

        chat_history.get_recent_channel_messages = filtered_get_recent

        self.register_slash_command(
            name="block",
            description="Add a user to the blocklist (bot ignores them entirely)",
            callback=self._block_cmd,
        )
        self.register_slash_command(
            name="unblock",
            description="Remove a user from the blocklist",
            callback=self._unblock_cmd,
        )

    async def on_unload(self):
        if self._original_on_message is not None:
            try:
                # Remove the instance attribute so the class method is exposed again
                del self.ctx.discord_client.on_message
            except AttributeError:
                self.ctx.discord_client.on_message = self._original_on_message
            self._original_on_message = None

        if self._original_get_recent is not None:
            chat_history.get_recent_channel_messages = self._original_get_recent
            self._original_get_recent = None

    async def _block_cmd(self, interaction: discord.Interaction, user: discord.User):
        uid = str(user.id)
        if uid in self._blocked:
            await interaction.response.send_message(
                f"{user.display_name} is already blocked", ephemeral=True
            )
            return
        self._blocked.add(uid)
        _save(self._blocked)
        await interaction.response.send_message(
            f"blocked {user.display_name} ({uid})", ephemeral=True
        )

    async def _unblock_cmd(self, interaction: discord.Interaction, user: discord.User):
        uid = str(user.id)
        if uid not in self._blocked:
            await interaction.response.send_message(
                f"{user.display_name} is not blocked", ephemeral=True
            )
            return
        self._blocked.discard(uid)
        _save(self._blocked)
        await interaction.response.send_message(
            f"unblocked {user.display_name}", ephemeral=True
        )
