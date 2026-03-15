"""Base class and types for the plugin system.

Plugins inherit from BasePlugin and register handlers/commands in on_load().
They interact with the bot through PluginBotContext, never directly.
"""

import logging
from enum import Enum
from typing import List, Optional, Any


class HookType(Enum):
    """Types of hooks a plugin can register."""
    ON_MESSAGE = "on_message"       # Called for every message the bot processes
    PRE_QUERY = "pre_query"         # Before LLM query, can modify prompt/context
    POST_QUERY = "post_query"       # After LLM response, can modify response
    ON_READY = "on_ready"           # Bot connected


class PluginBotContext:
    """Restricted interface plugins use to interact with the bot.

    Plugins never get a direct reference to OllamaBot. They interact
    through this proxy, which enforces boundaries.
    """

    def __init__(self, bot):
        self._bot = bot

    # --- Safe read access ---

    def get_context(self, server: int, channel: int) -> List[dict]:
        """Get conversation context (copy, not reference)."""
        ctx = self._bot.context.get(server, {}).get(channel, [])
        return list(ctx)

    @property
    def active_model(self) -> str:
        return self._bot.active_model

    @property
    def system_prompt(self) -> str:
        return self._bot.system_prompt

    # --- Safe write access ---

    def append_to_context(self, server: int, channel: int, entry: dict):
        """Add an entry to conversation context."""
        if server not in self._bot.context:
            self._bot.context[server] = {}
        if channel not in self._bot.context[server]:
            self._bot.context[server][channel] = []
        self._bot.context[server][channel].append(entry)

    async def query_llm(self, prompt: str, model: str = None, images: list = None) -> str:
        """Query the LLM through the bot's existing clients."""
        from models import is_claude_code_model
        m = model or self._bot.active_model
        if is_claude_code_model(m):
            resp, _ = await self._bot.claude_code_client.generate_with_search(prompt, m, images)
            return resp
        return await self._bot.ollama_client.generate(prompt, m, images)

    async def send_message(self, channel_id: int, content: str, **kwargs):
        """Send a message to a Discord channel."""
        channel = self._bot.get_channel(channel_id)
        if channel:
            await channel.send(content, **kwargs)

    @property
    def ollama_client(self):
        return self._bot.ollama_client

    @property
    def claude_client(self):
        return self._bot.claude_client

    @property
    def claude_code_client(self):
        return self._bot.claude_code_client

    @property
    def command_tree(self):
        return self._bot.tree

    @property
    def discord_client(self):
        """Access to the discord.Client for guild/channel lookups."""
        return self._bot


class BasePlugin:
    """Base class all plugins must inherit from.

    Subclass this and override on_load() to register your handlers.
    Use self.ctx (PluginBotContext) to interact with the bot.
    """

    # Subclasses override these
    name: str = "unnamed_plugin"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = []
    priority: int = 100  # Lower = runs first in hook chains

    def __init__(self, bot_context: PluginBotContext):
        self.ctx = bot_context
        self.logger = logging.getLogger(f"Plugin.{self.name}")
        self._slash_commands: List[dict] = []
        self._message_handlers: List[dict] = []
        self._hooks: List[dict] = []

    async def on_load(self):
        """Called when the plugin is loaded. Register commands/handlers here."""
        pass

    async def on_unload(self):
        """Called before the plugin is unloaded. Clean up resources."""
        pass

    async def self_test(self) -> bool:
        """Override to define smoke tests. Return True if passing."""
        return True

    def register_slash_command(self, name: str, description: str, callback, **kwargs):
        """Register a slash command. Synced on /sync_commands."""
        self._slash_commands.append({
            "name": name,
            "description": description,
            "callback": callback,
            **kwargs,
        })

    def register_message_handler(self, pattern: str, callback, priority: int = 100):
        """Register a regex-based message handler.

        If the callback returns True, the message is considered consumed
        and won't be passed to the LLM or other handlers.
        """
        self._message_handlers.append({
            "pattern": pattern,
            "callback": callback,
            "priority": priority,
        })

    def register_hook(self, hook_type: HookType, callback):
        """Register a lifecycle hook."""
        self._hooks.append({
            "hook_type": hook_type,
            "callback": callback,
        })
