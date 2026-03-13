"""Example plugin — reference this when creating new plugins.

This file is NOT loaded as a plugin (starts with underscore).
Copy and modify it to create your own plugins.
"""

# Imports: use standard library, third-party packages from requirements.txt,
# and bot modules (config, models, web_extractor, etc.)
import aiohttp
from plugin_base import BasePlugin


class ExamplePlugin(BasePlugin):
    """A minimal example plugin demonstrating all registration patterns."""

    # Required: unique name used for load/unload/reload
    name = "example"
    version = "1.0.0"
    description = "Example plugin showing available patterns"

    async def on_load(self):
        """Called when the plugin is loaded. Register everything here."""

        # --- Slash command ---
        # Registers /greet as a Discord slash command.
        # The callback receives a discord.Interaction.
        self.register_slash_command(
            name="greet",
            description="Say hello",
            callback=self._greet_command,
        )

        # --- Message handler ---
        # Regex pattern matched against every message the bot processes.
        # If callback returns True, the message is consumed (not sent to LLM).
        self.register_message_handler(
            pattern=r'^!example\b',
            callback=self._handle_example,
            priority=50,  # Lower = runs first. Default is 100.
        )

    async def on_unload(self):
        """Called before unload. Clean up resources (close sessions, etc.)."""
        pass

    async def self_test(self) -> bool:
        """Optional smoke test run during load. Return True if passing."""
        return True

    # --- Callbacks ---

    async def _greet_command(self, interaction):
        """Slash command callback. Must accept interaction as first arg."""
        await interaction.response.send_message(f"hey {interaction.user.display_name}")

    async def _handle_example(self, message):
        """Message handler callback. Receives discord.Message."""
        await message.channel.send("example plugin works!")
        return True  # Consumed — don't pass to LLM

    # --- Accessing bot state ---
    # self.ctx.active_model          — current LLM model name
    # self.ctx.system_prompt         — current system prompt
    # self.ctx.get_context(srv, ch)  — conversation history (copy)
    # self.ctx.query_llm(prompt)     — query the active LLM
    # self.ctx.send_message(ch, msg) — send to a Discord channel
    # self.ctx.ollama_client         — direct OllamaClient access
    # self.ctx.claude_code_client    — direct ClaudeCodeClient access
    # self.ctx.command_tree          — Discord CommandTree
    # self.ctx.discord_client        — the Discord bot client
    # self.logger                    — logger named "Plugin.<name>"
