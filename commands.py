"""Discord command handlers for Discord LLM Bot"""

import time
import asyncio
import logging
import os

import aiohttp
import discord
from discord import app_commands

from models import Txt2TxtModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up logging
logger = logging.getLogger("CommandHandlers")


class ModelSelectView(discord.ui.View):
    """Dropdown view for model selection."""

    def __init__(self, bot, options: list):
        super().__init__(timeout=60)
        self.bot = bot
        select = discord.ui.Select(
            placeholder="Pick a model...",
            options=options,
            min_values=1,
            max_values=1,
        )
        select.callback = self.on_select
        self.add_item(select)

    async def on_select(self, interaction: discord.Interaction):
        selected = interaction.data["values"][0]
        self.bot.active_model = selected
        self.bot.save_active_model()
        logger.info(f"Model switched to {selected} by {interaction.user.name}")
        await interaction.response.edit_message(
            content=f"Model switched to: **{selected}**",
            embed=None,
            view=None,
        )


class RestartConfirmView(discord.ui.View):
    """Confirm or revert bot code changes before restart."""

    def __init__(self, bot, author_id: int, test_passed: bool = True):
        super().__init__(timeout=120)
        self.bot = bot
        self.author_id = author_id
        self.test_passed = test_passed

        if not test_passed:
            self.confirm.label = "Apply Anyway & Restart"
            self.confirm.style = discord.ButtonStyle.red

    @discord.ui.button(label="Apply & Restart", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Only the requester can confirm.", ephemeral=True)
            return
        await interaction.response.edit_message(content="applying changes and restarting...", view=None)
        # Commit the changes
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A", cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", "[bot-self-modify] applied code changes",
            cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        # Save notification so the bot announces it's back after restart
        self.bot._state["restart_notify"] = {
            "channel_id": interaction.channel.id,
            "message": "ok i'm back, changes have been applied",
        }
        self.bot._save_state()
        await self.bot.request_restart(interaction.channel)

    @discord.ui.button(label="Revert", style=discord.ButtonStyle.red)
    async def revert(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Only the requester can revert.", ephemeral=True)
            return
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", ".", cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        await interaction.response.edit_message(content="Changes reverted.", view=None)


class PluginApplyView(discord.ui.View):
    """Confirm or revert plugin changes — hot-reloads instead of restarting."""

    MAX_AUTO_FIX_RETRIES = 2

    def __init__(self, bot, author_id: int, plugin_names: list, original_instruction: str = "", test_passed: bool = True):
        super().__init__(timeout=120)
        self.bot = bot
        self.author_id = author_id
        self.plugin_names = plugin_names  # plugins to reload after apply
        self.original_instruction = original_instruction
        self.test_passed = test_passed

        if not test_passed:
            self.apply.label = "Apply Anyway (Hot Reload)"
            self.apply.style = discord.ButtonStyle.red

    @discord.ui.button(label="Apply (Hot Reload)", style=discord.ButtonStyle.green)
    async def apply(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Only the requester can confirm.", ephemeral=True)
            return

        await interaction.response.edit_message(content="applying and reloading...", view=None)
        channel = interaction.channel

        # Git commit the plugin changes
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "plugins/", cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", "[plugin-edit] applied plugin changes",
            cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Hot-reload affected plugins, collecting errors
        results = []
        failures = {}  # name -> error_message
        for name in self.plugin_names:
            success, error = await self.bot.plugin_manager.reload_plugin_verbose(name)
            if success:
                results.append(f"`{name}`: ok")
            else:
                results.append(f"`{name}`: FAILED — {error}")
                failures[name] = error

        # Load any new plugins that weren't previously loaded
        discovered = self.bot.plugin_manager.discover_plugins()
        for name in discovered:
            if name not in self.bot.plugin_manager._plugins and name not in self.plugin_names:
                success, error = await self.bot.plugin_manager.load_plugin_verbose(name)
                if success:
                    results.append(f"`{name}`: loaded (new)")
                else:
                    results.append(f"`{name}`: FAILED — {error}")
                    failures[name] = error

        status = "\n".join(results) if results else "no plugins to reload"
        await channel.send(f"reload results:\n{status}")

        # If there are failures, attempt auto-fix
        if failures:
            await self._auto_fix_loop(channel, failures)

    async def _auto_fix_loop(self, channel, failures: dict):
        """Feed errors back to Claude Code for auto-fix retries."""
        from claude_code_client import RateLimitError

        for attempt in range(1, self.MAX_AUTO_FIX_RETRIES + 1):
            if not failures:
                break

            error_summary = "\n".join(
                f"- plugins/{name}.py: {error}" for name, error in failures.items()
            )
            fix_instruction = (
                f"The following plugins failed to load after editing. Fix the errors.\n\n"
                f"ERRORS:\n{error_summary}\n\n"
                f"Original request was: {self.original_instruction}"
            )

            await channel.send(
                f"auto-fix attempt {attempt}/{self.MAX_AUTO_FIX_RETRIES}..."
            )

            try:
                async with channel.typing():
                    response, exit_code = await self.bot.claude_code_client.run_plugin_edit(
                        fix_instruction,
                        model=self.bot.active_model,
                        existing_plugins=self.bot.plugin_manager.plugin_names,
                    )
            except RateLimitError:
                await channel.send("rate limited, can't auto-fix right now")
                break
            except Exception as e:
                await channel.send(f"auto-fix error: {e}")
                break

            if exit_code != 0:
                await channel.send(f"auto-fix edit failed (exit {exit_code})")
                break

            # Git commit the fix
            proc = await asyncio.create_subprocess_exec(
                "git", "add", "plugins/", cwd=PROJECT_DIR,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m",
                f"[plugin-autofix] attempt {attempt}",
                cwd=PROJECT_DIR,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            # Retry loading the failed plugins
            new_failures = {}
            for name in failures:
                success, error = await self.bot.plugin_manager.reload_plugin_verbose(name)
                if not success:
                    new_failures[name] = error

            if new_failures:
                await channel.send(
                    f"still failing: {', '.join(f'`{n}`' for n in new_failures)}"
                )
                failures = new_failures
            else:
                await channel.send("all plugins loaded successfully after auto-fix")
                # Show what changed
                await channel.send(f"fix: {response[:1500]}")
                return

        # If we exhausted retries, offer revert
        if failures:
            await channel.send(
                f"couldn't auto-fix after {self.MAX_AUTO_FIX_RETRIES} attempts. "
                f"failed plugins: {', '.join(f'`{n}`' for n in failures)}"
            )

    @discord.ui.button(label="Revert", style=discord.ButtonStyle.red)
    async def revert(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Only the requester can revert.", ephemeral=True)
            return
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", "--", "plugins/", cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        # Also revert any new plugin files
        proc = await asyncio.create_subprocess_exec(
            "git", "clean", "-fd", "plugins/", cwd=PROJECT_DIR,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        await interaction.response.edit_message(content="Plugin changes reverted.", view=None)


class CrashFixView(discord.ui.View):
    """Offer to auto-fix a crash based on log output."""

    def __init__(self, bot, log_context: str):
        super().__init__(timeout=300)
        self.bot = bot
        self.log_context = log_context

    @discord.ui.button(label="Fix this bug", style=discord.ButtonStyle.green)
    async def fix(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="looking at the logs and trying to fix it...", view=None)
        instruction = (
            "The bot crashed or encountered an error. Fix the bug based on the error in the logs. "
            "If the fix is obvious from the traceback, just fix it. If the error is environmental "
            "(network, DNS, etc.) and not a code bug, say so instead of making changes."
        )
        # Reuse the code change flow — this creates a snapshot, runs Claude Code, shows diff
        await self.bot._execute_code_change_with_logs(
            interaction.channel, interaction.user.id, instruction, self.log_context
        )

    @discord.ui.button(label="Ignore", style=discord.ButtonStyle.grey)
    async def ignore(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="crash report dismissed.", view=None)


class CommandHandlers:
    """Handles Discord slash commands"""

    def __init__(self, bot):
        self.bot = bot
        logger.info("CommandHandlers initialized")

    def setup_commands(self):
        """Register all slash commands"""
        logger.info("Setting up Discord slash commands")

        @self.bot.tree.command(name="clear", description="Clear context")
        async def clear(interaction: discord.Interaction):
            logger.info(f"Clear command called by {interaction.user.name}#{interaction.user.discriminator}")
            server = interaction.guild.id
            channel = interaction.channel.id

            if server in self.bot.context and channel in self.bot.context[server]:
                self.bot.context[server][channel] = []
                logger.debug(f"Cleared context for server {server}, channel {channel}")
            await interaction.response.send_message("Context cleared")
            logger.info("Context cleared successfully")

        @self.bot.tree.command(name="ask", description="Ask something")
        @app_commands.describe(question="Ask something")
        async def ask(interaction: discord.Interaction, question: str):
            logger.info(f"Ask command called by {interaction.user.name}#{interaction.user.discriminator} with question: {question[:50]}...")
            await interaction.response.defer(thinking=True)

            start = time.perf_counter()
            response = await self.bot.query_ollama(
                interaction.guild.id,
                interaction.channel.id,
                [{"role": "user", "content": question, "images": []}]
            )
            end = time.perf_counter()
            elapsed = end - start
            logger.info(f"Ask command processed in {elapsed:.3f} seconds")

            if isinstance(response, tuple):
                logger.debug("Sending embed and file response")
                await interaction.followup.send(embed=response[0], file=response[1])
            else:
                response_text = "\n".join(response) if isinstance(response, list) else response
                response_text += f"\n\n_Responded in {elapsed:.3f} seconds_"
                logger.debug(f"Sending text response: {response_text[:100]}...")
                await interaction.followup.send(response_text)

        @self.bot.tree.command(name='set_system_prompt')
        async def set_system_prompt(interaction: discord.Interaction, prompt: str):
            logger.info(f"Set system prompt command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            self.bot.system_prompt = prompt
            logger.debug("System prompt updated")
            await interaction.followup.send("Prompt updated")

        @self.bot.tree.command(name='reset_system_prompt')
        async def reset_system_prompt(interaction: discord.Interaction):
            logger.info(f"Reset system prompt command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            self.bot.system_prompt = self.bot.original_system_prompt
            logger.debug("System prompt reset to default")
            await interaction.followup.send("Prompt reset to default")

        @self.bot.tree.command(name='get_system_prompt')
        async def get_system_prompt(interaction: discord.Interaction):
            logger.info(f"Get system prompt command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            logger.debug("Sending current system prompt")
            from response_splitter import split_long_message
            chunks = split_long_message(self.bot.system_prompt)
            for chunk in chunks:
                await interaction.followup.send(chunk)

        @self.bot.tree.command(name="set_model", description="Switch the active LLM model")
        async def set_model(interaction: discord.Interaction):
            """Switch the active LLM model — shows an embed with a dropdown selector"""
            logger.info(f"Set model command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True, ephemeral=True)

            # Fetch available local models from Ollama
            local_models = await self._fetch_ollama_models()

            # Build the embed listing all models
            embed = discord.Embed(
                title="Select a Model",
                color=discord.Color.blurple(),
            )

            lines = []
            # Claude Code options first
            lines.append("**Claude Code (Subscription)**")
            lines.append("`  Claude Sonnet`")
            lines.append("`  Claude Opus`")

            if local_models:
                lines.append("")
                lines.append("**Local Models (Ollama)**")
                for m in local_models:
                    lines.append(f"`  {m['display_name']:40s} {m['size_str']}`")

            embed.description = "\n".join(lines)

            current = self.bot.active_model
            embed.set_footer(text=f"Current model: {current}")

            # Build dropdown options
            options = [
                discord.SelectOption(
                    label="Claude Sonnet",
                    value=Txt2TxtModel.CLAUDE_CODE.value,
                    description="Claude Code via subscription",
                    default=(current == Txt2TxtModel.CLAUDE_CODE.value),
                ),
                discord.SelectOption(
                    label="Claude Opus",
                    value=Txt2TxtModel.CLAUDE_CODE_OPUS.value,
                    description="Claude Code Opus via subscription",
                    default=(current == Txt2TxtModel.CLAUDE_CODE_OPUS.value),
                ),
                discord.SelectOption(
                    label="Qwen 3.5 35B (via Claude Code)",
                    value=Txt2TxtModel.CLAUDE_CODE_QWEN35.value,
                    description="Local Ollama model routed through Claude Code CLI",
                    default=(current == Txt2TxtModel.CLAUDE_CODE_QWEN35.value),
                ),
            ]
            for m in local_models:
                options.append(discord.SelectOption(
                    label=m["display_name"][:100],
                    value=m["value"],
                    description=f"{m['param_size']} · {m['size_str']}",
                    default=(current == m["value"]),
                ))

            # Cap at 25 options (Discord limit)
            options = options[:25]

            view = ModelSelectView(self.bot, options)
            await interaction.followup.send(embed=embed, view=view, ephemeral=True)

        @self.bot.tree.command(name="get_model", description="Show the current active LLM model")
        async def get_model(interaction: discord.Interaction):
            """Show the current active model"""
            logger.info(f"Get model command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.send_message(f"Current model: **{self.bot.active_model}**")

        @self.bot.tree.command(name="purge", description="Delete all messages in this channel")
        @app_commands.default_permissions(manage_messages=True)
        async def purge(interaction: discord.Interaction):
            """Delete all messages in the current channel."""
            logger.info(f"Purge command called by {interaction.user.name}#{interaction.user.discriminator} in #{interaction.channel.name}")
            if not interaction.channel.permissions_for(interaction.user).manage_messages:
                await interaction.response.send_message("You need the Manage Messages permission to use this.", ephemeral=True)
                return
            await interaction.response.defer(ephemeral=True)
            deleted_total = 0
            while True:
                deleted = await interaction.channel.purge(limit=100)
                deleted_total += len(deleted)
                if len(deleted) < 100:
                    break
            logger.info(f"Purged {deleted_total} messages in #{interaction.channel.name}")
            await interaction.followup.send(f"Deleted {deleted_total} messages.", ephemeral=True)

        @self.bot.tree.command(name="list_documents", description="List folders in My Documents (sandbox test)")
        async def list_documents(interaction: discord.Interaction):
            """Attempt to list /mnt/c/Users/Daniel/Documents — should be blocked by bwrap sandbox."""
            docs_path = "/mnt/c/Users/Daniel/Documents"
            try:
                entries = os.listdir(docs_path)
                await interaction.response.send_message(f"SANDBOX FAILURE: got {len(entries)} entries from {docs_path}")
            except (FileNotFoundError, OSError) as e:
                await interaction.response.send_message(f"Sandbox working: {e}")

        # ── Plugin management commands ────────────────────────────────

        @self.bot.tree.command(name="plugins", description="List loaded plugins")
        async def plugins(interaction: discord.Interaction):
            """Show all loaded plugins and their status."""
            info = self.bot.plugin_manager.list_plugins()
            if not info:
                await interaction.response.send_message("No plugins loaded.")
                return
            lines = []
            for p in info:
                status = "DISABLED" if p["disabled"] else "active"
                cmds = ", ".join(f"/{c}" for c in p["commands"]) if p["commands"] else "none"
                lines.append(
                    f"**{p['name']}** v{p['version']} [{status}] — {p['description'] or 'no description'}\n"
                    f"  commands: {cmds} | handlers: {p['message_handlers']} | hooks: {p['hooks']}"
                )
            await interaction.response.send_message("\n".join(lines))

        @self.bot.tree.command(name="reload_plugin", description="Hot-reload a plugin")
        @app_commands.describe(name="Plugin name to reload")
        async def reload_plugin(interaction: discord.Interaction, name: str):
            """Unload then re-load a plugin without restarting the bot."""
            logger.info(f"Reload plugin '{name}' requested by {interaction.user.name}")
            await interaction.response.defer(thinking=True)
            success = await self.bot.plugin_manager.reload_plugin(name)
            if success:
                await interaction.followup.send(f"Plugin `{name}` reloaded successfully.")
            else:
                await interaction.followup.send(f"Failed to reload plugin `{name}`. Check logs for details.")

        @self.bot.tree.command(name="load_plugin", description="Load a plugin")
        @app_commands.describe(name="Plugin name to load")
        async def load_plugin(interaction: discord.Interaction, name: str):
            """Load a plugin from the plugins/ directory."""
            logger.info(f"Load plugin '{name}' requested by {interaction.user.name}")
            await interaction.response.defer(thinking=True)
            success = await self.bot.plugin_manager.load_plugin(name)
            if success:
                await interaction.followup.send(f"Plugin `{name}` loaded successfully.")
            else:
                await interaction.followup.send(f"Failed to load plugin `{name}`. Check logs for details.")

        @self.bot.tree.command(name="unload_plugin", description="Unload a plugin")
        @app_commands.describe(name="Plugin name to unload")
        async def unload_plugin(interaction: discord.Interaction, name: str):
            """Unload a plugin, removing its commands and handlers."""
            logger.info(f"Unload plugin '{name}' requested by {interaction.user.name}")
            await interaction.response.defer(thinking=True)
            success = await self.bot.plugin_manager.unload_plugin(name)
            if success:
                await interaction.followup.send(f"Plugin `{name}` unloaded.")
            else:
                await interaction.followup.send(f"Plugin `{name}` not found or already unloaded.")

        @self.bot.tree.command(name="sync_commands")
        async def sync_commands(interaction: discord.Interaction):
            """Manually sync slash commands to Discord (rate-limited, use sparingly)"""
            logger.info(f"Sync commands called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            await self.bot.tree.sync()
            self.bot.plugin_manager._pending_sync = False
            await interaction.followup.send("Commands synced successfully.")

        logger.info("All Discord slash commands registered successfully")

    async def _fetch_ollama_models(self) -> list:
        """Query Ollama API for available local models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Ollama API returned status {resp.status}")
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return []

        models = []
        for m in data.get("models", []):
            name = m["name"]
            size_bytes = m.get("size", 0)
            param_size = m.get("details", {}).get("parameter_size", "?")

            # Shorten display name: strip hf.co/... and huihui_ai/... prefixes
            display = name
            for prefix in ("hf.co/", "huihui_ai/"):
                if display.startswith(prefix):
                    display = display[len(prefix):]
            size_gb = size_bytes / (1024 ** 3)
            if size_gb >= 1:
                size_str = f"{size_gb:.0f}GB"
            else:
                size_str = f"{size_gb * 1024:.0f}MB"

            models.append({
                "value": name,           # full Ollama model name for API calls
                "display_name": display, # shortened for UI
                "size_str": size_str,
                "param_size": param_size,
            })

        models.sort(key=lambda x: x["display_name"].lower())
        return models

