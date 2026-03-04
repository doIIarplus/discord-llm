"""Discord command handlers for Discord LLM Bot"""

import time
import traceback
from typing import Optional
import asyncio
import logging
import os

import aiohttp
import discord
from discord import app_commands

from image_generation import ImageGenerator
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

    def __init__(self, bot, author_id: int):
        super().__init__(timeout=120)
        self.bot = bot
        self.author_id = author_id

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


class CommandHandlers:
    """Handles Discord slash commands"""

    def __init__(self, bot):
        self.bot = bot
        self.image_gen = ImageGenerator()
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
            await interaction.followup.send(self.bot.system_prompt)

        @self.bot.tree.command(name="generate_image")
        @app_commands.describe(
            prompt="Prompt for image generation. Comma separated.",
            negative_prompt="Negative prompt. Same format as positive prompt, but for things you don't want to see.",
            seed="Seed for image generation. Same seed with same parameters will generate the same image. -1 for random",
            width="Width of image in pixels. Capped at 1500",
            height="Height of image in pixels. Capped at 2000",
            cfg_scale="Smaller values will produce more creative results. Larger values will conform to prompt more.",
            steps="Number of steps for image generation",
            upscale="Upscale factor (1.0 to 2.0)",
            allow_nsfw="Allow NSFW content generation"
        )
        async def generate_image(
            interaction: discord.Interaction,
            prompt: str,
            negative_prompt: Optional[str] = None,
            seed: int = -1,
            width: int = 832,
            height: int = 1216,
            cfg_scale: float = 3.0,
            steps: int = 30,
            upscale: float = 1.0,
            allow_nsfw: bool = True,
        ):
            logger.info(f"Generate image command called by {interaction.user.name}#{interaction.user.discriminator}")
            logger.debug(f"Image generation parameters: prompt='{prompt[:50]}...', seed={seed}, width={width}, height={height}, cfg_scale={cfg_scale}, steps={steps}, upscale={upscale}, allow_nsfw={allow_nsfw}")
            await interaction.response.defer(thinking=True)

            try:
                file_path, image_info, is_nsfw = await self.image_gen.generate_image(
                    prompt, negative_prompt, seed, width, height,
                    cfg_scale, steps, upscale, allow_nsfw
                )
                logger.info(f"Image generated successfully: {file_path}")
                logger.debug(f"Image info: steps={image_info.steps}, cfg={image_info.cfg_scale}, size={image_info.width}x{image_info.height}, seed={image_info.seed}")

                file = discord.File(fp=file_path, filename='generated.png')
                image_info_text = (
                    f"steps: {image_info.steps}, "
                    f"cfg: {image_info.cfg_scale}, "
                    f"size: {image_info.width}x{image_info.height}, "
                    f"seed: {image_info.seed}"
                )

                embed = discord.Embed()
                embed.set_image(url='attachment://generated.png')
                embed.set_footer(text=image_info_text)

                if is_nsfw:
                    file.spoiler = True
                    logger.debug("Marking image as NSFW")

                await interaction.followup.send(embed=embed, file=file)
                logger.info("Image sent to Discord successfully")

            except Exception as e:
                logger.error(f"Error during image generation: {e}", exc_info=True)
                await interaction.followup.send(f"An error occurred: {e}")

        @self.bot.tree.command(name="index_wiki")
        @app_commands.describe(
            clear_existing="Clear existing index before indexing (default: False)"
        )
        async def index_wiki(interaction: discord.Interaction, clear_existing: bool = False):
            """Index the MapleStory wiki dump for RAG"""
            logger.info(f"Index wiki command called by {interaction.user.name}#{interaction.user.discriminator}")
            logger.debug(f"Clear existing: {clear_existing}")
            await interaction.response.defer(thinking=True)

            wiki_path = "maplestorywikinet.xml"
            if not os.path.exists(wiki_path):
                logger.warning(f"Wiki file not found: {wiki_path}")
                await interaction.followup.send(f"Wiki file not found: {wiki_path}")
                return

            try:
                if clear_existing:
                    logger.info("Clearing existing collection")
                    self.bot.rag_system.clear_collection()
                    await interaction.followup.send("Cleared existing index. Starting indexing...")
                else:
                    logger.info("Starting wiki indexing")
                    await interaction.followup.send("Starting wiki indexing. This may take several minutes...")

                # Run indexing in background
                logger.debug("Running wiki indexing in background")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.bot.rag_system.index_wiki_dump,
                    wiki_path
                )
                logger.info("Wiki indexing completed successfully")

                stats = self.bot.rag_system.get_stats()
                logger.debug(f"Indexing stats: {stats}")
                await interaction.channel.send(
                    f"Wiki indexing complete!\n"
                    f"Total chunks indexed: {stats['total_chunks']}"
                )
            except Exception as e:
                logger.error(f"Error during wiki indexing: {e}", exc_info=True)
                traceback.print_exc()
                await interaction.channel.send(f"Error during indexing: {e}")

        @self.bot.tree.command(name="enable_rag")
        async def enable_rag(interaction: discord.Interaction):
            """Enable RAG for wiki content"""
            logger.info(f"Enable RAG command called by {interaction.user.name}#{interaction.user.discriminator}")
            self.bot.rag_enabled = True
            logger.debug("RAG enabled")
            await interaction.response.send_message("RAG enabled. Wiki context will be added to queries.")

        @self.bot.tree.command(name="disable_rag")
        async def disable_rag(interaction: discord.Interaction):
            """Disable RAG for wiki content"""
            logger.info(f"Disable RAG command called by {interaction.user.name}#{interaction.user.discriminator}")
            self.bot.rag_enabled = False
            logger.debug("RAG disabled")
            await interaction.response.send_message("RAG disabled. No wiki context will be added.")

        @self.bot.tree.command(name="search_wiki")
        @app_commands.describe(
            query="Search query for the wiki",
            n_results="Number of results to return (default: 3)"
        )
        async def search_wiki(interaction: discord.Interaction, query: str, n_results: int = 3):
            """Search the indexed wiki content"""
            logger.info(f"Search wiki command called by {interaction.user.name}#{interaction.user.discriminator} with query: {query[:50]}...")
            logger.debug(f"Number of results requested: {n_results}")
            await interaction.response.defer(thinking=True)

            try:
                results = self.bot.rag_system.search(query, n_results=n_results)
                logger.debug(f"Search returned {len(results)} results")

                if not results:
                    logger.info("No results found for search query")
                    await interaction.followup.send("No results found.")
                    return

                # Format results for Discord
                response = f"**Search results for:** {query}\n\n"
                for i, result in enumerate(results, 1):
                    # Truncate content for display
                    content = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                    response += f"**{i}. {result['title']}** (Score: {result['score']:.3f})\n{content}\n\n"

                # Split if too long
                if len(response) > 2000:
                    response = response[:1997] + "..."

                logger.debug(f"Sending search results: {response[:100]}...")
                await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Error during wiki search: {e}", exc_info=True)
                await interaction.followup.send(f"Error searching wiki: {e}")

        @self.bot.tree.command(name="rag_stats")
        async def rag_stats(interaction: discord.Interaction):
            """Get statistics about the indexed wiki content"""
            logger.info(f"RAG stats command called by {interaction.user.name}#{interaction.user.discriminator}")
            stats = self.bot.rag_system.get_stats()
            logger.debug(f"RAG stats: {stats}")
            await interaction.response.send_message(
                f"**RAG System Stats**\n"
                f"Total chunks: {stats['total_chunks']}\n"
                f"Collection: {stats['collection_name']}\n"
                f"RAG enabled: {self.bot.rag_enabled}"
            )

        @self.bot.tree.command(name="search")
        @app_commands.describe(
            query="What to search the web for",
            results="Number of results to return (default: 5)"
        )
        async def search(interaction: discord.Interaction, query: str, results: int = 5):
            """Search the web and get a summary from the LLM"""
            logger.info(f"Search command called by {interaction.user.name}#{interaction.user.discriminator} with query: {query[:50]}...")
            await interaction.response.defer(thinking=True)

            from web_extractor import web_search, format_search_results

            search_results = await web_search(query, max_results=results)
            if not search_results:
                await interaction.followup.send("No search results found (is TAVILY_API_KEY set?).")
                return

            # Summarize search results, then use as context for the LLM
            raw_context = format_search_results(search_results)
            search_summary = await self.bot.ollama_client.summarize_search_results(query, raw_context)

            llm_prompt = (
                f"Search Results Summary:\n{search_summary}\n\n"
                f"The user searched for: {query}"
            )

            response = await self.bot.query_ollama(
                interaction.guild.id,
                interaction.channel.id,
                [{"role": "user", "content": llm_prompt, "images": []}]
            )

            if isinstance(response, list):
                response_text = "\n".join(response)
            else:
                response_text = str(response)

            # Append short domain sources (deduplicated)
            seen = set()
            domains = []
            for r in search_results[:3]:
                try:
                    domain = r['url'].split('/')[2].removeprefix('www.')
                except (IndexError, AttributeError):
                    continue
                if domain not in seen:
                    seen.add(domain)
                    domains.append(domain)
            if domains:
                response_text += f"\n-# Sources: {', '.join(domains)}"

            if len(response_text) > 2000:
                response_text = response_text[:1997] + "..."

            await interaction.followup.send(response_text)

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

        @self.bot.tree.command(name="ping", description="Ping Google and show the latency")
        async def ping(interaction: discord.Interaction):
            """Ping Google and return the round-trip latency."""
            logger.info(f"Ping command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            try:
                start = time.perf_counter()
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://www.google.com", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        await resp.read()
                latency_ms = (time.perf_counter() - start) * 1000
                await interaction.followup.send(f"Pong! Google responded in **{latency_ms:.1f}ms**")
            except Exception as e:
                logger.error(f"Ping failed: {e}", exc_info=True)
                await interaction.followup.send(f"Ping failed: {e}")

        @self.bot.tree.command(name="list_documents", description="List all folders in My Documents")
        async def list_documents(interaction: discord.Interaction):
            """List all folder names inside the My Documents directory."""
            logger.info(f"List documents command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            docs_path = "/mnt/c/Users/Daniel/Documents"
            try:
                folders = sorted(
                    entry.name for entry in os.scandir(docs_path) if entry.is_dir()
                )
                if not folders:
                    await interaction.followup.send("No folders found in My Documents.")
                    return
                result = ", ".join(folders)
                if len(result) > 2000:
                    result = result[:1997] + "..."
                await interaction.followup.send(result)
            except Exception as e:
                logger.error(f"Error listing documents: {e}", exc_info=True)
                await interaction.followup.send(f"Error: {e}")

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

        @self.bot.tree.command(name="sync_commands")
        async def sync_commands(interaction: discord.Interaction):
            """Manually sync slash commands to Discord (rate-limited, use sparingly)"""
            logger.info(f"Sync commands called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            await self.bot.tree.sync()
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

