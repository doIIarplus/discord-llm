"""Discord LLM Bot - Main module"""

import asyncio
import json
import logging
import re
import os
import random
import sys
import time
import traceback
from typing import Dict, List

logger = logging.getLogger("Bot")

import aiohttp
import discord
from discord import app_commands
from discord.ext import tasks
from PIL import Image

from commands import CommandHandlers
from config import (
    CONTEXT_LIMIT,
    DISCORD_BOT_TOKEN,
    FILE_INPUT_FOLDER,
    GUILD_ID,
    IMAGE_RECOGNITION_MODEL,
    CHAT_MODEL,
    MAX_DISCORD_MESSAGE_LENGTH,
    OUTPUT_DIR_T2I,
    VISION_MODEL_CTX,
)
from image_generation import (
    ImageGenerator,
    choose_dimensions,
    choose_followup_dimensions,
    choose_source_dimensions,
    clean_edit_instruction,
)
from ollama_client import OllamaClient
from claude_client import ClaudeClient
from claude_code_client import ClaudeCodeClient, RateLimitError
from claude_code_client_pty import ClaudeCodeClientPTY
from models import is_claude_code_model, is_anthropic_model
from utils import encode_images_to_base64
from rag_system import RAGSystem
from web_extractor import extract_webpage_context, web_search, format_search_results, js_renderer
from file_parser import FileParser
from mention_extractor import extract_mention_context
from response_splitter import split_response_by_markers, split_response_by_paragraphs, split_long_message, calculate_typing_delay
from sandbox import safe_path, SandboxViolation
from plugin_base import HookType
import chat_history
from plugin_manager import PluginManager

# charlie was here

# Exit code that tells the wrapper script (run_bot.sh) to restart the bot
RESTART_EXIT_CODE = 42

# Keywords that suggest an image generation request
_IMAGE_GEN_KEYWORDS = re.compile(
    r'\b(generate|create|draw|make|paint|render|sketch)\b.{0,30}\b(image|picture|photo|illustration|art|drawing|painting)\b',
    re.IGNORECASE
)

# Keywords that suggest the user needs up-to-date / web-searchable info
_SEARCH_KEYWORDS = re.compile(
    r'\b(latest|recent|current|when|happening|happened|recently|today|yesterday|tonight|this week|this month|this year'
    r'|news|update|score|weather|price|stock|election|released|announced'
    r'|who won|who is winning|what happened|how much does|how much is'
    r'|in 202[4-9]|right now)\b',
    re.IGNORECASE
)

# Keywords that suggest the user wants the bot to modify its own code
_EDIT_CODE_TAG = re.compile(
    r'\[EDIT_CODE\](.*?)\[/EDIT_CODE\]',
    re.DOTALL
)


class OllamaBot(discord.Client):
    """Main Discord bot class"""

    def __init__(self):
        super().__init__(intents=discord.Intents.all())
        self.tree = app_commands.CommandTree(self)

        # Per-server per-channel context
        self.context: Dict[str, Dict[str, List[dict]]] = {}

        # Initialize clients
        self.ollama_client = OllamaClient()
        self.claude_client = ClaudeClient()
        # Use PTY client (persistent tmux session) if enabled, otherwise one-shot CLI
        use_pty = os.getenv("CLAUDE_USE_PTY", "").lower() in ("1", "true", "yes")
        if use_pty:
            self.claude_code_client = ClaudeCodeClientPTY(session_name="claude_bot")
            print("[bot] Using Claude Code PTY client (tmux session)")
        else:
            self.claude_code_client = ClaudeCodeClient()
            print("[bot] Using Claude Code CLI client (one-shot)")
        self.image_gen = ImageGenerator()
        self._last_search_sources: List[dict] = []

        # Active model (switchable via /set_model, persisted to disk)
        self._state_file = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self._state = self._load_state()
        self.active_model = self._state.get("active_model", CHAT_MODEL)

        # Initialize RAG system
        self.rag_system = RAGSystem()
        self.rag_enabled = False

        # Plugin system
        self.plugin_manager = PluginManager(self)

        # System prompts
        self.original_system_prompt = (
            "You are jaspt, a Discord bot. Your responses should be akin to that of a typical millenial texter: short, to the point, and mostly without punctuation. Do not offer any kind of assistance without being prompted. use slang *sparingly*. \n\n"
            "CONVERSATION FORMAT:\n"
            "The conversation history uses numbered [Turn N] tags. Each turn is a REAL message from a REAL user or your previous response. "
            "ONLY respond to the LAST turn. Do NOT invent, fabricate, or continue with additional user messages. "
            "Do NOT generate text inside [Turn] tags — only produce your own single response.\n\n"
            "MULTI-MESSAGE RESPONSES:\n"
            "When your response would naturally be multiple messages (like a greeting followed by information, "
            "or multiple distinct points), you can split them using the marker: ---MSG---\n"
            "Example:\n"
            "hey i can help you with that.\n"
            "---MSG---\n"
            "i found out this about what you asked\n\n"
            "Only use this for natural conversational breaks. Don't overuse it - most responses should be single messages. "
            "Use it when:\n"
            "- You want to greet then provide information\n"
            "- You have multiple distinct topics to address\n"
            "- A dramatic pause or separate thought would feel natural\n\n"
            "CODE EDITING:\n"
            "You ARE a Discord bot that can modify its own source code. You have a PLUGIN SYSTEM — "
            "new features should be implemented as plugins (hot-reloaded without restart). "
            "When a user asks you to add a feature, fix a bug, change behavior, or modify your code, "
            "include an [EDIT_CODE] tag in your response with a clear description of what to do. Format:\n"
            "[EDIT_CODE]detailed instruction of what to change[/EDIT_CODE]\n"
            "For NEW features or commands, phrase it as a plugin:\n"
            "- User: 'add a /hello command' → 'sure [EDIT_CODE]Create a new plugin that adds a /hello slash command that says hello[/EDIT_CODE]'\n"
            "- User: 'add a dice roller' → 'on it [EDIT_CODE]Create a new plugin with a /roll command that rolls dice[/EDIT_CODE]'\n"
            "For bug fixes or changes to existing behavior:\n"
            "- User: 'fix the logger error' → 'on it [EDIT_CODE]Add import logging and logger = logging.getLogger(__name__) to bot.py[/EDIT_CODE]'\n"
            "- User: 'you keep crashing when I send images' → 'lemme fix that [EDIT_CODE]Fix the image handling crash - check the error in bot.log and fix the root cause[/EDIT_CODE]'\n"
            "Include your normal casual response text OUTSIDE the tag. The tag content should be a clear, "
            "specific instruction — not conversational. Only use this when the user is genuinely asking for a code change. "
            "Do NOT use it for general questions about code or programming help.\n\n"
            "CLI TOOLS:\n"
            "You have access to CLI tools via Bash in the tools/ directory. Use them when users ask about "
            "Splitwise (bills, balances, expenses), scheduled tasks, web searches, or other tool-related actions. "
            "Run `python tools/<integration>/<tool>.py --help` to see usage. Key tools:\n"
            "- tools/splitwise/ — list_friends, get_balances, create_expense, delete_expense, list_groups, list_expenses (ONLY for discord_id=118567805678256128)\n"
            "- tools/scheduler/ — create_task (--once for one-shot reminders), list_tasks, delete_task\n"
            "- tools/web_search/search.py — search the web\n"
            "- tools/discord/send_message.py — send a message as the bot to any channel\n"
            "- tools/discord/get_channel_history.py — fetch recent messages from a channel\n"
            "- tools/discord/search_messages.py — search messages in the server\n"
            "- tools/discord/add_role.py / remove_role.py — manage user roles\n"
            "- tools/discord/list_roles.py — list server roles\n"
            "- tools/discord/react.py — add a reaction to a message\n"
            "- tools/discord/pin_message.py — pin/unpin a message\n"
            "- tools/discord/send_webhook.py — send Discord messages via webhook\n"
            "For reminders: use tools/scheduler/create_task.py --once with a command that calls tools/discord/send_message.py. "
            "Use the channel_id from [Current context] unless the user specifies a different channel. "
            "Example: create_task --name 'reminder' --schedule '0 9 30 3 *' --once "
            "--command 'python tools/discord/send_message.py --channel-id CHAN --content \"<@USER> reminder text\"'\n"
            "Always use these tools when the user's request matches their capabilities instead of making up answers.\n\n"
            "RELATIONSHIPS:\n"
            "You have a relationship score with each user (-10 to 10) shown in [Memory]. "
            "Adjust your warmth, effort, and willingness to help based on this score. "
            "For negative scores, be shorter and less enthusiastic. For positive scores, be warmer and go the extra mile. "
            "For hostile users (below -8), be extremely dismissive and refuse to engage."
        )
        self.system_prompt = self.original_system_prompt

        # Setup command handlers
        self.command_handlers = CommandHandlers(self)

    def _load_state(self) -> dict:
        """Load persisted bot state."""
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            print(f"Loaded saved state: {data}")
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self):
        """Persist bot state to disk."""
        self._state["active_model"] = self.active_model
        try:
            with open(self._state_file, "w") as f:
                json.dump(self._state, f)
        except OSError as e:
            print(f"Warning: could not save state: {e}")

    def save_active_model(self):
        """Persist current model selection to disk."""
        self._save_state()

    async def setup_hook(self):
        """Setup hook for Discord bot"""
        self.command_handlers.setup_commands()
        # Load all plugins before syncing commands
        await self.plugin_manager.load_all()
        # Sync to specific guild for instant command visibility
        guild = discord.Object(id=GUILD_ID)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        await js_renderer.start()

    def _read_recent_logs(self, max_lines: int = 200) -> str:
        """Read the last N lines from bot.log."""
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log")
        try:
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:])
        except FileNotFoundError:
            return ""

    async def _preload_image_pipeline(self):
        """Preload Flux + Ollama models used by the image-gen pipeline.

        Runs once at startup as a fire-and-forget background task. Cold-loading
        Flux on the first user request costs 4-5 minutes; warming it here at
        startup shifts that cost out of the user-facing critical path.
        """
        logger.info("[preload] starting image pipeline warmup (background)")
        t0 = time.perf_counter()
        try:
            # Flux is the biggest: ~33GB on disk, ~4-5 min to cold-load
            logger.info("[preload] warming Flux2 Klein pipeline...")
            t_flux = time.perf_counter()
            await self.image_gen.flux_client.warmup()
            logger.info("[preload] flux warm in %.1fs", time.perf_counter() - t_flux)

            # Classifier / rewriter text model
            logger.info("[preload] warming %s classifier...", CHAT_MODEL)
            t_cls = time.perf_counter()
            await self.image_gen.is_image_generation_task("generate a cat")
            logger.info("[preload] gemma warm in %.1fs", time.perf_counter() - t_cls)

            # NSFW classifier (uses gemma3:27b — already warm above, this is
            # cheap but exercises the full path)
            from utils import encode_image_downsized_to_base64
            sample = os.path.join(OUTPUT_DIR_T2I, "icon.png")
            if os.path.exists(sample):
                logger.info("[preload] warming NSFW classifier path...")
                t_ns = time.perf_counter()
                b64 = encode_image_downsized_to_base64(sample, max_side=256)
                await self.image_gen.ollama_client.classify_nsfw([b64])
                logger.info("[preload] NSFW path warm in %.1fs", time.perf_counter() - t_ns)

                # qwen3-vl is used by describe_image_for_edit for attached-image
                # edits — cold load is ~30s so preload here removes it from
                # the user-facing critical path.
                logger.info("[preload] warming qwen3-vl (edit descriptor)...")
                t_vl = time.perf_counter()
                try:
                    b64_edit = encode_image_downsized_to_base64(sample, max_side=256)
                    await self.image_gen.ollama_client.describe_image_for_edit(
                        b64_edit, "make it slightly brighter"
                    )
                    logger.info("[preload] qwen3-vl warm in %.1fs", time.perf_counter() - t_vl)
                except Exception as e:
                    logger.warning("[preload] qwen3-vl warmup failed: %s", e)
            else:
                logger.info("[preload] skipped vision warmup: no sample image at %s", sample)

            logger.info("[preload] image pipeline ready in %.1fs", time.perf_counter() - t0)
        except Exception as e:
            logger.exception("[preload] failed after %.1fs: %s",
                             time.perf_counter() - t0, e)

    @tasks.loop(hours=24)
    async def _claude_code_max_reminder(self):
        """Send a daily DM reminder to purchase Claude Code Max."""
        try:
            user = await self.fetch_user(134429572405002240)
            await user.send("hey buy claude code max already you keep forgetting https://claude.ai/upgrade")
        except Exception as e:
            print(f"Failed to send Claude Code Max reminder: {e}")

    @_claude_code_max_reminder.before_loop
    async def _before_reminder(self):
        await self.wait_until_ready()

    async def on_ready(self):
        """Called when the bot is fully connected. Send post-restart notification if pending."""
        print(f"Logged in as {self.user}")
        if not self._claude_code_max_reminder.is_running():
            self._claude_code_max_reminder.start()

        # Start PTY session eagerly so it's ready for the first message
        if hasattr(self.claude_code_client, 'ensure_session'):
            try:
                await self.claude_code_client.ensure_session()
                print("[bot] Claude Code PTY session started")
            except Exception as e:
                print(f"[bot] Failed to start PTY session: {e}")

        # Preload image-generation pipeline in the background so the first
        # user image request isn't hit with 3-5 minutes of cold-load latency.
        # This runs concurrently with normal bot operation — chat replies
        # still work while Flux is warming up.
        asyncio.create_task(self._preload_image_pipeline())
        notify = self._state.pop("restart_notify", None)
        if notify:
            self._save_state()  # Clear the notification from disk
            channel = self.get_channel(notify["channel_id"])
            if channel:
                try:
                    await channel.send(notify.get("message", "ok i'm back, changes have been applied"))
                except Exception as e:
                    print(f"Failed to send restart notification: {e}")

        # Check for crash sentinel written by run_bot.sh
        sentinel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_exit_code")
        if os.path.exists(sentinel_path):
            try:
                with open(sentinel_path) as f:
                    exit_code = f.read().strip()
                os.remove(sentinel_path)

                channel_id = self._state.get("last_active_channel_id")
                channel = self.get_channel(channel_id) if channel_id else None
                if channel:
                    logs = self._read_recent_logs()
                    # Extract just the traceback portion for the Discord message
                    traceback_lines = []
                    in_traceback = False
                    for line in logs.splitlines():
                        if line.startswith("Traceback"):
                            in_traceback = True
                            traceback_lines = [line]
                        elif in_traceback:
                            traceback_lines.append(line)
                    error_display = "\n".join(traceback_lines[-20:]) if traceback_lines else logs[-500:]
                    error_display = error_display[:1500]

                    await channel.send(
                        f"i just crashed (exit code {exit_code}). here's what went wrong:\n"
                        f"```\n{error_display}\n```"
                    )
                    from commands import CrashFixView
                    view = CrashFixView(self)
                    await channel.send("want me to try to fix it?", view=view)
                else:
                    print(f"Crash detected (exit {exit_code}) but no last active channel to report to")
            except Exception as e:
                print(f"Error handling crash sentinel: {e}")

    def pick_model(self, server: int, channel: int) -> str:
        """Pick the appropriate model based on context and active backend"""
        # Claude Code doesn't support images via CLI
        if is_claude_code_model(self.active_model):
            has_images = (server in self.context and
                channel in self.context[server] and
                self.context[server][channel] and
                self.context[server][channel][-1].get("images"))
            if has_images:
                return IMAGE_RECOGNITION_MODEL
            return self.active_model
        # Local models: switch to vision model when images are present
        if (server in self.context and
            channel in self.context[server] and
            self.context[server][channel] and
            self.context[server][channel][-1].get("images")):
            return IMAGE_RECOGNITION_MODEL
        return self.active_model

    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        # Ignore bot messages (including self)
        if message.author.bot:
            return

        # Ignore DMs (no guild)
        if message.guild is None:
            return

        # Record to persistent chat history (before any early returns)
        await chat_history.record_message(message)

        server = message.guild.id
        channel = message.channel.id

        # logger.info(f"[MSG-DEBUG] on_message: msg_id={message.id} author={message.author.display_name}"
        #              f"({message.author.id}) channel={channel} server={server} "
        #              f"content={message.content[:80]!r}")

        # Let plugins handle the message first (hot-swappable).
        # Plugins use LLM-based intent classification and return False for
        # messages they don't handle (e.g. code modification requests).
        if message.content.strip() and await self.plugin_manager.dispatch_message_handlers(message):
            return

        # Handle attachments
        image_files = []
        document_files = []

        for attachment in message.attachments:
            safe_filename = os.path.basename(attachment.filename)
            file_path = safe_path(os.path.join(FILE_INPUT_FOLDER, safe_filename))
            await attachment.save(file_path)

            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                image_files.append(file_path)
            else:
                document_files.append(file_path)

        # Determine response mode
        is_direct_mention = self.user in message.mentions
        is_reply_to_bot = False

        if message.reference:
            # Use cached resolved message when available
            ref_msg = message.reference.resolved
            if ref_msg is None:
                try:
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                except discord.NotFound:
                    ref_msg = None
            if ref_msg and ref_msg.author.id == self.user.id:
                is_reply_to_bot = True

        # Only respond to direct mentions and replies
        if not (is_direct_mention or is_reply_to_bot):
            return

        # Track last active channel for crash reporting
        self._state["last_active_channel_id"] = channel
        self._save_state()

        try:
            user_text = re.sub(r'<@!?\d+>', '', message.content).strip()

            # If replying to a message, inject the referenced message into context
            if message.reference and ref_msg and ref_msg.author.id != self.user.id:
                if server not in self.context:
                    self.context[server] = {}
                if channel not in self.context[server]:
                    self.context[server][channel] = []
                ctx = self.context[server][channel]
                ref_content = ref_msg.clean_content

                # Download and encode any attachments from the referenced message
                ref_images = []
                ref_doc_context = ""
                for att in ref_msg.attachments:
                    safe_filename = os.path.basename(att.filename)
                    file_path = safe_path(os.path.join(FILE_INPUT_FOLDER, f"ref_{safe_filename}"))
                    try:
                        await att.save(file_path)
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                            ref_images.extend(encode_images_to_base64([file_path]))
                        else:
                            content = FileParser.parse_file(file_path)
                            if content:
                                ref_doc_context += f"\n\n--- Content of {att.filename} ---\n{content}\n--------------------------\n"
                    except Exception as e:
                        print(f"Error processing ref attachment {att.filename}: {e}")
                    finally:
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass

                if ref_doc_context:
                    ref_content += f"\n\n[Attached Documents Context]{ref_doc_context}"

                # Only add if not already the last entry in context
                already_in_ctx = (
                    ctx and ctx[-1]["role"] == "user"
                    and ctx[-1]["content"] == ref_content
                )
                if not already_in_ctx and (ref_content or ref_images):
                    ctx.append({
                        "role": "user",
                        "name": ref_msg.author.display_name,
                        "content": ref_content or "(attachment)",
                        "timestamp": ref_msg.created_at.timestamp(),
                        "images": ref_images,
                    })
                    if len(ctx) > CONTEXT_LIMIT:
                        ctx.pop(0)

            fetched_sources = await self.build_context(message, server, False, image_files, document_files)
            # logger.info(f"[MSG-DEBUG] Calling _send_response: msg_id={message.id} channel={channel}")
            await self._send_response(message, server, channel, fetched_sources)

        except Exception as e:
            print(f"Error in on_message: {e}")
            print(traceback.format_exc())
            try:
                await message.channel.send(f"something broke: {e}")
            except Exception:
                pass

    async def build_context(
        self,
        message: discord.Message,
        server: int,
        strip_mention: bool = False,
        image_files: List[str] = None,
        document_files: List[str] = None
    ):
        """Build conversation context"""
        if image_files is None:
            image_files = []
        if document_files is None:
            document_files = []

        channel = message.channel.id

        if server not in self.context:
            self.context[server] = {}

        if channel not in self.context[server]:
            self.context[server][channel] = []

        prompt = (
            message.content
            if not strip_mention
            else message.clean_content.replace(f"@{self.user.name}", "").strip()
        )

        # Process documents and clean up files after parsing
        if document_files:
            doc_context = ""
            for doc_path in document_files:
                content = FileParser.parse_file(doc_path)
                if content:
                    filename = os.path.basename(doc_path)
                    doc_context += f"\n\n--- Content of {filename} ---\n{content}\n--------------------------\n"
                try:
                    os.remove(doc_path)
                except OSError:
                    pass

            if doc_context:
                prompt += f"\n\n[Attached Documents Context]{doc_context}"

        # Extract web page content if URLs are present
        webpage_context, fetched_sources = await extract_webpage_context(prompt)
        if webpage_context:
            prompt = f"{prompt}\n\n{webpage_context}"

        # Encode images (keep files for potential img2img editing)
        images = []
        if image_files:
            images = encode_images_to_base64(image_files)

        self.context[server][channel].append({
            "role": "user",
            "name": message.author.display_name,
            "discord_user_id": message.author.id,
            "content": prompt,
            "timestamp": time.time(),
            "images": images,
            "image_files": list(image_files),  # Keep paths for img2img editing
        })

        # Maintain context limit
        if len(self.context[server][channel]) > CONTEXT_LIMIT:
            self.context[server][channel].pop(0)

        return fetched_sources

    async def _send_response(
        self,
        message: discord.Message,
        server: int,
        channel: int,
        sources: List[dict] = None,
    ):
        """
        Generate and send response with natural typing delays for multiple messages.
        """
        async with message.channel.typing():
            start = time.perf_counter()
            response_data = await self.query_ollama(server, channel)
            end = time.perf_counter()
            elapsed = end - start
            print(f"Query took {elapsed:.2f}s")

        # Merge URL-fetched sources with search sources
        all_sources = list(sources or [])
        search_sources = getattr(self, '_last_search_sources', [])
        if search_sources:
            all_sources.extend(search_sources)
            self._last_search_sources = []

        # Handle image generation responses (tuple with embed + file)
        if isinstance(response_data, tuple):
            await message.channel.send(embed=response_data[0], file=response_data[1])
            return

        # Check for [EDIT_CODE] tags in the response — LLM decided a code change is needed
        # Join all text parts first since the tag may span multiple response items
        edit_instruction = None
        full_text = "\n".join(
            item for item in response_data if isinstance(item, str)
        )
        # Check if any plugin wants to suppress text BEFORE dispatching hooks
        # (e.g. TTS voice mode — we don't want to send text then audio)
        suppress_text = self.plugin_manager.should_suppress_text(message)
        logger.debug(f"[TTS-DEBUG] suppress_text={suppress_text} for user={message.author.id} "
                     f"(voice_mode check before POST_QUERY hook)")

        # Dispatch POST_QUERY hook (e.g. TTS voice mode generates + sends audio)
        logger.debug(f"[TTS-DEBUG] Dispatching POST_QUERY hook, full_text length={len(full_text)}")
        hook_results = await self.plugin_manager.dispatch_hook(
            HookType.POST_QUERY,
            message=message,
            response_text=full_text,
        )
        logger.debug(f"[TTS-DEBUG] POST_QUERY hook returned {len(hook_results)} results: {hook_results}")

        match = _EDIT_CODE_TAG.search(full_text)
        if match:
            edit_instruction = match.group(1).strip()
            full_text = _EDIT_CODE_TAG.sub('', full_text).strip()

        # Preserve any non-string items (images, etc.) and replace text with cleaned version
        non_text_items = [item for item in response_data if not isinstance(item, str)]
        response_data = [full_text] + non_text_items if full_text else non_text_items

        # Collect all text parts to figure out where to append sources
        # Claude Code uses paragraph breaks; local models use ---MSG--- markers
        splitter = split_response_by_paragraphs if is_anthropic_model(self.active_model) else split_response_by_markers
        all_parts = []
        for response_item in response_data:
            if isinstance(response_item, str) and response_item.strip():
                all_parts.extend(splitter(response_item))

        # Append source footnote with clickable links (deduplicated by domain)
        if all_sources and all_parts:
            seen_domains = set()
            source_links = []
            for s in all_sources:
                try:
                    domain = s['url'].split('/')[2].removeprefix('www.')
                except (IndexError, AttributeError):
                    continue
                if domain not in seen_domains:
                    seen_domains.add(domain)
                    source_links.append(f"[{domain}](<{s['url']}>)")
            if source_links:
                footnote = f"\n-# Sources: {' | '.join(source_links)}"
                last = all_parts[-1] + footnote
                if len(last) <= MAX_DISCORD_MESSAGE_LENGTH:
                    all_parts[-1] = last
                else:
                    all_parts.append(f"-# Sources: {' | '.join(source_links)}")

        # Send text parts with typing delays (unless a hook suppressed text, e.g. voice mode)
        logger.debug(f"[TTS-DEBUG] About to send text. suppress_text={suppress_text}, "
                     f"all_parts count={len(all_parts)}")
        if not suppress_text:
            for i, part in enumerate(all_parts):
                async with message.channel.typing():
                    delay = calculate_typing_delay(part)
                    delay *= random.uniform(0.9, 1.3)
                    await asyncio.sleep(delay)

                print(f"Sending response part {i+1}/{len(all_parts)}: {part[:50]}...")
                sent_msg = await message.channel.send(part)

                # Record bot response to persistent chat history
                await chat_history.record_bot_response(
                    guild_id=server,
                    channel_id=channel,
                    bot_user_id=self.user.id,
                    bot_name=self.user.display_name,
                    content=part,
                    message_id=sent_msg.id,
                    reply_to_message_id=message.id if i == 0 else None,
                )

                if i < len(all_parts) - 1:
                    await asyncio.sleep(random.uniform(0.3, 0.8))

            # Handle any image items
            for response_item in response_data:
                if isinstance(response_item, dict) and "image" in response_item:
                    file = discord.File(response_item["image"])
                    await message.channel.send(file=file)
        else:
            logger.debug(f"[TTS-DEBUG] Text suppressed — skipping {len(all_parts)} text parts")

        # If the LLM decided a code edit is needed, trigger it
        if edit_instruction:
            await self._execute_code_change(message, edit_instruction)

    async def _execute_code_change(self, message: discord.Message, instruction: str):
        """Run Claude Code to modify bot source, show diff, offer apply/revert.

        Routes to plugin-scoped edit when the instruction looks like a new feature
        or plugin modification. Falls back to core edit (with restart) otherwise.
        """
        # Detect if this should be a plugin edit
        plugin_keywords = [
            "plugin", "add feature", "new command", "add command", "add a command",
            "new feature", "add a feature",
        ]
        existing_plugins = self.plugin_manager.plugin_names
        instruction_lower = instruction.lower()

        is_plugin_edit = (
            any(kw in instruction_lower for kw in plugin_keywords)
            or any(p in instruction_lower for p in existing_plugins)
        )

        if is_plugin_edit:
            await self._execute_plugin_change(
                message.channel, message.author.id, instruction
            )
        else:
            await self._execute_code_change_with_logs(
                message.channel, message.author.id, instruction
            )

    async def _execute_plugin_change(
        self, channel, author_id: int, instruction: str
    ):
        """Run Claude Code scoped to plugin files, show diff, offer hot-reload."""
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Git snapshot
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m",
            f"[auto] pre-plugin-edit snapshot {time.strftime('%Y%m%d_%H%M%S')}",
            "--allow-empty", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        status_msg = await channel.send("on it, working on a plugin change...")

        try:
            async with channel.typing():
                response, exit_code = await self.claude_code_client.run_plugin_edit(
                    instruction,
                    model=self.active_model,
                    existing_plugins=self.plugin_manager.plugin_names,
                )
        except RateLimitError:
            reset = self.claude_code_client.rate_limit_resets_at or "unknown"
            await status_msg.edit(content=f"claude code is rate limited, resets at {reset}")
            return
        except Exception as e:
            await status_msg.edit(content=f"something went wrong: {e}")
            return

        if exit_code != 0:
            await status_msg.edit(content=f"plugin edit failed (exit {exit_code}):\n```\n{response[:1500]}\n```")
            return

        # Show diff (only plugins/ directory)
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--", "plugins/", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        diff = stdout.decode("utf-8", errors="replace")

        # Also check for new untracked files in plugins/
        proc = await asyncio.create_subprocess_exec(
            "git", "ls-files", "--others", "--exclude-standard", "plugins/", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        new_stdout, _ = await proc.communicate()
        new_files = new_stdout.decode("utf-8", errors="replace").strip()

        await status_msg.edit(content="done, here's what i changed:")

        if diff or new_files:
            display_parts = []
            if diff:
                diff_display = diff[:1500]
                if len(diff) > 1500:
                    diff_display += "\n... (truncated)"
                display_parts.append(diff_display)
            if new_files:
                display_parts.append(f"New files:\n{new_files}")
            await channel.send(f"```diff\n{''.join(display_parts)}\n```")
        else:
            # Check if changes were made outside plugins/
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", cwd=project_dir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            full_stdout, _ = await proc.communicate()
            if full_stdout.decode().strip():
                await channel.send("changes were made outside plugins/ — falling back to full edit flow")
                from commands import RestartConfirmView
                view = RestartConfirmView(self, author_id)
                await channel.send(response[:1500])
                await channel.send("apply changes and restart?", view=view)
                return
            await channel.send(f"no files changed.\n\nclaude said: {response[:1500]}")
            return

        # Summary
        summary = response[:1500] if len(response) > 1500 else response
        await channel.send(summary)

        # Determine which plugins to reload
        affected_plugins = []
        all_changed = (diff + "\n" + new_files) if new_files else diff
        for name in self.plugin_manager.discover_plugins():
            if f"plugins/{name}" in all_changed:
                affected_plugins.append(name)
        # If we couldn't detect specific plugins, reload all
        if not affected_plugins:
            affected_plugins = self.plugin_manager.plugin_names

        # ── Test step ──────────────────────────────────────────────────
        test_msg = await channel.send("running tests...")
        try:
            async with channel.typing():
                test_result = await self.claude_code_client.run_tests(
                    diff=diff,
                    change_type="plugin",
                    plugin_names=affected_plugins,
                    model="sonnet",
                )
        except Exception as e:
            await test_msg.edit(content=f"test runner error: {e}")
            test_result = None

        if test_result:
            status = "PASSED" if test_result.passed else "FAILED"
            await test_msg.edit(content=f"tests {status}")
            report = test_result.full_report[:1500]
            await channel.send(report)
            test_passed = test_result.passed
        else:
            test_passed = True  # If tests couldn't run, don't block

        from commands import PluginApplyView
        view = PluginApplyView(self, author_id, affected_plugins, original_instruction=instruction, test_passed=test_passed)
        label = "apply and hot-reload?" if test_passed else "tests failed. apply anyway, or revert?"
        await channel.send(label, view=view)

    async def _execute_code_change_with_logs(
        self, channel, author_id: int, instruction: str
    ):
        """Run Claude Code to modify bot source, show diff, offer apply/revert."""
        # Git snapshot for safety
        project_dir = os.path.dirname(os.path.abspath(__file__))
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m",
            f"[auto] pre-modification snapshot {time.strftime('%Y%m%d_%H%M%S')}",
            "--allow-empty", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        status_msg = await channel.send("on it, gimme a sec...")

        try:
            async with channel.typing():
                response, exit_code = await self.claude_code_client.run_code_edit(
                    instruction, model=self.active_model
                )
        except RateLimitError:
            reset = self.claude_code_client.rate_limit_resets_at or "unknown"
            await status_msg.edit(content=f"claude code is rate limited, resets at {reset}")
            return
        except Exception as e:
            await status_msg.edit(content=f"something went wrong: {e}")
            return

        if exit_code != 0:
            await status_msg.edit(content=f"code edit failed (exit {exit_code}):\n```\n{response[:1500]}\n```")
            return

        # Show diff
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", cwd=project_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        diff = stdout.decode("utf-8", errors="replace")

        await status_msg.edit(content="done, here's what i changed:")

        if diff:
            diff_display = diff[:1800]
            if len(diff) > 1800:
                diff_display += "\n... (truncated)"
            await channel.send(f"```diff\n{diff_display}\n```")
        else:
            await channel.send(f"no files changed.\n\nclaude said: {response[:1500]}")
            return

        # Summary
        summary = response[:1500] if len(response) > 1500 else response
        await channel.send(summary)

        # ── Test step ──────────────────────────────────────────────────
        test_msg = await channel.send("running tests...")
        try:
            async with channel.typing():
                test_result = await self.claude_code_client.run_tests(
                    diff=diff,
                    change_type="core",
                    model="sonnet",
                )
        except Exception as e:
            await test_msg.edit(content=f"test runner error: {e}")
            test_result = None

        if test_result:
            status = "PASSED" if test_result.passed else "FAILED"
            await test_msg.edit(content=f"tests {status}")
            report = test_result.full_report[:1500]
            await channel.send(report)
            test_passed = test_result.passed
        else:
            test_passed = True  # If tests couldn't run, don't block

        # Apply/Revert buttons
        from commands import RestartConfirmView
        view = RestartConfirmView(self, author_id, test_passed=test_passed)
        label = "apply changes and restart?" if test_passed else "tests failed. apply anyway, or revert?"
        await channel.send(label, view=view)

    def format_prompt(self, messages: List[dict]) -> str:
        """Format messages into a prompt with clear turn boundaries."""
        parts = []
        for i, msg in enumerate(messages, 1):
            if msg["role"] == "user":
                name = msg.get("name", "Unknown")
                uid = msg.get("discord_user_id", "")
                user_label = f"User ({name}, discord_id={uid})" if uid else f"User ({name})"
                parts.append(f"[Turn {i} | {user_label}]\n{msg['content']}\n[/Turn {i}]")
            else:
                parts.append(f"[Turn {i} | Assistant]\n{msg['content']}\n[/Turn {i}]")
        parts.append(f"[Turn {len(messages) + 1} | Assistant]")
        return "\n\n".join(parts) + "\n"

    def process_response(self, text: str, limit: int = MAX_DISCORD_MESSAGE_LENGTH) -> List:
        """Process response text, handling length limits"""
        # Remove thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Wrap bare URLs in <> to suppress Discord auto-embeds
        text = re.sub(r'(?<![<])(https?://\S+)', r'<\1>', text)
        # Split long text to fit Discord's message length limit
        parts = split_long_message(text.strip(), limit)
        return parts

    async def query_ollama(self, server: int, channel: int, override_messages: List[dict] = None):
        """Query Ollama for a response"""
        messages = override_messages or self.context[server][channel]
        user_content = messages[-1]['content']
        images = messages[-1].get("images", [])

        # Check if this is an image generation / edit task.
        # Three entry conditions:
        #   1. Direct keyword match ("generate/create/draw/... image/picture/...")
        #   2. Follow-up to a previously-generated bot image in context
        #   3. Fresh user-attached image (could be an edit request like
        #      "make her hair green" or just a visual question)
        has_recent_image_gen = any(
            msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", "")
            for msg in messages[-4:]
        )
        keyword_match = bool(_IMAGE_GEN_KEYWORDS.search(user_content))
        has_attached_image = bool(messages[-1].get("image_files"))
        logger.info(
            "[img] detection user_content_len=%d keyword_match=%s has_recent_image_gen=%s has_attached_image=%s",
            len(user_content), keyword_match, has_recent_image_gen, has_attached_image,
        )
        if keyword_match or has_recent_image_gen or has_attached_image:
            # Give the classifier enough context to disambiguate chat-about-image
            # from edit-this-image. Prefix based on what triggered us.
            classify_input = user_content
            if has_recent_image_gen and not keyword_match:
                for msg in reversed(messages[-4:]):
                    if msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", ""):
                        classify_input = f"[Previous: {msg['content']}]\nUser: {user_content}"
                        logger.info("[img] follow-up classify_input includes prev marker")
                        break
            elif has_attached_image and not keyword_match:
                classify_input = f"[User attached an image]\nUser: {user_content}"
                logger.info("[img] attached-image classify_input includes attachment marker")
            logger.info("[img] calling image-gen classifier")
            is_img_task = await self.image_gen.is_image_generation_task(classify_input)
            logger.info("[img] classifier verdict=%s", is_img_task)

            if is_img_task:
                img_pipeline_start = time.perf_counter()
                # Determine if this is a modification of a previous image or a fresh request
                is_modification = has_recent_image_gen and not keyword_match
                prev_seed = -1
                prev_prompt = None
                prev_image_path = None
                prev_width = 1024
                prev_height = 1024

                if is_modification:
                    # Extract previous prompt, seed, dims, and file path from context
                    for msg in reversed(messages[-4:]):
                        content = msg.get("content", "")
                        if msg.get("role") == "assistant" and "[Generated an image" in content:
                            prompt_match = re.search(r'\[Generated an image with the following prompt: (.+?)\]', content, re.DOTALL)
                            seed_match = re.search(r'seed: (\d+)', content)
                            size_match = re.search(r'size: (\d+)x(\d+)', content)
                            path_match = re.search(r'path: (.+?)\)', content)
                            if prompt_match:
                                prev_prompt = prompt_match.group(1)
                            if seed_match:
                                prev_seed = int(seed_match.group(1))
                            if size_match:
                                prev_width = int(size_match.group(1))
                                prev_height = int(size_match.group(2))
                            if path_match:
                                prev_image_path = path_match.group(1)
                            break
                    logger.info(
                        "[img] modification=True extracted prev_seed=%s prev_width=%s prev_height=%s "
                        "prev_image_path=%s prev_prompt_len=%s",
                        prev_seed, prev_width, prev_height, prev_image_path,
                        len(prev_prompt) if prev_prompt else None,
                    )
                else:
                    logger.info("[img] modification=False (fresh generation)")

                # Check for user-attached images for editing
                attached_image_path = None
                if images:
                    last_msg_images = messages[-1].get("image_files", [])
                    if last_msg_images:
                        attached_image_path = last_msg_images[0]
                        logger.info("[img] user attached image: %s", attached_image_path)

                try:
                    if attached_image_path and os.path.exists(attached_image_path):
                        # Case 2: User provided an image to edit.
                        #
                        # Per BFL's FLUX.2 Klein prompting guide:
                        #   "Reference images carry visual details. Your prompt
                        #    describes what should change or how elements should
                        #    combine — not what they look like."
                        #
                        # Pass the cleaned user instruction directly. Klein's
                        # reference conditioning extracts the visual content
                        # from the source image itself — re-describing it
                        # via a VLM only competes with that signal and causes
                        # identity drift.
                        with Image.open(attached_image_path) as src:
                            src_w, src_h = src.size
                        width, height = choose_source_dimensions(user_content, src_w, src_h)
                        logger.info(
                            "[img] branch=attached-edit source_dims=%dx%d chosen=%dx%d",
                            src_w, src_h, width, height,
                        )
                        prompt = clean_edit_instruction(user_content) or "enhance"
                        logger.info("[img] edit instruction: %r", prompt)
                        file_path, image_info, is_nsfw = await self.image_gen.edit_image(
                            prompt, attached_image_path, seed=-1,
                            width=width, height=height,
                        )
                    elif is_modification and prev_image_path and os.path.exists(prev_image_path):
                        # Case 1: Follow-up edit of a bot-generated image.
                        # Same BFL guidance as Case 2: pass the user's edit
                        # instruction directly, do not rewrite it into a full
                        # target description (that fights the reference
                        # conditioning and produces the "still cats" failure
                        # mode where the source visual signal beats the text).
                        width, height = choose_followup_dimensions(user_content, prev_width, prev_height)
                        logger.info(
                            "[img] branch=followup-edit source=%s prev_dims=%dx%d chosen=%dx%d",
                            prev_image_path, prev_width, prev_height, width, height,
                        )
                        prompt = clean_edit_instruction(user_content) or "enhance"
                        logger.info("[img] edit instruction: %r", prompt)
                        file_path, image_info, is_nsfw = await self.image_gen.edit_image(
                            prompt, prev_image_path, seed=-1,
                            width=width, height=height,
                        )
                    elif is_modification and prev_prompt:
                        # Fallback: no previous image file, re-generate with modified prompt
                        width, height = choose_followup_dimensions(user_content, prev_width, prev_height)
                        logger.info(
                            "[img] branch=prompt-only-fallback seed=%s chosen=%dx%d",
                            prev_seed, width, height,
                        )
                        logger.info("[img] rewriting prompt via modify_image_prompt")
                        prompt = (await self.ollama_client.modify_image_prompt(prev_prompt, user_content)).strip()
                        if not prompt:
                            logger.warning("[img] modify_image_prompt returned empty, falling back to original+user text")
                            prompt = f"{prev_prompt}, {user_content}"
                        logger.info("[img] modified prompt len=%d", len(prompt))
                        file_path, image_info, is_nsfw = await self.image_gen.generate_image(
                            prompt, seed=prev_seed, width=width, height=height,
                        )
                    else:
                        # Case 3: Brand new generation
                        prev_seed = -1
                        webpage_context, _ = await extract_webpage_context(user_content)
                        if webpage_context:
                            logger.info("[img] extracted webpage context len=%d", len(webpage_context))
                            user_content = f"{user_content}\n\n{webpage_context}"
                        logger.info("[img] rewriting prompt via generate_image_prompt")
                        prompt = (await self.image_gen.generate_image_prompt(user_content)).strip()
                        if not prompt:
                            logger.warning("[img] generate_image_prompt returned empty, falling back to raw user text")
                            prompt = user_content
                        logger.info("[img] rewritten prompt len=%d", len(prompt))
                        # Pick dims from the combined user query + rewritten prompt so
                        # keywords in either (e.g. "cityscape" in the rewrite) count.
                        width, height = choose_dimensions(f"{user_content} {prompt}")
                        logger.info(
                            "[img] branch=fresh chosen=%dx%d (from user + rewritten prompt)",
                            width, height,
                        )
                        file_path, image_info, is_nsfw = await self.image_gen.generate_image(
                            prompt, seed=prev_seed, width=width, height=height,
                        )
                    logger.info(
                        "[img] pipeline complete in %.2fs file=%s nsfw=%s",
                        time.perf_counter() - img_pipeline_start, file_path, is_nsfw,
                    )
                except Exception as img_err:
                    logger.exception(
                        "[img] pipeline failed after %.2fs",
                        time.perf_counter() - img_pipeline_start,
                    )
                    return [
                        f"image generation failed: {type(img_err).__name__}: {img_err}"
                    ]

                # Store image generation context for follow-up continuity (include path for img2img)
                if override_messages is None:
                    self.context[server][channel].append({
                        "role": "assistant",
                        "content": (
                            f"[Generated an image with the following prompt: {prompt}] "
                            f"(seed: {image_info.seed}, "
                            f"size: {image_info.width}x{image_info.height}, "
                            f"path: {file_path})"
                        ),
                        "timestamp": time.time(),
                    })

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

                return (embed, file)

        # Determine which backend we're using
        model = self.pick_model(server, channel)
        using_claude_code = is_claude_code_model(model)

        # Auto-fallback: if Claude Code is rate limited, fall back to local model
        if using_claude_code and self.claude_code_client.is_rate_limited:
            reset = self.claude_code_client.rate_limit_resets_at
            print(f"  [Claude Code rate limited, resets at {reset}, falling back to local model]")
            model = CHAT_MODEL
            using_claude_code = False

        # Check if the user's message needs a web search (heuristic + LLM)
        # Skip manual search pipeline when using Claude Code — it handles search itself
        search_summary = ""
        search_sources = []
        if not using_claude_code:
            # Strip Discord mentions for cleaner LLM input
            clean_query = re.sub(r'<@!?\d+>', '', user_content).strip()
            if _SEARCH_KEYWORDS.search(clean_query):
                print("  [search heuristic matched, checking with LLM...]")
                needs_search = await self.ollama_client.classify_search_task(clean_query)
                if needs_search:
                    search_query = await self.ollama_client.extract_search_query(clean_query)
                    print(f"  [searching: {search_query}]")
                    search_results = await web_search(search_query, max_results=5)
                    if search_results:
                        raw_context = format_search_results(search_results)
                        print(f"  [summarizing {len(raw_context)} chars of search results...]")
                        search_summary = await self.ollama_client.summarize_search_results(clean_query, raw_context)
                        print(f"  [summary: {len(search_summary)} chars]")
                        print(f"  [summary content: {search_summary}]")
                        search_sources = [
                            {"url": r["url"], "title": r["title"] or r["url"]}
                            for r in search_results[:3]
                        ]

        self._last_search_sources = search_sources

        # Extract active user IDs from conversation context for memory filtering
        active_user_ids = list({
            str(m.get("discord_user_id", ""))
            for m in messages if m.get("discord_user_id")
        })

        # Build prompt — PTY sessions keep their own history, so only send
        # the new message + fresh context. One-shot CLI needs the full history.
        is_pty = isinstance(self.claude_code_client, ClaudeCodeClientPTY)

        if is_pty and using_claude_code:
            # PTY mode: send only the latest user message with metadata
            last_msg = messages[-1]
            name = last_msg.get("name", "Unknown")
            uid = last_msg.get("discord_user_id", "")

            # Check for per-user personality override (e.g. TTS voice mode)
            personality_note = ""
            if uid:
                override = self.plugin_manager.get_system_prompt_override(int(uid))
                if override:
                    personality_note = f"\n[Personality override for this user: {override}]\n"

            prompt = f"{personality_note}[{name} (discord_id={uid})] {user_content}"

            # Attach fresh per-message context
            if search_summary:
                prompt = f"Search Results Summary:\n{search_summary}\n\n{prompt}"

            memory_context = chat_history.get_memory_context(str(server), channel_id=str(channel), active_user_ids=active_user_ids)
            if memory_context:
                prompt = f"{prompt}\n\n{memory_context}"

            # Include channel name for context isolation
            guild = self.get_guild(server)
            channel_obj = guild.get_channel(channel) if guild else None
            channel_name = channel_obj.name if channel_obj else str(channel)
            prompt += f"\n\n[Current context: #{channel_name}, guild_id={server}, channel_id={channel}]"

            mention_context = extract_mention_context(user_content, guild)
            if mention_context:
                prompt = f"{prompt}\n\n{mention_context}"

            if self.rag_enabled:
                wiki_context = self.rag_system.get_context_for_query(user_content)
                if wiki_context:
                    prompt = f"Wiki Context:\n{wiki_context}\n\n{prompt}"
        else:
            # One-shot CLI / Ollama: send full conversation history
            prompt = self.format_prompt(messages)

            if search_summary:
                prompt = f"Search Results Summary:\n{search_summary}\n\n{prompt}"

            if self.rag_enabled:
                user_question = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_question = msg.get("content", "")
                if user_question:
                    wiki_context = self.rag_system.get_context_for_query(user_question)
                    if wiki_context:
                        prompt = f"Wiki Context:\n{wiki_context}\n\n{prompt}"

            last_user_id = messages[-1].get("discord_user_id")
            system_prompt = self.system_prompt
            if last_user_id:
                override = self.plugin_manager.get_system_prompt_override(last_user_id)
                if override:
                    system_prompt = override
                    logger.debug(f"[TTS-DEBUG] Using personality override for user {last_user_id}")
            prompt = f"System: {system_prompt}\n" + prompt

            memory_context = chat_history.get_memory_context(str(server), channel_id=str(channel), active_user_ids=active_user_ids)
            if memory_context:
                prompt = f"{prompt}\n\n{memory_context}"

            prompt += f"\n\n[Current context: guild_id={server}, channel_id={channel}]"

            guild = self.get_guild(server)
            mention_context = extract_mention_context(user_content, guild)
            if mention_context:
                prompt = f"{prompt}\n\n{mention_context}"

        if images:
            print("Sending image")

        try:
            print(f"Using model: {model}")
            print(f"Prompt: {prompt}")

            if using_claude_code:
                try:
                    # PTY client: inject system prompt on first use
                    if is_pty and hasattr(self.claude_code_client, 'set_system_prompt'):
                        await self.claude_code_client.set_system_prompt(self.system_prompt)

                    raw_response, _ = await self.claude_code_client.generate_with_tools(
                        prompt, model, images
                    )
                    if raw_response == "No response from Claude Code.":
                        print("No response from Claude Code")
                        return ["No response from Claude Code."]
                except RateLimitError:
                    reset = self.claude_code_client.rate_limit_resets_at or "unknown"
                    print(f"  [Claude Code rate limited, resets at {reset}, falling back to local model]")
                    model = CHAT_MODEL
                    ctx = VISION_MODEL_CTX if images else None
                    raw_response = await self.ollama_client.generate(prompt, model, images, keep_alive=-1, num_ctx=ctx)

                    if raw_response == "No response from Ollama.":
                        return ["No response from Ollama."]
                except Exception as cc_err:
                    print(f"  [Claude Code error: {cc_err}, falling back to local model]")
                    model = CHAT_MODEL
                    ctx = VISION_MODEL_CTX if images else None
                    raw_response = await self.ollama_client.generate(prompt, model, images, keep_alive=-1, num_ctx=ctx)

                    if raw_response == "No response from Ollama.":
                        return ["No response from Ollama."]

            else:
                ctx = VISION_MODEL_CTX if images else None
                raw_response = await self.ollama_client.generate(prompt, model, images, keep_alive=-1, num_ctx=ctx)

                if raw_response == "No response from Ollama.":
                    print("No response from Ollama")
                    return ["No response from Ollama."]

            # Add response to context
            if override_messages is None:
                self.context[server][channel].append({
                    "role": "assistant",
                    "content": raw_response,
                    "timestamp": time.time(),
                })

            print(f"Response: {raw_response}")
            return self.process_response(raw_response)

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            backend = "Claude Code" if using_claude_code else "Ollama"
            return [f"Error communicating with {backend}: {e}"]

    async def request_restart(self, channel=None, reason=""):
        """Gracefully shut down the bot, signaling the wrapper to restart."""
        self._restart_requested = True
        if channel:
            try:
                await channel.send(f"Restarting... {reason}".strip())
            except Exception:
                pass
        await self.close()

    async def close(self):
        """Close the bot"""
        if hasattr(self.claude_code_client, 'shutdown'):
            await self.claude_code_client.shutdown()
        await js_renderer.stop()
        await super().close()


def main():
    """Main entry point"""
    bot = OllamaBot()
    bot.run(DISCORD_BOT_TOKEN)
    # After bot.run() returns, check if a restart was requested
    if getattr(bot, '_restart_requested', False):
        sys.exit(RESTART_EXIT_CODE)


if __name__ == "__main__":
    main()
