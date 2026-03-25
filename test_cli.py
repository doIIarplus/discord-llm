"""
Interactive CLI for testing the bot without Discord.

Mimics the Discord message flow: context building, LLM queries,
web extraction, web search, file attachments, and multi-user conversations.

Usage:
    python test_cli.py

Commands:
    /user <name>          Switch active user
    /attach <path>        Attach a file to the next message
    /clear                Clear conversation context
    /context              Show current context
    /search <query>       Web search via Tavily
    /model [name]         Show/switch active model (e.g. /model claude_code)
    /prompt               Show current system prompt
    /set_prompt <text>    Set system prompt
    /reset_prompt         Reset to default system prompt
    /plugins              List loaded plugins
    /reload_plugin <name> Hot-reload a plugin
    /load_plugin <name>   Load a plugin
    /unload_plugin <name> Unload a plugin
    /help                 Show this help
    /quit                 Exit
"""

import asyncio
import os
import re
import sys
import time
from typing import List

import aiohttp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONTEXT_LIMIT, CHAT_MODEL, IMAGE_RECOGNITION_MODEL, MAX_DISCORD_MESSAGE_LENGTH
from ollama_client import OllamaClient
from claude_code_client import ClaudeCodeClient, RateLimitError
from models import is_claude_code_model, Txt2TxtModel
from web_extractor import extract_webpage_context, web_search, format_search_results, js_renderer
from file_parser import FileParser
from response_splitter import split_response_by_markers, split_response_by_paragraphs, split_long_message

# Image gen heuristic from bot.py
_IMAGE_GEN_KEYWORDS = re.compile(
    r'\b(generate|create|draw|make|paint|render|sketch)\b.{0,30}\b(image|picture|photo|illustration|art|drawing|painting)\b',
    re.IGNORECASE
)

# Search heuristic from bot.py
_SEARCH_KEYWORDS = re.compile(
    r'\b(latest|recent|current|today|yesterday|tonight|this week|this month|this year'
    r'|news|update|score|weather|price|stock|election|released|announced'
    r'|who won|who is winning|what happened|how much does|how much is'
    r'|in 202[4-9]|right now)\b',
    re.IGNORECASE
)

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "blue": "\033[34m",
}


def c(text: str, *styles: str) -> str:
    """Colorize text."""
    prefix = "".join(COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{COLORS['reset']}"


class TestCLI:
    """Interactive CLI that mimics the Discord bot flow."""

    FAKE_SERVER = 1
    FAKE_CHANNEL = 1

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.claude_code_client = ClaudeCodeClient()
        self.active_model = CHAT_MODEL
        self.context: List[dict] = []
        self.system_prompt = (
            "Your responses should be akin to that of a typical millenial texter: short, to the point, and mostly without punctuation. Do not offer any kind of assistance without being prompted. use slang *sparingly*. \n\n"
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
            "CLI TOOLS:\n"
            "You have access to CLI tools via Bash in the tools/ directory. Use them when users ask about "
            "Splitwise (bills, balances, expenses), scheduled tasks, web searches, Discord actions, or other tool-related actions. "
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
            "Always use these tools when the user's request matches their capabilities instead of making up answers."
        )
        self.original_system_prompt = self.system_prompt
        self.current_user = "TestUser"
        self.pending_attachments: List[str] = []
        self.use_ddg = False
        self.plugin_manager = self._init_plugin_manager()

    def _init_plugin_manager(self):
        """Create a PluginManager with a minimal mock bot for CLI testing."""
        from plugin_manager import PluginManager

        # Minimal mock so plugins can load (they need ctx.discord_client, etc.)
        class _MockBot:
            def __init__(self, cli):
                self.context = {}
                self.active_model = cli.active_model
                self.system_prompt = cli.system_prompt
                self.ollama_client = cli.ollama_client
                self.claude_code_client = cli.claude_code_client
                self.tree = None  # No command tree in CLI mode
                self.rag_system = None
                self.rag_enabled = False

            def get_channel(self, _):
                return None

        mock = _MockBot(self)
        pm = PluginManager(mock)
        # Skip slash command registration in CLI mode
        pm._register_commands = lambda instance: None
        pm._unregister_commands = lambda instance: None
        return pm

    async def _fetch_ollama_models(self) -> list:
        """Query Ollama API for available local models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except Exception:
            return []

        models = []
        for m in data.get("models", []):
            name = m["name"]
            size_bytes = m.get("size", 0)
            param_size = m.get("details", {}).get("parameter_size", "?")

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
                "value": name,
                "display_name": display,
                "size_str": size_str,
                "param_size": param_size,
            })

        models.sort(key=lambda x: x["display_name"].lower())
        return models

    async def build_context(self, text: str, username: str):
        """Build context from user input, same logic as bot.py."""
        prompt = text

        # Process attached files
        if self.pending_attachments:
            doc_context = ""
            for path in self.pending_attachments:
                content = FileParser.parse_file(path)
                if content:
                    filename = os.path.basename(path)
                    doc_context += f"\n\n--- Content of {filename} ---\n{content}\n--------------------------\n"
            if doc_context:
                prompt += f"\n\n[Attached Documents Context]{doc_context}"
            self.pending_attachments = []

        # Extract web page content from URLs
        webpage_context, fetched_sources = await extract_webpage_context(prompt)
        if webpage_context:
            prompt = f"{prompt}\n\n{webpage_context}"

        self.context.append({
            "role": "user",
            "name": username,
            "content": prompt,
            "timestamp": time.time(),
            "images": [],
        })

        # Maintain context limit
        if len(self.context) > CONTEXT_LIMIT:
            self.context.pop(0)

        return fetched_sources

    def format_prompt(self, messages: List[dict]) -> str:
        """Format messages into a prompt string."""
        prompt = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            name = f"({msg['name']})" if msg["role"] == "user" and "name" in msg else ""
            prompt += f"{role} {name}: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt

    def process_response(self, text: str) -> List[str]:
        """Process response: strip thinking tags, split for length."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return split_long_message(text.strip(), MAX_DISCORD_MESSAGE_LENGTH)

    async def query(self) -> List[str]:
        """Query Ollama, same logic as bot.py's query_ollama."""
        messages = self.context
        user_content = messages[-1]["content"]

        # Check if this is an image generation task (fast heuristic first)
        # Also check if this could be a follow-up to a recent image generation
        has_recent_image_gen = any(
            msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", "")
            for msg in messages[-4:]
        )
        if _IMAGE_GEN_KEYWORDS.search(user_content) or has_recent_image_gen:
            # For follow-ups, give the classifier context about the recent image
            classify_input = user_content
            if has_recent_image_gen and not _IMAGE_GEN_KEYWORDS.search(user_content):
                for msg in reversed(messages[-4:]):
                    if msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", ""):
                        classify_input = f"[Previous: {msg['content']}]\nUser: {user_content}"
                        break

            from image_generation import ImageGenerator
            img_gen = ImageGenerator()
            is_img_task = await img_gen.is_image_generation_task(classify_input)

            if is_img_task:
                # Determine if this is a modification or a fresh request
                is_modification = has_recent_image_gen and not _IMAGE_GEN_KEYWORDS.search(user_content)
                prev_seed = -1
                prev_prompt = None

                if is_modification:
                    for msg in reversed(messages[-4:]):
                        content = msg.get("content", "")
                        if msg.get("role") == "assistant" and "[Generated an image" in content:
                            prompt_match = re.search(r'\[Generated an image with the following prompt: (.+?)\]', content, re.DOTALL)
                            seed_match = re.search(r'seed: (\d+)', content)
                            if prompt_match:
                                prev_prompt = prompt_match.group(1)
                            if seed_match:
                                prev_seed = int(seed_match.group(1))
                            break

                if is_modification and prev_prompt:
                    print(c(f"  [image modification: reusing seed {prev_seed}]", "yellow"))
                    simulated_prompt = (await self.ollama_client.modify_image_prompt(prev_prompt, user_content)).strip()
                    print(c(f"  [modified prompt: {simulated_prompt[:120]}...]", "yellow"))
                else:
                    print(c("  [new image generation]", "yellow"))
                    prev_seed = -1
                    simulated_prompt = (await self.ollama_client.generate_image_prompt(user_content)).strip()
                    print(c(f"  [SD prompt: {simulated_prompt[:120]}...]", "yellow"))

                # Store in context for follow-up continuity (same as bot.py)
                seed_display = prev_seed if prev_seed != -1 else "random"
                self.context.append({
                    "role": "assistant",
                    "content": (
                        f"[Generated an image with the following prompt: {simulated_prompt}] "
                        f"(seed: {seed_display}, size: 864x1280)"
                    ),
                    "timestamp": time.time(),
                })
                return [f"[Image would be generated | seed: {seed_display} | prompt: {simulated_prompt}]"]

        # Determine which backend we're using
        model = self.active_model
        using_claude_code = is_claude_code_model(model)

        # Auto-fallback: if Claude Code is rate limited, fall back to local model
        if using_claude_code and self.claude_code_client.is_rate_limited:
            reset = self.claude_code_client.rate_limit_resets_at
            print(c(f"  [Claude Code rate limited, resets at {reset}, falling back to local]", "red"))
            model = CHAT_MODEL
            using_claude_code = False

        # Check if the user's message needs a web search (heuristic + LLM)
        # Skip manual search pipeline when using Claude Code — it handles search itself
        search_summary = ""
        self._search_sources = []
        if not using_claude_code and _SEARCH_KEYWORDS.search(user_content):
            print(c("  [search heuristic matched, checking with LLM...]", "cyan"))
            needs_search = await self.ollama_client.classify_search_task(user_content)
            if needs_search:
                search_query = await self.ollama_client.extract_search_query(user_content)
                print(c(f"  [searching: {search_query}]", "cyan"))
                search_results = await web_search(search_query, max_results=5, use_ddg=self.use_ddg)
                if search_results:
                    raw_context = format_search_results(search_results)
                    print(c(f"  [found {len(search_results)} results, summarizing {len(raw_context)} chars...]", "cyan"))
                    search_summary = await self.ollama_client.summarize_search_results(user_content, raw_context)
                    print(c(f"  [summary: {len(search_summary)} chars]", "cyan"))
                    self._search_sources = [
                        {"url": r["url"], "title": r["title"] or r["url"]}
                        for r in search_results[:3]
                    ]
                else:
                    print(c("  [no search results]", "yellow"))
            else:
                print(c("  [LLM says no search needed]", "dim"))

        # Build prompt (web content already injected by build_context)
        prompt = self.format_prompt(messages)

        # Add search summary if available
        if search_summary:
            prompt = f"Search Results Summary:\n{search_summary}\n\n{prompt}"

        prompt = f"System: {self.system_prompt}\n" + prompt

        print(c(f"  [model: {model}]", "dim"))

        start = time.perf_counter()
        if using_claude_code:
            try:
                raw_response, _ = await self.claude_code_client.generate_with_tools(prompt, model)
                if raw_response == "No response from Claude Code.":
                    return ["No response from Claude Code."]
            except RateLimitError as rl_err:
                reset = self.claude_code_client.rate_limit_resets_at or "unknown"
                print(c(f"  [Claude Code rate limited, resets at {reset}, falling back to local]", "red"))
                model = CHAT_MODEL
                raw_response = await self.ollama_client.generate(prompt, model, keep_alive=-1)
                if raw_response == "No response from Ollama.":
                    return ["No response from Ollama."]
            except Exception as cc_err:
                print(c(f"  [Claude Code error: {cc_err}, falling back to local]", "red"))
                model = CHAT_MODEL
                raw_response = await self.ollama_client.generate(prompt, model, keep_alive=-1)
                if raw_response == "No response from Ollama.":
                    return ["No response from Ollama."]
        else:
            raw_response = await self.ollama_client.generate(prompt, model, keep_alive=-1)
            if raw_response == "No response from Ollama.":
                return ["No response from Ollama."]
        elapsed = time.perf_counter() - start

        # Add to context
        self.context.append({
            "role": "assistant",
            "content": raw_response,
            "timestamp": time.time(),
        })

        parts = self.process_response(raw_response)
        # Claude Code uses paragraph breaks; local models use ---MSG--- markers
        splitter = split_response_by_paragraphs if using_claude_code else split_response_by_markers
        final_parts = []
        for part in parts:
            final_parts.extend(splitter(part))

        print(c(f"  [{elapsed:.2f}s]", "dim"))
        return final_parts

    async def handle_search(self, query: str):
        """Handle /search command."""
        print(c(f"  Searching: {query}", "dim"))
        results = await web_search(query, use_ddg=self.use_ddg)
        if not results:
            print(c("  No results (is TAVILY_API_KEY set?)", "red"))
            return

        # Show raw results
        for i, r in enumerate(results, 1):
            print(c(f"  [{i}] {r['title']}", "cyan"))
            print(c(f"      {r['url']}", "dim"))
            snippet = r["content"][:120] + "..." if len(r["content"]) > 120 else r["content"]
            print(f"      {snippet}")
        print()

        # Feed to LLM directly (skip build_context to avoid re-fetching URLs)
        search_context = format_search_results(results)
        llm_prompt = (
            f"The user searched for: {query}\n\n"
            f"{search_context}\n\n"
            f"Based on these search results, provide a helpful answer to the user's query. "
            f"Base your answer ONLY on the data provided above - do not make up or guess any numbers, statistics, or facts not in the results. "
            f"If specific information isn't available, say so. Do not include URLs or source links - sources are handled separately."
        )

        self.context.append({
            "role": "user",
            "name": self.current_user,
            "content": llm_prompt,
            "timestamp": time.time(),
            "images": [],
        })
        response_parts = await self.query()
        self.print_bot_response(response_parts)

        # Sources
        sources = [r["url"] for r in results[:3]]
        print(c("\n  Sources:", "dim"))
        for url in sources:
            print(c(f"  - {url}", "dim"))

    def print_bot_response(self, parts: List[str], sources: List[dict] = None):
        """Print bot response parts with optional source footnote."""
        for i, part in enumerate(parts):
            if i > 0:
                print(c("  ---", "dim"))
            print(c(f"  {part}", "green"))
        if sources:
            seen = set()
            domains = []
            for s in sources:
                try:
                    domain = s['url'].split('/')[2].removeprefix('www.')
                except (IndexError, AttributeError):
                    continue
                if domain not in seen:
                    seen.add(domain)
                    domains.append(domain)
            if domains:
                print(c(f"\n  Sources: {', '.join(domains)}", "dim"))

    def show_context(self):
        """Show current conversation context."""
        if not self.context:
            print(c("  (empty)", "dim"))
            return
        for i, msg in enumerate(self.context):
            role = msg["role"]
            name = msg.get("name", "")
            content = msg["content"][:100]
            if len(msg["content"]) > 100:
                content += "..."
            tag = f"{role}"
            if name:
                tag += f" ({name})"
            color = "cyan" if role == "user" else "green"
            print(c(f"  [{i}] {tag}: {content}", color))

    async def handle_modify(self, instruction: str):
        """Handle /modify command — use Claude Code to edit bot source."""
        from claude_code_client import RateLimitError

        print(c(f"  Modifying bot: {instruction}", "yellow"))

        # Git snapshot
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m",
            f"[auto] pre-modification snapshot {time.strftime('%Y%m%d_%H%M%S')}",
            "--allow-empty",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        try:
            response, exit_code = await self.claude_code_client.run_code_edit(instruction)
        except RateLimitError as e:
            print(c(f"  Rate limited: {e}", "red"))
            return
        except Exception as e:
            print(c(f"  Error: {e}", "red"))
            return

        if exit_code != 0:
            print(c(f"  Code edit failed (exit {exit_code}):", "red"))
            print(f"  {response[:1500]}")
            return

        # Show diff
        proc = await asyncio.create_subprocess_exec(
            "git", "diff",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        diff = stdout.decode("utf-8", errors="replace")

        if diff:
            print(c("  Changes:", "yellow"))
            for line in diff.splitlines()[:60]:
                if line.startswith("+") and not line.startswith("+++"):
                    print(c(f"  {line}", "green"))
                elif line.startswith("-") and not line.startswith("---"):
                    print(c(f"  {line}", "red"))
                else:
                    print(c(f"  {line}", "dim"))
            if len(diff.splitlines()) > 60:
                print(c("  ... (truncated)", "dim"))
        else:
            print(c("  No files changed.", "dim"))

        print(c(f"\n  Summary: {response[:500]}", "green"))

        # Run tests before asking to apply
        if diff:
            print(c("\n  Running tests...", "yellow"))
            try:
                test_result = await self.claude_code_client.run_tests(
                    diff=diff,
                    change_type="core",
                )
                status_color = "green" if test_result.passed else "red"
                status_word = "PASSED" if test_result.passed else "FAILED"
                print(c(f"  Tests {status_word}", status_color))
                print(c(f"  {test_result.tier1_report}", "dim"))
                if test_result.tier2_report:
                    for line in test_result.tier2_report[:300].splitlines():
                        print(c(f"  {line}", "dim"))
                test_passed = test_result.passed
            except Exception as e:
                print(c(f"  Test runner error: {e}", "red"))
                test_passed = True  # Don't block if tests couldn't run
        else:
            test_passed = True

        # Ask to apply or revert
        if test_passed:
            answer = input(c("\n  Apply changes? [y/N] ", "yellow")).strip().lower()
        else:
            answer = input(c("\n  Tests FAILED. Apply anyway? [y/N] ", "red")).strip().lower()
        if answer == "y":
            proc = await asyncio.create_subprocess_exec(
                "git", "add", "-A",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", "[bot-self-modify] applied code changes",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            print(c("  Changes committed. Restart the CLI to pick them up.", "green"))
        else:
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", ".",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            print(c("  Changes reverted.", "yellow"))

    def print_help(self):
        print(c("\n  Commands:", "bold"))
        print("  /user <name>          Switch active user")
        print("  /attach <path>        Attach a file to next message")
        print("  /clear                Clear conversation context")
        print("  /context              Show current context")
        print("  /search <query>       Web search + LLM summary")
        print("  /ddg                  Toggle DuckDuckGo / Tavily search")
        print("  /model [name]         Show/switch active model (e.g. /model claude_code)")
        print("  /modify <instruction> Use Claude Code to modify bot source code")
        print("  /logs [N]             Show last N lines of bot.log (default 50)")
        print("  /prompt               Show current system prompt")
        print("  /set_prompt <text>    Set system prompt")
        print("  /reset_prompt         Reset to default system prompt")
        print("  /time <human time>    Convert to Discord timestamp (e.g. /time sunday 9am)")
        print("  /plugins              List loaded plugins")
        print("  /reload_plugin <name> Hot-reload a plugin")
        print("  /load_plugin <name>   Load a plugin")
        print("  /unload_plugin <name> Unload a plugin")
        print("  /help                 Show this help")
        print("  /quit                 Exit")
        print()
        print("  Any other input is sent as a message to the LLM.")
        print("  URLs in messages are auto-fetched via trafilatura.")
        print()

    async def run(self):
        """Main REPL loop."""
        await js_renderer.start()
        js_status = "on" if js_renderer.available else "off"
        print(c("\n  Discord LLM Bot — Test CLI", "bold"))
        print(c(f"  User: {self.current_user} | Model: {self.active_model} | JS renderer: {js_status}", "dim"))
        print(c("  Type /help for commands\n", "dim"))

        while True:
            try:
                prefix = c(f"{self.current_user}", "cyan")
                attachments_indicator = ""
                if self.pending_attachments:
                    count = len(self.pending_attachments)
                    attachments_indicator = c(f" [{count} file(s)]", "yellow")
                raw = input(f"{prefix}{attachments_indicator}> ")
            except (EOFError, KeyboardInterrupt):
                print(c("\n  bye", "dim"))
                break

            text = raw.strip()
            if not text:
                continue

            # --- Commands ---
            if text.startswith("/"):
                parts = text.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit":
                    print(c("  bye", "dim"))
                    break
                elif cmd == "/help":
                    self.print_help()
                elif cmd == "/user":
                    if arg:
                        self.current_user = arg
                        print(c(f"  Switched to user: {self.current_user}", "yellow"))
                    else:
                        print(c(f"  Current user: {self.current_user}", "dim"))
                elif cmd == "/attach":
                    if not arg:
                        print(c("  Usage: /attach <file_path>", "red"))
                    elif not os.path.isfile(arg):
                        print(c(f"  File not found: {arg}", "red"))
                    else:
                        self.pending_attachments.append(arg)
                        print(c(f"  Attached: {os.path.basename(arg)}", "yellow"))
                elif cmd == "/clear":
                    self.context = []
                    print(c("  Context cleared", "yellow"))
                elif cmd == "/context":
                    self.show_context()
                elif cmd == "/time":
                    if not arg:
                        print(c("  Usage: /time <human time>  (e.g. sunday 9am, tomorrow 3pm, in 2 hours)", "red"))
                    else:
                        from time_command import parse_time_input, format_discord_timestamp
                        from zoneinfo import ZoneInfo
                        tz = ZoneInfo('UTC')
                        dt = parse_time_input(arg, tz)
                        if dt is None:
                            print(c(f"  couldn't parse \"{arg}\"", "red"))
                        else:
                            print(c(f"  {format_discord_timestamp(dt)} (UTC)", "green"))
                elif cmd == "/search":
                    if not arg:
                        print(c("  Usage: /search <query>", "red"))
                    else:
                        await self.handle_search(arg)
                elif cmd == "/prompt":
                    print(c(f"  {self.system_prompt}", "dim"))
                elif cmd == "/set_prompt":
                    if arg:
                        self.system_prompt = arg
                        print(c("  System prompt updated", "yellow"))
                    else:
                        print(c("  Usage: /set_prompt <text>", "red"))
                elif cmd == "/reset_prompt":
                    self.system_prompt = self.original_system_prompt
                    print(c("  System prompt reset to default", "yellow"))
                elif cmd == "/ddg":
                    self.use_ddg = not self.use_ddg
                    engine = "DuckDuckGo" if self.use_ddg else "Tavily"
                    print(c(f"  Search engine: {engine}", "yellow"))
                elif cmd == "/model":
                    if arg:
                        # Try to match by enum name or value
                        matched = None
                        for m in Txt2TxtModel:
                            if arg.lower() in (m.name.lower(), m.value.lower()):
                                matched = m
                                break
                        if matched:
                            self.active_model = matched.value
                            print(c(f"  Model switched to: {matched.name} ({matched.value})", "yellow"))
                        else:
                            # Allow setting raw model string (e.g. an Ollama model)
                            self.active_model = arg
                            print(c(f"  Model switched to: {arg}", "yellow"))
                    else:
                        print(c(f"  Current model: {self.active_model}", "dim"))
                        # Show Claude Code options
                        print(c("  Claude Code (Subscription):", "dim"))
                        for val, label in [("claude-code", "Claude Sonnet"), ("claude-code-opus", "Claude Opus")]:
                            marker = " *" if val == self.active_model else ""
                            print(c(f"    {label:40s} {val}{marker}", "dim"))
                        # Show live Ollama models
                        print(c("  Local Models (Ollama):", "dim"))
                        ollama_models = await self._fetch_ollama_models()
                        if ollama_models:
                            for m in ollama_models:
                                marker = " *" if m["value"] == self.active_model else ""
                                print(c(f"    {m['display_name']:40s} {m['size_str']:>6s}  {m['param_size']}{marker}", "dim"))
                        else:
                            print(c("    (could not reach Ollama)", "red"))
                elif cmd == "/logs":
                    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log")
                    try:
                        with open(log_path, "r", errors="replace") as f:
                            lines = f.readlines()
                        n = int(arg) if arg else 50
                        for line in lines[-n:]:
                            print(c(f"  {line.rstrip()}", "dim"))
                    except FileNotFoundError:
                        print(c("  No bot.log found", "red"))
                elif cmd == "/modify":
                    if not arg:
                        print(c("  Usage: /modify <instruction>", "red"))
                    else:
                        await self.handle_modify(arg)
                elif cmd == "/plugins":
                    info = self.plugin_manager.list_plugins()
                    if not info:
                        print(c("  No plugins loaded. Use /load_plugin <name>", "dim"))
                    for p in info:
                        status = "DISABLED" if p["disabled"] else "active"
                        cmds = ", ".join(f"/{cn}" for cn in p["commands"]) if p["commands"] else "none"
                        print(c(f"  {p['name']} v{p['version']} [{status}] — {p['description'] or 'no description'}", "yellow"))
                        print(c(f"    commands: {cmds} | handlers: {p['message_handlers']} | hooks: {p['hooks']}", "dim"))
                elif cmd == "/reload_plugin":
                    if not arg:
                        print(c("  Usage: /reload_plugin <name>", "red"))
                    else:
                        success, error = await self.plugin_manager.reload_plugin_verbose(arg)
                        if success:
                            print(c(f"  Plugin '{arg}' reloaded successfully", "green"))
                        else:
                            print(c(f"  Failed to reload '{arg}': {error}", "red"))
                elif cmd == "/load_plugin":
                    if not arg:
                        print(c("  Usage: /load_plugin <name>", "red"))
                    else:
                        success, error = await self.plugin_manager.load_plugin_verbose(arg)
                        if success:
                            print(c(f"  Plugin '{arg}' loaded successfully", "green"))
                        else:
                            print(c(f"  Failed to load '{arg}': {error}", "red"))
                elif cmd == "/unload_plugin":
                    if not arg:
                        print(c("  Usage: /unload_plugin <name>", "red"))
                    else:
                        success = await self.plugin_manager.unload_plugin(arg)
                        if success:
                            print(c(f"  Plugin '{arg}' unloaded", "yellow"))
                        else:
                            print(c(f"  Plugin '{arg}' not found or not loaded", "red"))
                else:
                    print(c(f"  Unknown command: {cmd}", "red"))
                continue

            # --- 6-digit nhentai code ---
            if re.fullmatch(r'\d{6}', text):
                link = f"https://nhentai.net/g/{text}/"
                preview_page = f"https://nhentai.net/g/{text}/3"
                print(c(f"  {link}", "green"))
                print(c(f"  (preview image fetched from page 3 and embedded)", "cyan"))
                print()
                continue

            # --- Regular message ---
            sources = await self.build_context(text, self.current_user)
            response_parts = await self.query()
            # Merge URL-fetched sources with search sources
            all_sources = list(sources or [])
            search_sources = getattr(self, '_search_sources', [])
            if search_sources:
                all_sources.extend(search_sources)
                self._search_sources = []
            self.print_bot_response(response_parts, all_sources)
            print()


async def _async_main():
    cli = TestCLI()
    try:
        await cli.run()
    finally:
        await js_renderer.stop()


def main():
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
