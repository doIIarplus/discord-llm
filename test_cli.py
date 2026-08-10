"""
Interactive CLI for testing the bot without Discord.

Mimics the Discord message flow: context building, LLM queries,
web extraction, web search, file attachments, and multi-user conversations.

Usage:
    python test_cli.py

Commands:
    /user <name> [id]     Switch active user (optional discord_id for permission tests)
    /attach <path>        Attach a file to the next message
    /clear                Clear conversation context (and reset the Claude session)
    /session              Show this channel's resumable Claude session
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

import chat_history
import sys
import time
from typing import List

import aiohttp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLAUDE_RESUME_SESSIONS,
    CLAUDE_SESSION_MAX_TOKENS,
    CONTEXT_LIMIT,
    CHAT_MODEL,
    IMAGE_RECOGNITION_MODEL,
    MAX_DISCORD_MESSAGE_LENGTH,
)
from ollama_client import OllamaClient
from claude_code_client import ClaudeCodeClient, RateLimitError, SessionResumeError
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

# Mirror of bot.py's leaked-reasoning backstop (keep in sync).
_REASONING_LEAK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r'\bnot a code[- ]edit request\b',
        r"\bno request here\b",
        r'\bplain greeting\b',
        r'\bjust a (?:question|greeting|statement|comment|observation)\b.*\b(?:no|not)\b',
        r'\balready answered\b.*\bturns?\s*\d',
        r'\[/?Turn\b',
        r'\bthe user (?:is|just|wants|said|asked)\b.*\b(?:no|not|just|so I)\b',
    )
]


def _strip_reasoning_leak(text: str) -> str:
    """Drop ---MSG--- segments that are clearly leaked reasoning. See bot.py."""
    segments = text.split('---MSG---') if '---MSG---' in text else [text]

    def is_leak(seg: str) -> bool:
        s = seg.strip()
        return bool(s) and any(p.search(s) for p in _REASONING_LEAK_PATTERNS)

    kept = [seg for seg in segments if not is_leak(seg)]
    return '---MSG---'.join(kept) if kept else text


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
            "TONE/STYLE:\n"
            "Do NOT repeat the same filler word (e.g. 'bro', 'lol', 'lmao', 'ngl', 'fr', 'lowkey') more than once within a single response, or more than twice across a short window of consecutive responses. Vary your vocabulary — using the same filler repeatedly makes you sound like a broken record.\n\n"
            "NEVER NARRATE YOUR REASONING:\n"
            "Output ONLY the message you'd actually send in chat. Do NOT think out loud, do NOT analyze or classify "
            "the user's message, and do NOT explain your decision-making. Specifically, NEVER write things like "
            "'this is just a question', 'no request here, plain greeting', 'already answered in turn N', or any "
            "meta-commentary about what the user said or what you're about to do. Quoting the user's message back and "
            "labeling it is FORBIDDEN. If you genuinely need to deliberate, do it SILENTLY inside <think>...</think> "
            "tags — everything inside those tags is stripped and never shown. Anything outside the tags is sent "
            "verbatim, so it must read as a natural chat message, nothing else.\n\n"
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
            "- tools/resume/review.py — score a resume PDF against a role (rubric-based, explainable), optional GitHub enrichment\n"
            "- tools/discord/send_message.py — send a message as the bot to any channel\n"
            "- tools/discord/edit_message.py — edit a bot-sent message\n"
            "- tools/discord/delete_message.py — delete a message\n"
            "- tools/discord/get_channel_history.py — fetch recent messages from a channel\n"
            "- tools/discord/search_messages.py — search messages in the server\n"
            "- tools/discord/get_user.py — get user/member info (add --guild-id for nickname, roles, join date)\n"
            "- tools/discord/add_role.py / remove_role.py — manage user roles\n"
            "- tools/discord/list_roles.py — list server roles\n"
            "- tools/discord/list_emojis.py — list server custom emojis\n"
            "- tools/discord/emoji_stats.py — emoji usage counts (--source message|reaction|all, --unused, --backfill)\n"
            "- tools/discord/create_emoji.py — upload a custom emoji to the server\n"
            "- tools/discord/delete_emoji.py — delete a custom emoji (destructive)\n"
            "- tools/discord/set_nickname.py — set or clear a member's nickname\n"
            "- tools/discord/timeout_user.py — timeout a member (e.g. 10m, 1h, 7d)\n"
            "- tools/discord/react.py — add a reaction to a message\n"
            "- tools/discord/pin_message.py — pin/unpin a message\n"
            "- tools/discord/create_thread.py — create a thread (from message or standalone)\n"
            "- tools/discord/list_channels.py — list guild channels\n"
            "- tools/discord/create_channel.py — create a text, voice, or category channel\n"
            "- tools/discord/rename_channel.py — rename an existing channel\n"
            "- tools/discord/delete_channel.py — delete a channel\n"
            "- tools/discord/send_webhook.py — send Discord messages via webhook\n"
            "- tools/images/generate.py — draw an image with the diffusion model (Flux2 Klein)\n"
            "- tools/images/edit.py — edit an existing image with the diffusion model\n"
            "- tools/images/attach.py — attach an image you made yourself with code\n"
            "- tools/github/list_repos.py — list the owner's GitHub repos (OWNER ONLY)\n"
            "- tools/github/clone.py — get a local working copy, reusing one if it exists (OWNER ONLY)\n"
            "- tools/github/status.py — show what changed in a checkout (OWNER ONLY)\n"
            "- tools/github/commit_push.py — commit and push to the default branch (OWNER ONLY)\n"
            "- tools/agent/start_task.py — hand real coding work to a background agent (OWNER ONLY)\n"
            "- tools/agent/status.py — check on that job (OWNER ONLY)\n"
            "- tools/agent/cancel.py — stop it (OWNER ONLY)\n"
            "- tools/agent/push.py — publish a finished job's branch (OWNER ONLY)\n"
            "\n"
            "CODING WORK: there are two modes and picking right matters.\n"
            "ANSWERING a question about code ('how does X work', 'where is Y', 'is this "
            "a bug') — just do it yourself in this turn with Read/Grep/Bash on the repo "
            "and reply. Do not start a job for a question.\n"
            "DOING work — add a feature, fix a bug, refactor, write tests — call "
            "tools/agent/start_task.py with the repo and a FULL description of the task. "
            "The agent gets only what you write in --task, not this conversation, so "
            "include the actual requirements and any detail the user gave. It runs in an "
            "isolated git worktree with subagents available, and posts its own live "
            "progress to this channel.\n"
            "start_task returns immediately and the work is NOT done. Say what you kicked "
            "off in one short line and stop. Do not poll status.py in a loop, do not "
            "narrate progress yourself (the job's own message does that), and never claim "
            "it finished.\n"
            "When it finishes it commits to its own branch and pushes automatically, then "
            "posts buttons for opening a PR, viewing the diff, or deleting the branch. "
            "Nothing lands on the default branch, so this is safe — do NOT also run the "
            "github tools to push, and do not tell the user to push manually; the buttons "
            "already handle it. tools/agent/push.py is only a retry for when the automatic "
            "push failed (no write access, network).\n"
            "If the user wants changes to what it produced, start a NEW task describing "
            "the follow-up.\n"
            "The agent works in a sandbox at ~/git_projects and clones the repo itself, so "
            "there is no setup step — just give start_task.py the owner/repo.\n"
            "\n"
            "GITHUB WORKFLOW: list_repos.py when the user is vague about which repo "
            "('my bot repo') — match the name, don't guess a slug. Then clone.py, which "
            "REUSES an existing local checkout instead of making a second copy. Then edit "
            "files at the path it returns with your normal tools. Then status.py --diff to "
            "see exactly what you changed. Then commit_push.py.\n"
            "READ clone.py's response. If `reused: true` you are in the user's REAL working "
            "directory — if `uncommitted_changes` is non-zero that is THEIR work in "
            "progress, so commit only your own files via --add with explicit paths, never "
            "a blanket commit. If `reused: false` it's a scratch clone and you can be "
            "freer. Never --reset or --force-reset a reused checkout unless the user "
            "explicitly said to throw that work away.\n"
            "commit_push.py publishes DIRECTLY to the default branch — no PR, no review, "
            "live immediately. For anything beyond a trivial edit, run status.py --diff "
            "first and tell the user what you're about to publish. Write real commit "
            "messages: what changed and why, not 'update files'.\n"
            "These tools are restricted IN CODE to discord_id=118567805678256128. For "
            "anyone else they exit with a permission error — relay that plainly and do not "
            "work around it with raw git in Bash on someone else's behalf.\n"
            "\n"
            "IMAGES: you decide when a message wants a picture — there's no keyword trigger. "
            "You also decide HOW to make it, and the two options are good at opposite things.\n"
            "PREFER WRITING CODE (matplotlib, PIL/Pillow, SVG, graphviz) then "
            "tools/images/attach.py for: charts, graphs, plots, any data visualization; "
            "diagrams, flowcharts, timelines; anything with legible text, labels, numbers or "
            "axes; tables/scoreboards/calendars as images; precise geometry or exact colors. "
            "Diffusion models garble text and can't be trusted with real data — if the image "
            "carries information, draw it with code. Save the file under the project dir "
            "(api_out/ is fine) and register it with attach.py.\n"
            "USE tools/images/generate.py for: photographic, painterly, or imaginative "
            "visuals — scenes, characters, creatures, textures, 'draw me a X', album-art "
            "vibes. Anything where realism or aesthetics matter and exact text doesn't.\n"
            "USE tools/images/edit.py when someone wants an existing image changed ('make "
            "her hair green', 'remove the background'). Source images are in "
            "multimodal_input/ (user uploads) and api_out/ (things you generated). Describe "
            "ONLY what changes, not the whole scene.\n"
            "Any image from these three tools is attached to your reply automatically — do "
            "NOT try to send it with tools/discord/send_message.py, and don't paste the file "
            "path into chat. Just write your reply text normally; the picture rides along. "
            "If a request doesn't want an image, don't make one.\n"
            "For reminders: use tools/scheduler/create_task.py --once with a command that calls tools/discord/send_message.py. "
            "Use the channel_id from [Current context] unless the user specifies a different channel. "
            "Example: create_task --name 'reminder' --schedule '0 9 30 3 *' --once "
            "--command 'python tools/discord/send_message.py --channel-id CHAN --content \"<@USER> reminder text\"'\n"
            "Always use these tools when the user's request matches their capabilities instead of making up answers.\n"
            "ACT, DON'T ANNOUNCE: your process exits the moment you finish replying. "
            "Nothing runs in the background, so anything you said you'd 'go do' simply "
            "never happens. You may well remember this conversation next time someone "
            "pings you — but remembering is not doing, and the work still won't exist. "
            "This message is your only chance to act. So NEVER reply with "
            "intent instead of results: no 'lemme go check', 'i'll re-render it', 'gimme "
            "a sec', 'im on it', 'one moment'. Run the tools FIRST, then describe what "
            "you actually did. If someone asks for something you can build, build it in "
            "this turn before you answer. Saying you'll do it and stopping is the single "
            "worst thing you can do here — it reads as lying. If a task genuinely can't "
            "be done in one turn, say that plainly and say what you'd need; do not imply "
            "it's underway. Long tool work is fine — taking a few minutes and delivering "
            "beats replying instantly with a promise.\n"
            "The casual tone is style ONLY — this is a real server with real requests, not "
            "roleplay. When someone asks for a thing, they want the thing, not a character "
            "performance about the thing.\n"
            "PERMISSIONS: Discord and Splitwise tools enforce the requesting user's own permissions in code — the "
            "tool checks the triggering user's Discord roles (or Splitwise ownership) and hard-rejects actions they "
            "aren't allowed to perform, regardless of what you decide. If a tool returns a permission-denied error, "
            "tell the user plainly that they lack the required permission; do not retry or try to work around it.\n\n"
            "HONESTY / NO FABRICATION:\n"
            "Do NOT lie about what you did. Never claim an action was taken, a task/reminder was created or queued, "
            "a message was sent, a fix was applied, or a value was checked/verified unless you ACTUALLY ran the real "
            "tool for it and its output confirms it. No inventing task IDs, expense IDs, or message links. Don't say "
            "something is 'in the list', 'scheduled', 'done', or 'verified' unless you ran the listing/verification "
            "tool and saw it in the output.\n"
            "If you didn't do it or didn't check, just say so — 'i didn't actually run that yet' or 'idk, haven't "
            "checked' is fine. Guessing is ok ONLY if you clearly flag it as a guess; never present a guess as fact.\n"
            "Never describe a code or config change (e.g. a 'local time fix') as done when it wasn't. If you emit "
            "[EDIT_CODE], that's a REQUEST for a change, not proof it happened — don't report it as already applied.\n"
            "If you already said something untrue, correct it directly and plainly instead of doubling down.\n\n"
            "LONG-RUNNING / MONITORING TASKS:\n"
            "When asked to monitor, watch, wait, observe, or tail something, NEVER run a blocking "
            "`tail -f` or unbounded `sleep` — the Bash tool will time out with no output. Use capped "
            "windows: `timeout 60 tail -n 500 -f bot.log` or `sleep 45 && tail -n 300 bot.log`, and "
            "summarize what you saw. Pick a window (30s–120s) based on what the user asked for.\n\n"
            "RELATIONSHIPS:\n"
            "Each user in [Memory] has a relationship rank: hostile, cold, wary, neutral, friendly, close, "
            "or inner circle (worst to best). "
            "Adjust your warmth, effort, and willingness to help based on this rank. "
            "For hostile/cold/wary users, be shorter and less enthusiastic. For friendly/close/inner circle users, "
            "be warmer and go the extra mile. For hostile users, be extremely dismissive and refuse to engage.\n"
            "Each profile also carries a numeric score from -10 to 10. Disclosure rules:\n"
            "- If a user asks about THEIR OWN score, tell them the raw number from their own profile.\n"
            "- If a user asks about ANYONE ELSE's score, never reveal a raw number for that person. "
            "Instead, list only the top 5 friendliest users by name in rank order, highest first, with no numbers "
            "attached. Say nothing about the standing of anyone outside that top 5.\n"
            "Never estimate or invent a score. If a profile has no score, say you don't have one for them."
        )
        self.original_system_prompt = self.system_prompt
        self.current_user = "TestUser"
        # Requester identity used for tool permission enforcement. Defaults to
        # the owner + default guild so tools work out of the box; override with
        # `/user <name> [discord_id]` to test permission-denied paths.
        self.current_user_id = "118567805678256128"
        self.current_guild_id = "363154169294618625"
        # Stands in for a Discord channel so per-channel Claude sessions can be
        # exercised here (see /session and /clear).
        self.current_channel_id = "test-cli-channel"
        # Lazily started in query() — mirrors bot.py so Claude's tools/images/*
        # calls hit the resident Flux pipeline instead of loading their own.
        self.image_service = None
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

        # Split attachments into image files (for img2img) and documents
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        image_files: List[str] = []
        doc_files: List[str] = []
        for path in self.pending_attachments:
            ext = os.path.splitext(path)[1].lower()
            if ext in image_exts:
                image_files.append(path)
            else:
                doc_files.append(path)

        # Process document attachments (PDFs, code, text)
        if doc_files:
            doc_context = ""
            for path in doc_files:
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
            "image_files": image_files,
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
        text = _strip_reasoning_leak(text)
        return split_long_message(text.strip(), MAX_DISCORD_MESSAGE_LENGTH)

    # --- Resumable Claude sessions (mirrors bot.py) ------------------------

    def _system_prompt_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    def _session_pointer(self, model: str):
        """Return (resume_session_id, row) for this 'channel', or (None, None).

        Same invalidation rules as bot.py: a session bakes in its system prompt
        and model, so changing either starts a new one.
        """
        if not CLAUDE_RESUME_SESSIONS:
            return None, None
        row = chat_history.get_claude_session(
            self.current_guild_id, self.current_channel_id)
        if not row:
            return None, None
        if row.get("model") != model:
            print(c("  [session: model changed, starting fresh]", "dim"))
            chat_history.delete_claude_session(
                self.current_guild_id, self.current_channel_id)
            return None, None
        if row.get("system_prompt_hash") != self._system_prompt_hash():
            print(c("  [session: system prompt changed, starting fresh]", "dim"))
            chat_history.delete_claude_session(
                self.current_guild_id, self.current_channel_id)
            return None, None
        if (row.get("context_tokens") or 0) >= CLAUDE_SESSION_MAX_TOKENS:
            print(c(f"  [session: over {CLAUDE_SESSION_MAX_TOKENS} tokens, rotating]", "dim"))
            chat_history.delete_claude_session(
                self.current_guild_id, self.current_channel_id)
            return None, None
        print(c(f"  [session: resuming {row['session_id'][:8]} "
                f"(turn {(row.get('turns') or 0) + 1}, "
                f"{row.get('context_tokens')} tokens)]", "dim"))
        return row["session_id"], row

    def _record_session(self, model: str, meta: dict, row):
        """Persist the session pointer after a turn."""
        if not CLAUDE_RESUME_SESSIONS:
            return
        session_id = meta.get("session_id")
        if not session_id:
            return
        chat_history.upsert_claude_session(
            guild_id=self.current_guild_id,
            channel_id=self.current_channel_id,
            session_id=session_id,
            last_message_id=None,
            model=model,
            system_prompt_hash=self._system_prompt_hash(),
            memory_hash=None,
            context_tokens=meta.get("context_tokens") or 0,
            turns=((row.get("turns") or 0) if row else 0) + 1,
        )

    async def _ensure_image_service(self):
        """Start the local image service once, on first Claude query."""
        if self.image_service is not None:
            return self.image_service
        from image_generation import ImageGenerator
        from image_service import ImageService
        self.image_service = ImageService(ImageGenerator())
        await self.image_service.start()
        print(c("  [image service started for tools/images/*]", "dim"))
        return self.image_service

    async def query(self) -> List[str]:
        """Query Ollama, same logic as bot.py's query_ollama."""
        messages = self.context
        user_content = messages[-1]["content"]

        # Mirror bot.py: on the Claude Code backend, Claude owns the decision to
        # draw and the choice of diffusion vs. code, so the keyword/classifier
        # heuristic below is skipped entirely.
        claude_owns_images = (
            is_claude_code_model(self.active_model)
            and not self.claude_code_client.is_rate_limited
        )
        if claude_owns_images:
            print(c("  [image decisions delegated to Claude]", "dim"))

        # Check if this is an image generation task (fast heuristic first)
        # Also check if this could be a follow-up to a recent image generation
        has_recent_image_gen = any(
            msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", "")
            for msg in messages[-4:]
        )
        if not claude_owns_images and (
            _IMAGE_GEN_KEYWORDS.search(user_content) or has_recent_image_gen
        ):
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
                from image_generation import (
                    choose_dimensions,
                    choose_followup_dimensions,
                    choose_source_dimensions,
                )
                # Determine if this is a modification or a fresh request
                is_modification = has_recent_image_gen and not _IMAGE_GEN_KEYWORDS.search(user_content)
                prev_seed = -1
                prev_prompt = None
                prev_image_path = None
                prev_width = 1024
                prev_height = 1024

                if is_modification:
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

                # Check for user-attached image file on the current message
                attached_image_path = None
                last_msg_files = messages[-1].get("image_files", [])
                if last_msg_files:
                    attached_image_path = last_msg_files[0]

                if attached_image_path and os.path.exists(attached_image_path):
                    # Case 2: User attached an image to edit (img2img)
                    try:
                        from PIL import Image
                        with Image.open(attached_image_path) as src:
                            src_w, src_h = src.size
                    except Exception:
                        src_w, src_h = 1024, 1024
                    width, height = choose_source_dimensions(user_content, src_w, src_h)
                    print(c(f"  [image edit: user-attached {attached_image_path} -> {width}x{height}]", "yellow"))
                    simulated_prompt = (await self.ollama_client.generate_image_prompt(user_content)).strip()
                    mode = "edit"
                    source_path = attached_image_path
                elif is_modification and prev_image_path and os.path.exists(prev_image_path):
                    # Case 1: Follow-up edit of bot-generated image (img2img)
                    width, height = choose_followup_dimensions(user_content, prev_width, prev_height)
                    print(c(f"  [image edit: modifying {prev_image_path} -> {width}x{height}]", "yellow"))
                    simulated_prompt = (await self.ollama_client.modify_image_prompt(prev_prompt, user_content)).strip()
                    print(c(f"  [modified prompt: {simulated_prompt[:120]}...]", "yellow"))
                    mode = "edit"
                    source_path = prev_image_path
                elif is_modification and prev_prompt:
                    # Fallback: prior image file missing, re-generate with modified prompt
                    width, height = choose_followup_dimensions(user_content, prev_width, prev_height)
                    print(c(f"  [image modification: prompt-only, seed {prev_seed}, {width}x{height}]", "yellow"))
                    simulated_prompt = (await self.ollama_client.modify_image_prompt(prev_prompt, user_content)).strip()
                    print(c(f"  [modified prompt: {simulated_prompt[:120]}...]", "yellow"))
                    mode = "txt2img"
                    source_path = None
                else:
                    # Case 3: Brand new generation
                    prev_seed = -1
                    simulated_prompt = (await self.ollama_client.generate_image_prompt(user_content)).strip()
                    width, height = choose_dimensions(f"{user_content} {simulated_prompt}")
                    print(c(f"  [new image generation: {width}x{height}]", "yellow"))
                    print(c(f"  [flux prompt: {simulated_prompt[:120]}...]", "yellow"))
                    mode = "txt2img"
                    source_path = None

                # Store in context for follow-up continuity (same as bot.py).
                # Use a fake path so subsequent follow-ups can exercise the modification branch.
                seed_display = prev_seed if prev_seed != -1 else "random"
                fake_path = f"/tmp/test_cli_sim_{int(time.time()*1000)}.png"
                self.context.append({
                    "role": "assistant",
                    "content": (
                        f"[Generated an image with the following prompt: {simulated_prompt}] "
                        f"(seed: {seed_display}, size: {width}x{height}, path: {fake_path})"
                    ),
                    "timestamp": time.time(),
                })
                return [
                    f"[Image would be generated | mode: {mode} | "
                    f"source: {source_path} | seed: {seed_display} | "
                    f"size: {width}x{height} | prompt: {simulated_prompt}]"
                ]

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

        # Mirror bot.py: inject memory (profiles, relationship ranks, top 5)
        # scoped to the current user so disclosure behaviour is testable here.
        memory_context = chat_history.get_memory_context(
            self.current_guild_id,
            active_user_ids=[m["discord_user_id"] for m in messages if m.get("discord_user_id")],
            requesting_user_id=self.current_user_id,
        )
        if memory_context:
            prompt = f"{prompt}\n\n{memory_context}"

        prompt = f"System: {self.system_prompt}\n" + prompt

        print(c(f"  [model: {model}]", "dim"))

        start = time.perf_counter()
        claude_images = []
        if using_claude_code:
            try:
                import uuid as _uuid
                svc = await self._ensure_image_service()
                image_request_id = _uuid.uuid4().hex
                # Mirror bot.py: resume this "channel's" session when enabled,
                # so continuity is testable offline.
                resume_id, sess_row = self._session_pointer(model)
                meta = {}
                try:
                    try:
                        raw_response, _ = await self.claude_code_client.generate_with_tools(
                            prompt, model,
                            requester_user_id=self.current_user_id,
                            requester_guild_id=self.current_guild_id,
                            image_request_id=image_request_id,
                            resume_session_id=resume_id,
                            persist_session=CLAUDE_RESUME_SESSIONS,
                            meta=meta,
                        )
                    except SessionResumeError as e:
                        print(c(f"  [session resume failed, retrying fresh: {e}]", "red"))
                        chat_history.delete_claude_session(
                            self.current_guild_id, self.current_channel_id)
                        meta = {}
                        sess_row = None
                        raw_response, _ = await self.claude_code_client.generate_with_tools(
                            prompt, model,
                            requester_user_id=self.current_user_id,
                            requester_guild_id=self.current_guild_id,
                            image_request_id=image_request_id,
                            persist_session=CLAUDE_RESUME_SESSIONS,
                            meta=meta,
                        )
                except BaseException:
                    svc.discard(image_request_id)
                    raise
                claude_images = svc.drain(image_request_id)
                self._record_session(model, meta, sess_row)
            except RateLimitError as rl_err:
                reset = self.claude_code_client.rate_limit_resets_at or "unknown"
                print(c(f"  [Claude Code rate limited, resets at {reset}, falling back to local]", "red"))
                model = CHAT_MODEL
                raw_response = await self.ollama_client.generate(prompt, model, keep_alive=1800)
                if raw_response == "No response from Ollama.":
                    return ["No response from Ollama."]
            except Exception as cc_err:
                print(c(f"  [Claude Code error: {cc_err}, falling back to local]", "red"))
                model = CHAT_MODEL
                raw_response = await self.ollama_client.generate(prompt, model, keep_alive=1800)
                if raw_response == "No response from Ollama.":
                    return ["No response from Ollama."]
        else:
            raw_response = await self.ollama_client.generate(prompt, model, keep_alive=1800)
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

        # Surface images Claude produced this turn. In Discord these are
        # attached to the reply; here we print the paths so the flow is testable.
        for img in claude_images:
            kind = img.get("kind")
            detail = (
                f"seed: {img['seed']}, {img['width']}x{img['height']}"
                if img.get("seed") is not None
                else f"{img['width']}x{img['height']}, drawn with code"
            )
            nsfw = " | NSFW (would be spoilered)" if img.get("nsfw") else ""
            final_parts.append(
                f"[Attached image | {kind} | {detail}{nsfw} | {img['path']}]"
            )
            # Record the marker so follow-up edits can find the path, matching
            # what bot.py persists to chat history.
            self.context.append({
                "role": "assistant",
                "content": (
                    f"[Generated an image with the following prompt: "
                    f"{img.get('prompt', '')}] "
                    f"(seed: {img.get('seed')}, "
                    f"size: {img.get('width')}x{img.get('height')}, "
                    f"path: {img['path']})"
                ),
                "timestamp": time.time(),
            })

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
        print("  /user <name> [id]     Switch active user (optional discord_id for permission tests)")
        print("  /attach <path>        Attach a file to next message")
        print("  /clear                Clear conversation context (and reset the Claude session)")
        print("  /session              Show this channel's resumable Claude session")
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
                        parts = arg.split()
                        self.current_user = parts[0]
                        if len(parts) > 1:
                            self.current_user_id = parts[1]
                        print(c(f"  Switched to user: {self.current_user} (discord_id={self.current_user_id})", "yellow"))
                    else:
                        print(c(f"  Current user: {self.current_user} (discord_id={self.current_user_id})", "dim"))
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
                    # Match bot.py's /clear: drop the Claude session too.
                    if chat_history.get_claude_session(
                            self.current_guild_id, self.current_channel_id):
                        chat_history.delete_claude_session(
                            self.current_guild_id, self.current_channel_id)
                        print(c("  Context cleared (and Claude session reset)", "yellow"))
                    else:
                        print(c("  Context cleared", "yellow"))
                elif cmd == "/session":
                    row = chat_history.get_claude_session(
                        self.current_guild_id, self.current_channel_id)
                    if not row:
                        print(c(f"  No active session (resume enabled: {CLAUDE_RESUME_SESSIONS})", "yellow"))
                    else:
                        print(c(f"  session_id:     {row['session_id']}", "cyan"))
                        print(c(f"  turns:          {row.get('turns')}", "cyan"))
                        print(c(f"  context_tokens: {row.get('context_tokens')} "
                                f"(rotates at {CLAUDE_SESSION_MAX_TOKENS})", "cyan"))
                        print(c(f"  model:          {row.get('model')}", "cyan"))
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
