"""Discord LLM Bot - Main module"""

import asyncio
import hashlib
import json
import logging
import re
import os
import random
import sys
import time
import traceback
import uuid
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
    DM_ALLOWLIST,
    DM_GUILD_SENTINEL,
    FILE_INPUT_FOLDER,
    GUILD_ID,
    GUILD_ALLOWLIST,
    TOOL_INTEGRATIONS,
    tools_allowed_for,
    IMAGE_RECOGNITION_MODEL,
    CHAT_MODEL,
    CLAUDE_RESUME_SESSIONS,
    CLAUDE_SESSION_MAX_TOKENS,
    CLAUDE_SESSION_MAX_TURNS,
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
from agent_jobs import JobManager
from image_service import ImageService
from ollama_client import OllamaClient
from claude_client import ClaudeClient
from claude_code_client import ClaudeCodeClient, RateLimitError, SessionResumeError
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

# Keywords that suggest an image generation request.
# Two alternatives:
#   1. verb (generate/create/draw/...) within 120 chars of a visual noun
#      (image/picture/photo/...). The {0,120} gap lets verbose phrasings
#      through ("draw me a guy who communicates exclusively in reaction images"
#      has 51 chars between verb + noun).
#   2. standalone imperative form: "draw|paint|sketch|... + me/us/a/an/the/some"
#      catches "draw me a cat" where no explicit visual noun is named.
# False positives from #2 are filtered downstream by the LLM classifier
# (is_image_generation_task), so it's safe to be permissive here.
_IMAGE_GEN_KEYWORDS = re.compile(
    r'(?:'
    r'\b(?:generate|create|draw|make|paint|render|sketch|illustrate)\b'
    r'.{0,120}'
    r'\b(?:images?|pictures?|photos?|illustrations?|arts?|drawings?|paintings?|portraits?|gifs?|memes?|stickers?|emojis?)\b'
    r'|'
    r'\b(?:draw|paint|sketch|render|illustrate)\s+(?:me|us|a|an|the|some|that)\b'
    r')',
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

# High-confidence signatures of leaked chain-of-thought / decision narration that
# local models sometimes emit (without <think> tags) before their actual reply.
# Kept conservative to avoid stripping legitimate chat. See _strip_reasoning_leak.
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
    """Drop ---MSG--- segments that are clearly leaked reasoning, not real replies.

    Local models occasionally narrate their decision-making (e.g. classifying the
    user's message: 'Just "hi" — no request here, plain greeting.') and emit it as
    a leading message before the actual response. We split on the same marker the
    sender uses, drop any segment matching a high-confidence leak signature, and
    rejoin — but never drop everything (if all segments match, keep them as-is so
    we don't send nothing).
    """
    if '---MSG---' not in text:
        segments = [text]
    else:
        segments = text.split('---MSG---')

    def is_leak(seg: str) -> bool:
        s = seg.strip()
        if not s:
            return False
        return any(p.search(s) for p in _REASONING_LEAK_PATTERNS)

    kept = [seg for seg in segments if not is_leak(seg)]
    if not kept:
        return text  # everything looked like a leak — bail out, send original
    return '---MSG---'.join(kept)


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
        # Background coding jobs. Chat turns stay fast; real work runs detached
        # in a git worktree and streams progress into the channel.
        self.job_manager = JobManager(self, self.claude_code_client)
        # Loopback HTTP shim so Claude's tools/ CLIs can reach back into the bot
        # process — the resident Flux model, and now the job manager too.
        self.image_service = ImageService(self.image_gen, job_manager=self.job_manager)
        self._last_search_sources: List[dict] = []
        # One lock per (guild, channel) so two messages in the same channel
        # can't resume the same Claude session concurrently. Different channels
        # have different sessions and run in parallel freely.
        self._session_locks: Dict[tuple, asyncio.Lock] = {}

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
            "TONE/STYLE:\n"
            "Do NOT repeat the same filler word (e.g. 'bro', 'lol', 'lmao', 'ngl', 'fr', 'lowkey') more than once within a single response, or more than twice across a short window of consecutive responses. Vary your vocabulary — using the same filler repeatedly makes you sound like a broken record.\n\n"
            "CONVERSATION FORMAT:\n"
            "The conversation history uses numbered [Turn N] tags. Each turn is a REAL message from a REAL user or your previous response. "
            "ONLY respond to the LAST turn. Do NOT invent, fabricate, or continue with additional user messages. "
            "Do NOT generate text inside [Turn] tags — only produce your own single response.\n\n"
            "NEVER NARRATE YOUR REASONING:\n"
            "Output ONLY the message you'd actually send in chat. Do NOT think out loud, do NOT analyze or classify "
            "the user's message, and do NOT explain your decision-making. Specifically, NEVER write things like "
            "'this is just a question', 'not a code edit request', 'no request here, plain greeting', "
            "'already answered in turn N', or any meta-commentary about what the user said or what you're about to do. "
            "Quoting the user's message back and labeling it is FORBIDDEN. "
            "If you genuinely need to deliberate (e.g. whether to use [EDIT_CODE]), do it SILENTLY inside "
            "<think>...</think> tags — everything inside those tags is stripped and never shown. Anything outside the "
            "tags is sent verbatim to Discord, so it must read as a natural chat message, nothing else.\n\n"
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
            "You CAN modify your own source code, but ONLY when the user EXPLICITLY asks you to. "
            "The default is to NOT edit code. Conversation, questions, complaints, observations, and casual chat are "
            "NOT requests for code changes — just respond normally.\n"
            "\n"
            "TRIGGER [EDIT_CODE] ONLY for unambiguous, direct change requests, e.g.:\n"
            "- 'add a /hello command'\n"
            "- 'add a plugin that does X'\n"
            "- 'change your code so that X happens'\n"
            "- 'fix [specific bug] in [specific place]'\n"
            "- 'modify [file/feature] to do X'\n"
            "- 'update [config value] to X'\n"
            "- 'edit your system prompt to X'\n"
            "- 'make yourself stop doing X' (when X is clearly a code behavior, not a one-off)\n"
            "\n"
            "DO NOT trigger [EDIT_CODE] for:\n"
            "- questions: 'how does X work', 'why did you do X', 'what does this code do'\n"
            "- observations or complaints without a fix request: 'that was weird', 'you got that wrong', 'you're annoying'\n"
            "- requests for explanation: 'explain X', 'show me X', 'tell me about X'\n"
            "- requests to USE a CLI tool (those are not code changes — just run the tool)\n"
            "- one-off Discord actions: sending a message, reacting, scheduling a reminder, splitwise expense, etc.\n"
            "- general conversation, banter, or roleplay\n"
            "- ambiguous asks where 'change code' isn't clearly the ask\n"
            "\n"
            "If in doubt → DO NOT use the tag. Reply normally. The user will ask again more directly if they actually wanted a code change.\n"
            "\n"
            "When you DO emit [EDIT_CODE], NEVER also try to modify files yourself with Edit/Write/Bash — that path triggers "
            "an approval prompt and conflicts with the [EDIT_CODE] flow. Reading files for context is fine; mutating them is not.\n"
            "\n"
            "FORMAT:\n"
            "[EDIT_CODE]detailed, specific instruction of what to change[/EDIT_CODE]\n"
            "Include your normal casual response text OUTSIDE the tag. The tag content is a self-contained instruction "
            "for the underlying code agent — precise about file paths, function names, and the exact change, not conversational.\n"
            "\n"
            "EXAMPLES:\n"
            "- User: 'add a /hello command' → 'sure [EDIT_CODE]Create a new plugin that adds a /hello slash command that says hello[/EDIT_CODE]'\n"
            "- User: 'add a dice roller' → 'on it [EDIT_CODE]Create a new plugin with a /roll command that rolls dice[/EDIT_CODE]'\n"
            "- User: 'fix the logger error' → 'on it [EDIT_CODE]Add import logging and logger = logging.getLogger(__name__) to bot.py[/EDIT_CODE]'\n"
            "- User: 'switch the magic-link channel to 12345' → 'on it [EDIT_CODE]Change TARGET_CHANNEL_ID in plugins/gmail_magic_link.py to 12345[/EDIT_CODE]'\n"
            "Counter-examples (do NOT use the tag):\n"
            "- User: 'how does the voice transcriber work?' → just explain it in chat\n"
            "- User: 'that response was bad' → acknowledge, don't try to edit your prompt\n"
            "- User: 'send a reminder to X tomorrow' → use the scheduler CLI tool, not [EDIT_CODE]\n"
            "- User: 'split $50 with Jason' → use splitwise CLI tool, not [EDIT_CODE]\n\n"
            "CLI TOOLS:\n"
            "You have access to CLI tools via Bash in the tools/ directory. Use them when users ask about "
            "Splitwise (bills, balances, expenses), scheduled tasks, web searches, or other tool-related actions. "
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
            "- tools/doordash/ — order food on DoorDash (OWNER ONLY): search, menu, find_items, "
            "item_details, cart, order, address, payment_methods\n"
            "- tools/github/list_repos.py — list the owner's GitHub repos (OWNER ONLY)\n"
            "- tools/github/clone.py — get a local working copy, reusing one if it exists (OWNER ONLY)\n"
            "- tools/github/status.py — show what changed in a checkout (OWNER ONLY)\n"
            "- tools/github/commit_push.py — commit and push to the default branch (OWNER ONLY)\n"
            "- tools/agent/start_task.py — hand real coding work to a background agent (OWNER ONLY)\n"
            "- tools/agent/status.py — check on that job (OWNER ONLY)\n"
            "- tools/agent/cancel.py — stop it (OWNER ONLY)\n"
            "- tools/agent/push.py — publish a finished job's branch (OWNER ONLY)\n"
            "\n"
            "DOORDASH (OWNER ONLY — enforced in code, not just here):\n"
            "The flow is search.py -> menu.py (restaurants) or find_items.py (grocery/retail) "
            "-> cart.py add -> order.py preview -> order.py place --confirm.\n"
            "Every doordash command REQUIRES --intent: one plain-language line about who this "
            "is for and the goal behind it, e.g. --intent \"Summary: Help the user order lunch\". "
            "Not a restatement of the command.\n"
            "NEVER run order.py place until you have shown the user the actual items and the "
            "actual total from order.py preview and they have explicitly said yes to THAT. "
            "It spends real money and cannot be undone. 'order me a salad' is permission to "
            "build the cart and quote it, not to buy it. --confirm asserts they approved; "
            "passing it on your own is a real-money mistake.\n"
            "Everything else (search, menu, find_items, item_details, cart show/add/clear, "
            "order preview, order history, order receipt, address, payment_methods) is safe "
            "— just run it.\n"
            "order.py history only lists top-level items — it does NOT show modifiers or "
            "customizations. When someone asks what was actually on a past order (did that "
            "salad have chicken, what dressing, what size), get the order_uuid from history "
            "and run order.py receipt --order-uuid — it's read-only, so just run it.\n"
            "If a tool returns \"not_authenticated\" or \"no_access\", relay that message "
            "plainly and stop. Do not retry, and do not try to work around it.\n"
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
            "When a user asks you to monitor, watch, wait, observe, or tail something over time, "
            "NEVER run an open-ended blocking command like `tail -f file` or an unbounded `sleep` in Bash. "
            "The Bash tool has an internal timeout and will kill blocking commands, returning no output. "
            "Instead, use capped windows and polling:\n"
            "- `timeout 60 tail -n 500 -f bot.log` — bounded follow\n"
            "- `sleep 45 && tail -n 300 bot.log` — wait then snapshot\n"
            "- Multiple short tail snapshots with `sleep` between them if you need several samples\n"
            "Pick a window (30s–120s) based on how long the user said to wait, and report back what you observed. "
            "If the user asks you to 'check back in a few minutes', do one capped wait+read and summarize.\n\n"
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

    # --- Resumable Claude sessions ----------------------------------------
    # Each channel gets its own Claude Code CLI session, kept on disk so it
    # survives restarts. The session remembers what the model actually DID
    # (files written, URLs fetched) rather than just what was said, which the
    # re-rendered transcript can never convey. chat_history.db stays the source
    # of truth: any failure here degrades to the full-transcript path.

    def _session_lock(self, server, channel) -> asyncio.Lock:
        """Get (or create) the per-channel session lock."""
        key = (server, channel)
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock

    @staticmethod
    def _hash_text(text: str) -> str:
        """Short stable hash, used to detect prompt/memory changes."""
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def resume_enabled_for(self, using_claude_code: bool, is_pty: bool) -> bool:
        """Whether this turn should use a resumable session.

        The Claude Code backend only. The local Ollama backend has no session
        concept, and the PTY client already has its own persistent session.
        """
        return CLAUDE_RESUME_SESSIONS and using_claude_code and not is_pty

    def _session_is_stale(self, row: dict, model: str, system_prompt_hash: str):
        """Return a reason string if the stored session can't be reused, else None.

        A session bakes in its system prompt and model at creation, so changing
        either (via /set_system_prompt, /reset_system_prompt, /set_model) has to
        start a new one. Size limits are reported separately because they get a
        handoff summary rather than a plain reset.
        """
        if row.get("model") != model:
            return f"model changed ({row.get('model')} -> {model})"
        if row.get("system_prompt_hash") != system_prompt_hash:
            return "system prompt changed"
        return None

    def _session_is_full(self, row: dict):
        """Return a reason string if the session should be rotated, else None."""
        tokens = row.get("context_tokens") or 0
        if tokens >= CLAUDE_SESSION_MAX_TOKENS:
            return f"context {tokens} >= {CLAUDE_SESSION_MAX_TOKENS} tokens"
        turns = row.get("turns") or 0
        if CLAUDE_SESSION_MAX_TURNS and turns >= CLAUDE_SESSION_MAX_TURNS:
            return f"{turns} turns >= {CLAUDE_SESSION_MAX_TURNS}"
        return None

    async def _prepare_session(self, server, channel, model: str,
                               system_prompt_hash: str) -> dict:
        """Decide whether to resume, rotate, or start fresh for this channel.

        Returns {"row", "resume_id", "handoff", "reason"}. A None row means
        "start a fresh session"; `handoff` carries state rescued from a rotated
        session so the replacement isn't amnesiac about its own artifacts.
        Must be called under the channel's session lock.
        """
        row = chat_history.get_claude_session(str(server), str(channel))
        if not row:
            return {"row": None, "resume_id": None, "handoff": None,
                    "reason": "no stored session"}

        stale = self._session_is_stale(row, model, system_prompt_hash)
        if stale:
            logger.info("[session] %s/%s starting fresh: %s", server, channel, stale)
            chat_history.delete_claude_session(str(server), str(channel))
            return {"row": None, "resume_id": None, "handoff": None, "reason": stale}

        full = self._session_is_full(row)
        if full:
            logger.info("[session] %s/%s rotating: %s", server, channel, full)
            handoff = await self.claude_code_client.summarize_session_for_handoff(
                row["session_id"], model,
            )
            if handoff:
                logger.info("[session] handoff captured (%d chars)", len(handoff))
            else:
                logger.warning("[session] handoff unavailable, rotating without it")
            chat_history.delete_claude_session(str(server), str(channel))
            return {"row": None, "resume_id": None, "handoff": handoff,
                    "reason": f"rotated ({full})"}

        return {"row": row, "resume_id": row["session_id"], "handoff": None,
                "reason": None}

    def _build_session_delta_prompt(
        self,
        server,
        channel,
        messages: List[dict],
        user_content: str,
        search_summary,
        active_user_ids,
        row: dict,
        trigger_message_id=None,
    ) -> tuple:
        """Build the incremental prompt for a resumed session.

        The session already holds the conversation, so we send only what it
        hasn't seen: every message recorded in this channel since the watermark,
        plus the per-turn volatile context. Returns (prompt, memory_hash).

        Bot-authored rows are excluded deliberately — the session generated
        those replies itself, and record_bot_response writes them to the DB
        *after* the turn, so without this filter the model would re-read its own
        output as if a user had said it.
        """
        bot_user_id = str(self.user.id) if self.user else None
        delta = chat_history.get_channel_messages_since(
            str(server), str(channel), row.get("last_message_id"), limit=200,
        )

        lines = []
        for r in delta:
            if bot_user_id and str(r.get("author_id")) == bot_user_id:
                continue
            # The triggering message is added separately below using
            # user_content, which is richer (parsed docs, fetched URLs, image
            # notes) than the raw stored text.
            if trigger_message_id and str(r.get("message_id")) == str(trigger_message_id):
                continue
            body = r.get("content") or ""
            if r.get("image_summary"):
                body += f"\n[Attached image: {r['image_summary']}]"
            if not body.strip():
                continue
            lines.append(f"[{r.get('author_name', 'Unknown')} "
                         f"(discord_id={r.get('author_id', '')})] {body}")

        last_msg = messages[-1]
        name = last_msg.get("name", "Unknown")
        uid = last_msg.get("discord_user_id", "")

        # Per-user personality override goes in as a turn-level note, not the
        # system prompt — a session's system prompt is fixed at creation, and
        # the next message in this channel may come from a different user.
        personality_note = ""
        if uid:
            override = self.plugin_manager.get_system_prompt_override(int(uid))
            if override:
                personality_note = f"[Personality override for this user: {override}]\n"

        if lines:
            missed = "\n".join(lines)
            prompt = (
                f"{personality_note}[Messages in this channel since your last reply:]\n"
                f"{missed}\n\n[{name} (discord_id={uid})] {user_content}"
            )
        else:
            prompt = f"{personality_note}[{name} (discord_id={uid})] {user_content}"

        if search_summary:
            prompt = f"Search Results Summary:\n{search_summary}\n\n{prompt}"

        # Memory (user profiles / channel summaries / server events) changes
        # only when the summarizer runs, so re-send it just when it differs from
        # what this session already has.
        memory_context = chat_history.get_memory_context(
            str(server), channel_id=str(channel), active_user_ids=active_user_ids,
            requesting_user_id=(last_msg.get("discord_user_id") or None),
        )
        memory_hash = self._hash_text(memory_context) if memory_context else None
        if memory_context and memory_hash != row.get("memory_hash"):
            prompt = f"{prompt}\n\n[Updated server memory:]\n{memory_context}"

        guild = self.get_guild(server)
        mention_context = extract_mention_context(user_content, guild)
        if mention_context:
            prompt = f"{prompt}\n\n{mention_context}"

        if self.rag_enabled:
            wiki_context = self.rag_system.get_context_for_query(user_content)
            if wiki_context:
                prompt = f"Wiki Context:\n{wiki_context}\n\n{prompt}"

        return prompt, memory_hash

    async def _claude_turn_with_session(
        self,
        server,
        channel,
        model: str,
        messages: List[dict],
        user_content: str,
        search_summary,
        active_user_ids,
        requester_uid,
        images,
        full_prompt: str,
        image_request_id: str,
        trigger_message_id=None,
    ) -> str:
        """Run one Claude-with-tools turn against this channel's session.

        Resumes the channel's session when there is a usable one (sending only
        new messages), otherwise starts a fresh persisted session seeded with
        the full transcript in `full_prompt`. Returns the raw response text.

        Serialized per channel: two concurrent turns must not --resume the same
        session id, and the watermark write must not race.
        """
        system_prompt_hash = self._hash_text(self.system_prompt)

        async with self._session_lock(server, channel):
            sess = await self._prepare_session(
                server, channel, model, system_prompt_hash,
            )
            row = sess["row"]

            if row:
                prompt, memory_hash = self._build_session_delta_prompt(
                    server, channel, messages, user_content, search_summary,
                    active_user_ids, row, trigger_message_id=trigger_message_id,
                )
                turns = (row.get("turns") or 0)
            else:
                # Fresh session: full transcript, and fold in any handoff
                # rescued from the session this one replaces.
                prompt = full_prompt
                if sess["handoff"]:
                    prompt = (
                        "[Continuing an earlier conversation in this channel. "
                        "Handoff notes from it — these paths and findings are "
                        "real, reuse them instead of redoing the work:]\n"
                        f"{sess['handoff']}\n\n{prompt}"
                    )
                memory_hash = None
                turns = 0

            meta = {}
            try:
                raw_response, _ = await self.claude_code_client.generate_with_tools(
                    prompt, model, images,
                    requester_user_id=requester_uid,
                    requester_guild_id=server,
                    requester_channel_id=channel,
                    image_request_id=image_request_id,
                    resume_session_id=sess["resume_id"],
                    persist_session=True,
                    meta=meta,
                )
            except SessionResumeError as e:
                # The stored id is dead (session file pruned, cwd changed).
                # Drop it and retry once from scratch rather than erroring out.
                logger.warning("[session] %s/%s resume failed, retrying fresh: %s",
                               server, channel, e)
                chat_history.delete_claude_session(str(server), str(channel))
                meta = {}
                raw_response, _ = await self.claude_code_client.generate_with_tools(
                    full_prompt, model, images,
                    requester_user_id=requester_uid,
                    requester_guild_id=server,
                    requester_channel_id=channel,
                    image_request_id=image_request_id,
                    resume_session_id=None,
                    persist_session=True,
                    meta=meta,
                )
                memory_hash = None
                turns = 0

            session_id = meta.get("session_id")
            if session_id:
                chat_history.upsert_claude_session(
                    guild_id=str(server),
                    channel_id=str(channel),
                    session_id=session_id,
                    last_message_id=str(trigger_message_id) if trigger_message_id else None,
                    model=model,
                    system_prompt_hash=system_prompt_hash,
                    memory_hash=memory_hash,
                    context_tokens=meta.get("context_tokens") or 0,
                    turns=turns + 1,
                    handoff=sess["handoff"],
                )
                logger.info(
                    "[session] %s/%s %s id=%s turns=%d tokens=%s",
                    server, channel, "resumed" if row else "created",
                    session_id, turns + 1, meta.get("context_tokens"),
                )
            else:
                # No id came back — don't leave a stale pointer behind.
                logger.warning("[session] %s/%s no session_id returned; clearing pointer",
                               server, channel)
                chat_history.delete_claude_session(str(server), str(channel))

            return raw_response

    async def setup_hook(self):
        """Setup hook for Discord bot"""
        self.command_handlers.setup_commands()
        # Load all plugins before syncing commands
        await self.plugin_manager.load_all()
        # Sync to every allowlisted guild for instant command visibility.
        for gid in GUILD_ALLOWLIST:
            try:
                guild = discord.Object(id=gid)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info(f"Synced commands to guild {gid}")
            except Exception as e:
                logger.warning(f"Failed to sync commands to guild {gid}: {e}")
        await js_renderer.start()
        # Localhost shim so Claude-invoked tools/images/* drive the resident
        # Flux pipeline instead of loading their own copy of a 9B model.
        try:
            await self.image_service.start()
        except Exception as e:
            logger.error(f"Failed to start image service: {e}")

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

        # Guild lock audit: leave any guild the bot is in that isn't allowlisted.
        # Covers the case where the bot was added to another guild before this
        # safeguard existed, or before on_guild_join had a chance to fire.
        for g in list(self.guilds):
            if g.id not in GUILD_ALLOWLIST:
                logger.warning(
                    f"Found in disallowed guild {g.id} ({g.name!r}) — leaving"
                )
                try:
                    await g.leave()
                except Exception as e:
                    logger.error(f"Failed to leave disallowed guild {g.id}: {e}")

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

    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edits — update chat_history.db with the new content."""
        if after.author.bot:
            return
        # Guild lock: only allowlisted guilds are honored.
        if after.guild is not None and after.guild.id not in GUILD_ALLOWLIST:
            return
        # Allow edits in DMs from allowlisted users; otherwise require a guild
        if after.guild is None and after.author.id not in DM_ALLOWLIST:
            return
        if before.content == after.content:
            return  # Embed-only update (link preview etc.), not a real edit
        await chat_history.update_message_content(after.id, after.content or "")

    async def _record_reaction_event(self, payload, delta: int):
        """Shared body of the raw reaction add/remove handlers.

        Raw events are used so reactions on messages that aren't in the cache
        (anything older than this process) still count. Unicode emoji have no
        id and aren't tracked — the counters are about the server's own custom
        emoji set.
        """
        if payload.guild_id is None:
            return  # DM reaction — no guild-scoped emoji set
        if payload.guild_id not in GUILD_ALLOWLIST:
            return
        if payload.emoji is None or payload.emoji.id is None:
            return  # Unicode emoji
        try:
            await chat_history.record_reaction(
                str(payload.guild_id),
                str(payload.emoji.id),
                payload.emoji.name,
                int(bool(payload.emoji.animated)),
                delta=delta,
            )
        except Exception as e:
            logger.error(f"Failed to record reaction {payload.emoji.id}: {e}")

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Count a custom-emoji reaction being added."""
        await self._record_reaction_event(payload, delta=1)

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """Decrement when that reaction is taken back."""
        await self._record_reaction_event(payload, delta=-1)

    async def on_guild_join(self, guild: discord.Guild):
        """Auto-leave any guild that isn't allowlisted.

        Defense in depth: if someone adds the bot to another server (e.g. via
        an OAuth invite they shouldn't have), we exit immediately without ever
        responding to anything in that guild.
        """
        if guild.id not in GUILD_ALLOWLIST:
            logger.warning(
                f"Joined disallowed guild {guild.id} ({guild.name!r}) — leaving"
            )
            try:
                await guild.leave()
            except Exception as e:
                logger.error(f"Failed to leave disallowed guild {guild.id}: {e}")

    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        # Ignore bot messages (including self)
        if message.author.bot:
            return

        # Guild lock: only allowlisted guilds are honored. DMs (no guild) skip
        # this check — they're gated separately by DM_ALLOWLIST below.
        if message.guild is not None and message.guild.id not in GUILD_ALLOWLIST:
            return

        # DM handling: only allowlisted users may DM the bot.
        # In DMs, every message from an allowlisted user triggers a response
        # (no @mention required — DMs are inherently direct).
        is_dm = isinstance(message.channel, discord.DMChannel)
        if is_dm and message.author.id not in DM_ALLOWLIST:
            return

        # Record to persistent chat history (before any early returns).
        # For DMs, chat_history uses DM_GUILD_SENTINEL as guild_id.
        await chat_history.record_message(message)

        # For DMs, use the sentinel guild id for in-memory context keying
        # so /pick_model, build_context, and query_ollama all work uniformly.
        server = message.guild.id if message.guild else int(DM_GUILD_SENTINEL)
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

        # Determine response mode. DMs are inherently direct — always respond.
        if not is_dm:
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

            # Only respond to direct mentions and replies in guild channels
            if not (is_direct_mention or is_reply_to_bot):
                return

        # Track last active channel for crash reporting
        self._state["last_active_channel_id"] = channel
        self._save_state()

        try:
            user_text = re.sub(r'<@!?\d+>', '', message.content).strip()

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
        """Build conversation context from persistent chat history.

        Fetches the last CONTEXT_LIMIT messages from chat_history.db for this
        channel, so the bot sees all recent conversation — not just direct
        interactions. The current message's attachments (images/docs) and web
        extractions are still processed and attached to the final entry.
        """
        if image_files is None:
            image_files = []
        if document_files is None:
            document_files = []

        channel = message.channel.id
        bot_user_id = str(self.user.id)

        # Fetch recent messages from persistent DB (includes all users + bot)
        # We fetch CONTEXT_LIMIT - 1 because the current message will be appended
        db_messages = await asyncio.to_thread(
            chat_history.get_recent_channel_messages,
            guild_id=str(server),
            channel_id=str(channel),
            limit=CONTEXT_LIMIT - 1,
        )

        # Convert DB rows into the context format used by format_prompt/query_ollama
        ctx = []
        for row in db_messages:
            # Skip the current message if it was already recorded to DB
            if row["message_id"] == str(message.id):
                continue

            is_bot = row["author_id"] == bot_user_id
            entry = {
                "role": "assistant" if is_bot else "user",
                "content": row["content"] or "",
                "timestamp": row["created_at"],
            }

            if not is_bot:
                entry["name"] = row["author_name"]
                entry["discord_user_id"] = int(row["author_id"])

            # Include image summaries from the DB as context
            if row.get("image_summary"):
                entry["content"] += f"\n[Attached image: {row['image_summary']}]"

            # Include edit history so the bot knows about message edits
            if row.get("edit_history"):
                try:
                    edits = json.loads(row["edit_history"])
                    if edits:
                        original = edits[0]["content"]
                        entry["content"] += f"\n[This message was edited. Original: \"{original}\"]"
                except (json.JSONDecodeError, KeyError):
                    pass

            ctx.append(entry)

        # Build the current message's prompt with attachments and web context
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

        # Append the current message
        ctx.append({
            "role": "user",
            "name": message.author.display_name,
            "discord_user_id": message.author.id,
            "content": prompt,
            "timestamp": time.time(),
            "images": images,
            "image_files": list(image_files),  # Keep paths for img2img editing
            # Discord id of the triggering message — the watermark a resumed
            # Claude session advances to, so the next turn's delta starts here.
            "message_id": str(message.id),
        })

        # Store as the active context for query_ollama / pick_model
        if server not in self.context:
            self.context[server] = {}
        self.context[server][channel] = ctx

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
            sent_msg = await message.channel.send(embed=response_data[0], file=response_data[1])
            # Record the image gen marker to chat history so it persists across context rebuilds
            img_ctx = self.context.get(server, {}).get(channel, [])
            img_marker = next(
                (m["content"] for m in reversed(img_ctx)
                 if m.get("role") == "assistant" and "[Generated an image" in m.get("content", "")),
                None,
            )
            if img_marker:
                await chat_history.record_bot_response(
                    guild_id=server,
                    channel_id=channel,
                    bot_user_id=self.user.id,
                    bot_name=self.user.display_name,
                    content=img_marker,
                    message_id=sent_msg.id,
                    reply_to_message_id=message.id,
                )
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

            # Handle any image items (Claude-produced: diffusion or code-drawn)
            for response_item in response_data:
                if isinstance(response_item, dict) and "image" in response_item:
                    img_path = response_item["image"]
                    if not os.path.exists(img_path):
                        logger.warning("[img] queued image vanished before send: %s", img_path)
                        continue
                    info = response_item.get("info") or {}
                    file = discord.File(img_path, filename=os.path.basename(img_path))
                    if response_item.get("nsfw"):
                        file.spoiler = True

                    # Only diffusion output has generation params worth showing.
                    embed = None
                    if info.get("seed") is not None:
                        embed = discord.Embed()
                        embed.set_image(url=f"attachment://{file.filename}")
                        embed.set_footer(
                            text=(
                                f"steps: {info.get('steps')}, "
                                f"size: {info.get('width')}x{info.get('height')}, "
                                f"seed: {info.get('seed')}"
                            )
                        )

                    sent_img = await message.channel.send(file=file, embed=embed)

                    # Persist a marker so follow-up turns know an image exists
                    # and can reference/edit it by path.
                    marker = (
                        f"[Generated an image with the following prompt: "
                        f"{info.get('prompt', '')}] "
                        f"(seed: {info.get('seed')}, "
                        f"size: {info.get('width')}x{info.get('height')}, "
                        f"path: {img_path})"
                    )
                    await chat_history.record_bot_response(
                        guild_id=server,
                        channel_id=channel,
                        bot_user_id=self.user.id,
                        bot_name=self.user.display_name,
                        content=marker,
                        message_id=sent_img.id,
                        reply_to_message_id=None,
                    )
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
        # Backstop: drop leaked reasoning/decision narration that local models
        # sometimes emit without <think> tags (e.g. classifying the user's
        # message before answering). See _strip_reasoning_leak.
        text = _strip_reasoning_leak(text)
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

        # Image routing. On the Claude Code backend, Claude decides for itself
        # whether to draw something — and whether to use the diffusion model
        # (tools/images/generate.py) or write code and attach the result
        # (tools/images/attach.py). We skip the keyword/classifier heuristic
        # entirely in that case.
        #
        # The heuristic below remains the fallback for the local Ollama backend,
        # for rate-limited fallback, and for guilds with no tool access — where
        # Claude has no image tools, so the in-process pipeline is the only way
        # images can be produced at all. That pipeline runs inside the bot and
        # needs no Bash, so it stays available even in locked-down guilds.
        guild_tools = tools_allowed_for(server)
        tools_enabled = bool(guild_tools)

        _routing_model = self.pick_model(server, channel)
        claude_owns_images = (
            is_claude_code_model(_routing_model)
            and tools_enabled
            and not self.claude_code_client.is_rate_limited
        )
        if claude_owns_images:
            logger.info("[img] delegating image decisions to Claude")

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
        if not claude_owns_images and (
            keyword_match or has_recent_image_gen or has_attached_image
        ):
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
                        # Per BFL's FLUX.2 Klein prompting guide, the prompt
                        # should describe ONLY what changes — Klein gets the
                        # visual details from the source image. We use a small
                        # LLM call to normalize the user's natural-language
                        # request into one of BFL's documented edit patterns
                        # ("Replace X with Y", "Change the hair color to green",
                        # etc.) so vague/messy user input still produces
                        # well-formed instructions.
                        with Image.open(attached_image_path) as src:
                            src_w, src_h = src.size
                        width, height = choose_source_dimensions(user_content, src_w, src_h)
                        logger.info(
                            "[img] branch=attached-edit source_dims=%dx%d chosen=%dx%d",
                            src_w, src_h, width, height,
                        )
                        cleaned = clean_edit_instruction(user_content)
                        prompt = (
                            await self.ollama_client.normalize_edit_instruction(cleaned)
                        ).strip() if cleaned else "Enhance the image quality"
                        logger.info("[img] edit instruction (normalized): %r", prompt)
                        file_path, image_info, is_nsfw = await self.image_gen.edit_image(
                            prompt, attached_image_path, seed=-1,
                            width=width, height=height,
                        )
                    elif is_modification and prev_image_path and os.path.exists(prev_image_path):
                        # Case 1: Follow-up edit of a bot-generated image.
                        # Same BFL approach as Case 2 — normalize the user's
                        # input into a Klein-style edit instruction, do not
                        # rewrite it into a full target description.
                        width, height = choose_followup_dimensions(user_content, prev_width, prev_height)
                        logger.info(
                            "[img] branch=followup-edit source=%s prev_dims=%dx%d chosen=%dx%d",
                            prev_image_path, prev_width, prev_height, width, height,
                        )
                        cleaned = clean_edit_instruction(user_content)
                        prompt = (
                            await self.ollama_client.normalize_edit_instruction(cleaned)
                        ).strip() if cleaned else "Enhance the image quality"
                        logger.info("[img] edit instruction (normalized): %r", prompt)
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
        # guild_tools / tools_enabled were resolved at the top of this method,
        # since image routing depends on them too. An empty allowlist means this
        # guild gets no Bash at all (search-only path) — the one hard boundary
        # here. A partial allowlist keeps Bash, with each tool enforcing the
        # guild itself.

        # Resumable per-channel sessions. The full prompt is still built below:
        # it's local work (no API calls) and it's exactly what seeds a fresh
        # session. A resumed session replaces it with a delta prompt instead.
        resume_ok = self.resume_enabled_for(using_claude_code, is_pty) and tools_enabled

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

            memory_context = chat_history.get_memory_context(str(server), channel_id=str(channel), active_user_ids=active_user_ids, requesting_user_id=(messages[-1].get("discord_user_id") or None))
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
            override_note = ""
            if last_user_id:
                override = self.plugin_manager.get_system_prompt_override(last_user_id)
                if override:
                    if resume_ok:
                        # A session bakes its system prompt in at creation, and
                        # the next message here may come from a different user.
                        # Keep the session's prompt the base one and apply the
                        # override per turn (same trick as the PTY path).
                        override_note = f"[Personality override for this user: {override}]\n"
                    else:
                        system_prompt = override
                    logger.debug(f"[TTS-DEBUG] Using personality override for user {last_user_id}")
            prompt = f"System: {system_prompt}\n" + override_note + prompt

            memory_context = chat_history.get_memory_context(str(server), channel_id=str(channel), active_user_ids=active_user_ids, requesting_user_id=(messages[-1].get("discord_user_id") or None))
            if memory_context:
                prompt = f"{prompt}\n\n{memory_context}"

            prompt += f"\n\n[Current context: guild_id={server}, channel_id={channel}]"

            # Tell the model what it may actually use here. This is guidance so
            # it doesn't attempt denied tools and report a confusing failure —
            # the real enforcement is in the tools (tools/_guild_access.py) and,
            # for empty allowlists, in not passing Bash at all.
            if not tools_enabled:
                prompt += (
                    "\n\n[Tool access is DISABLED in this server. You cannot run Bash "
                    "or any tools/ CLI — including image generation. Do not claim to "
                    "perform tool actions; just chat. Web search is available.]"
                )
            elif guild_tools != TOOL_INTEGRATIONS:
                prompt += (
                    f"\n\n[Tool access in this server is limited to: "
                    f"{', '.join(sorted(guild_tools))}. Every other tools/ integration "
                    "is blocked and will reject you with a permission error — do not "
                    "attempt them, and do not try to work around the restriction (e.g. "
                    "by scripting the same action in Bash). If a user asks for "
                    "something only a blocked tool can do, say plainly that it isn't "
                    "enabled in this server.]"
                )

            # The Claude Code CLI has no vision input (allowedTools is
            # Bash,WebSearch,WebFetch), so attachments never reach it as images.
            # Surface the saved paths instead: Claude can't look at them, but it
            # can pass them to tools/images/edit.py.
            if claude_owns_images:
                attached = messages[-1].get("image_files") or []
                if attached:
                    prompt += (
                        "\n[The user attached "
                        f"{len(attached)} image(s), saved at: {', '.join(attached)}. "
                        "You cannot view them directly. If they're asking for an edit, "
                        "pass the path to tools/images/edit.py.]"
                    )
                # Most recent image this channel produced, for follow-up edits
                # ('make it bluer') without re-deriving state from chat text.
                prev = next(
                    (
                        m for m in reversed(messages[:-1])
                        if m.get("role") == "assistant"
                        and "[Generated an image" in m.get("content", "")
                    ),
                    None,
                )
                if prev:
                    path_match = re.search(r"path: (.+?)\)", prev["content"])
                    if path_match and os.path.exists(path_match.group(1)):
                        prompt += (
                            f"\n[Your most recent image in this channel: "
                            f"{path_match.group(1)} — pass this to "
                            f"tools/images/edit.py for follow-up edits.]"
                        )

            guild = self.get_guild(server)
            mention_context = extract_mention_context(user_content, guild)
            if mention_context:
                prompt = f"{prompt}\n\n{mention_context}"

        if images:
            print("Sending image")

        # Images Claude produced during this turn (diffusion via
        # tools/images/generate.py|edit.py, or code-drawn via attach.py).
        claude_images: List[dict] = []

        try:
            print(f"Using model: {model}")
            print(f"Prompt: {prompt}")

            if using_claude_code:
                try:
                    # PTY client: inject system prompt on first use
                    if is_pty and hasattr(self.claude_code_client, 'set_system_prompt'):
                        await self.claude_code_client.set_system_prompt(self.system_prompt)

                    # Identify the user who triggered the bot so CLI tools can
                    # enforce *their* Discord permissions (not the model's
                    # judgement). This is the trusted requester identity.
                    requester_uid = messages[-1].get("discord_user_id")
                    # Correlates any image tools/images/* produces during this
                    # turn so we can attach it to the reply below.
                    image_request_id = uuid.uuid4().hex
                    try:
                        if resume_ok:
                            # Resumable per-channel session: the model keeps its
                            # own memory of what it actually did, and we send
                            # only new messages.
                            raw_response = await self._claude_turn_with_session(
                                server=server,
                                channel=channel,
                                model=model,
                                messages=messages,
                                user_content=user_content,
                                search_summary=search_summary,
                                active_user_ids=active_user_ids,
                                requester_uid=requester_uid,
                                images=images,
                                full_prompt=prompt,
                                image_request_id=image_request_id,
                                trigger_message_id=messages[-1].get("message_id"),
                            )
                        elif tools_enabled:
                            raw_response, _ = await self.claude_code_client.generate_with_tools(
                                prompt, model, images,
                                requester_user_id=requester_uid,
                                requester_guild_id=server,
                                requester_channel_id=channel,
                                image_request_id=image_request_id,
                            )
                        else:
                            # No tools allowlisted for this guild: run without
                            # Bash entirely. This is the hard boundary — the
                            # model has no mechanism to execute anything.
                            raw_response, _ = await self.claude_code_client.generate_with_search(
                                prompt, model, images,
                            )
                    except BaseException:
                        # Don't leave orphaned images queued under this id —
                        # they'd never be attached and would leak memory.
                        self.image_service.discard(image_request_id)
                        raise
                    claude_images = self.image_service.drain(image_request_id)
                    if claude_images:
                        logger.info(
                            "[img] Claude produced %d image(s): %s",
                            len(claude_images),
                            ", ".join(
                                f"{i['kind']}:{os.path.basename(i['path'])}"
                                for i in claude_images
                            ),
                        )
                except RateLimitError:
                    reset = self.claude_code_client.rate_limit_resets_at or "unknown"
                    print(f"  [Claude Code rate limited, resets at {reset}, falling back to local model]")
                    model = CHAT_MODEL
                    ctx = VISION_MODEL_CTX if images else None
                    raw_response = await self.ollama_client.generate(prompt, model, images, keep_alive=1800, num_ctx=ctx)

                    if raw_response == "No response from Ollama.":
                        return ["No response from Ollama."]
                except Exception as cc_err:
                    print(f"  [Claude Code error: {cc_err}, falling back to local model]")
                    model = CHAT_MODEL
                    ctx = VISION_MODEL_CTX if images else None
                    raw_response = await self.ollama_client.generate(prompt, model, images, keep_alive=1800, num_ctx=ctx)

                    if raw_response == "No response from Ollama.":
                        return ["No response from Ollama."]

            else:
                ctx = VISION_MODEL_CTX if images else None
                raw_response = await self.ollama_client.generate(prompt, model, images, keep_alive=1800, num_ctx=ctx)

                if raw_response == "No response from Ollama.":
                    print("No response from Ollama")
                    return ["No response from Ollama."]

            print(f"Response: {raw_response}")
            parts = self.process_response(raw_response)
            # Append any images Claude made this turn. _send_response sends
            # dict items with an "image" key as files after the text parts,
            # so typing simulation and message splitting still apply.
            for img in claude_images:
                parts.append({
                    "image": img["path"],
                    "nsfw": img.get("nsfw", False),
                    "info": img,
                })
            return parts

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
        # Kill any in-flight coding jobs before the loop goes away, so their
        # worktrees get cleaned up rather than orphaned.
        try:
            await self.job_manager.shutdown()
        except Exception as e:
            logger.warning(f"agent job shutdown failed: {e}")
        await self.image_service.stop()
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
