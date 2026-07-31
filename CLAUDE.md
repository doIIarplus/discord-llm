# Discord LLM Bot

A Discord bot that integrates with locally-hosted LLMs via Ollama and Flux2 Klein 9B (via `diffusers`) for image generation. Responds to mentions/replies, supports multimodal input, web extraction, and optional RAG from a wiki dump.

## Running the Bot

```bash
source venv/bin/activate
python bot.py
```

Requires a `.env` file with `DISCORD_BOT_TOKEN`. Ollama must be running at `http://localhost:11434`. Flux2 Klein 9B is loaded in-process on first image request and auto-unloads after 5 minutes of inactivity (see [flux_client.py](flux_client.py)).

After changing slash commands, run `/sync_commands` once from Discord to push updates to Discord's API.

## Testing

**All changes to bot logic must be tested via the test CLI before being considered complete.** If a change adds or modifies functionality in `bot.py`, `web_extractor.py`, `ollama_client.py`, `commands.py`, or any module used in the message flow, the corresponding behavior must also be reflected and testable in [test_cli.py](test_cli.py). Update the test CLI whenever:
- A new feature is added (add a matching `/command` or ensure it works via normal messages)
- The context building, prompt formatting, or query flow changes
- New integrations are added (web search, file parsing, etc.)

```bash
source venv/bin/activate
python test_cli.py
```

Interactive CLI that mirrors the Discord bot's message flow without needing Discord. Supports multi-user conversations (`/user`), file attachments (`/attach`), web search (`/search`), context management, and system prompt changes. Type `/help` in the CLI for all commands.

**Quick smoke test** (non-interactive, verifies imports + LLM connectivity):
```bash
python -c "
import asyncio
from test_cli import TestCLI
async def t():
    cli = TestCLI()
    await cli.build_context('say pong', 'Test')
    r = await cli.query()
    assert r, 'No response'
    print('OK:', r[0][:50])
asyncio.run(t())
"
```

## Architecture

### Entry Point
- **[bot.py](bot.py)** — `OllamaBot(discord.Client)` main class. Handles `on_message`, builds per-server/per-channel context, orchestrates LLM queries, and sends responses with simulated typing delays.

### Core Modules
- **[config.py](config.py)** — All configuration: tokens, model names, paths, context limits. Reads from `.env`.
- **[models.py](models.py)** — Enums for `Txt2TxtModel` and `Txt2ImgModel`, plus `ImageInfo` dataclass.
- **[ollama_client.py](ollama_client.py)** — `OllamaClient`: async HTTP client for the Ollama API. Handles generation, image task classification, and NSFW classification.
- **[commands.py](commands.py)** — All Discord slash commands (registered via `app_commands`).
- **[image_generation.py](image_generation.py)** — `ImageGenerator`: wraps `FluxClient` and `OllamaClient`. Provides `generate_image` (txt2img) and `edit_image` (img2img). NSFW is detected post-generation and images are spoilered if flagged.
- **[flux_client.py](flux_client.py)** — `FluxClient`: wraps `Flux2KleinPipeline` from `diffusers`. Lazy-loads the model on first use and auto-unloads after 5 minutes of idle to free VRAM.

### Context & Response
- **[response_splitter.py](response_splitter.py)** — Splits responses on `---MSG---` markers and handles Discord's 2000-char message limit. Bot uses this to send multi-part messages with calculated typing delays.
- **[mention_extractor.py](mention_extractor.py)** — Resolves Discord `@user`, `#channel`, `@role` mentions into readable context injected into the prompt.
- **[web_extractor.py](web_extractor.py)** — Async URL extraction via `aiohttp` + `trafilatura` (strips boilerplate/ads/nav). Also provides `web_search()` via Tavily API for the `/search` command. Content truncated to 2000 chars.
- **[file_parser.py](file_parser.py)** — Parses uploaded files (PDF, text, code) and adds content to the prompt. Files are cleaned up after parsing.
- **[latex.py](latex.py)** — LaTeX-to-image rendering utilities (not currently used in the response pipeline).

### RAG System
- **[rag_system.py](rag_system.py)** — `RAGSystem`: ChromaDB-backed vector store using `all-mpnet-base-v2` embeddings. Indexes MediaWiki XML dumps. RAG is **disabled by default** (`self.rag_enabled = False`).
- **[wiki_parser.py](wiki_parser.py)** — Parses MediaWiki XML exports into chunks for indexing.
- **[chroma_db/](chroma_db/)** — Persistent ChromaDB storage (collection: `maplestory_wiki`).

### Memory System
- **[chat_history.py](chat_history.py)** — Persistent chat history and memory. Records all Discord messages to `chat_history.db` (SQLite). Provides `get_memory_context()` which returns user profiles and recent server events for prompt injection.
- **[tools/memory/summarize.py](tools/memory/summarize.py)** — Scheduled summarizer. Reads new messages from `chat_history.db`, calls Claude (Sonnet) to analyze them, and writes user profiles and server events back to the DB. Self-gates: only runs when there are new messages AND the server has been idle for 60+ minutes.

### Supporting
- **[utils.py](utils.py)** — Image-to-base64 helpers.

## Key Behaviors

**Response triggers**: Bot only responds when directly `@mentioned` or when someone replies to one of its messages.

**Context**: When the bot is triggered, it fetches the last `CONTEXT_LIMIT = 20` messages from `chat_history.db` for that channel. This means the bot sees **all** recent conversation (not just direct interactions), so users don't need to repeat context. The context is rebuilt from the DB on every trigger — nothing is lost on restart.

**Model selection**: If the last message has images attached, switches to `IMAGE_RECOGNITION_MODEL` (`qwen3-vl:32b`); otherwise uses `CHAT_MODEL` (`gemma-3-27b-it-abliterated`).

**Image generation detection**: On the Claude Code backend, Claude decides for
itself whether to draw and picks diffusion vs. code — see
[Images](#images-toolsimages). The keyword heuristic (generate/create/draw +
image/picture/photo) followed by an LLM classifier is now only the **fallback**
for the local Ollama backend and rate-limited fallback.

**Multi-message responses**: LLM can split its response with `---MSG---` markers; each part is sent as a separate Discord message with simulated typing delay.

**File cleanup**: Uploaded attachments in `multimodal_input/` are kept around so they can be reused as source images for img2img edits. Generated images in `api_out/` are not auto-cleaned.

**System prompt personality**: Millennial texter style — short, casual, no unsolicited help. Changeable at runtime via `/set_system_prompt` and `/reset_system_prompt`.

## Slash Commands

| Command | Description |
|---|---|
| `/clear` | Clear conversation context for the channel |
| `/ask <question>` | One-shot question (no context) |
| `/set_system_prompt` | Override system prompt |
| `/reset_system_prompt` | Restore default prompt |
| `/get_system_prompt` | Show current prompt |
| `/enable_rag` / `/disable_rag` | Toggle RAG wiki context |
| `/index_wiki` | Index a MediaWiki XML dump |
| `/search_wiki` | Search indexed wiki content |
| `/rag_stats` | Show RAG database stats |
| `/search <query>` | Search the web via Tavily, summarize with LLM |
| `/sync_commands` | Manually sync slash commands to Discord |

## Configuration (`.env`)

```
DISCORD_BOT_TOKEN=
GUILD_ID=363154169294618625                           # optional, defaults to this value
OLLAMA_API_URL=http://localhost:11434/api/generate     # optional
OLLAMA_MODEL=qwen2.5vl:72b                            # fallback, not the active chat model
IMAGE_RECOGNITION_MODEL=qwen3-vl:32b
NSFW_CLASSIFICATION_MODEL=qwen3-vl:32b
FLUX_MODEL_ID=black-forest-labs/FLUX.2-klein-9B      # optional, default shown
FILE_INPUT_FOLDER=/home/dollarplus/projects/discord_llm_bot/multimodal_input/
TAVILY_API_KEY=tvly-...                                # required for /search command
```

The active chat model (`CHAT_MODEL`) is hardcoded in [config.py](config.py) as `Txt2TxtModel.GEMMA3_27B_ABLITERATED`.

## Dependencies

Install with: `pip install -r requirements.txt`

Key packages: `discord.py`, `python-dotenv`, `aiohttp`, `trafilatura`, `tavily-python`, `chromadb`, `sentence-transformers`, `beautifulsoup4`, `pypdf`, `lxml`, `numpy`, `nltk`, `diffusers`, `transformers`, `torch`, `pillow`.

## CLI Tools

Standalone Python scripts in `tools/` that Claude can call via Bash. Each tool uses argparse, outputs JSON to stdout, and errors to stderr (exit 1). All tools are stateless and self-contained.

**Calling convention:** `python tools/<integration>/<tool>.py [args]`
**Discovery:** `python tools/<integration>/<tool>.py --help` for usage.

**IMPORTANT — keeping prompts in sync:** When adding, removing, or renaming CLI tools, also update the tool lists in the system prompts in [bot.py](bot.py) (`self.original_system_prompt`, "CLI TOOLS" section) and [test_cli.py](test_cli.py) (`self.system_prompt`, "CLI TOOLS" section). The LLM can only use tools it knows about from its prompt.

### Confirmation policy
- **Read-only tools** (list, get, search, stats): Execute immediately, report results.
- **Producing / additive tools** (generate or edit an image, render a chart, send a
  message, create a thread, schedule a task): **Execute immediately.** These are
  cheap and reversible, and the user asking for the thing *is* the confirmation.
  Do not describe what you are about to do and stop — that reads as a broken
  promise, because nothing runs between turns (see below).
- **Destructive / irreversible tools** (delete a channel or message, timeout a
  member, bulk role or nickname changes, deleting a Splitwise expense, anything
  that removes data): Describe the action and wait for explicit user confirmation
  before executing.

**No promises of future work.** Each Discord message is handled by a single
`claude -p` invocation that **exits when the reply is sent**. Nothing runs in the
background, and no agent loop continues afterwards. Resumable sessions (above)
change what you *remember* next turn — they do not give you time to work between
turns. So the turn that emits text is still the only chance to act: never answer
with intent ("lemme go pull that", "i'll re-render it", "gimme a sec") as a
substitute for acting. Do the work with tools **first**, then reply describing
what you actually did. If a task is genuinely too large for one turn, say so
plainly and state what you'd need, rather than implying it is underway.

### Access control

Access control is **enforced in code**, not just by prompt. The bot injects the *triggering* Discord user's identity into every tool subprocess as the `DISCORD_REQUESTING_USER_ID` / `DISCORD_REQUESTING_GUILD_ID` env vars (for the persistent PTY session, whose env is fixed at startup, it is written to `tools/discord/.request_context` instead). Tools read this trusted identity — not the `--user-id` the model passes — so a user cannot act beyond their own permissions even if the model is convinced to try.

- **Discord tools** ([tools/discord/_permissions.py](tools/discord/_permissions.py)): Every Discord tool calls `require_permission(...)`, which resolves the requesting user's Discord roles/permissions in the guild (with owner + Administrator bypass and channel-overwrite handling) and **hard-rejects** if they lack the permission the action needs. E.g. `delete_channel.py` requires Manage Channels; if the requester lacks it, the tool exits with a permission-denied error and never calls Discord. Mapping of tool → required permission lives in each tool (e.g. delete/create/rename channel → Manage Channels; timeout → Moderate Members; add/remove role → Manage Roles; delete/pin/edit message → Manage Messages; read tools → View Channel).
- **Splitwise tools** ([tools/splitwise/_auth.py](tools/splitwise/_auth.py)): Restricted in code to Discord user `118567805678256128` (dollarplus) via `require_owner()`. Any other requester is denied. Still decline politely in conversation, but the code is the backstop.
- **Fail closed**: if the requesting user cannot be verified, the action is denied.
- **Trust boundary**: this stops the normal failure mode (the model relaying a request from an unauthorized user). It is not a sandbox — a model with unrestricted Bash could bypass it. For a hard guarantee, move enforcement to a Claude Code PreToolUse hook.

When a tool returns a permission-denied error, relay it to the user plainly (they lack the required Discord permission); do not retry or attempt a workaround.

### Splitwise (`tools/splitwise/`)
Requires `SPLITWISE_API_KEY` in environment.

| Tool | Description |
|------|-------------|
| `get_current_user.py` | Get authenticated user's ID, name, email |
| `list_friends.py` | List all friends with IDs, names, emails, balances |
| `get_balances.py [--all]` | Show non-zero balances (--all includes zero) |
| `create_expense.py --amount N --description TEXT --split-with ID [ID ...] [--ratios R ...] [--shares S ...] [--group-id G] [--currency C] [--paid-by ID]` | Create expense (equal/ratio/custom split) |
| `delete_expense.py EXPENSE_ID` | Delete an expense |
| `list_groups.py` | List Splitwise groups and members |
| `get_group.py GROUP_ID` | Get group details and balances |
| `list_expenses.py [--limit N] [--friend-id ID] [--group-id ID] [--dated-after DATE] [--dated-before DATE]` | List recent expenses with filters |

**Workflow example:** To split $50 with "Jason":
1. `list_friends.py` → find Jason's user ID
2. `create_expense.py --amount 50 --description "Dinner" --split-with <jason_id>`

### Resumable Claude sessions (per channel)

Each Discord channel gets its own Claude Code CLI session, resumed on the next turn
instead of re-sending the chat transcript. Without this, every message was an
independent `claude -p --no-session-persistence` process that could not know what it
had just done — it would say "lemme go pull that" and the process would exit, making
the promise unkeepable. The session remembers what the model *actually did* (files
written, URLs that worked), which a re-rendered transcript can never convey.

- **Scope**: the Claude Code backend. The local Ollama backend and the PTY client
  (which has its own single global session) keep the full-transcript behavior.
- **Config**: `CLAUDE_RESUME_SESSIONS` (default **on**, set `0` to disable),
  `CLAUDE_SESSION_MAX_TOKENS` (default 180000), `CLAUDE_SESSION_MAX_TURNS` (0 = off).
- **Storage**: the `claude_sessions` table in `chat_history.db` holds only a pointer
  (`session_id`, watermark, model, prompt/memory hashes, token count). The
  conversation itself lives in `~/.claude/projects/<cwd-slug>/<session_id>.jsonl`,
  which is why sessions **survive bot restarts**. Those files grow; rotated ones are
  left behind (pruning is not automated).
- **Delta feeding**: a resumed turn sends only messages recorded since the watermark
  (`get_channel_messages_since`), so the bot still sees conversation it wasn't
  mentioned in — better than the old fixed 20-message window. **Bot-authored rows are
  excluded**: the session generated those replies itself, and `record_bot_response`
  writes them to the DB *after* the turn, so without the filter the model would
  re-read its own output as if a user had said it.
- **Invalidation**: changing the system prompt (`/set_system_prompt`,
  `/reset_system_prompt`) or the model (`/set_model`) starts a fresh session, since a
  session bakes both in at creation. Per-user personality overrides are therefore
  applied as a **turn-level note**, not via the system prompt — the next message in a
  channel may come from a different user.
- **Rotation**: past the token budget, the outgoing session writes a handoff note
  (exact file paths, working URLs, reusable commands, open threads) that seeds its
  replacement. Best-effort — falls back to a plain fresh session.
- **Concurrency**: one `asyncio.Lock` per `(guild, channel)`. `--fork-session` is
  deliberately not used — it would mint a new id and on-disk file every turn.
- **Failure handling**: a dead session id raises `SessionResumeError` (CLI exits 1 with
  `No conversation found with session ID: …`); the pointer is dropped and the turn
  retries once as a fresh session. `chat_history.db` remains the source of truth.
- **`/clear`** drops the session as well as the in-memory context.

### Images (`tools/images/`)
Claude decides **whether** an image is wanted and **how** to make it. There is no
keyword trigger on this path. Requires the bot process to be running — the tools
are thin HTTP clients, not standalone generators.

| Tool | Description |
|------|-------------|
| `generate.py PROMPT [--preset P] [--width N] [--height N] [--seed N] [--steps N]` | Draw with the diffusion model (Flux2 Klein) |
| `edit.py PROMPT --image PATH [--preset P] [...]` | Edit an existing image with the diffusion model |
| `attach.py PATH [--caption TEXT] [--nsfw-check]` | Attach an image Claude drew itself with code |

**Which one to use.** Diffusion garbles text and can't be trusted with real data,
so anything *carrying information* should be drawn with code and registered via
`attach.py`: charts, plots, diagrams, flowcharts, timelines, tables, scoreboards,
anything with labels/axes/numbers. `generate.py` is for photographic/painterly
output where realism matters and exact text doesn't.

- **[image_service.py](image_service.py)** — an aiohttp service the bot hosts on
  loopback (`IMAGE_SERVICE_HOST`/`IMAGE_SERVICE_PORT`, default `127.0.0.1:8766` —
  note **8766**, since spookie_merged uses 8765 on the same box). It exists because
  `FluxClient` holds Klein 9B **resident in the bot process**; a tool that imported
  Flux directly would load a second 9B model per subprocess. The tools are
  stdlib-only HTTP clients and must never import torch/diffusers.
- **Auth**: a token generated at startup, written to `tools/images/.image_service`
  with mode `0600`, removed on shutdown. Non-matching tokens get 401.
- **Path confinement**: `--image`/`PATH` args are `realpath`'d and must live under
  `PROJECT_DIR`, so a prompt-injected call can't pull arbitrary files off disk.
- **Delivery**: each Claude turn gets an `IMAGE_REQUEST_ID` (uuid4) forwarded through
  `generate_with_tools` → `_run_cli` → `_build_env` into the tool subprocess env.
  Tools echo it back; the service buckets images under it. After the turn `bot.py`
  calls `image_service.drain(id)` and appends `{"image": path}` dicts, which
  `_send_response` sends as files *after* the text — so typing simulation, `---MSG---`
  splitting, and NSFW spoilering all still apply. No path scraping from reply text.
- **Follow-up state**: the bot injects the last generated image's path (and user
  attachment paths) into the prompt, since the CLI runs with
  `--allowedTools Bash,WebSearch,WebFetch` and therefore **cannot see images**.

### Web Search (`tools/web_search/`)
Requires `TAVILY_API_KEY` in environment.

| Tool | Description |
|------|-------------|
| `search.py QUERY [--max-results N]` | Search the web, return raw results (no LLM summarization) |

### Discord (`tools/discord/`)
Requires `DISCORD_BOT_TOKEN` in env. Webhook tools also need `DISCORD_WEBHOOK_<NAME>` URLs.

| Tool | Description |
|------|-------------|
| `send_message.py --channel-id ID --content TEXT [--reply-to MSG_ID]` | Send message as the bot to any channel. Supports mentions. |
| `edit_message.py --channel-id ID --message-id ID --content TEXT` | Edit a bot-sent message |
| `delete_message.py --channel-id ID --message-id ID` | Delete a message (own or others with Manage Messages) |
| `get_channel_history.py --channel-id ID [--limit N] [--before MSG_ID] [--after MSG_ID] [--user-id ID]` | Fetch recent messages (max 100, default 10) |
| `search_messages.py --guild-id ID --query TEXT [--channel-id ID] [--author-id ID] [--max-results N]` | Search messages across the server |
| `get_user.py --user-id ID [--guild-id ID]` | Get user info (add guild-id for nickname, roles, join date) |
| `add_role.py --guild-id ID --user-id ID --role-id ID` | Add a role to a user |
| `remove_role.py --guild-id ID --user-id ID --role-id ID` | Remove a role from a user |
| `list_roles.py --guild-id ID` | List all server roles with IDs |
| `set_nickname.py --guild-id ID --user-id ID --nickname TEXT [--clear]` | Set or clear a member's nickname |
| `timeout_user.py --guild-id ID --user-id ID --duration DURATION [--remove]` | Timeout a member (e.g. 10m, 1h, 7d). Max 28d |
| `react.py --channel-id ID --message-id ID --emoji EMOJI` | Add reaction (Unicode or custom name:id) |
| `pin_message.py --channel-id ID --message-id ID [--unpin]` | Pin or unpin a message |
| `create_thread.py --channel-id ID --name TEXT [--message-id ID] [--content TEXT] [--auto-archive N]` | Create a thread (from message or standalone) |
| `list_channels.py --guild-id ID [--type N]` | List channels (0=text, 2=voice, 4=category) |
| `create_channel.py --guild-id ID --name TEXT [--type N] [--parent-id ID] [--topic TEXT]` | Create a text, voice, or category channel |
| `rename_channel.py --channel-id ID --name TEXT` | Rename an existing channel |
| `delete_channel.py --channel-id ID` | Delete a channel (irreversible) |
| `send_webhook.py --webhook NAME --content TEXT [--username NAME]` | Send message via webhook (different identity). Supports `<@USER_ID>` mentions. |

**Reminder workflow:** Combine scheduler `--once` with `send_message.py`:
```bash
python tools/scheduler/create_task.py \
  --name "reminder-buy-cream-puffs" \
  --schedule "0 9 30 3 *" \
  --once \
  --command "python tools/discord/send_message.py --channel-id 123456 --content '<@118567805678256128> Reminder: buy cream puffs'"
```

### RAG Wiki (`tools/rag/`)

| Tool | Description |
|------|-------------|
| `search.py QUERY [--n-results N]` | Search indexed wiki content |
| `index.py [--wiki-dump PATH] [--clear-existing]` | Index a MediaWiki XML dump |
| `stats.py` | Show ChromaDB collection statistics |

### Scheduler (`tools/scheduler/`)
For recurring tasks. Optional dependency: `pip install croniter`

| Tool | Description |
|------|-------------|
| `create_task.py --name NAME --schedule CRON --command CMD [--description TEXT] [--once]` | Create task (--once for one-shot reminders that auto-delete after running) |
| `list_tasks.py [--all]` | List scheduled tasks |
| `delete_task.py TASK_ID` | Delete a task |
| `run_due.py [--dry-run]` | Execute due tasks (called by system cron) |

**Cron setup for scheduler:**
```bash
* * * * * cd /home/dollarplus/projects/discord_llm_bot && /home/dollarplus/projects/discord_llm_bot/venv/bin/python tools/scheduler/run_due.py 2>&1
```

Logs are written to `scheduler.log` in the project root. Check it for task execution results, failures, and one-shot task cleanup.

### Memory (`tools/memory/`)

| Tool | Description |
|------|-------------|
| `summarize.py --guild-id ID [--dry-run] [--force]` | Summarize new chat history into user profiles and events. Self-gates on idle time (60m) and new messages. Use --force to skip idle check. |

The summarizer is registered as a scheduled task running every 5 minutes. Most invocations exit immediately (no new messages or server still active). When it does run, it calls Claude Sonnet to analyze messages and update `chat_history.db`.

### Resume Review (`tools/resume/`)
Modeled on interviewstreet/hiring-agent: a resume-to-score pipeline. Read-only.

| Tool | Description |
|------|-------------|
| `review.py --pdf PATH [--role-description TEXT] [--role-file PATH] [--github-username USER]` | Extract text from a resume PDF (pypdf), optionally enrich with public GitHub signals, and score the candidate with the local LLM across a fixed rubric (technical skills, experience, project quality, education, overall fit). Outputs JSON: `{overall_score, categories: [{name, score, reasoning}], summary}`. |

## File I/O

- Uploaded attachments temporarily saved to `multimodal_input/`, deleted after processing
- Generated images saved to `api_out/txt2img/` and `api_out/img2img/`
- LaTeX renders saved to `latex_images/` (if latex rendering is re-enabled)
- RAG vector DB: `chroma_db/`
- Scheduler tasks: `tools/scheduler/tasks.json`
- Chat history & memory: `chat_history.db` (messages, user profiles, server events, summarizer state)
