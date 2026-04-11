# Discord LLM Bot

A Discord bot that integrates with locally-hosted LLMs via Ollama and Stable Diffusion for image generation. Responds to mentions/replies, supports multimodal input, web extraction, and optional RAG from a wiki dump.

## Running the Bot

```bash
source venv/bin/activate
python bot.py
```

Requires a `.env` file with `DISCORD_BOT_TOKEN`. Ollama must be running at `http://localhost:11434` and Stable Diffusion at `http://127.0.0.1:7860`.

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
- **[image_generation.py](image_generation.py)** — `ImageGenerator`: wraps `StableDiffusionClient` and `OllamaClient` for text-to-image. NSFW is detected post-generation and images are spoilered if flagged.

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
- **[stable_diffusion_client.py](stable_diffusion_client.py)** — HTTP client for the Automatic1111 SD WebUI API (`/sdapi/v1/txt2img`).
- **[utils.py](utils.py)** — Image-to-base64 helpers.

## Key Behaviors

**Response triggers**: Bot only responds when directly `@mentioned` or when someone replies to one of its messages.

**Context**: Per-server, per-channel message history capped at `CONTEXT_LIMIT = 10` messages. Stored in memory only (lost on restart).

**Model selection**: If the last message has images attached, switches to `IMAGE_RECOGNITION_MODEL` (`qwen3-vl:32b`); otherwise uses `CHAT_MODEL` (`gemma-3-27b-it-abliterated`).

**Image generation detection**: Uses a fast keyword heuristic (generate/create/draw + image/picture/photo) before calling the LLM classifier, avoiding an Ollama round-trip for most messages.

**Multi-message responses**: LLM can split its response with `---MSG---` markers; each part is sent as a separate Discord message with simulated typing delay.

**File cleanup**: Uploaded attachments are deleted from `multimodal_input/` after being encoded/parsed. Generated images in `api_out/` are not auto-cleaned.

**System prompt personality**: Millennial texter style — short, casual, no unsolicited help. Changeable at runtime via `/set_system_prompt` and `/reset_system_prompt`.

## Slash Commands

| Command | Description |
|---|---|
| `/clear` | Clear conversation context for the channel |
| `/ask <question>` | One-shot question (no context) |
| `/set_system_prompt` | Override system prompt |
| `/reset_system_prompt` | Restore default prompt |
| `/get_system_prompt` | Show current prompt |
| `/generate_image` | Generate image via Stable Diffusion |
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
SD_API_URL=http://127.0.0.1:7860
FILE_INPUT_FOLDER=/home/dollarplus/projects/discord_llm_bot/multimodal_input/
TAVILY_API_KEY=tvly-...                                # required for /search command
```

The active chat model (`CHAT_MODEL`) is hardcoded in [config.py](config.py) as `Txt2TxtModel.GEMMA3_27B_ABLITERATED`.

## Dependencies

Install with: `pip install -r requirements.txt`

Key packages: `discord.py`, `python-dotenv`, `aiohttp`, `trafilatura`, `tavily-python`, `chromadb`, `sentence-transformers`, `beautifulsoup4`, `pypdf`, `lxml`, `numpy`, `nltk`.

Optional (better token counting in RAG): `transformers`, `torch`.

## CLI Tools

Standalone Python scripts in `tools/` that Claude can call via Bash. Each tool uses argparse, outputs JSON to stdout, and errors to stderr (exit 1). All tools are stateless and self-contained.

**Calling convention:** `python tools/<integration>/<tool>.py [args]`
**Discovery:** `python tools/<integration>/<tool>.py --help` for usage.

### Confirmation policy
- **Read-only tools** (list, get, search, stats): Execute immediately, report results.
- **Mutating tools** (create, delete, generate, schedule): Describe the action and wait for user confirmation before executing.

### Access control
- **Splitwise tools**: Only available to Discord user `118567805678256128` (dollarplus). If any other user requests Splitwise actions, politely decline — the tools are tied to dollarplus's personal Splitwise account. The requesting user's Discord ID is included in the prompt as `discord_id=`.

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

### Web Search (`tools/web_search/`)
Requires `TAVILY_API_KEY` in environment.

| Tool | Description |
|------|-------------|
| `search.py QUERY [--max-results N]` | Search the web, return raw results (no LLM summarization) |

### Stable Diffusion (`tools/stable_diffusion/`)
Requires SD WebUI running at `SD_API_URL`.

| Tool | Description |
|------|-------------|
| `generate.py --prompt TEXT [--negative-prompt TEXT] [--width N] [--height N] [--cfg-scale F] [--steps N] [--seed N]` | Generate image, returns file path |
| `start_server.py [--sd-url URL] [--chromaforge-dir PATH]` | Start the ChromaForge/SD WebUI server if not already running |

### nhentai (`tools/nhentai/`)

| Tool | Description |
|------|-------------|
| `fetch_preview.py CODE [--output-dir DIR]` | Fetch preview image for 6-digit code |

### Discord (`tools/discord/`)
Requires `DISCORD_BOT_TOKEN` in env. Webhook tools also need `DISCORD_WEBHOOK_<NAME>` URLs.

| Tool | Description |
|------|-------------|
| `send_message.py --channel-id ID --content TEXT [--reply-to MSG_ID]` | Send message as the bot to any channel. Supports mentions. |
| `get_channel_history.py --channel-id ID [--limit N] [--before MSG_ID] [--after MSG_ID] [--user-id ID]` | Fetch recent messages (max 100, default 10) |
| `search_messages.py --guild-id ID --query TEXT [--channel-id ID] [--author-id ID] [--max-results N]` | Search messages across the server |
| `add_role.py --guild-id ID --user-id ID --role-id ID` | Add a role to a user |
| `remove_role.py --guild-id ID --user-id ID --role-id ID` | Remove a role from a user |
| `list_roles.py --guild-id ID` | List all server roles with IDs |
| `react.py --channel-id ID --message-id ID --emoji EMOJI` | Add reaction (Unicode or custom name:id) |
| `pin_message.py --channel-id ID --message-id ID [--unpin]` | Pin or unpin a message |
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

## File I/O

- Uploaded attachments temporarily saved to `multimodal_input/`, deleted after processing
- Generated images saved to `api_out/txt2img/` and `api_out/img2img/`
- LaTeX renders saved to `latex_images/` (if latex rendering is re-enabled)
- RAG vector DB: `chroma_db/`
- Scheduler tasks: `tools/scheduler/tasks.json`
- Chat history & memory: `chat_history.db` (messages, user profiles, server events, summarizer state)
