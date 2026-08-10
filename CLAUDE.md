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
- **`emoji_usage` table** (in `chat_history.db`) — persistent counters for custom
  emoji, keyed `(guild_id, emoji_id, source)` where source is `message` or
  `reaction`. Message text is counted in `record_message()` and
  `record_bot_response()` via `CUSTOM_EMOJI_RE` (duplicates within one message
  each count); reactions are counted by `on_raw_reaction_add` /
  `on_raw_reaction_remove` in [bot.py](bot.py), using the **raw** events so
  reactions on uncached old messages still register. Counts are clamped at 0, so
  a removal of a reaction we never saw added can't go negative. Unicode emoji
  are not tracked (no id). Read it with
  `tools/discord/emoji_stats.py`. **Message counts can be backfilled**
  (`--backfill` rescans the `messages` table); **reaction counts only start from
  when this shipped** — Discord exposes no reaction history to replay, so there
  is no historical reaction data and early reaction numbers understate reality.
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
| `/emoji_stats [source] [unused] [limit]` | Custom emoji usage counts for the server (source: all/message/reaction; `unused` lists zero-count emojis; limit 1–100, default 25) |
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
DD_CLI_ACCESS_TOKEN=                                   # required for tools/doordash (no keychain in WSL)
```

The active chat model (`CHAT_MODEL`) is hardcoded in [config.py](config.py) as `Txt2TxtModel.CLAUDE_CODE` (the Claude Code CLI backend). To switch back to a local Ollama model, use `/set_model` at runtime or edit `CHAT_MODEL` in [config.py](config.py). The utility models (`SEARCH_UTILITY_MODEL`, `SEARCH_SUMMARIZATION_MODEL`, `NSFW_CLASSIFICATION_MODEL`, `IMAGE_RECOGNITION_MODEL`, `IMAGE_EDIT_DESCRIPTION_MODEL`) remain on Ollama.

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
  message, create a thread, schedule a task, **rebuilding a DoorDash cart with
  `order.py reorder`, applying or removing a promo with `promo.py apply/remove`,
  or dropping a wrong line with `cart.py remove`**): **Execute immediately.**
  These are cheap and reversible, and the user asking for the thing *is* the
  confirmation. Do not describe what you are about to do and stop — that reads as
  a broken promise, because nothing runs between turns (see below).
- **Destructive / irreversible tools** (delete a channel or message, timeout a
  member, bulk role or nickname changes, deleting a Splitwise expense, **placing
  a DoorDash order**, anything that removes data or spends money): Describe the
  action and wait for explicit user confirmation before executing. In the
  DoorDash integration `order.py place` is the *only* tool in this tier —
  everything else there is read-only or reversible.

**No promises of future work.** Each Discord message is handled by a single
`claude -p` invocation that **exits when the reply is sent**. Nothing runs in the
background, and no agent loop continues afterwards. Resumable sessions (above)
change what you *remember* next turn — they do not give you time to work between
turns. So the turn that emits text is still the only chance to act: never answer
with intent ("lemme go pull that", "i'll re-render it", "gimme a sec") as a
substitute for acting. Do the work with tools **first**, then reply describing
what you actually did. If a task is genuinely too large for one turn, say so
plainly and state what you'd need, rather than implying it is underway.

### Agentic coding (`tools/agent/`) — owner only

Conversational code work: "add logging to X", "fix the bug in Y". A chat turn
starts a **background job**; a full Claude Code agent (with subagents via `Task`)
does the work in an isolated worktree and streams progress into a Discord message
that edits itself.

| Tool | Description |
|---|---|
| `start_task.py --repo owner/repo --task "..."` | Kick off a job; returns a job_id immediately |
| `status.py [--job-id ID]` | Check the channel's current/latest job |
| `cancel.py [--job-id ID]` | Kill the agent and remove its worktree |
| `push.py [--job-id ID]` | Retry a push that failed (jobs push themselves) |

**Sandbox.** Everything happens under `~/git_projects` (`AGENT_WORKSPACE`), never
in `~/projects`. Repos are cloned on demand into
`~/git_projects/<owner>/<repo>`; each task gets a worktree under
`~/git_projects/.worktrees/<owner>__<repo>/<branch>`. This is deliberate — an
earlier design put worktrees inside the user's own checkout, which showed up as
an untracked `.worktrees/` in their `git status` and risked a blanket `git add`
sweeping up their work in progress.

**Flow.** Agent works → harness commits to `jaspt/<slug>-<jobid>` → **pushes the
branch automatically** → posts buttons. Auto-push is safe because it's always a
task branch, never the default branch. If the key lacks write access the push is
skipped (`PushDenied`), the work stays committed locally, and the message says so.

**Buttons** ([agent_views.py](agent_views.py)): *Open PR* (a link to GitHub's
prefilled compare page — no token needed; becomes a real API-backed button when
`GITHUB_TOKEN` is set), *View diff*, *Delete branch*. Action buttons are
restricted to whoever started the job, matching `RestartConfirmView` in
commands.py.

**Streaming** ([agent_events.py](agent_events.py)): `--output-format stream-json`
is parsed into a `ProgressState` and rendered into one message, edited at most
every 5s (`EDIT_INTERVAL`) so Discord's ratelimits aren't hit. The agent runs with
`--allowedTools Bash,Read,Write,Edit,Glob,Grep,Task,WebSearch,WebFetch` and
`--permission-mode acceptEdits`, with `cwd` set to the worktree so the *target
repo's* CLAUDE.md loads rather than this one.

One job per channel. `cancel` kills the subprocess and cleans up.

### GitHub (`tools/github/`) — owner only

Lets the bot work on arbitrary GitHub repos: clone, edit, commit, push.

| Tool | Description |
|---|---|
| `clone.py --repo owner/repo [--ref B] [--reset]` | Clone/refresh into `github_workspace/<owner>/<repo>`, print the path |
| `status.py --repo-path P [--diff]` | Show changed files, diffstat, optionally the diff |
| `commit_push.py --repo-path P -m MSG [--add ...] [--dry-run]` | Stage, commit, push to the **default branch** |

Workflow: `clone.py` → edit files at the printed path with normal tools →
`commit_push.py`. The workspace persists between turns and is gitignored.

- **Owner-only, enforced in code.** `_gh.require_owner()` checks the trusted
  `DISCORD_REQUESTING_USER_ID` against `GITHUB_OWNER_DISCORD_ID` (default
  `118567805678256128`) and fails closed. Same pattern as
  `tools/splitwise/_auth.py`. Also passes through the per-guild allowlist as the
  `github` integration, which is checked first.
- **Pushes straight to the default branch** — no PR, no review, live on push.
  `--dry-run` shows what would be committed without doing it.
- **Never force-pushes.** If the remote moved on, the push is rejected and the
  commit stays local rather than clobbering other commits.
- **Auth is the machine's SSH key**, so reachable repos = whatever that key
  grants. No token or `gh` CLI needed (`gh` is not installed).
- **Input hardening**: `--repo` must be a plain `owner/repo` slug (URL forms are
  normalized); `--repo-path` must resolve inside `github_workspace/`, so these
  tools cannot commit to *this* repo or anywhere else on disk.
- **Known gap**: the guild that has these tools also has unrestricted Bash, so a
  non-owner could still ask the bot to run `git push` directly. The owner check
  stops the sanctioned path, not every path. Closing it fully needs a PreToolUse
  hook on Bash.

### Per-guild tool allowlist

Which `tools/` integrations Claude may use is configured **per guild**, in `.env`:

```
GUILD_TOOLS_363154169294618625=*                    # everything
GUILD_TOOLS_1528742025711714425=web_search,images   # only these
GUILD_TOOLS_999999999999999999=                     # explicitly nothing
```

Valid names are the `tools/` directory names: `discord`, `doordash`, `flux`,
`images`, `memory`, `rag`, `resume`, `scheduler`, `splitwise`, `web_search`.
Unknown names are dropped with a warning; malformed `GUILD_TOOLS_<x>` keys are ignored.

**The default is deny-all.** A guild with no `GUILD_TOOLS_` entry gets no tools,
so a newly added server is safe until you opt it in. Note this cuts both ways —
if you don't give your home guild `=*`, tools (including image generation) stop
working there.

Enforcement is layered, and the tiers are *not* equally strong:

| Allowlist | Behavior | Strength |
|---|---|---|
| empty / unset | CLI runs with no Bash at all (`generate_with_search`) | **hard** — no execution mechanism exists |
| partial | Bash available; each tool checks the guild and rejects | **soft** — stops the model relaying a bad request, but Bash is not a sandbox |
| `*` | unrestricted | none |

- **[tools/_guild_access.py](tools/_guild_access.py)** — `require_integration("<name>")`
  at the top of each tool's `main()`. Reads the guild from the trusted
  `DISCORD_REQUESTING_GUILD_ID` env var (with the same `.request_context`
  fallback the Discord permission layer uses), never from model-supplied args.
- **Fails closed**: no verifiable guild → denied.
- **Scheduler exemption**: `run_due.py` sets `DISCORD_SYSTEM_CONTEXT=1`, so
  cron-launched tasks (reminders calling `send_message.py`) are not guild-scoped
  and keep working. `summarize.py` and `run_due.py` are ungated for the same
  reason.
- The system prompt states what's allowed in the current guild, so the model
  doesn't attempt denied tools — but that is guidance, not the boundary.

### Access control

Access control is **enforced in code**, not just by prompt. The bot injects the *triggering* Discord user's identity into every tool subprocess as the `DISCORD_REQUESTING_USER_ID` / `DISCORD_REQUESTING_GUILD_ID` env vars (for the persistent PTY session, whose env is fixed at startup, it is written to `tools/discord/.request_context` instead). Tools read this trusted identity — not the `--user-id` the model passes — so a user cannot act beyond their own permissions even if the model is convinced to try.

- **Discord tools** ([tools/discord/_permissions.py](tools/discord/_permissions.py)): Every Discord tool calls `require_permission(...)`, which resolves the requesting user's Discord roles/permissions in the guild (with owner + Administrator bypass and channel-overwrite handling) and **hard-rejects** if they lack the permission the action needs. E.g. `delete_channel.py` requires Manage Channels; if the requester lacks it, the tool exits with a permission-denied error and never calls Discord. Mapping of tool → required permission lives in each tool (e.g. delete/create/rename channel → Manage Channels; timeout → Moderate Members; add/remove role → Manage Roles; delete/pin/edit message → Manage Messages; read tools → View Channel;
`delete_emoji.py` → Manage Expressions, i.e. Manage Emojis and Stickers — the
same permission bit under both of Discord's names).
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

### DoorDash (`tools/doordash/`)
Thin subprocess wrappers around the installed `dd-cli` binary (v0.2.2,
`~/.local/bin/dd-cli`). No DoorDash API is reimplemented here — if the binary is
missing the tools exit 1 with `{"error": "dd-cli not installed"}`.

| Tool | Description |
|------|-------------|
| `search.py QUERY --intent TEXT [--limit N] [--lat F] [--lng F]` | Find nearby **restaurants**. Entry point; returns `stores[].store_id` |
| `find_nearby_stores.py [--vertical grocery\|alcohol\|convenience\|pets\|retail\|nv] [--max N] [--lat F] [--lng F] --intent TEXT` | **Non-restaurant** discovery entry point (default vertical `grocery`, default max 10). Fixed 16-mile radius; falls back to the default saved address when lat/lng are omitted. `stores[].store_id` feeds `find_items.py` |
| `store_details.py --store-id ID --intent TEXT` | Store name, image, business metadata — and the **only** source of `printable_address` (read-only) |
| `menu.py --store-id ID --intent TEXT` | Show a restaurant's menu (`menu_id` + `items[].item_id`) |
| `find_items.py --store-id ID QUERY [QUERY ...] --intent TEXT` | Search items in a **retail/grocery** store (empty for restaurants). Repeatable query |
| `item_details.py --kind restaurant\|retail --store-id ID --item-id ID [--menu-id ID] --intent TEXT` | Item pricing/description/customizations. `--kind restaurant` also needs `--menu-id` |
| `cart.py add --store-id ID --menu-id ID --items-json JSON [--cart-uuid U] [--fulfillment delivery\|pickup] [--group-cart] [--spend-limit-cents N] --intent TEXT` | Add items (`dd-cli cart add-items`). Appends to an existing open cart at that store unless `--cart-uuid` is given. `--spend-limit-cents` is the per-participant limit on a new host-pays-all group cart: it requires `--group-cart` and is rejected with `--cart-uuid` (validated in the wrapper) |
| `cart.py remove --cart-uuid U --cart-item-id ID --intent TEXT` | Drop one line from a cart, keeping the cart (`dd-cli cart remove-item`). `--cart-item-id` is the cart **LINE** id from `cart.py show` `items[].id`, **not** the menu `item_id` — call `cart.py show` first |
| `cart.py show \| clear --cart-uuid U --intent TEXT` | Show contents (no pricing) / empty and abandon (`cart show` / `cart delete`) |
| `cart.py list [--store-id ID] --intent TEXT` | List open carts |
| `order.py preview --cart-uuid U [--scheduled-time ISO8601] [--fulfillment delivery\|pickup] [--priority] [--no-apply-credits] --intent TEXT` | **Price the cart — no charge.** The only source of the real total (fees, tax, delivery) |
| `order.py place --cart-uuid U --confirm [--tip-cents N] [--scheduled-time ISO8601] [--fulfillment delivery\|pickup] [--priority] [--no-apply-credits] --intent TEXT` | **Submits the order. Spends real money, irreversible** (`dd-cli order submit`) |
| `order.py reorder --order-uuid U --intent TEXT` | Build a **new cart** from a past order, modifiers included. No charge; returns a `cart_uuid` for `preview` |
| `order.py checkout-url --cart-uuid U --intent TEXT` | Browser checkout link (read-only). Fallback for edits the CLI can't express: payment method, delivery address, in-browser tip |
| `order.py history [--max N] [--days N] --intent TEXT` | Recent order history |
| `order.py receipt --order-uuid U --intent TEXT` | Full itemized receipt for a past order, **including modifiers/options** (read-only) |
| `order.py status --order-uuid U --intent TEXT` | Whether a submitted order went through |
| `promo.py list --store-id ID --intent TEXT` | Campaign promos eligible at a store (read-only). Consumer- **and** store-scoped; an empty list is a normal answer, and it says nothing about which promos are on a given cart |
| `promo.py apply --cart-uuid U --promo-code CODE [--campaign-id ID] [--ad-group-id ID] [--ad-id ID] --intent TEXT` | Put a promo on a cart. Campaign promos need all four values from one `promo.py list` row; user-typed/referral codes need only `--promo-code` |
| `promo.py remove --cart-uuid U --promo-code CODE [--campaign-id ID] [--ad-group-id ID] [--ad-id ID] --intent TEXT` | Take a promo back off a cart. Pass the same flags `apply` was given |
| `address.py --intent TEXT [--set ADDRESS_ID]` | List saved addresses, or set the default (`address list` / `address set`) |
| `payment_methods.py --intent TEXT` | List saved cards (`payment-method list`) |

#### Cart item JSON schema (`--items-json`)

```json
[
  {
    "item_id": "i_21941681157",
    "item_name": "Avocado & Quinoa Superfood Ensalada",
    "quantity": 1,
    "nested_options": [
      {"id": "o_40817472508", "name": "Chipotle Vinaigrette (On the Side)", "quantity": 1,
       "options": [ {"id": "o_...", "name": "...", "quantity": 1} ]}
    ]
  }
]
```

- The key inside a `nested_options` entry is **`id`**, NOT `option_id`. Using
  `option_id` makes DoorDash reject the request with an *"option is nested at the
  wrong level"* error.
- A top-level `"options"` key on the item is **ignored** — modifiers must go
  under `nested_options`.
- Deeper combo/nested choices recurse via an `"options": [...]` array *inside* an
  option entry, using the same `{"id", "name", "quantity"}` shape.
- Prefixed ids (`i_` for items, `o_` for options) are correct — pass them
  verbatim as returned by `menu.py` / `item_details.py`, don't strip the prefix.
  **The prefix rule is asymmetric**: `cart.py add --items-json` wants the ids
  verbatim *with* their `i_`/`o_` prefixes, but `item_details.py --kind
  restaurant --item-id` wants the `i_` prefix **stripped** (`i_232…` → `232…`),
  because dd-cli's `restaurant-item-details` requires the bare numeric id.
- Items with required modifiers (size, dressing, …) will **fail to add** unless
  the required option ids are supplied. Get them from
  `item_details.py --kind restaurant --store-id ID --menu-id ID --item-id ID`.
- `cart.py add` appends to an existing open cart at that store unless
  `--cart-uuid` is given.

- **Owner-only, enforced in code.** `_dd.require_owner()` checks the trusted
  `DISCORD_REQUESTING_USER_ID` against `118567805678256128` and fails closed —
  same pattern as `tools/splitwise/_auth.py`. Also gated by the per-guild
  allowlist as the `doordash` integration, which is checked first.
- **`--intent` is mandatory on every tool.** dd-cli v0.2.2 requires it on every
  tool-backed command, so `run_dd` always appends it. It is a plain-language
  line about *who this is for and the goal* ("Summary: Help the user order
  lunch"), not a restatement of the command. DoorDash reviews this data.
- **`DD_CLI_ACCESS_TOKEN` is required.** dd-cli normally stores credentials in
  the OS keychain, which does not exist under WSL. Get a token by running
  `dd-cli export-token` on a desktop machine and set it in `.env`. Without it
  every command fails; `_dd.py` translates that into
  `{"error": "not_authenticated"}`, and a waitlisted account into
  `{"error": "no_access"}`. Relay either plainly and do not retry.
- **`order.py history` omits modifiers.** It returns only top-level items and
  does **NOT** include modifiers/customizations. To see what options were on a
  past order (e.g. whether chicken was added to a salad), use `order.py
  receipt`. Modifier data lives at
  `structuredContent.orders[].order_items[].options[].item_extra_option.{name, price_monetary_fields.display_string}`,
  with the base item at `order_items[].item.name`. One order per call — there is
  no batch mode.
- **`order.py reorder` is the fastest correct way to repeat a past order**,
  including all its modifiers — prefer it over rebuilding a cart by hand from
  `order.py history`, which omits modifiers entirely. It charges nothing, so run
  it immediately; feed the returned `cart_uuid` into `order.py preview`. Not
  every order is reorderable (check `success: false` + `fail_reason`), and the
  new cart **inherits the original order's fulfillment mode** — reordering a past
  pickup order silently produces a pickup cart, so confirm with `preview`.
- **Quote-affecting flags must be repeated on `place`.** Anything passed to
  `order.py preview` — `--scheduled-time`, `--fulfillment`, `--priority`,
  `--no-apply-credits` — has to be passed identically to `order.py place`, or the
  amount charged won't match the total the user approved. `--priority` is
  delivery-only and can't combine with `pickup` or `--scheduled-time` (the
  wrapper rejects both combinations).
- **`order.py place` is the one money-spending tool.** It refuses to run without
  `--confirm`, which asserts the user was shown the actual items and the actual
  `order.py preview` total and said yes to *that*. "Order me a salad" authorizes
  building and quoting a cart, not buying it. Everything else in this
  integration is read-only or reversible — run those immediately.
- `place` and `address --set` pass dd-cli's `-y` internally: these run as
  captured subprocesses with no tty, so dd-cli's own interactive prompt would
  hang until the 120s timeout. The `--confirm` gate is the real check.
- **Location**: dd-cli falls back to `DD_LAT`/`DD_LNG` and finally a Cupertino
  default. Prefer resolving coordinates first — `address.py` lists saved
  addresses; the `is_default` entry's lat/lng is "near me".

**Workflow example:** "order me a salad"
1. `address.py --intent "..."` → default address lat/lng
2. `search.py salad --lat ... --lng ... --intent "..."` → pick a store_id
3. `menu.py --store-id <id> --intent "..."` → menu_id + item_id
4. `cart.py add --store-id <id> --menu-id <mid> --items-json '[...]' --intent "..."` → cart_uuid
5. `order.py preview --cart-uuid <uuid> --intent "..."` → **show the user the items and total**
6. only after they say yes: `order.py place --cart-uuid <uuid> --confirm --intent "..."`

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
| `list_emojis.py --guild-id ID` | List server custom emojis (name, id, animated, mention form) |
| `create_emoji.py --guild-id ID --name NAME (--url URL \| --file PATH) [--roles ID ...]` | Upload a custom emoji. Requires Manage Expressions. png/jpeg/gif, ≤256KB; webp is converted to png |
| `delete_emoji.py --guild-id ID (--emoji-id ID \| --name NAME)` | Delete one custom emoji (irreversible). `--emoji-id` accepts `<:name:id>` |
| `emoji_stats.py --guild-id ID [--source message\|reaction\|all] [--unused] [--limit N] [--backfill]` | Emoji usage counts from the `emoji_usage` counter table (name, id, animated, count, last_used_at). `--unused` lists guild emojis with a zero count; `--backfill` recounts message usage from stored history (idempotent) |
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
