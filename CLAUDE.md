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

## File I/O

- Uploaded attachments temporarily saved to `multimodal_input/`, deleted after processing
- Generated images saved to `api_out/txt2img/` and `api_out/img2img/`
- LaTeX renders saved to `latex_images/` (if latex rendering is re-enabled)
- RAG vector DB: `chroma_db/`
