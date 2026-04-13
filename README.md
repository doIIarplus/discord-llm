# discord-llm

A Discord bot backed by locally-hosted LLMs (via Ollama) and a local Flux2 Klein 9B image pipeline (via `diffusers`). Responds to mentions and replies with chat, image analysis, image generation, image editing, web extraction, and optional RAG over a wiki dump.

## Capabilities

- **Chat** — multi-turn conversation with short, casual responses. Per-channel context capped at 10 messages.
- **Vision** — when a user attaches an image, the bot switches to a vision model (qwen3-vl:32b) to answer about what's in it.
- **Image generation** — "@bot generate an image of ..." triggers Flux2 Klein 9B. Dimension presets (square / portrait / landscape × default / HQ) are picked automatically from keywords in the request.
- **Image editing** — two flows, both via Flux2 Klein's native reference conditioning:
  - *Follow-up edit*: reply to a bot-generated image with "make her hair green" → the bot re-edits the previous image with the delta applied.
  - *Attached edit*: upload an image + "add a cowboy hat" → the bot analyzes the uploaded image with the vision model and generates an edited version.
- **Post-generation NSFW check** — outputs are classified by the vision model and spoilered if flagged.
- **Web content extraction** — URLs in messages are fetched, stripped to article text via `trafilatura`, and included as context.
- **Web search** — `/search <query>` via the Tavily API, summarized by the LLM.
- **Chat history + memory** — every message is recorded to `chat_history.db` and periodically summarized into per-user profiles and per-server event logs.
- **Plugins** — hot-swappable handlers under `plugins/` can intercept messages before the main LLM flow.
- **Claude Code integration** — the bot can route to the Claude Code CLI as an alternate backend (see `commands.py` → `/set_model`).

See [CLAUDE.md](CLAUDE.md) for the full module-by-module architecture.

## Requirements

- **Python 3.12** (tested; other versions may work)
- **NVIDIA GPU** with enough VRAM for Flux2 Klein 9B (~18 GB minimum, 96 GB recommended if you want to keep gemma3:27b + qwen3-vl:32b + Flux all resident)
- **CUDA toolkit** compatible with your torch build (we use `cu129` wheels for PyTorch 2.11)
- **Ollama** running locally at `http://localhost:11434`
- **WSL2 or native Linux** (the bot has been run on both; WSL2 disk I/O is slower which affects cold-load times)
- Roughly **200 GB of free disk** for the local models (Flux2 Klein ≈ 33 GB, each Ollama model ≈ 17-30 GB)

### Ollama configuration

Put this in `/etc/systemd/system/ollama.service.d/override.conf` so Ollama can hold multiple large models simultaneously without thrashing KV caches:

```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=4"
Environment="OLLAMA_KEEP_ALIVE=-1"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

### Ollama models to pull

```bash
ollama pull gemma3:27b           # chat + prompt rewriter + image-gen classifier
ollama pull qwen3-vl:32b          # vision (image recognition, NSFW, edit-prompt generation)
```

Optional (for the abliterated prompt rewriter, which is less likely to refuse edgy prompts):

```bash
ollama pull hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q8_0
```

### Python environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` is installed from PyTorch's own index to match the system CUDA version (see `requirements.txt` comments if you hit a driver mismatch).

### `.env` configuration

```
DISCORD_BOT_TOKEN=...
GUILD_ID=363154169294618625

# Ollama
OLLAMA_API_URL=http://localhost:11434/api/generate
IMAGE_RECOGNITION_MODEL=qwen3-vl:32b
NSFW_CLASSIFICATION_MODEL=qwen3-vl:32b
TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL=gemma3:27b

# Flux (defaults shown)
FLUX_MODEL_ID=black-forest-labs/FLUX.2-klein-9B

# Third-party APIs (optional)
TAVILY_API_KEY=tvly-...
ANTHROPIC_API_KEY=...
SPLITWISE_API_KEY=...

# File I/O
FILE_INPUT_FOLDER=/home/dollarplus/projects/discord_llm_bot/multimodal_input/
```

## Running

```bash
source venv/bin/activate
python bot.py
```

On first launch the bot will spend **~5 minutes** preloading Flux2 Klein, gemma3:27b, and qwen3-vl:32b in the background. Chat replies work immediately — only image generation is blocked on the preload completing. The cold-load cost is dominated by reading 33 GB of Flux weights from disk (especially slow on WSL2 VHDX filesystems).

After preload, per-request image gen/edit latency is **6-9 seconds warm** across all scenarios.

## Slash commands

Registered in [commands.py](commands.py). After adding or changing a command, run `/sync_commands` once from Discord to push the updates.

| Command | Description |
|---|---|
| `/clear` | Clear the channel's in-memory context |
| `/ask <question>` | One-shot question with no context |
| `/set_system_prompt` / `/reset_system_prompt` / `/get_system_prompt` | Manage the system prompt |
| `/set_model <model>` / `/get_model` | Switch the active chat backend |
| `/plugins` / `/reload_plugin` / `/load_plugin` / `/unload_plugin` | Manage runtime plugins |
| `/purge` | Delete all messages in the current channel |
| `/sync_commands` | Re-register slash commands with Discord |

## Testing

The bot flow can be exercised without Discord via the interactive test CLI:

```bash
source venv/bin/activate
python test_cli.py
```

Type `/help` once it's running to see all the simulated commands.

## Optimizing / profiling

Two developer-focused tools sit alongside the bot:

- [`benchmark_image_pipeline.py`](benchmark_image_pipeline.py) — exercises the bot's exact image-gen flow for three scenarios (fresh generation, attached-image edit, follow-up edit) and records per-stage wall-clock timings plus 5 Hz system metrics (CPU, RAM, disk I/O, GPU util, VRAM, power, temp) to JSONL. Use `--preload` to separate startup cost from per-request latency.

  ```bash
  python benchmark_image_pipeline.py --scenarios A B C --runs 5 --preload --label baseline
  ```

- [`tools/flux/generate.py`](tools/flux/generate.py) — standalone Flux2 Klein CLI for quick prompt testing and LoRA experimentation. Supports dimension presets, img2img, and stacked LoRAs from local files or URLs.

  ```bash
  python tools/flux/generate.py "an astronaut riding a horse on mars" --preset landscape-hq
  python tools/flux/generate.py "my character" --lora ./char.safetensors --lora-weight 0.9
  python tools/flux/generate.py "edit this" --image source.png --preset portrait
  ```

  Note: Flux1-trained LoRAs from Civitai will **not** load on Flux2 Klein — the architectures differ. Only use LoRAs explicitly trained for Flux2.

## CLI tools for the bot to invoke

The bot can call standalone scripts under `tools/` as a form of tool use. Each writes JSON to stdout and errors to stderr. See [CLAUDE.md](CLAUDE.md) for the full catalogue (Splitwise, Discord message/role operations, web search, RAG, scheduler, memory summarizer, etc.).

## Architecture overview

| Module | Purpose |
|---|---|
| [bot.py](bot.py) | Entry point. Handles `on_message`, orchestrates the image-gen pipeline and chat reply paths |
| [ollama_client.py](ollama_client.py) | Async HTTP client to Ollama. Classifier, prompt rewriter, NSFW classifier, edit-prompt builder |
| [flux_client.py](flux_client.py) | `Flux2KleinPipeline` wrapper with lazy load + warmup |
| [image_generation.py](image_generation.py) | Glue between `OllamaClient` and `FluxClient`. Dimension presets live here |
| [commands.py](commands.py) | Slash command registration |
| [chat_history.py](chat_history.py) | SQLite-backed message recording + memory system |
| [web_extractor.py](web_extractor.py) | Async URL extraction via `trafilatura` + web search via Tavily |
| [plugin_manager.py](plugin_manager.py) | Hot-swappable plugin dispatch |

## Troubleshooting

- **First image request is slow** — the background preload finished while you weren't looking. Check `bot.log` for `[preload] image pipeline ready in ...s`. If not present yet, wait.
- **"image generation failed: RuntimeError: Ollama API error"** — a required model isn't pulled. Run `ollama list` and compare against the model pulls above.
- **Flux cold loads repeatedly** — you're running with multiple bot processes or the pipeline is being unloaded externally. `FluxClient.UNLOAD_TIMEOUT` is pinned to 0; if it's unloading, something else is releasing the reference.
- **Ollama models keep thrashing in/out of VRAM** — check the systemd override is in place: `systemctl show ollama --property=Environment | grep OLLAMA_MAX_LOADED_MODELS`.
