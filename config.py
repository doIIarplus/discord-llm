"""Configuration settings for Discord LLM Bot"""

import os
import logging
from dotenv import load_dotenv

from models import Txt2TxtModel

# Root logger: INFO by default. Our own modules (below) go to DEBUG so the
# image-gen pipeline is fully traced. Third-party libraries stay quieter.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set our own loggers to DEBUG for really verbose tracing.
for _name in ("Bot", "ollama", "flux", "imagegen"):
    logging.getLogger(_name).setLevel(logging.DEBUG)

# But route the debug output through a handler that doesn't filter it.
# basicConfig's root handler has level INFO, so we need to lower it (or add
# a dedicated handler). Simplest: drop the root handler level to DEBUG and
# silence noisy third-party libraries explicitly.
logging.getLogger().setLevel(logging.DEBUG)
for _noisy in (
    "discord", "discord.client", "discord.gateway", "discord.http",
    "aiohttp", "aiohttp.access", "aiohttp.client", "aiohttp.internal",
    "urllib3", "httpx", "httpcore", "chromadb", "sentence_transformers",
    "PIL", "asyncio",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Create a logger for the config module
logger = logging.getLogger("Config")

load_dotenv()

# Discord Configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
GUILD_ID = int(os.getenv("GUILD_ID", "363154169294618625"))
logger.info(f"Discord configuration loaded. Guild ID: {GUILD_ID}")

# Ollama Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:35b-a3b-q8_0")
logger.info(
    f"Ollama configuration loaded. API URL: {OLLAMA_API_URL}, Model: {OLLAMA_MODEL}"
)

# Model Configuration
IMAGE_RECOGNITION_MODEL = os.getenv("IMAGE_RECOGNITION_MODEL", "qwen3-vl:32b")
# NSFW classifier and edit-prompt writer both use gemma3:27b instead of
# qwen3-vl because qwen3-vl is a reasoning model — its hidden chain-of-
# thought tokens consume the num_predict budget, leaving nothing for the
# actual response. gemma3 is multimodal + non-reasoning so responses are
# fast (~0.3s) and deterministic.
NSFW_CLASSIFICATION_MODEL = os.getenv("NSFW_CLASSIFICATION_MODEL", "gemma3:27b")
IMAGE_EDIT_DESCRIPTION_MODEL = os.getenv("IMAGE_EDIT_DESCRIPTION_MODEL", "gemma3:27b")
CHAT_MODEL = Txt2TxtModel.GEMMA3_27B.value
SEARCH_UTILITY_MODEL = Txt2TxtModel.GEMMA3_27B.value
SEARCH_SUMMARIZATION_MODEL = Txt2TxtModel.QWEN3_VL.value
TEXT_TO_IMAGE_MODEL = "..."
TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL = os.getenv(
    "TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL",
    "gemma3:27b",
)
logger.info("Model configurations loaded")

# Flux2 Klein 9B Configuration
# Klein is a distilled/fast variant. The HF model card uses guidance_scale=1.0
# and num_inference_steps=4; those are the recommended defaults.
FLUX_MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
FLUX_DEFAULT_STEPS = 4
FLUX_DEFAULT_GUIDANCE = 1.0

# Project root (used to anchor all file paths)
PROJECT_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))

# File Paths — always relative to project root
FILE_INPUT_FOLDER = os.path.join(PROJECT_DIR, "multimodal_input")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "api_out")
OUTPUT_DIR_T2I = os.path.join(OUTPUT_DIR, "txt2img")
OUTPUT_DIR_I2I = os.path.join(OUTPUT_DIR, "img2img")

# Context Configuration
CONTEXT_LIMIT = 10
VISION_MODEL_CTX = 32768  # Cap context window for vision models to avoid OOM (default 256K is way too much)

# Discord Message Configuration
MAX_DISCORD_MESSAGE_LENGTH = 1900

# Memory / Chat History
MEMORY_SUMMARIZE_BATCH_SIZE = 200   # Max messages per summarization batch
MEMORY_MAX_PROFILE_EVENTS = 10      # Max recent events injected into prompt
MEMORY_IDLE_MINUTES = 10            # Summarize only after this many minutes of inactivity
MEMORY_MAX_PROFILE_CHARS = 2000     # Compact user profiles beyond this length
MEMORY_MAX_CHANNEL_CHARS = 1500     # Compact channel summaries beyond this length
MEMORY_MAX_EVENTS = 50              # Keep only the most recent N events per guild
# Channel allowlist for chat history recording. Empty list = record ALL channels.
# Set via comma-separated channel IDs in .env: MEMORY_CHANNEL_ALLOWLIST=123,456,789
_allowlist_raw = os.getenv("MEMORY_CHANNEL_ALLOWLIST", "")
MEMORY_CHANNEL_ALLOWLIST: set = set(
    cid.strip() for cid in _allowlist_raw.split(",") if cid.strip()
)

# Create output directories
os.makedirs(OUTPUT_DIR_T2I, exist_ok=True)
os.makedirs(OUTPUT_DIR_I2I, exist_ok=True)
logger.info(f"Output directories created: {OUTPUT_DIR_T2I}, {OUTPUT_DIR_I2I}")

# Tavily Web Search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Anthropic Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MONTHLY_BUDGET = float(os.getenv("CLAUDE_MONTHLY_BUDGET", "50.0"))

# Splitwise
SPLITWISE_API_KEY = os.getenv("SPLITWISE_API_KEY", "")

logger.info("Configuration module initialization complete")
