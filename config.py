"""Configuration settings for Discord LLM Bot"""

import os
import logging
from dotenv import load_dotenv

from models import Txt2TxtModel

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger for the config module
logger = logging.getLogger("Config")

load_dotenv()

# Discord Configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
GUILD_ID = 363154169294618625
logger.info(f"Discord configuration loaded. Guild ID: {GUILD_ID}")

# Ollama Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:72b")
logger.info(
    f"Ollama configuration loaded. API URL: {OLLAMA_API_URL}, Model: {OLLAMA_MODEL}"
)

# Model Configuration
IMAGE_RECOGNITION_MODEL = os.getenv("IMAGE_RECOGNITION_MODEL", "qwen3-vl:32b")
NSFW_CLASSIFICATION_MODEL = os.getenv("NSFW_CLASSIFICATION_MODEL", "qwen3-vl:32b")
CHAT_MODEL = Txt2TxtModel.GEMMA3_27B_ABLITERATED.value
TEXT_TO_IMAGE_MODEL = "..."
TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL = os.getenv(
    "TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL",
    "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q8_0",
)
logger.info("Model configurations loaded")

# Stable Diffusion Configuration
SD_API_URL = os.getenv("SD_API_URL", "http://127.0.0.1:7860")
logger.info(f"Stable Diffusion configuration loaded. API URL: {SD_API_URL}")

# File Paths
FILE_INPUT_FOLDER = os.getenv(
    "FILE_INPUT_FOLDER", "/home/dollarplus/projects/discord_llm_bot/multimodal_input/"
)
OUTPUT_DIR = "api_out"
OUTPUT_DIR_T2I = os.path.join(OUTPUT_DIR, "txt2img")
OUTPUT_DIR_I2I = os.path.join(OUTPUT_DIR, "img2img")

# Context Configuration
CONTEXT_LIMIT = 10

# Discord Message Configuration
MAX_DISCORD_MESSAGE_LENGTH = 1900

# Create output directories
os.makedirs(OUTPUT_DIR_T2I, exist_ok=True)
os.makedirs(OUTPUT_DIR_I2I, exist_ok=True)
logger.info(f"Output directories created: {OUTPUT_DIR_T2I}, {OUTPUT_DIR_I2I}")

# Bot personality for autonomous response decisions
BOT_PERSONALITY = os.getenv("BOT_PERSONALITY", """
- Helpful and knowledgeable about technology, programming, and AI
- Enjoys discussing interesting technical problems
- Will offer assistance when someone seems stuck
- Friendly but not overly chatty
""".strip())

# Number of recent messages to consider for autonomous response context
AUTONOMOUS_CONTEXT_MESSAGES = int(os.getenv("AUTONOMOUS_CONTEXT_MESSAGES", "10"))

# Channels where autonomous responses are enabled (empty list = all channels)
# Format: comma-separated channel IDs, e.g., "123456789,987654321"
_autonomous_channels_str = os.getenv(
    "AUTONOMOUS_RESPONSE_CHANNELS",
    "1381051356894334999,1171545202486431745"
)
AUTONOMOUS_RESPONSE_CHANNELS = (
    [int(ch.strip()) for ch in _autonomous_channels_str.split(",") if ch.strip()]
    if _autonomous_channels_str
    else []
)

logger.info("Configuration module initialization complete")
