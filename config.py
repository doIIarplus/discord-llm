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
NSFW_CLASSIFICATION_MODEL = os.getenv("NSFW_CLASSIFICATION_MODEL", "qwen3-vl:32b")
CHAT_MODEL = Txt2TxtModel.GEMMA3_27B.value
SEARCH_UTILITY_MODEL = Txt2TxtModel.GEMMA3_27B.value
SEARCH_SUMMARIZATION_MODEL = Txt2TxtModel.QWEN3_VL.value
TEXT_TO_IMAGE_MODEL = "..."
TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL = os.getenv(
    "TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL",
    "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q8_0",
)
logger.info("Model configurations loaded")

# Stable Diffusion Configuration
SD_API_URL = os.getenv("SD_API_URL", "http://127.0.0.1:7860")
logger.info(f"Stable Diffusion configuration loaded. API URL: {SD_API_URL}")

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
