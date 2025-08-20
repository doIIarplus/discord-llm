"""Configuration settings for Discord LLM Bot"""

import os
from dotenv import load_dotenv

load_dotenv()

# Discord Configuration
DISCORD_BOT_TOKEN = os.getenv(
    "DISCORD_BOT_TOKEN",
    ""
)
GUILD_ID = 363154169294618625

# Ollama Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:72b")

# Model Configuration
IMAGE_RECOGNITION_MODEL = os.getenv("IMAGE_RECOGNITION_MODEL", "qwen2.5vl:72b")
NSFW_CLASSIFICATION_MODEL = os.getenv("NSFW_CLASSIFICATION_MODEL", "qwen2.5vl:7b")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss:120b")
TEXT_TO_IMAGE_MODEL = "..."
TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL = os.getenv(
    "TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL",
    "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q8_0"
)

# Stable Diffusion Configuration
SD_API_URL = os.getenv("SD_API_URL", "http://127.0.0.1:7860")

# File Paths
FILE_INPUT_FOLDER = os.getenv(
    "FILE_INPUT_FOLDER",
    "/home/dollarplus/projects/discord_llm_bot/multimodal_input/"
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