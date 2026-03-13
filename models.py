"""Data models and enums for Discord LLM Bot"""

from dataclasses import dataclass
from enum import Enum


class Txt2ImgModel(Enum):
    PONY = "aponyDiffusionV6XL_v6StartWithThisOne"
    CYBERREALISTIC_PONY = "cyberrealisticPony_v110"
    PONY_REALISM = "ponyRealism_V23"
    FLUX = "flux_dev"
    FLAX = "plantMilkModelSuite_flax"
    HEMP2 = "plantMilkModelSuite_hempII"
    WALNUT = "plantMilkModelSuite_walnut"


class Txt2TxtModel(Enum):
    GEMMA3_27B = "gemma3:27b"
    DEEPSEEK_R1_70B = "deepseek-r1:70b"
    GEMMA3_27B_ABLITERATED = "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q8_0"
    QWEN_72B = "qwen2.5vl:72b"
    DEEPSEX = "hf.co/TheBloke/deepsex-34b-GGUF:Q8_0"
    GPT_OSS = "gpt-oss:120b"
    GPT_ABLITERATED = "huihui_ai/gpt-oss-abliterated:120b"
    GPT_OSS_20b = "gpt-oss:20b"
    QWEN3_VL = "qwen3-vl:32b"
    QWEN35_35B = "qwen3.5:35b-a3b-q8_0"
    # Claude Code CLI backends
    CLAUDE_CODE = "claude-code"
    CLAUDE_CODE_OPUS = "claude-code-opus"
    # Ollama models routed through Claude Code CLI
    CLAUDE_CODE_QWEN35 = "cc-qwen3.5:35b-a3b-q8_0"


# All models that route through the Claude Code CLI (Anthropic or Ollama-backed)
CLAUDE_CODE_MODELS = {
    Txt2TxtModel.CLAUDE_CODE.value,
    Txt2TxtModel.CLAUDE_CODE_OPUS.value,
    Txt2TxtModel.CLAUDE_CODE_QWEN35.value,
}

# Models that are Anthropic Claude (for response splitting, search behavior)
_ANTHROPIC_MODELS = {Txt2TxtModel.CLAUDE_CODE.value, Txt2TxtModel.CLAUDE_CODE_OPUS.value}


def is_claude_code_model(model_value: str) -> bool:
    """Check if a model routes through the Claude Code CLI."""
    return model_value in CLAUDE_CODE_MODELS


def is_anthropic_model(model_value: str) -> bool:
    """Check if a model is an Anthropic Claude model (not Ollama-via-CC)."""
    return model_value in _ANTHROPIC_MODELS


@dataclass
class ImageInfo:
    sampler_name: str
    steps: int
    cfg_scale: float
    width: int
    height: int
    seed: int
