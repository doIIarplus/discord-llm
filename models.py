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


@dataclass
class ImageInfo:
    sampler_name: str
    steps: int
    cfg_scale: float
    width: int
    height: int
    seed: int