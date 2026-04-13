"""Image generation module for Discord LLM Bot.

Uses Flux2 Klein 9B for native text-to-image and image-to-image generation.
"""

import logging
import re
import time
from typing import Optional, Tuple

from PIL import Image

from models import ImageInfo
from ollama_client import OllamaClient
from flux_client import FluxClient
from utils import encode_image_downsized_to_base64

logger = logging.getLogger("imagegen")


_HQ_RE = re.compile(
    r'\b(hq|hi[-\s]?res|hi[-\s]?quality|high[-\s]?res(?:olution)?|'
    r'high[-\s]?quality|ultra[-\s]?hd|4k|8k|ultra[-\s]?detail(?:ed)?|'
    r'detailed|masterpiece)\b',
    re.IGNORECASE,
)

_PORTRAIT_RE = re.compile(
    r'\b(portrait|headshot|selfie|character|person|man|woman|girl|boy|guy|lady|'
    r'face|full[-\s]?body|fullbody|profile\s*pic|vertical|tall[-\s]?shot|'
    r'standing)\b',
    re.IGNORECASE,
)

_LANDSCAPE_RE = re.compile(
    r'\b(landscape|scenery|vista|panorama|panoramic|horizon|cityscape|seascape|'
    r'skyline|wide[-\s]?shot|wide[-\s]?angle|horizontal|aerial|'
    r'establishing\s*shot)\b',
    re.IGNORECASE,
)

# (orientation, hq) -> (width, height). Flux2 Klein tolerates up to 1536/side.
DIMENSION_PRESETS = {
    ("square",    False): (1024, 1024),
    ("square",    True):  (1536, 1536),
    ("portrait",  False): (832,  1216),
    ("portrait",  True):  (1024, 1536),
    ("landscape", False): (1216, 832),
    ("landscape", True):  (1536, 1024),
}


def _detect_orientation(text: str) -> Optional[str]:
    """Return 'portrait', 'landscape', or None if both or neither match."""
    p = bool(_PORTRAIT_RE.search(text))
    l = bool(_LANDSCAPE_RE.search(text))
    if p and not l:
        return "portrait"
    if l and not p:
        return "landscape"
    return None


def choose_dimensions(text: str) -> Tuple[int, int]:
    """Pick (width, height) from keywords in the user's request.
    Square 1024x1024 is the default when orientation is ambiguous."""
    hq = bool(_HQ_RE.search(text))
    orientation = _detect_orientation(text) or "square"
    return DIMENSION_PRESETS[(orientation, hq)]


def classify_dimensions(width: int, height: int) -> Tuple[str, bool]:
    """Reverse a (w, h) pair into (orientation, hq) for inheritance logic."""
    aspect = width / height if height else 1.0
    if aspect < 0.85:
        orientation = "portrait"
    elif aspect > 1.18:
        orientation = "landscape"
    else:
        orientation = "square"
    hq = max(width, height) >= 1536
    return orientation, hq


def choose_followup_dimensions(
    text: str, prev_width: int, prev_height: int
) -> Tuple[int, int]:
    """For follow-up edits: inherit prev dims; override only when the user
    explicitly asks for a different orientation or HQ in their follow-up."""
    prev_orientation, prev_hq = classify_dimensions(prev_width, prev_height)
    orientation = _detect_orientation(text) or prev_orientation
    hq = bool(_HQ_RE.search(text)) or prev_hq
    return DIMENSION_PRESETS[(orientation, hq)]


def choose_source_dimensions(
    text: str, source_width: int, source_height: int
) -> Tuple[int, int]:
    """For user-attached img2img: match the source's aspect ratio / HQ status,
    with user-text keywords as overrides."""
    source_orientation, source_hq = classify_dimensions(source_width, source_height)
    orientation = _detect_orientation(text) or source_orientation
    hq = bool(_HQ_RE.search(text)) or source_hq
    return DIMENSION_PRESETS[(orientation, hq)]


class ImageGenerator:
    """Handles image generation and editing tasks."""

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.flux_client = FluxClient()

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,  # unused by Flux, kept for compat
        seed: int = -1,
        width: int = 1024,
        height: int = 1024,
        cfg_scale: float = 1.0,
        steps: int = 4,
        **_kwargs,
    ) -> Tuple[str, ImageInfo, bool]:
        """Generate a new image from text.

        Returns (file_path, ImageInfo, is_nsfw).
        """
        _ = negative_prompt  # Flux2 Klein doesn't use negative prompts
        logger.info(
            "generate_image seed=%s %dx%d steps=%d cfg=%s prompt_len=%d",
            seed, width, height, steps, cfg_scale, len(prompt),
        )
        t0 = time.perf_counter()
        file_path, image_info = await self.flux_client.generate(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=cfg_scale,
        )
        logger.info("flux gen done in %.2fs -> %s", time.perf_counter() - t0, file_path)

        # NSFW check (downsampled to keep vision token count small)
        t1 = time.perf_counter()
        image_base64 = encode_image_downsized_to_base64(file_path, max_side=512)
        logger.info("nsfw check: downsized base64 len=%d bytes", len(image_base64))
        is_nsfw = await self.ollama_client.classify_nsfw([image_base64])
        logger.info(
            "nsfw classification done in %.2fs: %s",
            time.perf_counter() - t1, "NSFW" if is_nsfw else "SFW",
        )

        return file_path, image_info, is_nsfw

    async def edit_image(
        self,
        prompt: str,
        image_path: str,
        seed: int = -1,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        cfg_scale: float = 1.0,
    ) -> Tuple[str, ImageInfo, bool]:
        """Edit an existing image based on a text prompt.

        Flux2 Klein uses the source image as a reference/condition rather
        than a noisy init, so there is no strength knob — the pipeline
        decides divergence based on the prompt.

        Returns (file_path, ImageInfo, is_nsfw).
        """
        logger.info(
            "edit_image source=%s seed=%s %dx%d steps=%d cfg=%s prompt_len=%d",
            image_path, seed, width, height, steps, cfg_scale, len(prompt),
        )
        source_image = Image.open(image_path).convert("RGB")
        logger.debug("source image opened: %dx%d mode=%s",
                     *source_image.size, source_image.mode)

        t0 = time.perf_counter()
        file_path, image_info = await self.flux_client.edit(
            prompt=prompt,
            image=source_image,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=cfg_scale,
        )
        logger.info("flux edit done in %.2fs -> %s", time.perf_counter() - t0, file_path)

        # NSFW check (downsampled to keep vision token count small)
        t1 = time.perf_counter()
        image_base64 = encode_image_downsized_to_base64(file_path, max_side=512)
        logger.info("nsfw check: downsized base64 len=%d bytes", len(image_base64))
        is_nsfw = await self.ollama_client.classify_nsfw([image_base64])
        logger.info(
            "nsfw classification done in %.2fs: %s",
            time.perf_counter() - t1, "NSFW" if is_nsfw else "SFW",
        )

        return file_path, image_info, is_nsfw

    async def is_image_generation_task(self, prompt: str) -> bool:
        """Check if a prompt is requesting image generation."""
        logger.info("classify_image_task prompt_len=%d", len(prompt))
        t0 = time.perf_counter()
        result = await self.ollama_client.classify_image_task(prompt)
        logger.info("classify_image_task result=%s in %.2fs",
                    result, time.perf_counter() - t0)
        return result

    async def generate_image_prompt(self, user_prompt: str) -> str:
        """Generate an optimized prompt for image generation."""
        logger.info("generate_image_prompt user_prompt_len=%d", len(user_prompt))
        t0 = time.perf_counter()
        result = await self.ollama_client.generate_image_prompt(user_prompt)
        logger.info(
            "generate_image_prompt -> rewritten_len=%d in %.2fs",
            len(result), time.perf_counter() - t0,
        )
        return result
