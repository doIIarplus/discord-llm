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


_MENTION_RE = re.compile(r'<@!?\d+>|<#\d+>|<@&\d+>')
_URL_RE = re.compile(r'https?://\S+')


def clean_edit_instruction(user_text: str) -> str:
    """Strip Discord mentions, URLs, and excess whitespace from a user's
    raw message so it can be passed directly to Flux as an edit prompt.

    Flux2 Klein uses the source image as a reference/condition, so the
    edit prompt should describe ONLY what to change — not re-describe
    what's already in the image. Passing the user's cleaned-up text
    directly preserves identity better than VLM-generated descriptions
    that tend to hallucinate details.
    """
    s = _MENTION_RE.sub('', user_text)
    s = _URL_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


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
    """For user-attached img2img: stay close to the source's dimensions.

    Preserves the source's aspect ratio and (by default) size. Only scales
    up to an HQ preset if the user explicitly asks for higher resolution —
    because every pixel of upscale forces Flux to hallucinate content and
    drift from the source identity.
    """
    hq_requested = bool(_HQ_RE.search(text))
    if hq_requested:
        # Explicit HQ ask: upscale to the matching preset
        source_orientation, _ = classify_dimensions(source_width, source_height)
        orientation = _detect_orientation(text) or source_orientation
        return DIMENSION_PRESETS[(orientation, True)]

    # Default: keep source size, snapped to multiples of 64 and clamped
    # into Flux's valid range [256, 1536]. Flux produces weak results
    # below ~512 on the shortest side, so bump up if the source is tiny.
    aspect = source_width / source_height if source_height else 1.0
    shortest = min(source_width, source_height)
    if shortest < 512:
        scale = 512 / shortest
        target_w = int(round(source_width * scale))
        target_h = int(round(source_height * scale))
    else:
        target_w, target_h = source_width, source_height

    target_w = (min(1536, max(256, target_w)) // 64) * 64
    target_h = (min(1536, max(256, target_h)) // 64) * 64
    return target_w, target_h


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
        cfg_scale: float = 5.0,
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
