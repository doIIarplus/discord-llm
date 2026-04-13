"""Image generation module for Discord LLM Bot.

Uses Flux2 Klein 9B for native text-to-image and image-to-image generation.
"""

from typing import Optional, Tuple

from PIL import Image

from models import ImageInfo
from ollama_client import OllamaClient
from flux_client import FluxClient
from utils import encode_image_to_base64


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
        cfg_scale: float = 3.0,
        steps: int = 4,
        **_kwargs,
    ) -> Tuple[str, ImageInfo, bool]:
        """Generate a new image from text.

        Returns (file_path, ImageInfo, is_nsfw).
        """
        _ = negative_prompt  # Flux2 Klein doesn't use negative prompts
        file_path, image_info = await self.flux_client.generate(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=cfg_scale,
        )

        # NSFW check
        image_base64 = encode_image_to_base64(file_path)
        is_nsfw = await self.ollama_client.classify_nsfw([image_base64])

        return file_path, image_info, is_nsfw

    async def edit_image(
        self,
        prompt: str,
        image_path: str,
        seed: int = -1,
        strength: float = 0.7,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        cfg_scale: float = 3.0,
    ) -> Tuple[str, ImageInfo, bool]:
        """Edit an existing image based on a text prompt.

        Args:
            prompt: What the edited image should look like.
            image_path: Path to the source image.
            seed: Random seed (-1 for random).
            strength: How much to change (0.0 = no change, 1.0 = complete redraw).

        Returns (file_path, ImageInfo, is_nsfw).
        """
        source_image = Image.open(image_path).convert("RGB")

        file_path, image_info = await self.flux_client.edit(
            prompt=prompt,
            image=source_image,
            seed=seed,
            strength=strength,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=cfg_scale,
        )

        # NSFW check
        image_base64 = encode_image_to_base64(file_path)
        is_nsfw = await self.ollama_client.classify_nsfw([image_base64])

        return file_path, image_info, is_nsfw

    async def is_image_generation_task(self, prompt: str) -> bool:
        """Check if a prompt is requesting image generation."""
        return await self.ollama_client.classify_image_task(prompt)

    async def generate_image_prompt(self, user_prompt: str) -> str:
        """Generate an optimized prompt for image generation."""
        return await self.ollama_client.generate_image_prompt(user_prompt)
