"""Image generation plugin — /generate_image slash command."""

import logging
from typing import Optional

import discord
from discord import app_commands

from plugin_base import BasePlugin

logger = logging.getLogger("Plugin.image_gen")


class ImageGenPlugin(BasePlugin):
    name = "image_gen"
    version = "1.0.0"
    description = "Generate images via Stable Diffusion with /generate_image"

    async def on_load(self):
        from image_generation import ImageGenerator
        self._image_gen = ImageGenerator()

        self.register_slash_command(
            name="generate_image",
            description="Generate an image using Stable Diffusion",
            callback=self._generate_image_command,
        )

    async def _generate_image_command(
        self,
        interaction: discord.Interaction,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: int = -1,
        width: int = 832,
        height: int = 1216,
        cfg_scale: float = 3.0,
        steps: int = 30,
        upscale: float = 1.0,
        allow_nsfw: bool = True,
    ):
        logger.info(f"Generate image called by {interaction.user.name}")
        await interaction.response.defer(thinking=True)

        try:
            file_path, image_info, is_nsfw = await self._image_gen.generate_image(
                prompt, negative_prompt, seed, width, height,
                cfg_scale, steps, upscale, allow_nsfw,
            )

            file = discord.File(fp=file_path, filename="generated.png")
            info_text = (
                f"steps: {image_info.steps}, "
                f"cfg: {image_info.cfg_scale}, "
                f"size: {image_info.width}x{image_info.height}, "
                f"seed: {image_info.seed}"
            )

            embed = discord.Embed()
            embed.set_image(url="attachment://generated.png")
            embed.set_footer(text=info_text)

            if is_nsfw:
                file.spoiler = True

            await interaction.followup.send(embed=embed, file=file)
        except Exception as e:
            logger.error(f"Image generation error: {e}", exc_info=True)
            await interaction.followup.send(f"An error occurred: {e}")
