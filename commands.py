"""Discord command handlers for Discord LLM Bot"""

import time
from typing import Optional

import discord
from discord import app_commands

from image_generation import ImageGenerator


class CommandHandlers:
    """Handles Discord slash commands"""
    
    def __init__(self, bot):
        self.bot = bot
        self.image_gen = ImageGenerator()
        
    def setup_commands(self):
        """Register all slash commands"""
        
        @self.bot.tree.command(name="clear", description="Clear context")
        async def clear(interaction: discord.Interaction):
            server = interaction.guild.id
            channel = interaction.channel.id
            
            if server in self.bot.context and channel in self.bot.context[server]:
                self.bot.context[server][channel] = []
            await interaction.response.send_message("Context cleared")
            
        @self.bot.tree.command(name="ask", description="Ask something")
        @app_commands.describe(question="Ask something")
        async def ask(interaction: discord.Interaction, question: str):
            await interaction.response.defer(thinking=True)
            
            start = time.perf_counter()
            response = await self.bot.query_ollama(
                interaction.guild.id, 
                interaction.channel.id,
                [{"role": "user", "content": question, "images": []}]
            )
            end = time.perf_counter()
            elapsed = end - start
            
            if isinstance(response, tuple):
                await interaction.followup.send(embed=response[0], file=response[1])
            else:
                response_text = "\n".join(response) if isinstance(response, list) else response
                response_text += f"\n\n_Responded in {elapsed:.3f} seconds_"
                await interaction.followup.send(response_text)
                
        @self.bot.tree.command(name='set_system_prompt')
        async def set_system_prompt(interaction: discord.Interaction, prompt: str):
            await interaction.response.defer(thinking=True)
            self.bot.system_prompt = prompt
            await interaction.followup.send("Prompt updated")
            
        @self.bot.tree.command(name='reset_system_prompt')
        async def reset_system_prompt(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            self.bot.system_prompt = self.bot.original_system_prompt
            await interaction.followup.send("Prompt reset to default")
            
        @self.bot.tree.command(name='get_system_prompt')
        async def get_system_prompt(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            await interaction.followup.send(self.bot.system_prompt)
            
        @self.bot.tree.command(name="generate_image")
        @app_commands.describe(
            prompt="Prompt for image generation. Comma separated.",
            negative_prompt="Negative prompt. Same format as positive prompt, but for things you don't want to see.",
            seed="Seed for image generation. Same seed with same parameters will generate the same image. -1 for random",
            width="Width of image in pixels. Capped at 1500",
            height="Height of image in pixels. Capped at 2000",
            cfg_scale="Smaller values will produce more creative results. Larger values will conform to prompt more.",
            steps="Number of steps for image generation",
            upscale="Upscale factor (1.0 to 2.0)",
            allow_nsfw="Allow NSFW content generation"
        )
        async def generate_image(
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
            await interaction.response.defer(thinking=True)
            
            try:
                file_path, image_info, is_nsfw = await self.image_gen.generate_image(
                    prompt, negative_prompt, seed, width, height, 
                    cfg_scale, steps, upscale, allow_nsfw
                )
                
                file = discord.File(fp=file_path, filename='generated.png')
                image_info_text = (
                    f"steps: {image_info.steps}, "
                    f"cfg: {image_info.cfg_scale}, "
                    f"size: {image_info.width}x{image_info.height}, "
                    f"seed: {image_info.seed}"
                )
                
                embed = discord.Embed()
                embed.set_image(url='attachment://generated.png')
                embed.set_footer(text=image_info_text)
                
                if is_nsfw:
                    file.spoiler = True
                    
                await interaction.followup.send(embed=embed, file=file)
                
            except Exception as e:
                await interaction.followup.send(f"An error occurred: {e}")