"""Discord command handlers for Discord LLM Bot"""

import time
from typing import Optional
import asyncio
import os

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
        
        @self.bot.tree.command(name="index_wiki")
        @app_commands.describe(
            clear_existing="Clear existing index before indexing (default: False)"
        )
        async def index_wiki(interaction: discord.Interaction, clear_existing: bool = False):
            """Index the MapleStory wiki dump for RAG"""
            await interaction.response.defer(thinking=True)
            
            wiki_path = "maplestorywikinet.xml"
            if not os.path.exists(wiki_path):
                await interaction.followup.send(f"Wiki file not found: {wiki_path}")
                return
            
            try:
                if clear_existing:
                    self.bot.rag_system.clear_collection()
                    await interaction.followup.send("Cleared existing index. Starting indexing...")
                else:
                    await interaction.followup.send("Starting wiki indexing. This may take several minutes...")
                
                # Run indexing in background
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    self.bot.rag_system.index_wiki_dump,
                    wiki_path
                )
                
                stats = self.bot.rag_system.get_stats()
                await interaction.channel.send(
                    f"✅ Wiki indexing complete!\n"
                    f"Total chunks indexed: {stats['total_chunks']}"
                )
            except Exception as e:
                await interaction.channel.send(f"❌ Error during indexing: {e}")
        
        @self.bot.tree.command(name="enable_rag")
        async def enable_rag(interaction: discord.Interaction):
            """Enable RAG for wiki content"""
            self.bot.rag_enabled = True
            await interaction.response.send_message("✅ RAG enabled. Wiki context will be added to queries.")
        
        @self.bot.tree.command(name="disable_rag")
        async def disable_rag(interaction: discord.Interaction):
            """Disable RAG for wiki content"""
            self.bot.rag_enabled = False
            await interaction.response.send_message("❌ RAG disabled. No wiki context will be added.")
        
        @self.bot.tree.command(name="search_wiki")
        @app_commands.describe(
            query="Search query for the wiki",
            n_results="Number of results to return (default: 3)"
        )
        async def search_wiki(interaction: discord.Interaction, query: str, n_results: int = 3):
            """Search the indexed wiki content"""
            await interaction.response.defer(thinking=True)
            
            try:
                results = self.bot.rag_system.search(query, n_results=n_results)
                
                if not results:
                    await interaction.followup.send("No results found.")
                    return
                
                # Format results for Discord
                response = f"**Search results for:** {query}\n\n"
                for i, result in enumerate(results, 1):
                    # Truncate content for display
                    content = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                    response += f"**{i}. {result['title']}** (Score: {result['score']:.3f})\n{content}\n\n"
                
                # Split if too long
                if len(response) > 2000:
                    response = response[:1997] + "..."
                
                await interaction.followup.send(response)
            except Exception as e:
                await interaction.followup.send(f"Error searching wiki: {e}")
        
        @self.bot.tree.command(name="rag_stats")
        async def rag_stats(interaction: discord.Interaction):
            """Get statistics about the indexed wiki content"""
            stats = self.bot.rag_system.get_stats()
            await interaction.response.send_message(
                f"**RAG System Stats**\n"
                f"Total chunks: {stats['total_chunks']}\n"
                f"Collection: {stats['collection_name']}\n"
                f"RAG enabled: {self.bot.rag_enabled}"
            )