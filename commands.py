"""Discord command handlers for Discord LLM Bot"""

import time
import traceback
from typing import Optional
import asyncio
import logging
import os

import discord
from discord import app_commands

from image_generation import ImageGenerator
from tools.base import registry as tool_registry

# Set up logging
logger = logging.getLogger("CommandHandlers")


class CommandHandlers:
    """Handles Discord slash commands"""
    
    def __init__(self, bot):
        self.bot = bot
        self.image_gen = ImageGenerator()
        logger.info("CommandHandlers initialized")
        
    def setup_commands(self):
        """Register all slash commands"""
        logger.info("Setting up Discord slash commands")
        
        @self.bot.tree.command(name="clear", description="Clear context")
        async def clear(interaction: discord.Interaction):
            logger.info(f"Clear command called by {interaction.user.name}#{interaction.user.discriminator}")
            server = interaction.guild.id
            channel = interaction.channel.id
            
            if server in self.bot.context and channel in self.bot.context[server]:
                self.bot.context[server][channel] = []
                logger.debug(f"Cleared context for server {server}, channel {channel}")
            await interaction.response.send_message("Context cleared")
            logger.info("Context cleared successfully")
            
        @self.bot.tree.command(name="ask", description="Ask something")
        @app_commands.describe(question="Ask something")
        async def ask(interaction: discord.Interaction, question: str):
            logger.info(f"Ask command called by {interaction.user.name}#{interaction.user.discriminator} with question: {question[:50]}...")
            await interaction.response.defer(thinking=True)
            
            start = time.perf_counter()
            response = await self.bot.query_ollama(
                interaction.guild.id, 
                interaction.channel.id,
                [{"role": "user", "content": question, "images": []}]
            )
            end = time.perf_counter()
            elapsed = end - start
            logger.info(f"Ask command processed in {elapsed:.3f} seconds")
            
            if isinstance(response, tuple):
                logger.debug("Sending embed and file response")
                await interaction.followup.send(embed=response[0], file=response[1])
            else:
                response_text = "\n".join(response) if isinstance(response, list) else response
                response_text += f"\n\n_Responded in {elapsed:.3f} seconds_"
                logger.debug(f"Sending text response: {response_text[:100]}...")
                await interaction.followup.send(response_text)
                
        @self.bot.tree.command(name='set_system_prompt')
        async def set_system_prompt(interaction: discord.Interaction, prompt: str):
            logger.info(f"Set system prompt command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            self.bot.system_prompt = prompt
            logger.debug("System prompt updated")
            await interaction.followup.send("Prompt updated")
            
        @self.bot.tree.command(name='reset_system_prompt')
        async def reset_system_prompt(interaction: discord.Interaction):
            logger.info(f"Reset system prompt command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            self.bot.system_prompt = self.bot.original_system_prompt
            logger.debug("System prompt reset to default")
            await interaction.followup.send("Prompt reset to default")
            
        @self.bot.tree.command(name='get_system_prompt')
        async def get_system_prompt(interaction: discord.Interaction):
            logger.info(f"Get system prompt command called by {interaction.user.name}#{interaction.user.discriminator}")
            await interaction.response.defer(thinking=True)
            logger.debug("Sending current system prompt")
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
            logger.info(f"Generate image command called by {interaction.user.name}#{interaction.user.discriminator}")
            logger.debug(f"Image generation parameters: prompt='{prompt[:50]}...', seed={seed}, width={width}, height={height}, cfg_scale={cfg_scale}, steps={steps}, upscale={upscale}, allow_nsfw={allow_nsfw}")
            await interaction.response.defer(thinking=True)
            
            try:
                file_path, image_info, is_nsfw = await self.image_gen.generate_image(
                    prompt, negative_prompt, seed, width, height, 
                    cfg_scale, steps, upscale, allow_nsfw
                )
                logger.info(f"Image generated successfully: {file_path}")
                logger.debug(f"Image info: steps={image_info.steps}, cfg={image_info.cfg_scale}, size={image_info.width}x{image_info.height}, seed={image_info.seed}")
                
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
                    logger.debug("Marking image as NSFW")
                    
                await interaction.followup.send(embed=embed, file=file)
                logger.info("Image sent to Discord successfully")
                
            except Exception as e:
                logger.error(f"Error during image generation: {e}", exc_info=True)
                await interaction.followup.send(f"An error occurred: {e}")
        
        @self.bot.tree.command(name="index_wiki")
        @app_commands.describe(
            clear_existing="Clear existing index before indexing (default: False)"
        )
        async def index_wiki(interaction: discord.Interaction, clear_existing: bool = False):
            """Index the MapleStory wiki dump for RAG"""
            logger.info(f"Index wiki command called by {interaction.user.name}#{interaction.user.discriminator}")
            logger.debug(f"Clear existing: {clear_existing}")
            await interaction.response.defer(thinking=True)
            
            wiki_path = "maplestorywikinet.xml"
            if not os.path.exists(wiki_path):
                logger.warning(f"Wiki file not found: {wiki_path}")
                await interaction.followup.send(f"Wiki file not found: {wiki_path}")
                return
            
            try:
                if clear_existing:
                    logger.info("Clearing existing collection")
                    self.bot.rag_system.clear_collection()
                    await interaction.followup.send("Cleared existing index. Starting indexing...")
                else:
                    logger.info("Starting wiki indexing")
                    await interaction.followup.send("Starting wiki indexing. This may take several minutes...")
                
                # Run indexing in background
                logger.debug("Running wiki indexing in background")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    self.bot.rag_system.index_wiki_dump,
                    wiki_path
                )
                logger.info("Wiki indexing completed successfully")
                
                stats = self.bot.rag_system.get_stats()
                logger.debug(f"Indexing stats: {stats}")
                await interaction.channel.send(
                    f"✅ Wiki indexing complete!\n"
                    f"Total chunks indexed: {stats['total_chunks']}"
                )
            except Exception as e:
                logger.error(f"Error during wiki indexing: {e}", exc_info=True)
                traceback.print_exc()
                await interaction.channel.send(f"❌ Error during indexing: {e}")
        
        @self.bot.tree.command(name="enable_rag")
        async def enable_rag(interaction: discord.Interaction):
            """Enable RAG for wiki content"""
            logger.info(f"Enable RAG command called by {interaction.user.name}#{interaction.user.discriminator}")
            self.bot.rag_enabled = True
            logger.debug("RAG enabled")
            await interaction.response.send_message("✅ RAG enabled. Wiki context will be added to queries.")
        
        @self.bot.tree.command(name="disable_rag")
        async def disable_rag(interaction: discord.Interaction):
            """Disable RAG for wiki content"""
            logger.info(f"Disable RAG command called by {interaction.user.name}#{interaction.user.discriminator}")
            self.bot.rag_enabled = False
            logger.debug("RAG disabled")
            await interaction.response.send_message("❌ RAG disabled. No wiki context will be added.")
        
        @self.bot.tree.command(name="search_wiki")
        @app_commands.describe(
            query="Search query for the wiki",
            n_results="Number of results to return (default: 3)"
        )
        async def search_wiki(interaction: discord.Interaction, query: str, n_results: int = 3):
            """Search the indexed wiki content"""
            logger.info(f"Search wiki command called by {interaction.user.name}#{interaction.user.discriminator} with query: {query[:50]}...")
            logger.debug(f"Number of results requested: {n_results}")
            await interaction.response.defer(thinking=True)
            
            try:
                results = self.bot.rag_system.search(query, n_results=n_results)
                logger.debug(f"Search returned {len(results)} results")
                
                if not results:
                    logger.info("No results found for search query")
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
                
                logger.debug(f"Sending search results: {response[:100]}...")
                await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Error during wiki search: {e}", exc_info=True)
                await interaction.followup.send(f"Error searching wiki: {e}")
        
        @self.bot.tree.command(name="rag_stats")
        async def rag_stats(interaction: discord.Interaction):
            """Get statistics about the indexed wiki content"""
            logger.info(f"RAG stats command called by {interaction.user.name}#{interaction.user.discriminator}")
            stats = self.bot.rag_system.get_stats()
            logger.debug(f"RAG stats: {stats}")
            await interaction.response.send_message(
                f"**RAG System Stats**\n"
                f"Total chunks: {stats['total_chunks']}\n"
                f"Collection: {stats['collection_name']}\n"
                f"RAG enabled: {self.bot.rag_enabled}"
            )

        # Tool system commands
        @self.bot.tree.command(name="enable_tools")
        async def enable_tools(interaction: discord.Interaction):
            """Enable tool calling for the LLM"""
            logger.info(f"Enable tools command called by {interaction.user.name}#{interaction.user.discriminator}")
            self.bot.tools_enabled = True
            await interaction.response.send_message("✅ Tools enabled. The bot can now use tools to help answer questions.")

        @self.bot.tree.command(name="disable_tools")
        async def disable_tools(interaction: discord.Interaction):
            """Disable tool calling for the LLM"""
            logger.info(f"Disable tools command called by {interaction.user.name}#{interaction.user.discriminator}")
            self.bot.tools_enabled = False
            await interaction.response.send_message("❌ Tools disabled. The bot will respond without using tools.")

        @self.bot.tree.command(name="list_tools")
        async def list_tools(interaction: discord.Interaction):
            """List all available tools"""
            logger.info(f"List tools command called by {interaction.user.name}#{interaction.user.discriminator}")

            tools = tool_registry.get_all()
            if not tools:
                await interaction.response.send_message("No tools available.")
                return

            # Group by category
            by_category = {}
            for tool in tools:
                cat = tool.category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(tool)

            response = "**Available Tools:**\n\n"
            for category, cat_tools in sorted(by_category.items()):
                response += f"**{category.title()}**\n"
                for tool in cat_tools:
                    response += f"• `{tool.name}`: {tool.description}\n"
                response += "\n"

            response += f"\nTools enabled: {'✅' if self.bot.tools_enabled else '❌'}"

            if len(response) > 2000:
                response = response[:1997] + "..."

            await interaction.response.send_message(response)

        @self.bot.tree.command(name="tool_info")
        @app_commands.describe(tool_name="Name of the tool to get info about")
        async def tool_info(interaction: discord.Interaction, tool_name: str):
            """Get detailed information about a specific tool"""
            logger.info(f"Tool info command called for {tool_name}")

            tool = tool_registry.get(tool_name)
            if not tool:
                await interaction.response.send_message(f"Tool '{tool_name}' not found.")
                return

            response = f"**Tool: {tool.name}**\n"
            response += f"Category: {tool.category}\n"
            response += f"Description: {tool.description}\n\n"

            if tool.parameters:
                response += "**Parameters:**\n"
                for param in tool.parameters:
                    req = "required" if param.required else "optional"
                    response += f"• `{param.name}` ({param.param_type.value}, {req})\n"
                    response += f"  {param.description}\n"
                    if param.enum:
                        response += f"  Choices: {', '.join(param.enum)}\n"
            else:
                response += "No parameters required.\n"

            await interaction.response.send_message(response)

        logger.info("All Discord slash commands registered successfully")