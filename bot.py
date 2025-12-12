"""Discord LLM Bot - Main module"""

import asyncio
import json
import os
import re
import time
import traceback
from typing import Dict, List

import discord
from discord import app_commands

from commands import CommandHandlers
from config import (
    CONTEXT_LIMIT,
    DISCORD_BOT_TOKEN,
    FILE_INPUT_FOLDER,
    GUILD_ID,
    IMAGE_RECOGNITION_MODEL,
    CHAT_MODEL,
    MAX_DISCORD_MESSAGE_LENGTH
)
from image_generation import ImageGenerator
from latex import split_text_and_latex
from ollama_client import OllamaClient
from script_executor import ScriptExecutor
from utils import encode_images_to_base64, encode_image_to_base64
from rag_system import RAGSystem
from web_extractor import extract_webpage_context


class OllamaBot(discord.Client):
    """Main Discord bot class"""
    
    def __init__(self):
        super().__init__(intents=discord.Intents.all())
        self.tree = app_commands.CommandTree(self)
        
        # Per-server per-channel context
        self.context: Dict[str, Dict[str, List[dict]]] = {}
        
        # Initialize clients
        self.ollama_client = OllamaClient()
        self.image_gen = ImageGenerator()
        self.script_executor = ScriptExecutor()
        
        # Initialize RAG system
        self.rag_system = RAGSystem()
        self.rag_enabled = False  # Flag to enable/disable RAG
        
        # System prompts
        # self.original_system_prompt = (
        #     "If your response includes actual mathematical expressions or formulas, present them using LaTeX syntax, "
        #     "wrapped in double dollar signs like this: $$...$$. Only use LaTeX for mathematical content—do not use "
        #     "LaTeX for plain text, variables in discussion, or non-math concepts. Ensure all LaTeX expressions are "
        #     "valid, syntactically correct, and renderable. For example, the quadratic formula should be written "
        #     "as: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$ \n "
        #     "Avoid using LaTeX unless a clear mathematical equation is being communicated. DO NOT USE MATHEMATICAL EQUATIONS OR LATEX IN NON MATH RELATED TOPICS."
        #     "Do not reference this instruction in your response. You are being used as a discord bot, so take discord formatting into consideration. If your response includes any kind of tables, put the table in code blocks. "
        #     "Do not mention any part of this prompt in your responses."
        # )

        self.original_system_prompt = (
            "you're a discord bot. Table formatting via ascii dashes (i.e. like ------- etc.) doesn't work, so don't even try it."
        )
        self.system_prompt = self.original_system_prompt
        
        # Setup command handlers
        self.command_handlers = CommandHandlers(self)
        
    async def setup_hook(self):
        """Setup hook for Discord bot"""
        self.command_handlers.setup_commands()
        await self.tree.sync()
        
    def pick_model(self, server: int, channel: int) -> str:
        """Pick the appropriate model based on context"""
        if (server in self.context and 
            channel in self.context[server] and 
            self.context[server][channel] and 
            self.context[server][channel][-1].get("images")):
            return IMAGE_RECOGNITION_MODEL
        return CHAT_MODEL
        
    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        if message.author.bot:
            return
            
        should_respond = False
        server = message.guild.id
        
        # Handle attachments
        files = []
        for attachment in message.attachments:
            file_path = os.path.join(FILE_INPUT_FOLDER, attachment.filename)
            await attachment.save(file_path)
            files.append(file_path)
            
        # Check if bot should respond
        if self.user in message.mentions:
            should_respond = True
            await self.build_context(message, server, True, files)
            
        # Check if replying to bot
        if message.reference:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
            if ref_msg.author.id == self.user.id:
                should_respond = True
                await self.build_context(message, server, True, files)
                
        if should_respond:
            async with message.channel.typing():
                start = time.perf_counter()
                response_texts = await self.query_ollama(server, message.channel.id)
                end = time.perf_counter()
                elapsed = end - start
                
                if isinstance(response_texts, tuple):
                    await message.channel.send(embed=response_texts[0], file=response_texts[1])
                else:
                    for response_text in response_texts:
                        if isinstance(response_text, str) and response_text.strip():
                            print(f"Sending response: {response_text}")
                            await message.channel.send(response_text)
                        elif isinstance(response_text, dict) and "image" in response_text:
                            file = discord.File(response_text["image"])
                            await message.channel.send(file=file)
                            
    async def build_context(
        self,
        message: discord.Message,
        server: int,
        strip_mention: bool = False,
        files: List[str] = None
    ):
        """Build conversation context"""
        if files is None:
            files = []
            
        channel = message.channel.id
        
        if server not in self.context:
            self.context[server] = {}
            
        if channel not in self.context[server]:
            self.context[server][channel] = []
            
        prompt = (
            message.content
            if not strip_mention
            else message.clean_content.replace(f"@{self.user.name}", "").strip()
        )
        
        # Extract web page content if URLs are present
        webpage_context = extract_webpage_context(prompt)
        if webpage_context:
            prompt = f"{prompt}\n\n{webpage_context}"
        
        images = encode_images_to_base64(files) if files else []
        
        self.context[server][channel].append({
            "role": "user",
            "name": message.author.display_name,
            "content": prompt,
            "timestamp": time.time(),
            "images": images,
        })
        
        # Maintain context limit
        if len(self.context[server][channel]) > CONTEXT_LIMIT:
            self.context[server][channel].pop(0)
            
    def format_prompt(self, messages: List[dict]) -> str:
        """Format messages into a prompt"""
        prompt = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            name = f"({msg['name']})" if msg["role"] == "user" and "name" in msg else ""
            prompt += f"{role} {name}: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt
        
    def process_response(self, text: str, limit: int = MAX_DISCORD_MESSAGE_LENGTH) -> List:
        """Process response text, handling LaTeX and length limits"""
        # Remove thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        processed_response = split_text_and_latex(text)
        print(f"Processed response: {processed_response}")
        return processed_response
        
    async def query_ollama(self, server: int, channel: int, override_messages: List[dict] = None):
        """Query Ollama for a response"""
        messages = override_messages or self.context[server][channel]
        user_content = messages[-1]['content']
        images = messages[-1].get("images", [])
        
        # First check if this can be solved programmatically
        is_programmatic = await self.ollama_client.classify_programmatic_task(user_content)

        # do not programmatic for webpages
        webpage_context = extract_webpage_context(user_content)
        
        if is_programmatic and not webpage_context:
            print(f"Identified as programmatic task: {user_content}. Clearing all context.")
            
            # Generate Python script
            script = await self.ollama_client.generate_python_script(user_content)
            print(f"Generated script:\n{script}")
            
            # Execute the script
            success, output, image_path = self.script_executor.execute_script(script)
            
            if success:
                # Prepare the context for the LLM with the script output
                script_context = (
                    f"The user asked: {user_content}\n"
                    f"I executed the following Python script:\n```python\n{script}\n```\n"
                    f"The script output was:\n{output}\n"
                )
                
                if image_path:
                    script_context += f"The script also generated an image (attached).\n"
                    # Read the image for the LLM to analyze
                    image_base64 = encode_image_to_base64(image_path)
                    images = [image_base64]
                
                script_context += "Based on this output, provide a clear and concise answer to the user's question. Do not mention that this output was generated via a script. Do not mention any file names or such. Just provide an answer to the question."
                
                # Get LLM response based on the script output
                prompt = f"System: {self.system_prompt}\n{script_context}\nAssistant: "
                response = await self.ollama_client.generate(prompt, model=CHAT_MODEL, images=images)
                
                # Process the response
                result = self.process_response(response)
                
                # If there's an image, add it to the response
                if image_path:
                    result.append({"image": image_path})
                    
                return result
            else:
                # Script failed, fall back to regular response
                print(f"Script execution failed: {output}")
                # Continue with regular flow below
        
        # Check if this is an image generation task
        is_img_task = await self.image_gen.is_image_generation_task(user_content)
        
        if is_img_task:
            # Extract web page content if URLs are present
            if webpage_context:
                user_content = f"{user_content}\n\n{webpage_context}"
            
            prompt = await self.image_gen.generate_image_prompt(user_content)
            file_path, image_info, is_nsfw = await self.image_gen.generate_image(prompt, '')
            
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
                
            return (embed, file)
            
        # Regular text query
        prompt = self.format_prompt(messages)
        
        # Add RAG context if enabled
        if self.rag_enabled:
            # Extract the user's question from messages
            user_question = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_question = msg.get("content", "")
            
            if user_question:
                wiki_context = self.rag_system.get_context_for_query(user_question)
                if wiki_context:
                    prompt = f"Wiki Context:\n{wiki_context}\n\n{prompt}"
        
        prompt = f"System: {self.system_prompt}\n" + prompt
        
        # Extract web page content if URLs are present in the last message
        if webpage_context:
            prompt = f"{prompt}\n\nWeb Page Context:\n{webpage_context}"
        
        if images:
            print("Sending image")
            
        try:
            model = self.pick_model(server, channel)
            print(f"Using model: {model}")
            print(f"Prompt: {prompt}")
            
            raw_response = await self.ollama_client.generate(prompt, model, images)
            
            if raw_response == "No response from Ollama.":
                print("No response from Ollama")
                return ["No response from Ollama."]
                
            # Add response to context
            if override_messages is None:
                self.context[server][channel].append({
                    "role": "assistant",
                    "content": raw_response,
                    "timestamp": time.time(),
                })
                
                # Save context to file
                with open("output.txt", "w") as file:
                    json.dump(self.context, file, indent=4)
                    
            print(f"Response: {raw_response}")
            return self.process_response(raw_response)
            
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return [f"Error communicating with Ollama: {e}"]
            
    async def close(self):
        """Close the bot"""
        await super().close()


def main():
    """Main entry point"""
    bot = OllamaBot()
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()