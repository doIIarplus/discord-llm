"""Discord LLM Bot - Main module"""

import asyncio
import re
import os
import random
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
from ollama_client import OllamaClient
from utils import encode_images_to_base64
from rag_system import RAGSystem
from web_extractor import extract_webpage_context, web_search, format_search_results, js_renderer
from file_parser import FileParser
from mention_extractor import extract_mention_context
from response_splitter import split_response_by_markers, split_long_message, calculate_typing_delay

# Keywords that suggest an image generation request
_IMAGE_GEN_KEYWORDS = re.compile(
    r'\b(generate|create|draw|make|paint|render|sketch)\b.{0,30}\b(image|picture|photo|illustration|art|drawing|painting)\b',
    re.IGNORECASE
)

# Keywords that suggest the user needs up-to-date / web-searchable info
_SEARCH_KEYWORDS = re.compile(
    r'\b(latest|recent|current|today|yesterday|tonight|this week|this month|this year'
    r'|news|update|score|weather|price|stock|election|released|announced'
    r'|who won|who is winning|what happened|how much does|how much is'
    r'|in 202[4-9]|right now)\b',
    re.IGNORECASE
)


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
        self._last_search_sources: List[dict] = []

        # Initialize RAG system
        self.rag_system = RAGSystem()
        self.rag_enabled = False

        # System prompts
        self.original_system_prompt = (
            "Your responses should be akin to that of a typical millenial texter: short, to the point, and mostly without punctuation. Do not offer any kind of assistance without being prompted. use slang *sparingly*. \n\n"
            "MULTI-MESSAGE RESPONSES:\n"
            "When your response would naturally be multiple messages (like a greeting followed by information, "
            "or multiple distinct points), you can split them using the marker: ---MSG---\n"
            "Example:\n"
            "hey i can help you with that.\n"
            "---MSG---\n"
            "i found out this about what you asked\n\n"
            "Only use this for natural conversational breaks. Don't overuse it - most responses should be single messages. "
            "Use it when:\n"
            "- You want to greet then provide information\n"
            "- You have multiple distinct topics to address\n"
            "- A dramatic pause or separate thought would feel natural"
        )
        self.system_prompt = self.original_system_prompt

        # Setup command handlers
        self.command_handlers = CommandHandlers(self)

    async def setup_hook(self):
        """Setup hook for Discord bot"""
        self.command_handlers.setup_commands()
        # NOTE: tree.sync() is intentionally omitted here.
        # Run it manually (e.g. via a one-off script) after changing commands,
        # as Discord rate-limits this call.
        await js_renderer.start()

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
        # Ignore bot messages (including self)
        if message.author.bot:
            return

        server = message.guild.id
        channel = message.channel.id

        # Handle attachments
        image_files = []
        document_files = []

        for attachment in message.attachments:
            safe_filename = os.path.basename(attachment.filename)
            file_path = os.path.join(FILE_INPUT_FOLDER, safe_filename)
            await attachment.save(file_path)

            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                image_files.append(file_path)
            else:
                document_files.append(file_path)

        # Determine response mode
        is_direct_mention = self.user in message.mentions
        is_reply_to_bot = False

        if message.reference:
            # Use cached resolved message when available
            ref_msg = message.reference.resolved
            if ref_msg is None:
                try:
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                except discord.NotFound:
                    ref_msg = None
            if ref_msg and ref_msg.author.id == self.user.id:
                is_reply_to_bot = True

        # Only respond to direct mentions and replies
        if is_direct_mention or is_reply_to_bot:
            fetched_sources = await self.build_context(message, server, False, image_files, document_files)
            await self._send_response(message, server, channel, fetched_sources)

    async def build_context(
        self,
        message: discord.Message,
        server: int,
        strip_mention: bool = False,
        image_files: List[str] = None,
        document_files: List[str] = None
    ):
        """Build conversation context"""
        if image_files is None:
            image_files = []
        if document_files is None:
            document_files = []

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

        # Process documents and clean up files after parsing
        if document_files:
            doc_context = ""
            for doc_path in document_files:
                content = FileParser.parse_file(doc_path)
                if content:
                    filename = os.path.basename(doc_path)
                    doc_context += f"\n\n--- Content of {filename} ---\n{content}\n--------------------------\n"
                try:
                    os.remove(doc_path)
                except OSError:
                    pass

            if doc_context:
                prompt += f"\n\n[Attached Documents Context]{doc_context}"

        # Extract web page content if URLs are present
        webpage_context, fetched_sources = await extract_webpage_context(prompt)
        if webpage_context:
            prompt = f"{prompt}\n\n{webpage_context}"

        # Encode images and clean up files
        images = []
        if image_files:
            images = encode_images_to_base64(image_files)
            for img_path in image_files:
                try:
                    os.remove(img_path)
                except OSError:
                    pass

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

        return fetched_sources

    async def _send_response(
        self,
        message: discord.Message,
        server: int,
        channel: int,
        sources: List[dict] = None,
    ):
        """
        Generate and send response with natural typing delays for multiple messages.
        """
        async with message.channel.typing():
            start = time.perf_counter()
            response_data = await self.query_ollama(server, channel)
            end = time.perf_counter()
            elapsed = end - start
            print(f"Query took {elapsed:.2f}s")

        # Merge URL-fetched sources with search sources
        all_sources = list(sources or [])
        search_sources = getattr(self, '_last_search_sources', [])
        if search_sources:
            all_sources.extend(search_sources)
            self._last_search_sources = []

        # Handle image generation responses (tuple with embed + file)
        if isinstance(response_data, tuple):
            await message.channel.send(embed=response_data[0], file=response_data[1])
            return

        # Collect all text parts to figure out where to append sources
        all_parts = []
        for response_item in response_data:
            if isinstance(response_item, str) and response_item.strip():
                all_parts.extend(split_response_by_markers(response_item))

        # Append source footnote with clickable links (deduplicated by domain)
        if all_sources and all_parts:
            seen_domains = set()
            source_links = []
            for s in all_sources:
                try:
                    domain = s['url'].split('/')[2].removeprefix('www.')
                except (IndexError, AttributeError):
                    continue
                if domain not in seen_domains:
                    seen_domains.add(domain)
                    source_links.append(f"[{domain}](<{s['url']}>)")
            if source_links:
                footnote = f"\n-# Sources: {' | '.join(source_links)}"
                last = all_parts[-1] + footnote
                if len(last) <= MAX_DISCORD_MESSAGE_LENGTH:
                    all_parts[-1] = last
                else:
                    all_parts.append(f"-# Sources: {' | '.join(source_links)}")

        # Send text parts with typing delays
        for i, part in enumerate(all_parts):
            async with message.channel.typing():
                delay = calculate_typing_delay(part)
                delay *= random.uniform(0.9, 1.3)
                await asyncio.sleep(delay)

            print(f"Sending response part {i+1}/{len(all_parts)}: {part[:50]}...")
            await message.channel.send(part)

            if i < len(all_parts) - 1:
                await asyncio.sleep(random.uniform(0.3, 0.8))

        # Handle any image items
        for response_item in response_data:
            if isinstance(response_item, dict) and "image" in response_item:
                file = discord.File(response_item["image"])
                await message.channel.send(file=file)

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
        """Process response text, handling length limits"""
        # Remove thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Wrap bare URLs in <> to suppress Discord auto-embeds
        text = re.sub(r'(?<![<])(https?://\S+)', r'<\1>', text)
        # Split long text to fit Discord's message length limit
        parts = split_long_message(text.strip(), limit)
        return parts

    async def query_ollama(self, server: int, channel: int, override_messages: List[dict] = None):
        """Query Ollama for a response"""
        messages = override_messages or self.context[server][channel]
        user_content = messages[-1]['content']
        images = messages[-1].get("images", [])

        # Check if this is an image generation task (fast heuristic first)
        # Also check if this could be a follow-up to a recent image generation
        has_recent_image_gen = any(
            msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", "")
            for msg in messages[-4:]
        )
        if _IMAGE_GEN_KEYWORDS.search(user_content) or has_recent_image_gen:
            # For follow-ups, give the classifier context about the recent image
            classify_input = user_content
            if has_recent_image_gen and not _IMAGE_GEN_KEYWORDS.search(user_content):
                for msg in reversed(messages[-4:]):
                    if msg.get("role") == "assistant" and "[Generated an image" in msg.get("content", ""):
                        classify_input = f"[Previous: {msg['content']}]\nUser: {user_content}"
                        break
            is_img_task = await self.image_gen.is_image_generation_task(classify_input)

            if is_img_task:
                # Determine if this is a modification of a previous image or a fresh request
                is_modification = has_recent_image_gen and not _IMAGE_GEN_KEYWORDS.search(user_content)
                prev_seed = -1
                prev_prompt = None

                if is_modification:
                    # Extract previous prompt and seed from context
                    for msg in reversed(messages[-4:]):
                        content = msg.get("content", "")
                        if msg.get("role") == "assistant" and "[Generated an image" in content:
                            prompt_match = re.search(r'\[Generated an image with the following prompt: (.+?)\]', content, re.DOTALL)
                            seed_match = re.search(r'seed: (\d+)', content)
                            if prompt_match:
                                prev_prompt = prompt_match.group(1)
                            if seed_match:
                                prev_seed = int(seed_match.group(1))
                            break

                if is_modification and prev_prompt:
                    # Modification: minimally edit the previous prompt, reuse seed
                    print(f"  [image modification: reusing seed {prev_seed}, modifying prompt]")
                    prompt = (await self.ollama_client.modify_image_prompt(prev_prompt, user_content)).strip()
                    print(f"  [modified prompt: {prompt}]")
                else:
                    # New generation: fresh prompt, random seed
                    prev_seed = -1
                    # Extract web page content if URLs are present
                    webpage_context, _ = await extract_webpage_context(user_content)
                    if webpage_context:
                        user_content = f"{user_content}\n\n{webpage_context}"
                    prompt = (await self.image_gen.generate_image_prompt(user_content)).strip()

                file_path, image_info, is_nsfw = await self.image_gen.generate_image(prompt, '', seed=prev_seed)

                # Store image generation context for follow-up continuity
                if override_messages is None:
                    self.context[server][channel].append({
                        "role": "assistant",
                        "content": (
                            f"[Generated an image with the following prompt: {prompt}] "
                            f"(seed: {image_info.seed}, "
                            f"size: {image_info.width}x{image_info.height})"
                        ),
                        "timestamp": time.time(),
                    })

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

        # Check if the user's message needs a web search (heuristic + LLM)
        search_summary = ""
        search_sources = []
        # Strip Discord mentions for cleaner LLM input
        clean_query = re.sub(r'<@!?\d+>', '', user_content).strip()
        if _SEARCH_KEYWORDS.search(clean_query):
            print("  [search heuristic matched, checking with LLM...]")
            needs_search = await self.ollama_client.classify_search_task(clean_query)
            if needs_search:
                search_query = await self.ollama_client.extract_search_query(clean_query)
                print(f"  [searching: {search_query}]")
                search_results = await web_search(search_query, max_results=5)
                if search_results:
                    raw_context = format_search_results(search_results)
                    print(f"  [summarizing {len(raw_context)} chars of search results...]")
                    search_summary = await self.ollama_client.summarize_search_results(clean_query, raw_context)
                    print(f"  [summary: {len(search_summary)} chars]")
                    print(f"  [summary content: {search_summary}]")
                    search_sources = [
                        {"url": r["url"], "title": r["title"] or r["url"]}
                        for r in search_results[:3]
                    ]

        self._last_search_sources = search_sources

        # Regular text query
        prompt = self.format_prompt(messages)

        # Add search summary if available
        if search_summary:
            prompt = f"Search Results Summary:\n{search_summary}\n\n{prompt}"

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

        system_prompt = self.system_prompt
        prompt = f"System: {system_prompt}\n" + prompt

        # Extract Discord mentions (users, channels, roles) from the last message
        guild = self.get_guild(server)
        mention_context = extract_mention_context(user_content, guild)
        if mention_context:
            prompt = f"{prompt}\n\n{mention_context}"

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

            print(f"Response: {raw_response}")
            return self.process_response(raw_response)

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return [f"Error communicating with Ollama: {e}"]

    async def close(self):
        """Close the bot"""
        await js_renderer.stop()
        await super().close()


def main():
    """Main entry point"""
    bot = OllamaBot()
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
