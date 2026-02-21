"""Ollama API client for Discord LLM Bot"""

import traceback
import time
from typing import List, Optional
from aiohttp import ClientSession

from config import (
    OLLAMA_API_URL,
    CHAT_MODEL,
    SEARCH_UTILITY_MODEL,
    SEARCH_SUMMARIZATION_MODEL,
    NSFW_CLASSIFICATION_MODEL,
    TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL
)


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self):
        self.api_url = OLLAMA_API_URL

    async def generate(
        self,
        prompt: str,
        model: str = CHAT_MODEL,
        images: Optional[List[str]] = None
    ) -> str:
        """Generate a response from Ollama"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }

            if images:
                payload["images"] = images

            start_time = time.perf_counter()
            async with ClientSession() as session:
                async with session.post(self.api_url, json=payload) as resp:
                    data = await resp.json()
                    response = data.get("response", "No response from Ollama.")

            duration = time.perf_counter() - start_time
            print(f"Ollama generation took {duration:.2f}s")
            return response
        except Exception as e:
            print(f"Error in Ollama generate: {e}")
            print(traceback.format_exc())
            raise

    async def classify_image_task(self, prompt: str) -> bool:
        """Check if a prompt is requesting image generation"""
        system_prompt = (
            "You are a classifier that classifies whether a prompt is an instruction to generate an image or not. Your response should "
            "only contain two possible outcomes: Yes and No. Yes if the prompt contains an instruction to generate an image, and No if it does not. "
            "For example, prompts containing 'create an image' or 'generate an image of', etc. will return 'Yes'. Not all prompts that contain words like "
            "'image' or 'picture' should return 'yes'. For example, prompts such as 'what is in this image' should return No"
        )

        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        response = await self.generate(full_prompt, model=CHAT_MODEL)

        return "yes" in response.lower()

    async def generate_image_prompt(self, prompt: str) -> str:
        """Generate an image generation prompt from user input"""
        system_prompt = (
            "You are a tool that generates prompts for image generation tasks for diffusion-based image generation models. "
            "Given the user prompt in plain text, output a highly detailed, caption-like prompt. Attempt to be as specific as possible. "
            "For example, if the prompt is 'generate a photorealistic image of charlie puth', a possible prompt "
            "could be 'high definition, photorealistic depiction of charlie puth'. It's ok to have the prompt be more descriptive and akin to natural language. "
            "If the user input includes raw text, for example, 'generate a picture of a sign that says Hello', make sure to keep that information, in this case, the text 'Hello', in the final prompt"
        )

        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        return await self.generate(full_prompt, model=TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL)

    async def modify_image_prompt(self, original_prompt: str, modification: str) -> str:
        """Modify an existing image generation prompt based on user instructions"""
        system_prompt = (
            "You are a tool that modifies image generation prompts. You will receive an original prompt and a user's modification request. "
            "Output ONLY the modified prompt. Keep the original prompt as intact as possible — only change the parts that the user's request requires. "
            "Preserve all style, quality, and detail descriptors from the original unless the user explicitly asks to change them. "
            "Do not add commentary, explanations, or anything other than the modified prompt."
        )

        full_prompt = (
            f"System: {system_prompt}\n"
            f"Original prompt: {original_prompt}\n"
            f"User modification: {modification}\n"
            f"Assistant: "
        )
        return await self.generate(full_prompt, model=TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL)

    async def classify_search_task(self, prompt: str) -> bool:
        """Check if a prompt requires up-to-date information from the web"""
        system_prompt = (
            "You are a classifier that determines whether a user's message requires searching the web for up-to-date information. "
            "Your response should only contain two possible outcomes: Yes and No. "
            "Output Yes if the message asks about current events, recent news, live data (weather, stocks, scores), "
            "or anything that requires information newer than your training data. "
            "Output No if the message is casual conversation, asks about general knowledge, or doesn't need fresh data. "
            "Examples:\n"
            "'what's the latest news about AI' -> Yes\n"
            "'who won the super bowl this year' -> Yes\n"
            "'what's the weather in new york' -> Yes\n"
            "'how does photosynthesis work' -> No\n"
            "'hey what's up' -> No\n"
            "'tell me a joke' -> No"
        )

        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        response = await self.generate(full_prompt, model=SEARCH_UTILITY_MODEL)
        return "yes" in response.lower()

    async def extract_search_query(self, prompt: str) -> str:
        """Extract a concise search query from a user's message"""
        system_prompt = (
            "You are a tool that extracts a concise web search query from a user's message. "
            "Output ONLY the search query, nothing else. No quotes, no explanation. "
            "Examples:\n"
            "User: 'hey do you know what's going on with the AI regulations in the EU'\n"
            "AI regulations EU 2026\n"
            "User: 'who won the NBA finals'\n"
            "NBA finals winner 2026\n"
            "User: 'what's the current price of bitcoin'\n"
            "bitcoin price today"
        )

        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        response = await self.generate(full_prompt, model=SEARCH_UTILITY_MODEL)
        return response.strip()

    async def summarize_search_results(self, user_query: str, search_context: str) -> str:
        """Pre-process search results into a concise summary for the conversation context."""
        system_prompt = (
            "You are a research assistant. Summarize the following search results to answer "
            "the user's question. Include ALL relevant numbers, statistics, names, and facts "
            "exactly as stated in the results. Be concise but complete. "
            "Do not include URLs or source links. Do not add information not present in the results."
        )

        full_prompt = (
            f"System: {system_prompt}\n"
            f"Search Results:\n{search_context}\n\n"
            f"User Question: {user_query}\n"
            f"Assistant: "
        )
        return await self.generate(full_prompt, model=SEARCH_SUMMARIZATION_MODEL)

    async def classify_nsfw(self, images: List[str]) -> bool:
        """Classify if an image is NSFW"""
        system_prompt = (
            "You are a classifier that classifies images as NSFW (not safe for work) or "
            "SFW (safe for work). Your response should only contain two possible outcomes: "
            "NSFW and SFW. Output NSFW if the image contains explicit or potentially sensitive material that makes it "
            "not suitable for all audiences, and SFW if it is safe for all audiences. Consider depictions of women's bare legs or legs that "
            "display a significant portion of the thighs as 'NSFW' as well."
        )

        user_prompt = "Is this image NSFW?"
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: "

        response = await self.generate(full_prompt, model=NSFW_CLASSIFICATION_MODEL, images=images)
        return "nsfw" in response.lower()
