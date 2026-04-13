"""Ollama API client for Discord LLM Bot"""

import logging
import traceback
import time
from typing import List, Optional
from aiohttp import ClientSession, ClientTimeout

# Ollama model cold-loads + vision inference on big images can easily exceed
# aiohttp's 5-minute default. 20 minutes is more than enough headroom.
_OLLAMA_TIMEOUT = ClientTimeout(total=1200, sock_read=1200, sock_connect=30)

from config import (
    OLLAMA_API_URL,
    CHAT_MODEL,
    SEARCH_UTILITY_MODEL,
    SEARCH_SUMMARIZATION_MODEL,
    NSFW_CLASSIFICATION_MODEL,
    TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL,
    IMAGE_EDIT_DESCRIPTION_MODEL,
)

logger = logging.getLogger("ollama")


def _truncate(s: str, n: int = 160) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...[+{len(s)-n} chars]"


def _sanitize_prompt_output(raw: str) -> str:
    """Clean up a chat model's response that was supposed to be a single
    image-generation prompt.

    Handles common misbehaviors:
    - Preamble ("Okay, here are a few options...")
    - Multi-option responses ("Option 1: ...")
    - Markdown formatting (**bold**, headings, bullets)
    - Code fences
    - Follow-up question blocks at the end
    - Trailing commentary
    """
    import re
    text = raw.strip()
    if not text:
        return text

    # If the model emitted multiple options, grab the first one's content.
    # Look for "Option N" or "Option N:" headers.
    option_match = re.search(
        r'(?:^|\n)\s*(?:\*\*)?Option\s*1[^\n:]*:?\s*(?:\*\*)?\s*\n+(.+?)'
        r'(?=\n\s*(?:\*\*)?Option\s*2|\Z)',
        text, re.DOTALL | re.IGNORECASE,
    )
    if option_match:
        text = option_match.group(1).strip()

    # Strip code fences if the whole thing is wrapped
    fence = re.match(r'^```[a-zA-Z]*\n?(.+?)\n?```\s*$', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Drop preamble lines like "Here is the prompt:", "Okay, ..."
    preamble_patterns = [
        r"^(okay|ok|sure|here('?s| is)[^:\n]*:|let me[^:\n]*:|prompt:|alright[^:\n]*:|based on[^:\n]*:)",
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, "", text, count=1, flags=re.IGNORECASE).lstrip(":").strip()

    # Strip surrounding quotes
    if len(text) >= 2 and text[0] in '"\u201c\u2018' and text[-1] in '"\u201d\u2019':
        text = text[1:-1].strip()

    # Drop trailing "To refine the prompt further, please tell me..." follow-up blocks
    cutoffs = [
        r'\n\s*To help me refine', r'\n\s*To refine', r'\n\s*Let me know if',
        r'\n\s*Do you want', r'\n\s*Would you like', r'\n\s*Please tell me',
        r'\n\s*\*\s*What', r'\n\s*Hope this helps',
    ]
    for pat in cutoffs:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[:m.start()].rstrip()

    # Collapse repeated whitespace / newlines into single spaces since we want
    # one paragraph for a diffusion prompt
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip remaining markdown emphasis markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)

    return text.strip()


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self):
        self.api_url = OLLAMA_API_URL

    async def generate(
        self,
        prompt: str,
        model: str = CHAT_MODEL,
        images: Optional[List[str]] = None,
        keep_alive: Optional[int] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        think: Optional[bool] = None,
    ) -> str:
        """Generate a response from Ollama.

        Args:
            keep_alive: Ollama keep_alive value. -1 = stay loaded forever,
                        None = use Ollama's default timeout.
            num_ctx: Context window size. Defaults to None (Ollama model default).
                     Set explicitly for vision models to avoid allocating huge
                     KV caches (e.g. qwen3-vl defaults to 256K).
            num_predict: Hard cap on output tokens. Use for concise responses
                         (classifier, edit prompt rewriter) where model may
                         over-generate despite system prompt constraints.
            think: Enable/disable thinking mode for reasoning models (qwen3,
                   qwen3-vl). Set to False when you want fast, direct output
                   without hidden chain-of-thought tokens eating num_predict
                   budget.
        """
        n_images = len(images) if images else 0
        logger.info(
            "POST %s model=%s prompt_len=%d images=%d keep_alive=%s num_ctx=%s num_predict=%s think=%s",
            self.api_url, model, len(prompt), n_images, keep_alive, num_ctx, num_predict, think,
        )
        logger.debug("prompt preview: %s", _truncate(prompt, 300))
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }

            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            if think is not None:
                payload["think"] = think

            options = {}
            if num_ctx is not None:
                options["num_ctx"] = num_ctx
            if num_predict is not None:
                options["num_predict"] = num_predict
            if options:
                payload["options"] = options

            if images:
                payload["images"] = images

            start_time = time.perf_counter()
            async with ClientSession(timeout=_OLLAMA_TIMEOUT) as session:
                async with session.post(self.api_url, json=payload) as resp:
                    data = await resp.json()
                    if resp.status >= 400 or "error" in data:
                        err = data.get("error") or f"HTTP {resp.status}"
                        logger.error("api error model=%s status=%s error=%s",
                                     model, resp.status, err)
                        raise RuntimeError(f"Ollama API error ({model}): {err}")
                    if "response" not in data:
                        logger.error("missing response field model=%s body=%s",
                                     model, _truncate(str(data), 300))
                        raise RuntimeError(
                            f"Ollama returned no 'response' field for {model}: {data}"
                        )
                    response = data["response"]

            duration = time.perf_counter() - start_time
            logger.info(
                "response model=%s took=%.2fs response_len=%d eval_count=%s prompt_eval_count=%s",
                model, duration, len(response),
                data.get("eval_count"), data.get("prompt_eval_count"),
            )
            logger.debug("response preview: %s", _truncate(response, 300))
            return response
        except Exception as e:
            logger.error("exception in generate model=%s: %s", model, e)
            logger.debug("traceback:\n%s", traceback.format_exc())
            raise

    async def classify_image_task(self, prompt: str) -> bool:
        """Check if a prompt is requesting image generation OR an edit of
        either a previously generated bot image or a user-attached image."""
        system_prompt = (
            "You are a classifier. Decide whether the user's message is a request to "
            "generate a new image OR to modify / edit an existing image. "
            "Your response must be exactly 'Yes' or 'No', nothing else.\n\n"
            "Return 'Yes' for:\n"
            "- Direct generation requests: 'generate an image of a cat', 'draw me a dragon', "
            "'create a picture of a sunset', 'make me a photo of...', 'paint a portrait of...'\n"
            "- Edit/modification requests when the prompt contains either:\n"
            "    * a '[Previous: [Generated an image...]]' marker (follow-up on a bot image), OR\n"
            "    * a '[User attached an image]' marker (editing a freshly uploaded image)\n"
            "  In those cases, almost any natural-language visual instruction is an edit request:\n"
            "    'make her hair green'\n"
            "    'change the background to a beach'\n"
            "    'add a hat'\n"
            "    'remove the cat'\n"
            "    'make it nighttime'\n"
            "    'higher quality'\n"
            "    'portrait version'\n"
            "    'zoom in on the face'\n"
            "    'turn this into an oil painting'\n"
            "    'make him a cowboy'\n"
            "\n"
            "Return 'No' for:\n"
            "- Questions ABOUT an existing image ('what is in this image?', 'who is that?', "
            "'describe this', 'is this real?', 'do i look cute')\n"
            "- Normal conversation unrelated to the image\n"
            "- Requests for information, advice, search, or code\n"
            "- Refusals or 'nevermind' messages\n"
            "\n"
            "If the message contains a marker AND the user's text describes a visual change, "
            "the answer is Yes. If the user is only asking a question about the image, it is No."
        )

        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        response = await self.generate(
            full_prompt, model=CHAT_MODEL, keep_alive=-1, num_ctx=4096, think=False,
        )

        return "yes" in response.lower()

    async def generate_image_prompt(self, prompt: str) -> str:
        """Generate an image generation prompt from user input"""
        system_prompt = (
            "You are a tool that rewrites a user's image request into a detailed prompt for "
            "a diffusion image model. Output ONLY the rewritten prompt text.\n\n"
            "STRICT OUTPUT RULES — the response must:\n"
            "- Be a single natural-language paragraph, no line breaks\n"
            "- Contain NO markdown, headings, bullet points, numbered lists, or code blocks\n"
            "- Contain NO quotation marks around the prompt\n"
            "- Contain NO explanations, preamble, or commentary ('Here is...', 'Okay...', etc.)\n"
            "- Contain NO follow-up questions or requests for clarification\n"
            "- Never offer multiple options or variations\n"
            "- If the user's request is ambiguous, silently make reasonable assumptions "
            "and commit to one interpretation\n\n"
            "The rewritten prompt should be highly visual, include details about composition, "
            "style, lighting, and subject. If the user input includes literal text to render "
            "(e.g. 'a sign that says Hello'), preserve that text exactly in the output.\n\n"
            "Example input: 'generate a photorealistic image of charlie puth'\n"
            "Example output: photorealistic portrait of charlie puth, soft studio lighting, "
            "detailed facial features, shallow depth of field, high resolution"
        )

        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        raw = await self.generate(
            full_prompt, model=TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL,
            keep_alive=-1, num_ctx=4096, think=False,
        )
        return _sanitize_prompt_output(raw)

    async def modify_image_prompt(self, original_prompt: str, modification: str) -> str:
        """Modify an existing image generation prompt based on user instructions"""
        system_prompt = (
            "You are a tool that edits an existing image generation prompt based on a user's "
            "change request. Output ONLY the modified prompt text.\n\n"
            "STRICT OUTPUT RULES — the response must:\n"
            "- Be a single natural-language paragraph, no line breaks\n"
            "- Contain NO markdown, headings, bullet points, numbered lists, or code blocks\n"
            "- Contain NO quotation marks around the prompt\n"
            "- Contain NO preamble, commentary, or explanation\n"
            "- Contain NO follow-up questions\n"
            "- Never offer multiple options or variations\n"
            "- Silently commit to one interpretation of any ambiguity\n\n"
            "Preserve as much of the original prompt as possible. Only change the parts the "
            "user's modification explicitly affects. Keep all style, quality, lighting, and "
            "detail descriptors from the original unless the user asks to change them."
        )

        full_prompt = (
            f"System: {system_prompt}\n"
            f"Original prompt: {original_prompt}\n"
            f"User modification: {modification}\n"
            f"Assistant:"
        )
        raw = await self.generate(
            full_prompt, model=TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL,
            keep_alive=-1, num_ctx=4096, think=False,
        )
        return _sanitize_prompt_output(raw)

    async def describe_image_for_edit(
        self, image_base64: str, user_instruction: str
    ) -> str:
        """Use the vision model to look at an attached image and build a concise
        edit prompt combining a short description of the source with the user's
        instruction.

        Used when the user uploads a fresh image and asks for a visual change
        ("make her hair green") — we don't have an original diffusion prompt to
        modify, so we synthesize one via the VLM.
        """
        system_prompt = (
            "You write concise image-editing prompts. Given a source image and a "
            "user edit instruction, output ONE SENTENCE (max 50 words) describing "
            "what the edited image should look like.\n\n"
            "Rules:\n"
            "- 50 words maximum, no more\n"
            "- Single sentence or fragment, no line breaks\n"
            "- No preamble, no markdown, no options, no follow-up questions\n"
            "- Never say 'here is', 'based on', 'the edited image', etc.\n"
            "- Mention subject, composition, and the requested change only\n"
            "- Preserve the source's pose/lighting/style unless the edit says otherwise"
        )
        user_prompt = f"Edit instruction: {user_instruction}\n\nOutput the prompt now:"
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        raw = await self.generate(
            full_prompt,
            # Deliberately not qwen3-vl: its thinking mode swallows num_predict
            # and returns empty. gemma3:27b is multimodal and non-reasoning.
            model=IMAGE_EDIT_DESCRIPTION_MODEL,
            images=[image_base64],
            num_ctx=4096,
            num_predict=150,
            keep_alive=-1,
        )
        return _sanitize_prompt_output(raw)

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

        response = await self.generate(
            full_prompt,
            model=NSFW_CLASSIFICATION_MODEL,
            images=images,
            num_ctx=4096,
            num_predict=16,  # "NSFW"/"SFW" is 1-2 tokens
            keep_alive=-1,
            think=False,
        )
        return "nsfw" in response.lower()
