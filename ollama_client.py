"""Ollama API client for Discord LLM Bot"""

import traceback
from typing import List, Optional
from aiohttp import ClientSession

from config import (
    OLLAMA_API_URL,
    CHAT_MODEL,
    IMAGE_RECOGNITION_MODEL,
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
                
            async with ClientSession() as session:
                async with session.post(self.api_url, json=payload) as resp:
                    data = await resp.json()
                    return data.get("response", "No response from Ollama.")
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
            "Given the user prompt in plain text, output a diffusion model friendly prompt. Attempt to be as specific as possible, "
            "and separate tags and concepts with commas. For example, if the prompt is 'generate a photorealistic image of charlie puth', a possible prompt "
            "could be 'high definition, photorealistic, charlie puth, 1man'. It's ok to have the tags be more descriptive and akin to natural language. "
            "for example, for the above prompt, a tag could be 'picture of charlie puth' instead of 'charlie puth, photograph'. "
            "Do not include tags such as 'trending on artstation'. "
        )
        
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        return await self.generate(full_prompt, model=TEXT_TO_IMAGE_PROMPT_GENERATION_MODEL)
        
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
        
    async def classify_programmatic_task(self, prompt: str) -> bool:
        """Check if a request can be solved programmatically with Python"""
        system_prompt = (
            "You are a classifier that determines if a user request can be solved by writing and executing a Python script. "
            "Respond with only 'Yes' or 'No'. "
            "Examples of programmatic tasks (answer Yes):\n"
            "- Mathematical calculations (algebra, calculus, statistics)\n"
            "- Counting characters or words in text\n"
            "- Drawing geometric shapes or plots\n"
            "- Data analysis or transformations\n"
            "- Algorithm implementations\n"
            "- Pattern matching or string manipulation\n"
            "- Generating visualizations or charts\n"
            "- Logical puzzles that can be computed\n"
            "Examples of non-programmatic tasks (answer No):\n"
            "- General knowledge questions\n"
            "- Opinions or subjective discussions\n"
            "- Current events or news\n"
            "- Creative writing or storytelling\n"
            "- Philosophical questions\n"
            "- Conversational responses\n"
            "- Image generation tasks that can't be easily solved via python scripts, but are more targeted towards stable diffusion like models."
        )
        
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        response = await self.generate(full_prompt, model=CHAT_MODEL)
        
        return "yes" in response.lower()
        
    async def generate_python_script(self, prompt: str) -> str:
        """Generate Python code to solve a programmatic request"""
        system_prompt = (
            "You are a Python code generator. Given a user request, generate ONLY Python code that solves the problem. "
            "Do not include any explanations, comments about the code, or markdown formatting. "
            "Output ONLY the raw Python code that can be directly executed. "
            "The code should print the final result or create visualizations as needed. "
            "For visualizations, use matplotlib and save to a file. "
            "Make sure all output is clear and the answer to the question is obvious from the output. "
            "REJECT any requests that might lead to leaking sensitive information about the system, or causing damage to the host, such as leaking user details, deleting files, or revealing file / folder structure. "
            "Also reject requests that would obviously take too long to compute, such as trying to calculate the 1 millionth fibonacci number or something ridiculous like that."
        )
        
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
        response = await self.generate(full_prompt, model=CHAT_MODEL)
        
        # Clean up the response - remove markdown code blocks if present
        if response.startswith("```python"):
            response = response[9:]  # Remove ```python
        elif response.startswith("```"):
            response = response[3:]  # Remove ```
            
        if response.endswith("```"):
            response = response[:-3]  # Remove trailing ```
            
        return response.strip()