"""Image generation module for Discord LLM Bot"""

from typing import Optional, Tuple

from models import ImageInfo
from ollama_client import OllamaClient
from stable_diffusion_client import StableDiffusionClient
from utils import encode_image_to_base64


class ImageGenerator:
    """Handles image generation tasks"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.sd_client = StableDiffusionClient()
        
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: int = -1,
        width: int = 832,
        height: int = 1216,
        cfg_scale: float = 3.0,
        steps: int = 30,
        upscale: float = 1.0,
        allow_nsfw: bool = True,
    ) -> Tuple[str, ImageInfo, bool]:
        """Generate an image using Stable Diffusion"""
        
        # Apply constraints
        width = min(1500, width)
        height = min(2000, height)
        steps = min(60, max(2, steps))
        upscale = min(2, max(1, upscale))
        cfg_scale = min(7, max(1.5, cfg_scale))
        
        # Build prompts
        baseline_positive_prompt = (
            ""
        )

        nsfw_negative = "nude, nsfw, naked, sex, sexy, nipples, areola, bare skin, penis, vagina, lingerie, boobs, boobies, breasts, underboob, sideboob, cleavage, see-through, transparent clothing, thong, cameltoe, upskirt, lewd, erotic, suggestive, sexually explicit, topless, exposed genitals, exposed breasts, porn, hentai, pubic hair, butt, ass, spread legs, open legs, skimpy, sensual pose, seductive, sex appeal, fetish, overly detailed anatomy, sexual content, "

        if not allow_nsfw:
            baseline_positive_prompt += "pg rated, safe for work, fully clothed, "

        baseline_negative_prompt = (
            "bad quality, worst quality, lowres, jpeg artifacts, bad anatomy, bad hands, multiple views, signature, watermark, censored, ugly, (messy), abstract, (too_many:1.3), "
        )

        if "kaling" in prompt:
            prompt = prompt.replace("kaling", "kaling, ms,  <lora:Kaling-ILXL-V1:1> ")
        
        if "len" in prompt:
            prompt = prompt.replace("len", "len, maplestory, long hair,white hair,ponytail,long sleeves,streaked hair, red eyes,rabbit ears,animal ears,white jacket, jacket,shoulder cutout,wide sleeves,hair bow, boots,skirt,pleated skirt, black skirt,fingerless gloves,thighhighs, white thighhighs, <lora:Len2:1>, ")
        
        if "ren" in prompt:
            prompt = prompt.replace("ren", "len, maplestory, long hair,white hair,ponytail,long sleeves,streaked hair, red eyes,rabbit ears,animal ears,white jacket, jacket,shoulder cutout,wide sleeves,hair bow, boots,skirt,pleated skirt, black skirt,fingerless gloves,thighhighs, white thighhighs, <lora:Len2:1>, ")

        if not allow_nsfw:
            baseline_negative_prompt += nsfw_negative
        
        all_negatives = baseline_negative_prompt.split(",")
        all_negatives = [item.strip() for item in all_negatives]

        for negative in all_negatives:
            if negative in prompt:
                prompt = prompt.replace(negative, "")
                    
        prompt = baseline_positive_prompt + prompt
        negative_prompt = baseline_negative_prompt + (negative_prompt or "")
        
        sampler = "Euler"
        
        # Build payload
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler,
            "n_iter": 1,
            "batch_size": 1,
            "override_settings": {
                "CLIP_stop_at_last_layers": 1,
            },
        }
        
        # Handle upscaling
        if upscale > 1.0:
            payload["enable_hr"] = True
            payload["hr_upscaler"] = "4x_foolhardy_Remacri"
            payload["hr_scale"] = upscale
            payload["hr_sampler_name"] = sampler
            payload["hr_second_pass_steps"] = steps
            payload["denoising_strength"] = 0.5
            
        # Add face detailer
        payload["alwayson_scripts"] = {
            "ADetailer": {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt"
                    }
                ]
            }
        }
        
        # Generate image
        file_path, image_info = self.sd_client.call_txt2img_api(**payload)
        
        # Check for NSFW
        image_base64 = encode_image_to_base64(file_path)
        is_nsfw = await self.ollama_client.classify_nsfw([image_base64])
        
        return file_path, image_info, is_nsfw
        
    async def is_image_generation_task(self, prompt: str) -> bool:
        """Check if a prompt is requesting image generation"""
        return await self.ollama_client.classify_image_task(prompt)
        
    async def generate_image_prompt(self, user_prompt: str) -> str:
        """Generate an optimized prompt for image generation"""
        return await self.ollama_client.generate_image_prompt(user_prompt)