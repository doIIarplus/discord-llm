"""Stable Diffusion API client for Discord LLM Bot"""

import base64
import json
import os
import urllib.request
from typing import Tuple

from config import SD_API_URL, OUTPUT_DIR_T2I, OUTPUT_DIR_I2I
from models import ImageInfo
from utils import timestamp


class StableDiffusionClient:
    """Client for interacting with Stable Diffusion API"""
    
    def __init__(self):
        self.api_url = SD_API_URL
        
    def call_api(self, api_endpoint: str, **payload) -> dict:
        """Make a call to the Stable Diffusion API"""
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.api_url}/{api_endpoint}"
        request = urllib.request.Request(
            url,
            headers={"Content-Type": "application/json"},
            data=data,
        )
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))
        
    def decode_and_save_base64(self, base64_str: str, save_path: str):
        """Decode base64 string and save to file"""
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))
            
    def call_txt2img_api(self, **payload) -> Tuple[str, ImageInfo]:
        """Call text-to-image API and save the result"""
        response = self.call_api("sdapi/v1/txt2img", **payload)
        info = json.loads(response.get("info"))
        
        image_info = ImageInfo(
            sampler_name=info.get("sampler_name"),
            steps=info.get("steps"),
            cfg_scale=info.get("cfg_scale"),
            width=info.get("width"),
            height=info.get("height"),
            seed=info.get("seed"),
        )
        
        for index, image in enumerate(response.get("images")):
            save_path = os.path.join(OUTPUT_DIR_T2I, f"txt2img-{timestamp()}-{index}.png")
            self.decode_and_save_base64(image, save_path)
            
        return save_path, image_info
        
    def call_img2img_api(self, **payload) -> str:
        """Call image-to-image API and save the result"""
        response = self.call_api("sdapi/v1/img2img", **payload)
        for index, image in enumerate(response.get("images")):
            save_path = os.path.join(OUTPUT_DIR_I2I, f"img2img-{timestamp()}-{index}.png")
            self.decode_and_save_base64(image, save_path)
        return save_path