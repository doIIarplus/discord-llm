"""Utility functions for Discord LLM Bot"""

import base64
import datetime
import io
import time
from typing import List

from sandbox import safe_path


def timestamp() -> str:
    """Generate a timestamp string for file naming"""
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path: str) -> str:
    """Encode a file to base64 string"""
    path = safe_path(path)
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    return encode_file_to_base64(image_path)


def encode_images_to_base64(image_paths: List[str]) -> List[str]:
    """Encode multiple image files to base64 strings"""
    return [encode_image_to_base64(path) for path in image_paths]


def encode_image_downsized_to_base64(image_path: str, max_side: int = 512) -> str:
    """Encode an image downsized to fit within max_side on the longest edge.

    Used to send smaller payloads to vision-model classifiers (e.g. NSFW
    check) where full resolution is wasted on token budget.
    """
    from PIL import Image

    image_path = safe_path(image_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            if w >= h:
                new_w = max_side
                new_h = int(h * max_side / w)
            else:
                new_h = max_side
                new_w = int(w * max_side / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")