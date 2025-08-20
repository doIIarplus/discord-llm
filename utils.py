"""Utility functions for Discord LLM Bot"""

import base64
import datetime
import time
from typing import List


def timestamp() -> str:
    """Generate a timestamp string for file naming"""
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path: str) -> str:
    """Encode a file to base64 string"""
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    return encode_file_to_base64(image_path)


def encode_images_to_base64(image_paths: List[str]) -> List[str]:
    """Encode multiple image files to base64 strings"""
    return [encode_image_to_base64(path) for path in image_paths]