"""Flux2 Klein 9B image generation client.

Wraps the Flux2KleinPipeline from diffusers for text-to-image and
image-to-image generation. Lazy-loads the model on first use and
auto-unloads after 5 minutes of inactivity to free VRAM.
"""

import asyncio
import os
import time
import threading
from typing import Optional, Tuple

import torch
from PIL import Image

from config import (
    PROJECT_DIR,
    OUTPUT_DIR_T2I,
    OUTPUT_DIR_I2I,
    FLUX_MODEL_ID,
    FLUX_DEFAULT_STEPS,
    FLUX_DEFAULT_GUIDANCE,
)
from models import ImageInfo

# Auto-unload after this many seconds of inactivity
UNLOAD_TIMEOUT = 300  # 5 minutes


class FluxClient:
    """Flux2 Klein 9B image generation client with lazy loading and auto-unload."""

    def __init__(self):
        self._pipe = None
        self._last_used: float = 0
        self._lock = threading.Lock()
        self._unload_timer: Optional[threading.Timer] = None

    def _ensure_loaded(self):
        """Load the pipeline if not already in VRAM."""
        if self._pipe is not None:
            self._last_used = time.time()
            return

        with self._lock:
            if self._pipe is not None:
                return

            print("[flux] Loading Flux2 Klein 9B model...")
            start = time.perf_counter()

            from diffusers import Flux2KleinPipeline

            self._pipe = Flux2KleinPipeline.from_pretrained(
                FLUX_MODEL_ID,
                torch_dtype=torch.bfloat16,
            )
            self._pipe.to("cuda")

            duration = time.perf_counter() - start
            print(f"[flux] Model loaded in {duration:.1f}s")
            self._last_used = time.time()
            self._schedule_unload()

    def _schedule_unload(self):
        """Schedule auto-unload after UNLOAD_TIMEOUT seconds."""
        if self._unload_timer:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(UNLOAD_TIMEOUT, self._check_unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _check_unload(self):
        """Unload if idle for UNLOAD_TIMEOUT seconds."""
        if self._pipe is None:
            return
        idle = time.time() - self._last_used
        if idle >= UNLOAD_TIMEOUT:
            self.unload()
        else:
            # Not idle enough, reschedule
            remaining = UNLOAD_TIMEOUT - idle
            self._unload_timer = threading.Timer(remaining, self._check_unload)
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def unload(self):
        """Unload the model from VRAM."""
        with self._lock:
            if self._pipe is not None:
                del self._pipe
                self._pipe = None
                torch.cuda.empty_cache()
                print("[flux] Model unloaded from VRAM")
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None

    def _generate_sync(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        seed: int = -1,
        width: int = 1024,
        height: int = 1024,
        steps: int = FLUX_DEFAULT_STEPS,
        guidance_scale: float = FLUX_DEFAULT_GUIDANCE,
    ) -> Tuple[str, ImageInfo]:
        """Synchronous generation (called via asyncio.to_thread).

        Flux2 Klein uses the `image` arg as a reference/condition, not a noisy
        init, so there is no `strength` parameter — the pipeline decides how
        much the output diverges from the reference based on the prompt alone.
        """
        self._ensure_loaded()

        width = min(1536, max(256, width))
        height = min(1536, max(256, height))
        width = (width // 64) * 64
        height = (height // 64) * 64
        steps = min(20, max(1, steps))

        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if image is not None:
            kwargs["image"] = image.convert("RGB").resize((width, height), Image.LANCZOS)

        print(f"[flux] Generating {'img2img' if image else 'txt2img'}: "
              f"{width}x{height}, {steps} steps, seed={seed}")
        start = time.perf_counter()

        result = self._pipe(**kwargs)
        output_image = result.images[0]

        duration = time.perf_counter() - start
        print(f"[flux] Generated in {duration:.1f}s")

        # Save to disk
        timestamp = int(time.time() * 1000)
        if image is not None:
            save_dir = OUTPUT_DIR_I2I
            filename = f"img2img-{timestamp}-0.png"
        else:
            save_dir = OUTPUT_DIR_T2I
            filename = f"txt2img-{timestamp}-0.png"

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        output_image.save(save_path)

        info = ImageInfo(
            sampler_name="Flux2Klein",
            steps=steps,
            cfg_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )

        self._last_used = time.time()
        self._schedule_unload()

        return save_path, info

    async def generate(
        self,
        prompt: str,
        seed: int = -1,
        width: int = 1024,
        height: int = 1024,
        steps: int = FLUX_DEFAULT_STEPS,
        guidance_scale: float = FLUX_DEFAULT_GUIDANCE,
    ) -> Tuple[str, ImageInfo]:
        """Generate a new image from text. Returns (file_path, ImageInfo)."""
        return await asyncio.to_thread(
            self._generate_sync,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
        )

    async def edit(
        self,
        prompt: str,
        image: Image.Image,
        seed: int = -1,
        width: int = 1024,
        height: int = 1024,
        steps: int = FLUX_DEFAULT_STEPS,
        guidance_scale: float = FLUX_DEFAULT_GUIDANCE,
    ) -> Tuple[str, ImageInfo]:
        """Edit an existing image with a text prompt. Returns (file_path, ImageInfo)."""
        return await asyncio.to_thread(
            self._generate_sync,
            prompt=prompt,
            image=image,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
        )
