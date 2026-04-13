"""Flux2 Klein 9B image generation client.

Wraps the Flux2KleinPipeline from diffusers for text-to-image and
image-to-image generation. Lazy-loads the model on first use and
auto-unloads after 5 minutes of inactivity to free VRAM.
"""

import asyncio
import json
import logging
import os
import time
import threading
from typing import Optional, Tuple

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from config import (
    PROJECT_DIR,
    OUTPUT_DIR_T2I,
    OUTPUT_DIR_I2I,
    FLUX_MODEL_ID,
    FLUX_DEFAULT_STEPS,
    FLUX_DEFAULT_GUIDANCE,
)
from models import ImageInfo

logger = logging.getLogger("flux")

# Auto-unload after this many seconds of inactivity.
# 0 disables auto-unload entirely (keep the pipeline resident for the
# lifetime of the process). Cold loads are expensive (~60s from warm disk,
# up to 4+ minutes under memory pressure) so we prefer pinning.
UNLOAD_TIMEOUT = 0  # disabled


def _vram_str() -> str:
    """Return a compact VRAM usage string."""
    try:
        if not torch.cuda.is_available():
            return "cpu"
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"alloc={allocated:.1f}GB reserved={reserved:.1f}GB total={total:.1f}GB"
    except Exception:
        return "unknown"


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
            logger.debug("pipe warm, reusing (%s)", _vram_str())
            self._last_used = time.time()
            return

        with self._lock:
            if self._pipe is not None:
                return

            logger.info("cold load of %s (pre-load: %s)", FLUX_MODEL_ID, _vram_str())
            start = time.perf_counter()

            from diffusers import Flux2KleinPipeline

            self._pipe = Flux2KleinPipeline.from_pretrained(
                FLUX_MODEL_ID,
                torch_dtype=torch.bfloat16,
            )
            logger.info("pipeline constructed (%.1fs), moving to cuda", time.perf_counter() - start)
            mv_start = time.perf_counter()
            self._pipe.to("cuda")
            logger.info("moved to cuda in %.1fs", time.perf_counter() - mv_start)

            duration = time.perf_counter() - start
            logger.info("cold load complete total=%.1fs (%s)", duration, _vram_str())
            self._last_used = time.time()
            self._schedule_unload()

    def _schedule_unload(self):
        """Schedule auto-unload. No-op when UNLOAD_TIMEOUT is 0 (pinned mode)."""
        if UNLOAD_TIMEOUT <= 0:
            return
        if self._unload_timer:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(UNLOAD_TIMEOUT, self._check_unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _check_unload(self):
        """Unload if idle for UNLOAD_TIMEOUT seconds."""
        if UNLOAD_TIMEOUT <= 0 or self._pipe is None:
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

    async def warmup(self):
        """Preload the pipeline by triggering a minimal generation.

        Intended for bot startup: cold-loading Flux on the first user request
        costs 60+ seconds. Calling this once at startup shifts that cost out
        of the user-facing critical path.
        """
        logger.info("warmup start")
        t0 = time.perf_counter()
        # Trigger _ensure_loaded via a real generate of a tiny image
        await self.generate(
            prompt="warmup", seed=0, width=256, height=256, steps=1, guidance_scale=1.0,
        )
        logger.info("warmup done in %.1fs", time.perf_counter() - t0)

    def unload(self):
        """Unload the model from VRAM."""
        with self._lock:
            if self._pipe is not None:
                del self._pipe
                self._pipe = None
                torch.cuda.empty_cache()
                logger.info("unloaded from vram (%s)", _vram_str())
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
        mode = "img2img" if image is not None else "txt2img"
        logger.info(
            "generate_sync mode=%s requested %dx%d steps=%d guidance=%s seed=%s",
            mode, width, height, steps, guidance_scale, seed,
        )
        logger.debug("prompt: %s", prompt)

        self._ensure_loaded()

        orig_w, orig_h = width, height
        width = min(1536, max(256, width))
        height = min(1536, max(256, height))
        width = (width // 64) * 64
        height = (height // 64) * 64
        steps = min(20, max(1, steps))
        if (width, height) != (orig_w, orig_h):
            logger.info("clamped dims %dx%d -> %dx%d", orig_w, orig_h, width, height)

        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            seed = int(torch.randint(0, 2**32, (1,)).item())
            generator = torch.Generator(device="cuda").manual_seed(seed)
            logger.info("seed was -1, randomized to %d", seed)

        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if image is not None:
            src_w, src_h = image.size
            kwargs["image"] = image.convert("RGB").resize((width, height), Image.LANCZOS)
            logger.info("img2img source %dx%d resized to %dx%d", src_w, src_h, width, height)

        logger.info(
            "pipeline call mode=%s %dx%d steps=%d guidance=%s seed=%d (%s)",
            mode, width, height, steps, guidance_scale, seed, _vram_str(),
        )
        start = time.perf_counter()

        result = self._pipe(**kwargs)
        output_image = result.images[0]

        duration = time.perf_counter() - start
        logger.info("pipeline done in %.2fs", duration)

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

        # Embed generation metadata in PNG text chunks so the params can be
        # recovered later via PIL / exiftool / a1111-compatible tooling.
        meta = PngInfo()
        sampler = type(self._pipe.scheduler).__name__
        # Stable Diffusion WebUI / civitai-compatible "parameters" block
        params_block = (
            f"{prompt}\n"
            f"Steps: {steps}, Sampler: {sampler}, CFG scale: {guidance_scale}, "
            f"Seed: {seed}, Size: {width}x{height}, "
            f"Model: {FLUX_MODEL_ID}"
        )
        meta.add_text("parameters", params_block)
        # Explicit machine-readable keys
        meta.add_text("flux_prompt", prompt)
        meta.add_text("flux_seed", str(seed))
        meta.add_text("flux_steps", str(steps))
        meta.add_text("flux_guidance_scale", str(guidance_scale))
        meta.add_text("flux_width", str(width))
        meta.add_text("flux_height", str(height))
        meta.add_text("flux_sampler", sampler)
        meta.add_text("flux_model", FLUX_MODEL_ID)
        meta.add_text("flux_mode", "img2img" if image is not None else "txt2img")
        meta.add_text("flux_duration_s", f"{duration:.3f}")
        if image is not None:
            meta.add_text("flux_source_size", f"{src_w}x{src_h}")

        output_image.save(save_path, pnginfo=meta)
        logger.info("saved %s", save_path)

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
        import telemetry
        t0 = time.perf_counter()
        err = None
        path = None
        info = None
        try:
            path, info = await asyncio.to_thread(
                self._generate_sync,
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            return path, info
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            try:
                await telemetry.record_image_generation(
                    mode="txt2img",
                    model=FLUX_MODEL_ID,
                    prompt=prompt,
                    width=info.width if info else width,
                    height=info.height if info else height,
                    steps=info.steps if info else steps,
                    guidance_scale=info.cfg_scale if info else guidance_scale,
                    seed=info.seed if info else seed,
                    sampler=info.sampler_name if info else None,
                    output_path=path,
                    duration_s=time.perf_counter() - t0,
                    error=err,
                )
            except Exception as tel_err:
                logger.debug("telemetry record failed: %s", tel_err)

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
        import telemetry
        t0 = time.perf_counter()
        src_w, src_h = image.size
        err = None
        path = None
        info = None
        try:
            path, info = await asyncio.to_thread(
                self._generate_sync,
                prompt=prompt,
                image=image,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            return path, info
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            try:
                await telemetry.record_image_generation(
                    mode="img2img",
                    model=FLUX_MODEL_ID,
                    prompt=prompt,
                    source_width=src_w,
                    source_height=src_h,
                    width=info.width if info else width,
                    height=info.height if info else height,
                    steps=info.steps if info else steps,
                    guidance_scale=info.cfg_scale if info else guidance_scale,
                    seed=info.seed if info else seed,
                    sampler=info.sampler_name if info else None,
                    output_path=path,
                    duration_s=time.perf_counter() - t0,
                    error=err,
                )
            except Exception as tel_err:
                logger.debug("telemetry record failed: %s", tel_err)
