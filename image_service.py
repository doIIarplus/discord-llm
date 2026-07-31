"""Localhost-only HTTP shim over the bot's resident Flux pipeline.

Why this exists
---------------
`FluxClient` lazy-loads Flux2 Klein 9B into VRAM inside the bot process and
holds it for 5 minutes. Claude-invoked CLI tools run as separate subprocesses,
so a tool that imported Flux directly would load a *second* copy of a 9B model
on every call — a VRAM spike plus a cold model load per image.

Instead the bot runs this tiny aiohttp service and `tools/images/*.py` become
thin HTTP clients. One model copy, no reload, and the existing NSFW
classification / auto-unload behavior is reused verbatim.

The service also exposes /attach, which registers an image Claude produced
*programmatically* (matplotlib, PIL, an SVG it rendered) for delivery through
the same path. Diffusion and code-drawn images therefore reach Discord
identically, and Claude picks whichever fits the request.

Security
--------
- Binds to loopback only (IMAGE_SERVICE_HOST, default 127.0.0.1).
- Requires a bearer token generated at startup and written to
  IMAGE_SERVICE_TOKEN_FILE with mode 0600. Anything without it gets 401.
- `image_path` on /edit is confined to PROJECT_DIR so a prompt-injected tool
  call can't read arbitrary files off disk into a generated image.

Pending attachments
-------------------
Each Claude turn gets a request id (env `IMAGE_REQUEST_ID`, forwarded to tool
subprocesses). Tools echo it back on each call, and the service records the
produced paths under that id. After Claude's turn the bot drains the bucket and
attaches the images to its normal reply — so typing simulation, response
splitting, and NSFW spoilering all still apply. No file scraping, no regex.
"""

import asyncio
import logging
import os
import secrets
from collections import defaultdict
from typing import Dict, List, Optional

from aiohttp import web
from PIL import Image

from config import (
    IMAGE_SERVICE_HOST,
    IMAGE_SERVICE_PORT,
    IMAGE_SERVICE_TOKEN_FILE,
    PROJECT_DIR,
)

logger = logging.getLogger("image_service")

# Klein tolerates up to 1536/side; keep dims to multiples of 64.
_MIN_SIDE, _MAX_SIDE = 256, 1536


def _clamp_side(value, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    v = min(_MAX_SIDE, max(_MIN_SIDE, v))
    return (v // 64) * 64


class ImageService:
    """Wraps an ImageGenerator behind a token-guarded loopback HTTP API."""

    def __init__(self, image_gen):
        self.image_gen = image_gen
        self.token = secrets.token_urlsafe(32)
        self._runner: Optional[web.AppRunner] = None
        # request_id -> [{path, seed, width, height, nsfw, kind}]
        self._pending: Dict[str, List[dict]] = defaultdict(list)
        # Flux is a single GPU pipeline; serialize so two tool calls can't
        # interleave into the same CUDA context.
        self._lock = asyncio.Lock()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/generate", self._handle_generate)
        app.router.add_post("/edit", self._handle_edit)
        app.router.add_post("/attach", self._handle_attach)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, IMAGE_SERVICE_HOST, IMAGE_SERVICE_PORT)
        await site.start()
        self._write_token_file()
        logger.info(
            "image service listening on %s:%s (token file %s)",
            IMAGE_SERVICE_HOST, IMAGE_SERVICE_PORT, IMAGE_SERVICE_TOKEN_FILE,
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        # Only remove the token file if it's still OURS. Another instance (a
        # live bot, a test harness) may have rewritten it in the meantime, and
        # deleting that would silently break its CLI tools.
        try:
            with open(IMAGE_SERVICE_TOKEN_FILE) as f:
                lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
            if len(lines) >= 3 and lines[2] != self.token:
                logger.info("token file belongs to another instance, leaving it alone")
                return
            os.unlink(IMAGE_SERVICE_TOKEN_FILE)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("could not remove image service token file: %s", e)

    def _write_token_file(self) -> None:
        os.makedirs(os.path.dirname(IMAGE_SERVICE_TOKEN_FILE), exist_ok=True)
        # Create with 0600 from the start — never briefly world-readable.
        fd = os.open(IMAGE_SERVICE_TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"{IMAGE_SERVICE_HOST}\n{IMAGE_SERVICE_PORT}\n{self.token}\n")

    # ---------- pending attachments ----------

    def drain(self, request_id: str) -> List[dict]:
        """Pop and return every image produced under `request_id`."""
        if not request_id:
            return []
        return self._pending.pop(request_id, [])

    def discard(self, request_id: str) -> None:
        self._pending.pop(request_id, None)

    # ---------- handlers ----------

    def _authorized(self, request: web.Request) -> bool:
        header = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        return secrets.compare_digest(header[len(prefix):], self.token)

    async def _handle_health(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response({"status": "ok"})

    def _safe_project_path(self, raw: str):
        """Resolve `raw` and confine it to PROJECT_DIR.

        Returns (path, None) on success or (None, error_message). Tool arguments
        are model-driven input, so they must never be able to name a file
        outside the project tree.
        """
        if not raw:
            return None, "path is required"
        resolved = os.path.realpath(raw)
        if not resolved.startswith(PROJECT_DIR + os.sep):
            return None, f"path must be inside {PROJECT_DIR}"
        if not os.path.exists(resolved):
            return None, f"path does not exist: {raw}"
        return resolved, None

    async def _handle_attach(self, request: web.Request) -> web.Response:
        """Register an image Claude created itself (chart, diagram, SVG render).

        No Flux involvement — this just queues an existing file for delivery so
        code-drawn and diffusion images share one path to Discord.
        """
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        path, err = self._safe_project_path(body.get("path") or "")
        if err:
            return web.json_response({"error": err}, status=400)

        ext = os.path.splitext(path)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            return web.json_response(
                {"error": f"unsupported image extension: {ext or '(none)'}"}, status=400
            )

        is_nsfw = False
        if body.get("nsfw_check"):
            # Opt-in: pointless latency for a chart Claude just plotted, but
            # available when it renders something user-supplied.
            try:
                from utils import encode_image_downsized_to_base64
                b64 = encode_image_downsized_to_base64(path, max_side=512)
                is_nsfw = await self.image_gen.ollama_client.classify_nsfw([b64])
            except Exception as e:
                logger.warning("attach nsfw check failed, assuming SFW: %s", e)

        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception as e:
            return web.json_response(
                {"error": f"not a readable image: {type(e).__name__}: {e}"}, status=400
            )

        result = {
            "path": path,
            "seed": None,
            "width": width,
            "height": height,
            "steps": None,
            "nsfw": bool(is_nsfw),
            "kind": "attach",
            "prompt": (body.get("caption") or "").strip(),
        }

        request_id = (body.get("request_id") or "").strip()
        if request_id:
            self._pending[request_id].append(result)
            logger.info("registered programmatic image for request %s: %s", request_id, path)
        else:
            logger.warning("attach with no request_id, will not auto-attach: %s", path)

        return web.json_response(result)

    async def _handle_generate(self, request: web.Request) -> web.Response:
        return await self._run(request, kind="generate")

    async def _handle_edit(self, request: web.Request) -> web.Response:
        return await self._run(request, kind="edit")

    async def _run(self, request: web.Request, kind: str) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return web.json_response({"error": "prompt is required"}, status=400)

        width = _clamp_side(body.get("width"), 1024)
        height = _clamp_side(body.get("height"), 1024)
        try:
            seed = int(body.get("seed", -1))
        except (TypeError, ValueError):
            seed = -1
        try:
            steps = max(1, min(50, int(body.get("steps", 4))))
        except (TypeError, ValueError):
            steps = 4

        image_path = None
        if kind == "edit":
            image_path, err = self._safe_project_path(body.get("image_path") or "")
            if err:
                return web.json_response({"error": f"image_path: {err}"}, status=400)

        try:
            async with self._lock:
                if kind == "edit":
                    file_path, info, is_nsfw = await self.image_gen.edit_image(
                        prompt, image_path, seed=seed, width=width, height=height,
                        steps=steps,
                    )
                else:
                    file_path, info, is_nsfw = await self.image_gen.generate_image(
                        prompt, seed=seed, width=width, height=height, steps=steps,
                    )
        except Exception as e:
            logger.exception("image service %s failed", kind)
            return web.json_response(
                {"error": f"{type(e).__name__}: {e}"}, status=500
            )

        result = {
            "path": file_path,
            "seed": info.seed,
            "width": info.width,
            "height": info.height,
            "steps": info.steps,
            "nsfw": bool(is_nsfw),
            "kind": kind,
            "prompt": prompt,
        }

        request_id = (body.get("request_id") or "").strip()
        if request_id:
            self._pending[request_id].append(result)
            logger.info("recorded image for request %s: %s", request_id, file_path)
        else:
            # Still usable — Claude gets the path back — but the bot won't
            # auto-attach it, so make that visible rather than silent.
            logger.warning("image produced with no request_id, will not auto-attach: %s", file_path)

        return web.json_response(result)
