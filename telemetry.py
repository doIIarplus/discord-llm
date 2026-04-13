"""Telemetry DB for image generation + Ollama API calls.

Records every Ollama generate() call and every Flux image generation to a
local SQLite database so the full request/response history can be audited
after the fact (debugging, latency analysis, prompt archaeology).

The database lives at `generation_history.db` in the project root and is
write-behind async — callers use the `record_*` helpers which do their
work on a background thread via `asyncio.to_thread` so the main async
loop isn't blocked on SQLite I/O.

Tables:
    ollama_calls
        Every call to the Ollama API with full prompt + response text,
        timing, and eval counts.

    image_generations
        Every Flux2 Klein generation (txt2img + img2img) with the prompt,
        seed, dimensions, steps, cfg, duration, and output path.
"""

import asyncio
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("telemetry")

_DB_PATH = os.path.join(
    os.path.realpath(os.path.dirname(os.path.abspath(__file__))),
    "generation_history.db",
)

_CONN_LOCK = threading.Lock()
_INIT_DONE = False


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sync():
    global _INIT_DONE
    with _CONN_LOCK:
        if _INIT_DONE:
            return
        conn = _get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ollama_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_len INTEGER NOT NULL,
                    num_images INTEGER NOT NULL DEFAULT 0,
                    num_ctx INTEGER,
                    num_predict INTEGER,
                    keep_alive INTEGER,
                    think INTEGER,
                    response TEXT,
                    response_len INTEGER,
                    eval_count INTEGER,
                    prompt_eval_count INTEGER,
                    duration_s REAL NOT NULL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_ollama_ts ON ollama_calls(ts);
                CREATE INDEX IF NOT EXISTS ix_ollama_model ON ollama_calls(model);

                CREATE TABLE IF NOT EXISTS image_generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_len INTEGER NOT NULL,
                    source_path TEXT,
                    source_width INTEGER,
                    source_height INTEGER,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    steps INTEGER NOT NULL,
                    guidance_scale REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    sampler TEXT,
                    output_path TEXT,
                    duration_s REAL NOT NULL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_imgs_ts ON image_generations(ts);
                CREATE INDEX IF NOT EXISTS ix_imgs_mode ON image_generations(mode);
                """
            )
            conn.commit()
        finally:
            conn.close()
        _INIT_DONE = True
        logger.info("initialized telemetry db at %s", _DB_PATH)


def _record_ollama_sync(
    model: str,
    prompt: str,
    response: Optional[str],
    *,
    num_images: int = 0,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    keep_alive: Optional[int] = None,
    think: Optional[bool] = None,
    eval_count: Optional[int] = None,
    prompt_eval_count: Optional[int] = None,
    duration_s: float = 0.0,
    error: Optional[str] = None,
):
    _init_sync()
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO ollama_calls
               (ts, model, prompt, prompt_len, num_images, num_ctx, num_predict,
                keep_alive, think, response, response_len,
                eval_count, prompt_eval_count, duration_s, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), model, prompt, len(prompt), num_images,
                num_ctx, num_predict, keep_alive,
                (1 if think else 0) if think is not None else None,
                response, len(response) if response else 0,
                eval_count, prompt_eval_count, duration_s, error,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning("failed to record ollama call: %s", e)
    finally:
        conn.close()


def _record_image_sync(
    *,
    mode: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    sampler: Optional[str] = None,
    source_path: Optional[str] = None,
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    output_path: Optional[str] = None,
    duration_s: float = 0.0,
    error: Optional[str] = None,
):
    _init_sync()
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO image_generations
               (ts, mode, model, prompt, prompt_len, source_path,
                source_width, source_height, width, height,
                steps, guidance_scale, seed, sampler, output_path,
                duration_s, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), mode, model, prompt, len(prompt),
                source_path, source_width, source_height,
                width, height, steps, guidance_scale, seed,
                sampler, output_path, duration_s, error,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning("failed to record image generation: %s", e)
    finally:
        conn.close()


async def record_ollama_call(**kwargs):
    """Async wrapper — runs the sqlite insert on a background thread."""
    try:
        await asyncio.to_thread(_record_ollama_sync, **kwargs)
    except Exception as e:
        logger.warning("record_ollama_call failed: %s", e)


async def record_image_generation(**kwargs):
    """Async wrapper — runs the sqlite insert on a background thread."""
    try:
        await asyncio.to_thread(_record_image_sync, **kwargs)
    except Exception as e:
        logger.warning("record_image_generation failed: %s", e)
