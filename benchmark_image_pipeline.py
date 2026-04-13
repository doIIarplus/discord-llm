#!/usr/bin/env python3
"""Image generation pipeline benchmark.

Measures cold-start and warm-start latency for three scenarios that mirror
the Discord bot's image handling paths. Emits per-stage timings plus a
5Hz sample of system-wide CPU / RAM / disk / GPU metrics per run.

Usage:
    venv/bin/python benchmark_image_pipeline.py --scenarios A B C --runs 3
    venv/bin/python benchmark_image_pipeline.py --scenarios A --runs 5 --skip-cold
    venv/bin/python benchmark_image_pipeline.py --label baseline

Writes JSONL to --output (default: benchmark_results.jsonl) and prints
per-run summaries to stdout.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Any

import psutil

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# pynvml is optional; we fall back to cpu-only metrics if not present
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception as _e:
    _NVML_HANDLE = None
    _NVML = False

OLLAMA_HOST = "http://localhost:11434"
SOURCE_IMAGE = PROJECT_ROOT / "api_out" / "txt2img" / "icon.png"

DEFAULT_PROMPTS = {
    "A": "generate an image of a brown cat in space",
    "B": "make her hair green",
    "C": "change the background to a beach",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    start: float
    end: float
    duration: float


@dataclass
class Sample:
    ts: float
    cpu_pct: float
    ram_used_gb: float
    disk_read_mb: float
    disk_write_mb: float
    gpu_util: Optional[float] = None
    gpu_mem_gb: Optional[float] = None
    gpu_mem_pct: Optional[float] = None
    gpu_power_w: Optional[float] = None
    gpu_temp_c: Optional[float] = None


# ---------------------------------------------------------------------------
# System sampler — 5 Hz background thread
# ---------------------------------------------------------------------------

class SystemSampler:
    def __init__(self, interval_s: float = 0.2):
        self.interval = interval_s
        self.samples: List[Sample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._start_ts: float = 0.0
        self._prev_disk = None

    def start(self):
        self._start_ts = time.perf_counter()
        self._prev_disk = psutil.disk_io_counters()
        psutil.cpu_percent(interval=None)  # prime
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                now = time.perf_counter() - self._start_ts
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                dr = (disk.read_bytes - self._prev_disk.read_bytes) / 1024**2
                dw = (disk.write_bytes - self._prev_disk.write_bytes) / 1024**2
                self._prev_disk = disk

                sample = Sample(
                    ts=now,
                    cpu_pct=cpu,
                    ram_used_gb=mem.used / 1024**3,
                    disk_read_mb=dr,
                    disk_write_mb=dw,
                )
                if _NVML:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                        sample.gpu_util = float(util.gpu)
                        sample.gpu_mem_gb = meminfo.used / 1024**3
                        sample.gpu_mem_pct = 100 * meminfo.used / meminfo.total
                        sample.gpu_power_w = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000
                        sample.gpu_temp_c = float(
                            pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
                        )
                    except Exception:
                        pass
                self.samples.append(sample)
            except Exception:
                # Don't let a sampling error take down the benchmark
                pass


# ---------------------------------------------------------------------------
# Stage tracker
# ---------------------------------------------------------------------------

class StageTracker:
    def __init__(self):
        self.stages: List[Stage] = []
        self._start_ts: float = 0.0

    def start(self):
        self._start_ts = time.perf_counter()
        self.stages = []

    def wrap(self, name: str) -> "_StageCtx":
        return _StageCtx(self, name)


class _StageCtx:
    def __init__(self, tracker: StageTracker, name: str):
        self.tracker = tracker
        self.name = name
        self.t0: float = 0.0

    async def __aenter__(self):
        self.t0 = time.perf_counter() - self.tracker._start_ts
        return self

    async def __aexit__(self, *args):
        t1 = time.perf_counter() - self.tracker._start_ts
        self.tracker.stages.append(
            Stage(name=self.name, start=self.t0, end=t1, duration=t1 - self.t0)
        )


# ---------------------------------------------------------------------------
# Cold-state reset helpers
# ---------------------------------------------------------------------------

async def unload_ollama_models():
    """Force every loaded Ollama model out of memory by sending keep_alive=0."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{OLLAMA_HOST}/api/ps") as r:
            data = await r.json()
            loaded = [m["name"] for m in data.get("models", [])]
        if not loaded:
            print("[cold] no ollama models currently loaded")
            return
        print(f"[cold] unloading {len(loaded)} ollama model(s): {loaded}")
        for name in loaded:
            payload = {"model": name, "prompt": "", "keep_alive": 0, "stream": False}
            try:
                async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload) as r2:
                    await r2.text()
            except Exception as e:
                print(f"[cold] unload error for {name}: {e}")

        # Poll /api/ps until empty (or timeout)
        for _ in range(30):
            async with session.get(f"{OLLAMA_HOST}/api/ps") as r:
                data = await r.json()
                if not data.get("models"):
                    break
            await asyncio.sleep(0.5)
        async with session.get(f"{OLLAMA_HOST}/api/ps") as r:
            data = await r.json()
            still = [m["name"] for m in data.get("models", [])]
            if still:
                print(f"[cold] WARNING: still loaded after unload: {still}")
            else:
                print("[cold] all ollama models unloaded")


async def reset_flux_client(gen):
    import torch, gc
    gen.flux_client.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[cold] flux pipeline released + cuda cache emptied")


async def preload_all(gen):
    """Preload Flux + Ollama models. Simulates what the bot should do at startup.
    Prints the preload cost separately from measured runs."""
    import time
    print("[preload] warming up pipeline...")

    # 1) Flux cold load via a tiny 256x256/1-step generate
    t_flux = time.perf_counter()
    await gen.flux_client.warmup()
    flux_dt = time.perf_counter() - t_flux
    print(f"[preload] flux warmed in {flux_dt:.1f}s")

    # 2) gemma3:27b warm via a trivial classify call (reuses existing prompt path)
    t_gemma = time.perf_counter()
    await gen.is_image_generation_task("generate a cat")
    gemma_dt = time.perf_counter() - t_gemma
    print(f"[preload] gemma3 warmed in {gemma_dt:.1f}s")

    # 3) qwen3-vl:32b warm via a trivial NSFW call
    from utils import encode_image_downsized_to_base64
    t_qwen = time.perf_counter()
    b64 = encode_image_downsized_to_base64(str(SOURCE_IMAGE), max_side=256)
    await gen.ollama_client.classify_nsfw([b64])
    qwen_dt = time.perf_counter() - t_qwen
    print(f"[preload] qwen3-vl warmed in {qwen_dt:.1f}s")

    total = flux_dt + gemma_dt + qwen_dt
    print(f"[preload] total startup cost: {total:.1f}s")
    return {"flux_warmup_s": flux_dt, "gemma_warmup_s": gemma_dt,
            "qwen_warmup_s": qwen_dt, "total_s": total}


# ---------------------------------------------------------------------------
# Scenario implementations — identical to bot.py's image flow
# ---------------------------------------------------------------------------

async def scenario_a_fresh_gen(gen, tracker, user_input: str):
    """Mirrors bot.py Case 3 (brand new generation)."""
    from image_generation import choose_dimensions
    from utils import encode_image_downsized_to_base64

    async with tracker.wrap("classify_image_task"):
        is_img = await gen.is_image_generation_task(user_input)
        if not is_img:
            raise RuntimeError("classifier said not image task")

    async with tracker.wrap("generate_image_prompt"):
        prompt = (await gen.generate_image_prompt(user_input)).strip()

    width, height = choose_dimensions(f"{user_input} {prompt}")

    async with tracker.wrap("flux_generate"):
        file_path, info = await gen.flux_client.generate(
            prompt=prompt, seed=42, width=width, height=height,
            steps=4, guidance_scale=1.0,
        )

    async with tracker.wrap("classify_nsfw"):
        b64 = encode_image_downsized_to_base64(file_path, max_side=512)
        is_nsfw = await gen.ollama_client.classify_nsfw([b64])

    return {
        "file_path": file_path,
        "nsfw": is_nsfw,
        "width": width,
        "height": height,
        "rewritten_prompt": prompt,
    }


async def scenario_b_attached_edit(gen, tracker, user_input: str, source_path: str):
    """Mirrors bot.py Case 2 (user-attached image edit)."""
    from image_generation import choose_source_dimensions
    from utils import encode_image_downsized_to_base64
    from PIL import Image

    with Image.open(source_path) as src:
        src_w, src_h = src.size
    width, height = choose_source_dimensions(user_input, src_w, src_h)

    # Bot.py builds this exact classify_input for attached-image case
    classify_input = f"[User attached an image]\nUser: {user_input}"
    async with tracker.wrap("classify_image_task"):
        is_img = await gen.is_image_generation_task(classify_input)
        if not is_img:
            raise RuntimeError("classifier said not image task")

    async with tracker.wrap("describe_image_for_edit"):
        src_b64 = encode_image_downsized_to_base64(source_path, max_side=512)
        prompt = (
            await gen.ollama_client.describe_image_for_edit(src_b64, user_input)
        ).strip()

    async with tracker.wrap("flux_edit"):
        file_path, info = await gen.flux_client.edit(
            prompt=prompt,
            image=Image.open(source_path).convert("RGB"),
            seed=42, width=width, height=height,
            steps=4, guidance_scale=1.0,
        )

    async with tracker.wrap("classify_nsfw"):
        b64 = encode_image_downsized_to_base64(file_path, max_side=512)
        is_nsfw = await gen.ollama_client.classify_nsfw([b64])

    return {
        "file_path": file_path,
        "nsfw": is_nsfw,
        "width": width,
        "height": height,
        "rewritten_prompt": prompt,
        "source_dims": [src_w, src_h],
    }


async def scenario_c_followup_edit(
    gen, tracker, user_input: str,
    prev_prompt: str, prev_image_path: str,
    prev_seed: int, prev_w: int, prev_h: int,
):
    """Mirrors bot.py Case 1 (follow-up edit on bot-generated image)."""
    from image_generation import choose_followup_dimensions
    from utils import encode_image_downsized_to_base64
    from PIL import Image

    width, height = choose_followup_dimensions(user_input, prev_w, prev_h)

    # Bot.py builds this classify_input for the follow-up case
    prev_marker = (
        f"[Generated an image with the following prompt: {prev_prompt}] "
        f"(seed: {prev_seed}, size: {prev_w}x{prev_h}, path: {prev_image_path})"
    )
    classify_input = f"[Previous: {prev_marker}]\nUser: {user_input}"
    async with tracker.wrap("classify_image_task"):
        is_img = await gen.is_image_generation_task(classify_input)
        if not is_img:
            raise RuntimeError("classifier said not image task")

    async with tracker.wrap("modify_image_prompt"):
        prompt = (
            await gen.ollama_client.modify_image_prompt(prev_prompt, user_input)
        ).strip()

    async with tracker.wrap("flux_edit"):
        file_path, info = await gen.flux_client.edit(
            prompt=prompt,
            image=Image.open(prev_image_path).convert("RGB"),
            seed=prev_seed, width=width, height=height,
            steps=4, guidance_scale=1.0,
        )

    async with tracker.wrap("classify_nsfw"):
        b64 = encode_image_downsized_to_base64(file_path, max_side=512)
        is_nsfw = await gen.ollama_client.classify_nsfw([b64])

    return {
        "file_path": file_path,
        "nsfw": is_nsfw,
        "width": width,
        "height": height,
        "rewritten_prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_one(gen, scenario: str, label: str, args) -> dict:
    tracker = StageTracker()
    sampler = SystemSampler(interval_s=0.2)

    tracker.start()
    sampler.start()
    t_total0 = time.perf_counter()
    err = None
    result = None
    try:
        if scenario == "A":
            result = await scenario_a_fresh_gen(
                gen, tracker, args.prompt_a or DEFAULT_PROMPTS["A"]
            )
        elif scenario == "B":
            result = await scenario_b_attached_edit(
                gen, tracker,
                args.prompt_b or DEFAULT_PROMPTS["B"],
                str(SOURCE_IMAGE),
            )
        elif scenario == "C":
            result = await scenario_c_followup_edit(
                gen, tracker,
                args.prompt_c or DEFAULT_PROMPTS["C"],
                prev_prompt=args.prev_prompt,
                prev_image_path=args.prev_image,
                prev_seed=args.prev_seed,
                prev_w=args.prev_w,
                prev_h=args.prev_h,
            )
        else:
            raise ValueError(f"unknown scenario {scenario}")
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        total = time.perf_counter() - t_total0
        sampler.stop()

    return {
        "label": label,
        "scenario": scenario,
        "total_s": total,
        "stages": [asdict(s) for s in tracker.stages],
        "samples": [asdict(s) for s in sampler.samples],
        "result": result,
        "error": err,
    }


def print_run(r: dict):
    print(
        f"[{r['label']:10s}] scenario={r['scenario']} total={r['total_s']:.2f}s "
        f"{'OK' if not r['error'] else 'ERROR'}"
    )
    for s in r["stages"]:
        print(f"    {s['name']:28s} {s['duration']:7.2f}s")
    if r["error"]:
        first_line = r["error"].split("\n")[0]
        print(f"    ERROR: {first_line}")
    if r["result"]:
        rs = r["result"]
        print(
            f"    -> {rs.get('file_path')} {rs.get('width')}x{rs.get('height')} "
            f"nsfw={rs.get('nsfw')}"
        )


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=["A", "B", "C"],
                    choices=["A", "B", "C"])
    ap.add_argument("--runs", type=int, default=3,
                    help="warm runs per scenario after the cold run")
    ap.add_argument("--skip-cold", action="store_true")
    ap.add_argument("--preload", action="store_true",
                    help="Warm up all models before the first measurement "
                         "(simulates bot startup preload). Preload cost is "
                         "reported separately and cold runs are skipped.")
    ap.add_argument("--output", default="benchmark_results.jsonl")
    ap.add_argument("--label", default=None,
                    help="label prefix for this batch (e.g. 'baseline', 'opt1')")
    ap.add_argument("--prompt-a", default=None)
    ap.add_argument("--prompt-b", default=None)
    ap.add_argument("--prompt-c", default=None)

    # Scenario C requires a previous gen marker — we'll auto-populate from
    # scenario A's result if it was run first. Otherwise these can be set.
    ap.add_argument("--prev-prompt", default=None)
    ap.add_argument("--prev-image", default=None)
    ap.add_argument("--prev-seed", type=int, default=42)
    ap.add_argument("--prev-w", type=int, default=1024)
    ap.add_argument("--prev-h", type=int, default=1024)
    args = ap.parse_args()

    batch_label = args.label or time.strftime("%Y%m%d-%H%M%S")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    from image_generation import ImageGenerator
    gen = ImageGenerator()

    preload_stats = None
    if args.preload:
        # Unload everything first to get a clean preload measurement
        print("\n=========== preload (simulates bot startup) ===========")
        await unload_ollama_models()
        await reset_flux_client(gen)
        await asyncio.sleep(2)
        preload_stats = await preload_all(gen)

    all_results = []
    for scenario in args.scenarios:
        print(f"\n=========== scenario {scenario} ===========")

        # If scenario C and no prev was provided, try to reuse a result from A
        if scenario == "C" and not args.prev_prompt:
            prev_a = next(
                (r for r in all_results
                 if r["scenario"] == "A" and r.get("result") and not r.get("error")),
                None,
            )
            if prev_a:
                rs = prev_a["result"]
                args.prev_prompt = rs["rewritten_prompt"]
                args.prev_image = rs["file_path"]
                args.prev_w = rs["width"]
                args.prev_h = rs["height"]
                print(f"[C] reusing from A: {args.prev_image}")
            else:
                print("[C] WARNING: no scenario A result to build follow-up from; "
                      "using placeholder")
                args.prev_prompt = "a brown cat floating in space"
                args.prev_image = str(SOURCE_IMAGE)

        # In preload mode we skip the per-scenario cold-unload because the
        # first request after preload IS the "warm first request" path.
        if not args.skip_cold and not args.preload:
            await unload_ollama_models()
            await reset_flux_client(gen)
            await asyncio.sleep(3)
            r = await run_one(gen, scenario, f"{batch_label}:{scenario}:cold", args)
            print_run(r)
            all_results.append(r)

        for i in range(args.runs):
            r = await run_one(gen, scenario, f"{batch_label}:{scenario}:warm_{i+1}", args)
            print_run(r)
            all_results.append(r)

    # Append to JSONL
    with output.open("a") as f:
        for r in all_results:
            out = dict(r)
            if preload_stats is not None:
                out["preload"] = preload_stats
            f.write(json.dumps(out) + "\n")
    print(f"\n[bench] wrote {len(all_results)} runs to {output}")

    # Summary table
    print("\n====== summary ======")
    if preload_stats:
        print(f"Preload: {preload_stats['total_s']:.1f}s "
              f"(flux {preload_stats['flux_warmup_s']:.1f}s, "
              f"gemma {preload_stats['gemma_warmup_s']:.1f}s, "
              f"qwen-vl {preload_stats['qwen_warmup_s']:.1f}s)")
    for scenario in args.scenarios:
        runs = [r for r in all_results if r["scenario"] == scenario]
        if not runs:
            continue
        print(f"\nScenario {scenario}:")
        for r in runs:
            tag = r["label"].rsplit(":", 1)[-1]
            err = "" if not r["error"] else " ERROR"
            print(f"  {tag:12s} {r['total_s']:7.2f}s{err}")


if __name__ == "__main__":
    asyncio.run(main())
