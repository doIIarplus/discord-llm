#!/usr/bin/env python3
"""Standalone Flux2 Klein 9B CLI with LoRA support.

Generate images directly from the command line without going through the
Discord bot. Supports LoRA stacking, img2img reference conditioning, and
dimension presets.

Note: Flux1 LoRAs will NOT load on Flux2 Klein — the architectures differ.
Only use LoRAs specifically trained for Flux2 / Flux2 Klein.
"""

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FLUX_MODEL_ID, OUTPUT_DIR_T2I, OUTPUT_DIR_I2I


PRESETS = {
    "square":       (1024, 1024),
    "square-hq":    (1536, 1536),
    "portrait":     (832,  1216),
    "portrait-hq":  (1024, 1536),
    "landscape":    (1216, 832),
    "landscape-hq": (1536, 1024),
}

LORA_CACHE = Path.home() / ".cache" / "flux_cli_loras"


def resolve_lora_path(src: str) -> str:
    """Return a local path for a LoRA. If src is a URL, download to cache."""
    if src.startswith(("http://", "https://")):
        LORA_CACHE.mkdir(parents=True, exist_ok=True)
        name = src.rsplit("/", 1)[-1].split("?")[0]
        if not name.endswith((".safetensors", ".ckpt", ".pt")):
            name = hashlib.md5(src.encode()).hexdigest()[:12] + ".safetensors"
        dest = LORA_CACHE / name
        if dest.exists():
            print(f"[lora] cached: {dest}")
            return str(dest)
        print(f"[lora] downloading {src} -> {dest}")
        urllib.request.urlretrieve(src, dest)
        return str(dest)
    return str(Path(src).expanduser().resolve())


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s 'an astronaut on mars'\n"
            "  %(prog)s 'anime girl' --lora ./char.safetensors --lora-weight 0.9\n"
            "  %(prog)s 'scene' --lora a.safetensors --lora b.safetensors \\\n"
            "      --lora-weight 0.7 --lora-weight 1.0\n"
            "  %(prog)s 'portrait of a woman' --preset portrait-hq --num 4\n"
            "  %(prog)s 'edit this' --image source.png --preset landscape\n"
            "  %(prog)s 'hq scene' --lora https://host/lora.safetensors\n"
        ),
    )
    p.add_argument("prompt", help="Text prompt")
    p.add_argument("--seed", type=int, default=-1,
                   help="Random seed (default: random). With --num N, seeds "
                        "increment from this value.")
    p.add_argument("--steps", type=int, default=4,
                   help="Inference steps (default: 4 — Klein is distilled)")
    p.add_argument("--guidance", "--cfg", type=float, default=1.0,
                   help="CFG scale (default: 1.0 — Klein recommended)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="square",
                   help="Dimension preset (default: square = 1024x1024)")
    p.add_argument("--width", type=int, help="Override preset width")
    p.add_argument("--height", type=int, help="Override preset height")
    p.add_argument("--lora", action="append", default=[], metavar="PATH|URL",
                   help="LoRA file path or HTTPS URL. Repeat for multiple LoRAs.")
    p.add_argument("--lora-weight", action="append", type=float, default=[],
                   metavar="FLOAT",
                   help="Weight for each --lora (matches by position). "
                        "Unmatched LoRAs default to 1.0.")
    p.add_argument("--image", help="Source image for img2img reference conditioning")
    p.add_argument("--num", "-n", type=int, default=1,
                   help="Number of images to generate (default: 1)")
    p.add_argument("--output", "-o",
                   help="Output file (with extension) or output dir. "
                        "Default: api_out/txt2img/ or api_out/img2img/")
    p.add_argument("--model", default=FLUX_MODEL_ID,
                   help=f"HF model ID (default: {FLUX_MODEL_ID})")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve dimensions from preset, optionally overridden by --width/--height
    w, h = PRESETS[args.preset]
    if args.width:
        w = args.width
    if args.height:
        h = args.height
    # Klein tolerates up to 1536/side; round to multiples of 64
    w = (min(1536, max(256, w)) // 64) * 64
    h = (min(1536, max(256, h)) // 64) * 64

    # Pair LoRAs with weights (positional; unmatched → 1.0)
    weights = list(args.lora_weight) + [1.0] * (len(args.lora) - len(args.lora_weight))
    lora_specs: List[Tuple[str, float]] = [
        (resolve_lora_path(path), weight)
        for path, weight in zip(args.lora, weights)
    ]

    # Load source image up front so we fail fast on a bad path
    src_image = None
    if args.image:
        src_image = Image.open(args.image).convert("RGB").resize((w, h), Image.LANCZOS)

    # Load pipeline
    print(f"[flux] loading {args.model}...")
    t0 = time.perf_counter()
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    print(f"[flux] loaded in {time.perf_counter() - t0:.1f}s")

    # Apply LoRAs (stacked with set_adapters so multiple can mix)
    if lora_specs:
        adapter_names = []
        for i, (path, weight) in enumerate(lora_specs):
            adapter = f"lora_{i}"
            print(f"[lora] loading {path} weight={weight}")
            pipe.load_lora_weights(path, adapter_name=adapter)
            adapter_names.append(adapter)
        pipe.set_adapters(
            adapter_names,
            adapter_weights=[weight for _, weight in lora_specs],
        )

    # Decide output path(s)
    out_kind = "img2img" if src_image is not None else "txt2img"
    single_file = False
    if args.output:
        out_target = Path(args.output).expanduser().resolve()
        if out_target.suffix:  # looks like a file (e.g. ".png")
            out_dir = out_target.parent
            single_file = True
        else:
            out_dir = out_target
    else:
        out_dir = Path(OUTPUT_DIR_I2I if src_image is not None else OUTPUT_DIR_T2I)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    for i in range(args.num):
        if args.seed >= 0:
            seed = args.seed + i
        else:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs = dict(
            prompt=args.prompt,
            height=h,
            width=w,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
        if src_image is not None:
            kwargs["image"] = src_image

        print(
            f"[gen {i+1}/{args.num}] {w}x{h} steps={args.steps} "
            f"cfg={args.guidance} seed={seed}"
        )
        t0 = time.perf_counter()
        img = pipe(**kwargs).images[0]
        dt = time.perf_counter() - t0

        if single_file and args.num == 1:
            save_path = out_target
        else:
            ts = int(time.time() * 1000)
            save_path = out_dir / f"cli-{out_kind}-{ts}-{seed}.png"
        img.save(save_path)
        print(f"  saved: {save_path} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
