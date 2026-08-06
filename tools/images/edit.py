#!/usr/bin/env python3
"""Edit an existing image with the diffusion model (Flux2 Klein).

Use this to modify a user-attached image or one you generated earlier —
"make the hair green", "replace the background with a beach", "add a hat".

Prompting note (BFL's Klein guidance): describe ONLY what changes, not the whole
target scene. Klein takes the visual details from the source image, so
"change the car to red" beats "a red car on a street at sunset with...".

Source images live in multimodal_input/ (user attachments) and api_out/
(previously generated). The path must be inside the project directory.

The edited image is attached to the bot's reply automatically.

Examples:
  edit.py 'change the hair color to green' --image multimodal_input/pic.png
  edit.py 'replace the background with a snowy forest' --image api_out/txt2img/x.png
"""

import argparse

from _client import PRESETS, emit, post, resolve_dims
import os as _ga_os, sys as _ga_sys
_ga_sys.path.insert(0, _ga_os.path.join(_ga_os.path.dirname(_ga_os.path.abspath(__file__)), '..'))
from _guild_access import require_integration


def main():
    # Per-guild tool gating (tools/_guild_access.py). The guild id comes
    # from the trusted env var the bot injects, never from arguments.
    require_integration('images')
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt",
                   help="The change to make. Describe only what differs.")
    p.add_argument("--image", required=True,
                   help="Source image path (must be inside the project dir)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="square",
                   help="Output dimension preset (default: square). Prefer one "
                        "matching the source aspect ratio.")
    p.add_argument("--width", type=int, help="Override preset width")
    p.add_argument("--height", type=int, help="Override preset height")
    p.add_argument("--seed", type=int, default=-1,
                   help="Seed for reproducibility (default: random)")
    p.add_argument("--steps", type=int, default=4,
                   help="Inference steps (default: 4 — Klein is distilled)")
    args = p.parse_args()

    width, height = resolve_dims(args.preset, args.width, args.height)
    result = post("/edit", {
        "prompt": args.prompt,
        "image_path": args.image,
        "width": width,
        "height": height,
        "seed": args.seed,
        "steps": args.steps,
    })
    emit(result)


if __name__ == "__main__":
    main()
