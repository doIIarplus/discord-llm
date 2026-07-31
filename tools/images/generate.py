#!/usr/bin/env python3
"""Generate an image with the diffusion model (Flux2 Klein) via the bot's
resident pipeline.

Use this for photographic, painterly, or imaginative visuals — scenes,
characters, creatures, textures, album-art vibes. It is the WRONG tool when the
image needs legible text, exact numbers, or precise layout: diffusion models
garble text. For charts, diagrams, flowcharts, or anything with labels, write
code instead (matplotlib/PIL/SVG) and register the result with attach.py.

The generated image is attached to the bot's reply automatically — you do not
need to send it via tools/discord/send_message.py. Just describe it in your
reply text as you normally would.

Examples:
  generate.py 'a red panda astronaut floating over Jupiter, film grain'
  generate.py 'cyberpunk alley at night, neon puddles' --preset landscape
  generate.py 'portrait of an old fisherman' --preset portrait-hq --seed 42
"""

import argparse

from _client import PRESETS, emit, post, resolve_dims


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", help="What to draw. Be visually specific.")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="square",
                   help="Dimension preset (default: square = 1024x1024)")
    p.add_argument("--width", type=int, help="Override preset width")
    p.add_argument("--height", type=int, help="Override preset height")
    p.add_argument("--seed", type=int, default=-1,
                   help="Seed for reproducibility (default: random). Reuse a "
                        "previous seed to keep composition stable.")
    p.add_argument("--steps", type=int, default=4,
                   help="Inference steps (default: 4 — Klein is distilled)")
    args = p.parse_args()

    width, height = resolve_dims(args.preset, args.width, args.height)
    result = post("/generate", {
        "prompt": args.prompt,
        "width": width,
        "height": height,
        "seed": args.seed,
        "steps": args.steps,
    })
    emit(result)


if __name__ == "__main__":
    main()
