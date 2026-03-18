#!/usr/bin/env python3
"""Generate an image using the Stable Diffusion WebUI API.

Calls the txt2img endpoint and saves the result to disk.
Returns the file path and generation parameters as JSON.
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

DEFAULT_SD_URL = os.environ.get("SD_API_URL", "http://127.0.0.1:7860")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_out", "txt2img"
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="Image description")
    parser.add_argument("--negative-prompt", default="", help="What to avoid")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sd-url", default=DEFAULT_SD_URL,
                        help="Stable Diffusion API URL")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
        "seed": args.seed,
    }

    url = f"{args.sd_url}/sdapi/v1/txt2img"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, headers={"Content-Type": "application/json"}, data=data,
    )

    try:
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        error(f"Stable Diffusion API error: {e}")

    info = json.loads(result.get("info", "{}"))
    images = result.get("images", [])
    if not images:
        error("No images returned from API")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(args.output_dir, f"txt2img-{timestamp}-0.png")
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(images[0]))

    output({
        "file_path": os.path.abspath(file_path),
        "seed": info.get("seed"),
        "parameters": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "width": info.get("width", args.width),
            "height": info.get("height", args.height),
            "cfg_scale": info.get("cfg_scale", args.cfg_scale),
            "steps": info.get("steps", args.steps),
            "sampler": info.get("sampler_name"),
        },
    })


if __name__ == "__main__":
    main()
