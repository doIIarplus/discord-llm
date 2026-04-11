#!/usr/bin/env python3
"""Start the ChromaForge/Stable Diffusion WebUI server.

Checks if the SD API is already reachable. If not, launches
`bash webui.sh` in the ChromaForge directory as a background process.
Returns JSON with the server status.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

DEFAULT_SD_URL = os.environ.get("SD_API_URL", "http://127.0.0.1:7860")
CHROMAFORGE_DIR = os.path.expanduser("~/projects/chromaforge")


def is_server_running(sd_url: str, timeout: int = 5) -> bool:
    """Check if the SD WebUI API is reachable."""
    try:
        req = urllib.request.Request(f"{sd_url}/sdapi/v1/sd-models")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sd-url", default=DEFAULT_SD_URL,
        help="Stable Diffusion API URL (default: %(default)s)",
    )
    parser.add_argument(
        "--chromaforge-dir", default=CHROMAFORGE_DIR,
        help="Path to the ChromaForge directory (default: %(default)s)",
    )
    args = parser.parse_args()

    if is_server_running(args.sd_url):
        output({
            "status": "already_running",
            "message": "Stable Diffusion WebUI is already running.",
            "url": args.sd_url,
        })

    if not os.path.isdir(args.chromaforge_dir):
        error(f"ChromaForge directory not found: {args.chromaforge_dir}")

    webui_script = os.path.join(args.chromaforge_dir, "webui.sh")
    if not os.path.isfile(webui_script):
        error(f"webui.sh not found in {args.chromaforge_dir}")

    try:
        log_path = os.path.join(args.chromaforge_dir, "webui.log")
        log_file = open(log_path, "a")
        proc = subprocess.Popen(
            ["bash", "webui.sh"],
            cwd=args.chromaforge_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as e:
        error(f"Failed to start server: {e}")

    output({
        "status": "started",
        "message": "Stable Diffusion WebUI server is starting in the background.",
        "pid": proc.pid,
        "url": args.sd_url,
        "log": log_path,
    })


if __name__ == "__main__":
    main()
