#!/usr/bin/env python3
"""Fetch a preview image for a 6-digit nhentai code.

Uses Playwright for JavaScript rendering and saves the preview image to disk.
Returns the file path and URL as JSON.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import output, error

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "api_out")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("code", help="6-digit nhentai code")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not re.fullmatch(r"\d{6}", args.code):
        error(f"Invalid code: '{args.code}'. Must be exactly 6 digits.")

    os.makedirs(args.output_dir, exist_ok=True)

    link = f"https://nhentai.net/g/{args.code}/"
    preview_page = f"https://nhentai.net/g/{args.code}/3"

    try:
        import asyncio
        from bs4 import BeautifulSoup
        # Import the project's JS renderer
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from web_extractor import js_renderer
        import aiohttp

        async def _fetch():
            html = await js_renderer.render(preview_page, timeout=20000)
            if not html:
                return None, None

            soup = BeautifulSoup(html, "html.parser")
            img_tag = (
                soup.select_one("div.thumbnail-container a img")
                or soup.select_one("div#gallery-container img")
                or soup.select_one("img")
            )
            img_url = img_tag["src"] if img_tag else None
            if not img_url:
                return None, None

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    img_url,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://nhentai.net/"},
                ) as resp:
                    if resp.status != 200:
                        return img_url, None
                    img_data = await resp.read()

            ext = img_url.rsplit(".", 1)[-1].split("?")[0] or "jpg"
            file_path = os.path.join(args.output_dir, f"nhentai_{args.code}.{ext}")
            with open(file_path, "wb") as f:
                f.write(img_data)
            return img_url, os.path.abspath(file_path)

        img_url, file_path = asyncio.run(_fetch())

        if file_path:
            output({"file_path": file_path, "url": link, "image_url": img_url})
        else:
            # Fallback: return link without preview image
            output({"file_path": None, "url": link, "message": "Could not fetch preview image"})

    except ImportError as e:
        error(f"Missing dependency: {e}. Needs: playwright, beautifulsoup4, aiohttp")
    except Exception as e:
        error(f"Failed to fetch preview: {e}")


if __name__ == "__main__":
    main()
