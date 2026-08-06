#!/usr/bin/env python3
"""Attach an image YOU created programmatically to the bot's reply.

This is the preferred path whenever the image can be *drawn with code* rather
than hallucinated by a diffusion model. Write a script (matplotlib, PIL/Pillow,
plain SVG, graphviz), save a PNG somewhere under the project directory, then
register it here and it rides along with your reply.

Prefer this over images/generate.py for:
  - charts, graphs, plots, any data visualization
  - diagrams, flowcharts, architecture sketches, timelines
  - anything with legible text, labels, numbers, or axes
  - tables rendered as images, scoreboards, calendars
  - precise geometry, exact colors, or exact brand text

Use images/generate.py instead for photographic or painterly content, where
precision doesn't matter and realism does.

Save output under the project dir (e.g. api_out/ or a temp path inside it).
Supported: .png .jpg .jpeg .gif .webp .bmp

Examples:
  attach.py api_out/gpq_chart.png
  attach.py api_out/diagram.png --caption 'request flow'
  attach.py api_out/user_supplied_render.png --nsfw-check
"""

import argparse

from _client import emit, post
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
    p.add_argument("path",
                   help="Path to the image you created (inside the project dir)")
    p.add_argument("--caption", default="",
                   help="Optional note about what this image is (for the bot's log)")
    p.add_argument("--nsfw-check", action="store_true",
                   help="Run the NSFW classifier before attaching. Skipped by "
                        "default since code-drawn charts don't need it; enable "
                        "when rendering user-supplied content.")
    args = p.parse_args()

    result = post("/attach", {
        "path": args.path,
        "caption": args.caption,
        "nsfw_check": args.nsfw_check,
    })
    emit(result)


if __name__ == "__main__":
    main()
