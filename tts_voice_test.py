#!/usr/bin/env python3
"""Generate TTS parameter sweep samples to find the best voice clone settings.

Usage: python tts_voice_test.py [--voice NAME] [--text TEXT]

Generates samples across parameter combos and saves to api_out/tts_test/.
Compare with the original reference audio to find the best match.
"""

import argparse
import asyncio
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="trump", help="Voice name to test")
    parser.add_argument("--text", default=None, help="Custom text to speak")
    parser.add_argument("--exaggeration", type=float, nargs="+", default=None,
                        help="Specific exaggeration values to test")
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=None,
                        help="Specific cfg_weight values to test")
    parser.add_argument("--temperature", type=float, nargs="+", default=None,
                        help="Specific temperature values to test")
    parser.add_argument("--repetition-penalty", type=float, nargs="+", default=None,
                        help="Specific repetition_penalty values to test")
    parser.add_argument("--min-p", type=float, nargs="+", default=None,
                        help="Specific min_p values to test")
    parser.add_argument("--top-p", type=float, nargs="+", default=None,
                        help="Specific top_p values to test")
    parser.add_argument("--out-dir", default="api_out/tts_test",
                        help="Output directory for samples")
    args = parser.parse_args()

    # Default test text — pick something with natural pauses, emphasis, and varied tone
    test_text = args.text or (
        "We're going to make this country so great again, believe me. "
        "Nobody knows more about winning than I do. "
        "And let me tell you, the results are going to be tremendous."
    )

    # Build parameter grid from whatever was specified
    param_axes = {}
    if args.exaggeration:
        param_axes["exaggeration"] = args.exaggeration
    if args.cfg_weight:
        param_axes["cfg_weight"] = args.cfg_weight
    if args.temperature:
        param_axes["temperature"] = args.temperature
    if args.repetition_penalty:
        param_axes["repetition_penalty"] = args.repetition_penalty
    if args.min_p:
        param_axes["min_p"] = args.min_p
    if args.top_p:
        param_axes["top_p"] = args.top_p

    # Default sweep if nothing specified
    if not param_axes:
        param_axes = {
            "exaggeration": [0.3, 0.5, 0.7, 0.85, 1.0, 1.3],
            "cfg_weight": [0.2, 0.3, 0.5, 0.7],
        }

    # Generate all combinations
    param_names = list(param_axes.keys())
    param_values = list(param_axes.values())
    combos = list(itertools.product(*param_values))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Voice: {args.voice}")
    print(f"Text: {test_text}")
    for name, vals in param_axes.items():
        print(f"  {name}: {vals}")
    print(f"Total samples: {len(combos)}")
    print(f"Output dir: {out_dir}")
    print()

    # Load engine
    from tts_engines.chatterbox_engine import ChatterboxEngine
    engine = ChatterboxEngine()
    await engine.initialize()

    results = []

    for combo in combos:
        params = dict(zip(param_names, combo))

        # Build filename from params
        parts = [args.voice] + [f"{k[:4]}{v:.2f}" for k, v in params.items()]
        fname = "_".join(parts) + ".wav"
        out_path = out_dir / fname

        desc = ", ".join(f"{k}={v:.2f}" for k, v in params.items())
        print(f"  Generating: {desc} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        result = await engine.generate(
            text=test_text,
            voice=args.voice,
            output_path=out_path,
            **params,
        )

        elapsed = time.perf_counter() - t0
        print(f"{result.duration_seconds:.1f}s audio in {elapsed:.1f}s")
        results.append({
            "file": fname,
            "params": params,
            "duration": result.duration_seconds,
            "gen_time": elapsed,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"Generated {len(results)} samples in {out_dir}/")
    print(f"{'='*70}")
    print(f"{'File':<55} {'Duration':>8} {'GenTime':>8}")
    print(f"{'-'*55} {'-'*8} {'-'*8}")
    for r in results:
        print(f"{r['file']:<55} {r['duration']:>7.1f}s {r['gen_time']:>7.1f}s")

    print(f"\nReference audio: voices/{args.voice}.*")
    print("Listen to each sample and compare voice tone, pacing, and emotion to the reference.")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
