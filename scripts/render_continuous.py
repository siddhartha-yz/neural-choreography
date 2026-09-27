"""Render the continuous 07.5 through 10.2 motion study."""

import argparse
import os
import sys
from pathlib import Path
from scripts.render import PROFILES
from zanim_scenes.continuous import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="final")
    parser.add_argument("--time", type=float)
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Render a tail from this absolute scene time",
    )
    parser.add_argument("--end", type=float, help="Stop at this absolute scene time")
    args = parser.parse_args()
    os.environ.setdefault(
        "ZANIM_TYPST", str(Path(__file__).with_name("typst_binding.py").resolve())
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ.setdefault("ZANIM_CACHE_DIR", str(Path("media/zanim-cache").resolve()))
    scene = build(*PROFILES[args.profile])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.time is None:
        scene.render_video(
            args.output,
            start=args.start,
            end=args.end,
            workers=4,
            verify_random_access=True,
        )
    else:
        scene.render_frame(args.output, time=args.time)
    duration = (scene.duration if args.end is None else args.end) - args.start
    print(f"Continuous study: {duration:.2f}s -> {args.output}")


if __name__ == "__main__":
    main()
