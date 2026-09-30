"""Render the complete continuous 03.1 through 15.5 Zanim film."""

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
    parser.add_argument("--timeline", type=Path, help="Write chapter navigation JSON")
    args = parser.parse_args()
    os.environ.setdefault(
        "ZANIM_TYPST", str(Path(__file__).with_name("typst_binding.py").resolve())
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ.setdefault("ZANIM_CACHE_DIR", str(Path("media/zanim-cache").resolve()))
    scene = build(*PROFILES[args.profile])
    if args.timeline:
        import json

        args.timeline.parent.mkdir(parents=True, exist_ok=True)
        args.timeline.write_text(
            json.dumps(
                {"duration": scene.duration, "chapters": scene.chapter_marks},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.time is None:
        from scripts.zanim_render_window import render_window, verify_window

        end = scene.duration if args.end is None else args.end
        window = render_window(scene, args.start, end)
        checks = verify_window(scene, window, args.start, end)
        print(
            f"Render window: {len(window._registry)} / {len(scene._registry)} objects; {checks} pixel comparisons passed",
            flush=True,
        )
        scene = window
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
