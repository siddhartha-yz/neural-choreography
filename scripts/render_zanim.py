"""Render migrated scenes using Zanim's native 2D/3D renderer."""

import argparse
import os
import shutil
import sys
from pathlib import Path
from scripts.render import PROFILES
from zanim_scenes import (
    transposed_conv,
    optimization,
    multi_channel,
    convolution,
    mechanisms,
    attention,
    recurrent,
    foundations,
    vision_language,
)

BUILDERS = {"13.10": transposed_conv.build, "06.4": multi_channel.build}


for episode in optimization.CONFIG:
    BUILDERS[episode] = lambda width=1920, height=1080, fps=60, episode=episode: (
        optimization.build(episode, width, height, fps)
    )


for episode in convolution.EPISODE_IDS:
    BUILDERS[episode] = lambda width=1920, height=1080, fps=60, episode=episode: (
        convolution.build(episode, width, height, fps)
    )


for episode in mechanisms.EPISODE_IDS:
    BUILDERS[episode] = lambda width=1920, height=1080, fps=60, episode=episode: (
        mechanisms.build(episode, width, height, fps)
    )


for episode in attention.EPISODE_IDS:
    BUILDERS[episode] = lambda width=1920, height=1080, fps=60, episode=episode: (
        attention.build(episode, width, height, fps)
    )


for episode in recurrent.EPISODE_IDS:
    BUILDERS[episode] = lambda width=1920, height=1080, fps=60, episode=episode: (
        recurrent.build(episode, width, height, fps)
    )


for module in (foundations, vision_language):
    for episode in module.EPISODE_IDS:
        BUILDERS[episode] = (
            lambda width=1920, height=1080, fps=60, episode=episode, module=module: (
                module.build(episode, width, height, fps)
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", choices=BUILDERS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="final")
    parser.add_argument("--time", type=float)
    args = parser.parse_args()
    build = BUILDERS[args.episode]
    os.environ.setdefault(
        "ZANIM_CACHE_DIR",
        str(Path(__file__).resolve().parents[1] / "media" / "zanim-cache"),
    )
    if not os.environ.get("ZANIM_TYPST") and not shutil.which("typst"):
        os.environ["ZANIM_TYPST"] = str(
            Path(__file__).with_name("typst_binding.py").resolve()
        )
        os.environ["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        )
    scene = build(*PROFILES[args.profile])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.time is None:
        scene.render_video(args.output, workers=4, verify_random_access=True)
    else:
        scene.render_frame(args.output, time=args.time)
    print(f"{args.episode}: Zanim, {scene.duration:.2f}s → {args.output}")


if __name__ == "__main__":
    main()
