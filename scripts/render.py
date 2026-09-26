"""Render one episode with explicit, reproducible output settings."""

import argparse
import ast
from pathlib import Path
import subprocess
import sys
from scripts.engines import engine_for

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"preview": (854, 480, 15), "final": (1920, 1080, 60)}


def scene_for(episode: str) -> tuple[Path, str]:
    scenes = {path.parent.name: path for path in (ROOT / "episodes").glob("*/scene.py")}
    if episode not in scenes:
        raise ValueError(f"Unknown episode {episode!r}; choose from: {', '.join(sorted(scenes))}")
    path = scenes[episode]
    classes = [node.name for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name.startswith("Episode")]
    if len(classes) != 1:
        raise ValueError(f"Expected exactly one Episode class in {path}")
    return path, classes[0]


def command(episode: str, profile: str, output: Path) -> list[str]:
    path, scene = scene_for(episode)
    width, height, fps = PROFILES[profile]
    if engine_for(episode) == 'zanim':
        target = output.resolve() / 'videos' / 'zanim' / f'{episode}-{profile}.mp4'
        return [sys.executable, '-m', 'scripts.render_zanim', episode,
                '--profile', profile, '--output', str(target)]
    return [sys.executable, "-m", "manim", str(path), scene,
            "-r", f"{width},{height}", "--fps", str(fps), "--format", "mp4",
            "--media_dir", str(output.resolve()), "-o", f"{episode}-{profile}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="Episode directory, e.g. 03.1")
    parser.add_argument("--profile", choices=PROFILES, default="preview")
    parser.add_argument("--output", type=Path, help="Media directory; defaults to media/<profile>")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without rendering")
    args = parser.parse_args()
    try:
        cmd = command(args.episode, args.profile, args.output or ROOT / "media" / args.profile)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        import shlex
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
