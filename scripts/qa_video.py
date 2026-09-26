"""Validate video metadata and extract timestamped frames for human review."""

import argparse
from fractions import Fraction
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.render import PROFILES
from scripts.episode_info import introduction


def inspect_metadata(metadata: dict, profile: str) -> tuple[dict, float, list[str]]:
    streams = metadata.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("Expected exactly one video stream")
    video = videos[0]
    duration = float(video.get("duration") or metadata.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video has no valid duration")
    width, height, fps = PROFILES[profile]
    problems = []
    if (video.get("width"), video.get("height")) != (width, height):
        problems.append(f"Expected {width}×{height}, got {video.get('width')}×{video.get('height')}")
    try:
        actual_fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        actual_fps = 0
    if abs(actual_fps - fps) > 0.01:
        problems.append(f"Expected {fps} fps, got {actual_fps:g}")
    if video.get("codec_name") != "h264":
        problems.append("Expected H.264 video")
    if any(stream.get("codec_type") == "audio" for stream in streams):
        problems.append("Expected silent video without an audio track")
    return video, duration, problems


def frame_times(duration: float, fps: float, requested: list[float] | None) -> list[float]:
    # Seek to the final frame's start, rather than to EOF.
    last = max(0.0, duration - 1 / fps)
    values = requested if requested is not None else [0, min(0.5, last), min(duration / 4, last), min(duration / 2, last), min(3 * duration / 4, last)]
    if any(not math.isfinite(value) or value < 0 or value > last for value in values):
        raise ValueError(f"Frame times must be between 0 and {last:.6f} seconds")
    return sorted(set([*values, last]))


def review(video_path: Path, output: Path, profile: str, requested: list[float] | None,
           frame_width: int = 640, copy_video: bool = False) -> int:
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path)
    ], text=True))
    video, duration, problems = inspect_metadata(metadata, profile)
    # Decode every video frame, not just the timestamps used for the contact sheet.
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video_path),
                    "-map", "0:v:0", "-f", "null", "-"],
                   stdout=subprocess.DEVNULL, check=True)
    try:
        fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0
    times = frame_times(duration, fps if fps > 0 else PROFILES[profile][2], requested)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, timestamp in enumerate(times):
        name = f"frame-{index:02d}.png"
        target = output / name
        target.unlink(missing_ok=True)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-ss", f"{timestamp:.9f}",
            "-frames:v", "1", "-vf", f"scale={min(frame_width, video['width'])}:-2", str(target)
        ], check=True)
        if not target.exists() or target.stat().st_size == 0:
            raise ValueError(f"No frame extracted at {timestamp:.6f} s")
        frames.append({"time": timestamp, "file": name})
    report = {"video": str(video_path), "profile": profile, "duration": duration,
              "problems": problems, "frames": frames, "metadata": metadata,
              "visual_review": "pending", "full_decode": "passed"}
    if copy_video:
        target = output / "video.mp4"
        if target.resolve() != video_path.resolve():
            shutil.copy2(video_path, target)
        report["review_video"] = "video.mp4"
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_review_page(output, report)
    print(f"Review: {output / 'index.html'}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return int(bool(problems))


def write_review_page(output: Path, report: dict) -> None:
    video_path = Path(report["video"])
    episode = video_path.stem.split("-")[0]
    heading, explanation = introduction(episode)
    if not explanation:
        heading = video_path.name
    figures = "".join(f'<figure><a href="{frame["file"]}"><img src="{frame["file"]}" alt="Frame"></a><figcaption>{frame["time"]:.3f} s</figcaption></figure>' for frame in report["frames"])
    player = '<video controls preload="none" style="width:100%;max-width:960px" src="video.mp4"></video>' if report.get("review_video") else ''
    status = "; ".join(report["problems"]) if report["problems"] else "视频规格检查通过。"
    (output / "index.html").write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(heading)}</title>'
        '<style>body{background:#151515;color:#eee;font:16px sans-serif;margin:24px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
        'figure{margin:0}img{width:100%}figcaption{padding:8px}</style>'
        f'<h1>{html.escape(heading)}</h1><p class="intro">{html.escape(explanation)}</p><p>{html.escape(status)}</p>'
        '<p>画面待审核：检查标签遮挡、运动连续性、画面层次与转场。关键帧不能替代完整观看。</p>'
        f'{player}<main>{figures}</main></html>')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="final")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", help="Key timestamps in seconds; final frame is always added")
    parser.add_argument("--frame-width", type=int, choices=range(64, 3841), default=640,
                        metavar="64..3840", help="Frame export width, capped at source width")
    parser.add_argument("--copy-video", action="store_true", help="Include a playable copy in the review directory")
    args = parser.parse_args()
    try:
        return review(args.video.resolve(strict=True), args.output.resolve(), args.profile, args.times,
                      args.frame_width, args.copy_video)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"QA failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
