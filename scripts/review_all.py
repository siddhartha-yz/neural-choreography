"""Render episodes in isolated media directories and build a review index."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import subprocess
import sys

from scripts.render import ROOT, PROFILES, command, scene_for
from scripts.episode_info import introduction
from scripts.engines import engine_for


def review_episode(episode: str, output: Path, profile: str = "preview") -> dict:
    destination = output / episode
    destination.mkdir(parents=True, exist_ok=True)
    engine = engine_for(episode)
    media = ROOT / "media" / (f"batch-zanim-{profile}" if engine == "zanim" else f"batch-{profile}") / episode
    result = {"engine": engine, "episode": episode, "profile": profile, "status": "failed", "visual_review": "pending"}
    with (destination / "run.log").open("w") as log:
        render = subprocess.run(command(episode, profile, media), cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
        if render.returncode:
            result["error"] = f"Render exited with {render.returncode}"
            return result
        videos = list(media.rglob(f"{episode}-{profile}.mp4"))
        if len(videos) != 1:
            result["error"] = f"Expected one output video, found {len(videos)}"
            return result
        qa_options = ["--frame-width", "1920", "--copy-video"] if profile == "final" else []
        qa = subprocess.run([sys.executable, "-m", "scripts.qa_video", str(videos[0]),
                             "--profile", profile, "--output", str(destination), *qa_options],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if qa.returncode not in (0, 1):
            result["error"] = f"Frame extraction exited with {qa.returncode}"
            return result
    report = json.loads((destination / "report.json").read_text())
    result.update(status="passed" if qa.returncode == 0 else "metadata-failed",
                  duration=report["duration"], problems=report["problems"], full_decode=report["full_decode"])
    return result


def write_index(output: Path, results: list[dict], profile: str = "preview") -> None:
    results = sorted(results, key=lambda row: tuple(map(int, row["episode"].split("."))))
    (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    cards = []
    for row in results:
        episode = row["episode"]
        heading, explanation = introduction(episode)
        if row["status"] == "failed":
            detail = html.escape(row.get("error", "Unknown failure"))
            content = f'<p>{detail}</p>'
        else:
            detail = html.escape("; ".join(row["problems"])) or "视频规格通过；画面待审核"
            content = (f'<a href="{episode}/index.html"><img loading="lazy" '
                       f'src="{episode}/frame-05.png" alt="{episode} 最后一帧"></a>'
                       f'<p>{row["duration"]:.2f} 秒 · {detail}</p>'
                       f'<a href="{episode}/index.html">查看关键帧</a>')
        cards.append(f'<article><h2>{html.escape(heading)}</h2><p>{html.escape(explanation)}</p>{content} '
                     f'<a href="{episode}/run.log">日志</a></article>')
    passed = sum(row["status"] == "passed" for row in results)
    title = "全集成片验证" if profile == "final" else "全集预览审核"
    quality = "1080p60 成片" if profile == "final" else "480p15 预览"
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>点还在动 · {title}</title><style>'
        'body{background:#151515;color:#eee;font:16px system-ui;margin:24px}'
        'a{color:#8dd8ee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}'
        'article{border:1px solid #444;padding:16px;border-radius:12px}img{width:100%}'
        f'</style><h1>点还在动 · {title}</h1>'
        f'<p>已处理 {len(results)} 集 · 视频规格通过 {passed} 集</p>'
        f'<p>下方展示末帧，点击查看关键帧。全部为 {quality}；自动检查不代表人工过审，'
        '也不验证完整视频的运动与节奏。</p><main>' + "".join(cards) + '</main></html>')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", help="Default: all episodes")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--profile", choices=PROFILES, default="preview")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episodes = list(dict.fromkeys(args.episodes or sorted(path.parent.name for path in (ROOT / "episodes").glob("*/scene.py"))))
    for episode in episodes:
        try:
            scene_for(episode)
        except ValueError as error:
            parser.error(str(error))
    output = (args.output or ROOT / "media" / ("review-final" if args.profile == "final" else "review-all")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(review_episode, episode, output, args.profile): episode for episode in episodes}
        for future in as_completed(futures):
            episode = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"episode": episode, "profile": args.profile, "status": "failed", "error": str(error), "visual_review": "pending"}
            results.append(result)
            write_index(output, results, args.profile)
            print(f"[{len(results)}/{len(episodes)}] {episode}: {result['status']}", flush=True)
    print(f"Review index: {output / 'index.html'}")
    return int(any(row["status"] != "passed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
