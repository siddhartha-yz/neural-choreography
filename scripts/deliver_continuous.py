"""Package a rendered complete film with chapter navigation and provenance.

Run render_continuous first, then this command. QA must pass before the
playback page is written. It does not mark artistic review as approved.
"""

import argparse
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path


def playback_page(timeline, filename, poster):
    chapters = timeline["chapters"]
    options = "".join(
        f'<option value="{m["start"]:.6f}">{html.escape(m["title"])}</option>'
        for m in chapters
    )
    buttons = "".join(
        f'<button data-time="{m["start"]:.6f}">{html.escape(m["title"])}</button>'
        for m in chapters
    )
    minutes, seconds = divmod(round(timeline["duration"]), 60)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>点还在动 · 完整版</title>
<style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#000;color:#d9e2e9;font:14px/1.6 system-ui,"Noto Sans CJK SC",sans-serif}}main{{max-width:1600px;margin:auto;padding:16px 24px 32px}}header{{display:flex;justify-content:space-between;align-items:baseline;gap:12px}}h1{{font-size:17px;letter-spacing:.12em;font-weight:450}}small,footer{{color:#81909c}}video{{display:block;width:100%;background:#000;max-height:calc(100vh - 150px)}}.controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}}button,select,a.download{{font:inherit;color:#a6b7c4;background:#080c0f;border:1px solid #26323b;border-radius:4px;padding:6px 12px;cursor:pointer}}button:hover,button:focus-visible,a:hover{{color:#58c4dd;border-color:#58c4dd}}select{{min-width:230px}}a{{color:#58c4dd;text-decoration:none}}details{{margin-top:16px}}summary{{cursor:pointer;color:#81909c}}.chapters{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;margin-top:12px}}.chapters button{{text-align:left}}footer{{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;margin-top:18px}}.cinema header,.cinema .controls,.cinema details,.cinema footer{{display:none}}.cinema main{{max-width:none;padding:0}}.cinema video{{max-height:100vh;height:100vh}}@media(max-width:600px){{main{{padding:10px}}header small{{display:none}}video{{max-height:none}}.controls{{gap:6px}}select{{width:100%}}}}
</style></head><body><main>
<header><h1>点还在动 / Neural Choreography</h1><small>49 段 · {minutes} 分 {seconds:02d} 秒 · 1080p60 · 静音</small></header>
<video id="film" controls autoplay muted playsinline preload="metadata" poster="{poster}" src="{html.escape(filename)}"></video>
<div class="controls"><select id="chapter" aria-label="选择章节">{options}</select><button id="next">下一段</button><button id="restart">从头播放</button><button id="full">全屏</button><button id="cinema">纯享</button><a class="download" href="{html.escape(filename)}" download>下载整片</a></div>
<details><summary>完整章节目录</summary><nav class="chapters" aria-label="49 个章节">{buttons}</nav></details>
<footer><span>第 3–15 章 · Zanim 完整版</span><a href="review/index.html">视频检查</a><a href="manifest.json">版本记录</a><span>F 全屏 · Esc 退出纯享</span></footer>
</main><script>
const film=document.getElementById('film'),chapter=document.getElementById('chapter');
const marks=Array.from(chapter.options,option=>Number(option.value));
function seek(t){{film.currentTime=t;film.play().catch(()=>{{}});}}
chapter.onchange=()=>seek(Number(chapter.value));
document.querySelectorAll('[data-time]').forEach(button=>button.onclick=()=>seek(Number(button.dataset.time)));
film.addEventListener('timeupdate',()=>{{let i=0;while(i+1<marks.length&&film.currentTime>=marks[i+1])i++;chapter.selectedIndex=i;}});
document.getElementById('next').onclick=()=>seek(marks[Math.min(chapter.selectedIndex+1,marks.length-1)]);
document.getElementById('restart').onclick=()=>seek(0);
function full(){{film.requestFullscreen?.().catch(()=>{{}});}}
document.getElementById('full').onclick=full;
document.getElementById('cinema').onclick=()=>document.body.classList.add('cinema');
document.addEventListener('keydown',event=>{{if(event.key==='Escape')document.body.classList.remove('cinema');if(event.key.toLowerCase()==='f'&&event.target.tagName!=='SELECT')full();}});
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument(
        "--assembly", type=Path, help="Optional segment provenance JSON"
    )
    args = parser.parse_args()
    video = args.video.resolve()
    out = video.parent
    timeline = json.loads(args.timeline.read_text())
    chapters = timeline["chapters"]
    from scripts.engines import ZANIM_EPISODES

    codes = [m["title"].split(" · ")[-1] for m in chapters]
    if len(codes) != 49 or set(codes) != ZANIM_EPISODES or len(set(codes)) != 49:
        raise ValueError("Timeline must cover all 49 chapters exactly once")
    times = [0, 14.8, 16.3, 18, 27.8, 35.983333, 36, 368.033333, 368.05]
    times += [m["start"] + 8 for m in chapters[30:]]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.qa_video",
            str(video),
            "--profile",
            "final",
            "--output",
            str(out / "review"),
            "--times",
            *map(str, times),
            "--frame-width",
            "1920",
        ],
        check=True,
    )
    report = json.loads((out / "review/report.json").read_text())
    if (
        report["problems"]
        or report["full_decode"] != "passed"
        or abs(report["duration"] - timeline["duration"]) > 1 / 60
    ):
        raise ValueError("Film QA/duration failed")
    repository = Path(__file__).resolve().parents[1]
    sources = sorted((repository / "zanim_scenes").rglob("*.py")) + [
        repository / "scripts/render_continuous.py",
        repository / "scripts/zanim_render_window.py",
        repository / "scripts/deliver_continuous.py",
        repository / "scripts/qa_video.py",
        repository / "requirements-zanim.txt",
    ]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(
        status="complete_zanim_edition",
        chapters=chapters,
        duration=report["duration"],
        resolution=[1920, 1080],
        fps=60,
        codec="h264",
        audio="none",
        full_decode="passed",
        artistic_review="not_automatically_approved",
        video_sha256=digest(video),
        source_sha256={str(p.relative_to(repository)): digest(p) for p in sources},
    )
    if args.assembly:
        manifest["assembly"] = json.loads(args.assembly.read_text())
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "index.html").write_text(
        playback_page(timeline, video.name, "review/frame-04.png")
    )

    # Portable chapter/subtitle data, usable by players without burning extra text.
    def stamp(seconds):
        millis = round(seconds * 1000)
        h, millis = divmod(millis, 3600000)
        m, millis = divmod(millis, 60000)
        s, millis = divmod(millis, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"

    cues = ["WEBVTT", ""]
    for i, m in enumerate(chapters):
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else report["duration"]
        cues += [f"{stamp(m['start'])} --> {stamp(end)}", m["title"], ""]
    (out / "chapters.vtt").write_text("\n".join(cues))
    print("Complete film:", out / "index.html")


if __name__ == "__main__":
    main()
