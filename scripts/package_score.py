"""Master a listening draft and pair it with unchanged existing video frames."""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path


def run(command):
    return subprocess.run(command, check=True, text=True, capture_output=True)


def loudness(ffmpeg, path):
    result = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-18:TP=-1.5:LRA=8:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    begin = result.stderr.rfind("{")
    end = result.stderr.rfind("}")
    return json.loads(result.stderr[begin : end + 1])


def page(chapters):
    options = "".join(
        f'<option value="{c["start"]:.6f}">{html.escape(c["title"])}</option>'
        for c in chapters
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>点还在动 · 配乐试听 01</title>
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#000;color:#b8c7d1;font:14px/1.6 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:18px 24px}}header{{display:flex;align-items:baseline;justify-content:space-between;gap:12px}}h1{{font-size:17px;font-weight:450;letter-spacing:.1em}}small,footer{{color:#758491}}video{{display:block;width:100%;max-height:calc(100vh - 155px);background:#000}}nav{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0;align-items:center}}button,select,a{{font:inherit;background:#080c0f;border:1px solid #26323b;border-radius:4px;padding:6px 12px;color:#a6b7c4;cursor:pointer;text-decoration:none}}button:hover,a:hover{{color:#58c4dd;border-color:#58c4dd}}footer{{font-size:12px;margin-top:16px}}.cinema header,.cinema nav,.cinema footer{{display:none}}.cinema main{{max-width:none;padding:0}}.cinema video{{height:100vh;max-height:100vh}}@media(max-width:600px){{main{{padding:10px}}header small{{display:none}}video{{max-height:none}}select{{width:100%}}}}
</style></head><body><main>
<header><h1>点还在动 · 声音草稿 01</h1><small>2 分钟 · 原创配乐试听</small></header>
<video id="film" controls playsinline preload="metadata" poster="poster.png" src="neural-choreography-score-preview.mp4"></video>
<nav><button id="play">播放 / 暂停</button><select id="chapter" aria-label="章节">{options}</select><button id="mute" aria-pressed="false">对比静音</button><button id="cinema">纯享</button><a href="score.mp3">仅听配乐</a><a href="neural-choreography-score-preview.mp4" download>下载样片</a></nav>
<footer>点击播放开启声音 · F 全屏 · Esc 退出纯享</footer>
</main><script>
const film=document.getElementById('film'),chapter=document.getElementById('chapter'),mute=document.getElementById('mute');
const marks=Array.from(chapter.options,o=>Number(o.value));
film.volume=.8;
document.getElementById('play').onclick=()=>film.paused?film.play().catch(()=>{{}}):film.pause();
chapter.onchange=()=>{{film.currentTime=Number(chapter.value);film.play().catch(()=>{{}});}};
film.addEventListener('timeupdate',()=>{{let i=0;while(i+1<marks.length&&film.currentTime>=marks[i+1])i++;chapter.selectedIndex=i;}});
mute.onclick=()=>{{film.muted=!film.muted;}};
film.addEventListener('volumechange',()=>{{mute.textContent=film.muted?'恢复配乐':'对比静音';mute.setAttribute('aria-pressed',String(film.muted));}});
document.getElementById('cinema').onclick=()=>document.body.classList.add('cinema');
document.addEventListener('keydown',e=>{{if(e.key==='Escape')document.body.classList.remove('cinema');if(e.key.toLowerCase()==='f'&&e.target.tagName!=='SELECT')film.requestFullscreen?.().catch(()=>{{}});}});
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    out = args.output.resolve()
    score = json.loads((out / "score.json").read_text())
    duration = score["duration"]
    timeline = json.loads(args.timeline.read_text())
    chapters = [c for c in timeline["chapters"] if c["start"] < duration - 1e-6]
    raw = out / "score-unmastered.wav"
    measurement = loudness(args.ffmpeg, raw)
    values = {
        "measured_I": "input_i",
        "measured_TP": "input_tp",
        "measured_LRA": "input_lra",
        "measured_thresh": "input_thresh",
        "offset": "target_offset",
    }
    norm = "loudnorm=I=-18:TP=-1.5:LRA=8:linear=true:print_format=json:"
    norm += ":".join(f"{key}={measurement[value]}" for key, value in values.items())
    wav = out / "score.wav"
    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-i",
            str(raw),
            "-af",
            norm,
            "-ar",
            "48000",
            "-c:a",
            "pcm_s24le",
            str(wav),
        ]
    )
    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-i",
            str(wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-metadata",
            "title=点还在动 · 声音草稿 01",
            str(out / "score.mp3"),
        ]
    )
    metadata = [";FFMETADATA1", "title=Neural Choreography — original score sketch 01"]
    for i, chapter in enumerate(chapters):
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else duration
        metadata += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(chapter['start'] * 1000)}",
            f"END={round(end * 1000)}",
            f"title={chapter['title']}",
        ]
    meta = out / "preview-chapters.ffmeta"
    meta.write_text("\n".join(metadata) + "\n")
    video = out / "neural-choreography-score-preview.mp4"
    run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-i",
            str(args.video.resolve()),
            "-i",
            str(wav),
            "-i",
            str(meta),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "2",
            "-map_chapters",
            "2",
            "-t",
            str(duration),
            "-frames:v",
            str(round(duration * 60)),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            str(video),
        ]
    )
    run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(video),
            "-f",
            "null",
            "-",
        ]
    )
    info = json.loads(
        run(
            [
                args.ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-show_chapters",
                "-of",
                "json",
                str(video),
            ]
        ).stdout
    )
    visual = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    if (
        int(visual["nb_read_frames"]) != round(duration * 60)
        or visual["width"] != 1920
        or visual["height"] != 1080
        or visual["avg_frame_rate"] != "60/1"
        or audio["sample_rate"] != "48000"
        or audio["channels"] != 2
        or len(info["chapters"]) != len(chapters)
        or abs(float(info["format"]["duration"]) - duration) > 1 / 60
    ):
        raise ValueError("Preview media metadata check failed")

    # Compare decoded frames at three representative times with the silent film.
    comparisons = []
    for time in [17.0, 27.8, 107.0]:

        def pixels(path):
            return subprocess.run(
                [
                    args.ffmpeg,
                    "-v",
                    "error",
                    "-ss",
                    str(time),
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout

        a, b = pixels(args.video), pixels(video)
        if not a or a != b:
            raise ValueError(f"Video pixels changed at {time}")
        comparisons.append(dict(time=time, sha256=hashlib.sha256(a).hexdigest()))
    measured = loudness(args.ffmpeg, video)
    if abs(float(measured["input_i"]) + 18) > 0.7 or float(measured["input_tp"]) > -1:
        raise ValueError("Encoded soundtrack loudness/headroom check failed")
    poster = args.video.parent / "review/frame-04.png"
    shutil.copy2(poster, out / "poster.png")
    (out / "index.html").write_text(page(chapters))
    report = dict(
        status="listening_draft",
        duration=duration,
        audio=audio,
        video={
            key: visual[key]
            for key in [
                "codec_name",
                "width",
                "height",
                "avg_frame_rate",
                "nb_read_frames",
            ]
        },
        chapters=len(chapters),
        full_decode="passed",
        original_video_pixel_comparisons=comparisons,
        loudness=measured,
        artistic_review="pending_listening",
        source_sha256={
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [Path(__file__), Path(__file__).with_name("compose_score.py")]
        },
        original_video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        score_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
    )
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "player": str(out / "index.html"),
                "duration": duration,
                "integrated_lufs": measured["input_i"],
                "true_peak_dbfs": measured["input_tp"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
