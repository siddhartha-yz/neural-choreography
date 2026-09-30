"""Master a listening draft and pair it with unchanged existing video frames."""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def run(command):
    return subprocess.run(command, check=True, text=True, capture_output=True)


def archive_piano(out, video_name="neural-choreography-piano.mp4", recorded=False):
    """Portable playback and score data; the large master WAV stays separate."""
    names = [
        "index.html",
        video_name,
        "poster.png",
        "score.mp3",
        "score.json",
        "report.json",
        "manifest.json",
        "timeline.json",
        "chapters.vtt",
        "MUSIC-CREDITS.txt" if recorded else "PIANO-SAMPLE-CREDITS.txt",
    ]
    names.append("edit.json" if recorded else "nocturne.mid")
    archive = out / (
        "neural-choreography-classical-player.zip"
        if recorded
        else "neural-choreography-piano-player.zip"
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in names:
            bundle.write(out / name, name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Piano edition archive integrity failed")
    return archive


def loudness(ffmpeg, path, integrated=-18, peak=-1.5, dynamic_range=8):
    result = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"loudnorm=I={integrated}:TP={peak}:LRA={dynamic_range}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    begin = result.stderr.rfind("{")
    end = result.stderr.rfind("}")
    return json.loads(result.stderr[begin : end + 1])


def page(
    chapters,
    title="点还在动 · 声音草稿 01",
    credit=None,
    duration=120,
    filename="neural-choreography-score-preview.mp4",
    movements=None,
    recorded=False,
):
    title = html.escape(title)
    filename = html.escape(filename, quote=True)
    minutes, seconds = divmod(round(duration), 60)
    edition = (
        "古典钢琴配乐版"
        if recorded
        else "钢琴夜曲完整版"
        if duration > 120
        else "原创配乐试听"
    )
    navigation = ""
    if movements:
        choices = "".join(
            f'<option value="{m["start"]:.6f}">{html.escape(m["title"])} · '
            f"{html.escape(m['key'])}</option>"
            for m in movements
        )
        navigation = f'<select id="movement" aria-label="音乐段落">{choices}</select>'
    attribution = ""
    if credit:
        attribution = (
            f' · <a href="{html.escape(credit["source_url"], quote=True)}">'
            f"钢琴采样：{html.escape(credit['author'])} / FreePats</a>"
            f' · <a href="{html.escape(credit["license_url"], quote=True)}">'
            f"{html.escape(credit['license'])}</a>"
        )
    if recorded:
        attribution = (
            ' · <a href="MUSIC-CREDITS.txt">录音来源与署名</a>'
            ' · <a href="https://creativecommons.org/licenses/by-sa/4.0/">'
            "本配乐影片：CC BY-SA 4.0</a>"
        )
    options = "".join(
        f'<option value="{c["start"]:.6f}">{html.escape(c["title"])}</option>'
        for c in chapters
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#000;color:#b8c7d1;font:14px/1.6 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:18px 24px}}header{{display:flex;align-items:baseline;justify-content:space-between;gap:12px}}h1{{font-size:17px;font-weight:450;letter-spacing:.1em}}small,footer{{color:#758491}}video{{display:block;width:100%;max-height:calc(100vh - 155px);background:#000}}nav{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0;align-items:center}}button,select,a{{font:inherit;background:#080c0f;border:1px solid #26323b;border-radius:4px;padding:6px 12px;color:#a6b7c4;cursor:pointer;text-decoration:none}}button:hover,a:hover{{color:#58c4dd;border-color:#58c4dd}}footer{{font-size:12px;margin-top:16px}}.cinema header,.cinema nav,.cinema footer{{display:none}}.cinema main{{max-width:none;padding:0}}.cinema video{{height:100vh;max-height:100vh}}@media(max-width:600px){{main{{padding:10px}}header small{{display:none}}video{{max-height:none}}select{{width:100%}}}}
</style></head><body><main>
<header><h1>{title}</h1><small>{minutes} 分 {seconds:02d} 秒 · {edition}</small></header>
<video id="film" controls playsinline preload="metadata" poster="poster.png" src="{filename}"></video>
<nav><button id="play">播放 / 暂停</button><select id="chapter" aria-label="章节">{options}</select>{navigation}<button id="mute" aria-pressed="false">对比静音</button><button id="cinema">纯享</button><a href="score.mp3">仅听配乐</a><a href="{filename}" download>下载视频</a></nav>
<footer>点击播放开启声音 · F 全屏 · Esc 退出纯享{attribution}</footer>
</main><script>
const film=document.getElementById('film'),chapter=document.getElementById('chapter'),mute=document.getElementById('mute');
const marks=Array.from(chapter.options,o=>Number(o.value));
film.volume=.8;
document.getElementById('play').onclick=()=>film.paused?film.play().catch(()=>{{}}):film.pause();
chapter.onchange=()=>{{film.currentTime=Number(chapter.value);film.play().catch(()=>{{}});}};
film.addEventListener('timeupdate',()=>{{let i=0;while(i+1<marks.length&&film.currentTime>=marks[i+1])i++;chapter.selectedIndex=i;}});
const movement=document.getElementById('movement');
if(movement){{const points=Array.from(movement.options,o=>Number(o.value));movement.onchange=()=>{{film.currentTime=Number(movement.value);film.play().catch(()=>{{}});}};film.addEventListener('timeupdate',()=>{{let i=0;while(i+1<points.length&&film.currentTime>=points[i+1])i++;movement.selectedIndex=i;}});}}
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
    credit = score.get("sample_credits")
    recorded = score["status"] == "complete_classical_edition"
    credit_text = (
        score["credits_text"]
        if recorded
        else (
            f"Piano samples: {credit['work']} by {credit['author']}; "
            f"SF2: {credit['sf2_conversion']}; {credit['license']}; "
            f"{credit['source_url']}; {credit['license_url']}; {credit['use']}"
            if credit
            else "Original numerical instrument synthesis"
        )
    )
    duration = score["duration"]
    timeline = json.loads(args.timeline.read_text())
    complete = score["status"] in (
        "complete_piano_edition",
        "complete_classical_edition",
    )
    if complete and (
        abs(duration - timeline["duration"]) > 1 / 48000
        or len(timeline["chapters"]) != 49
    ):
        raise ValueError("Complete score must match all 49 film chapters")
    chapters = [c for c in timeline["chapters"] if c["start"] < duration - 1e-6]
    raw = out / "score-unmastered.wav"
    target_i, target_peak, target_range = (-22, -2, 12) if recorded else (-18, -1.5, 8)
    measurement = loudness(args.ffmpeg, raw, target_i, target_peak, target_range)
    values = {
        "measured_I": "input_i",
        "measured_TP": "input_tp",
        "measured_LRA": "input_lra",
        "measured_thresh": "input_thresh",
        "offset": "target_offset",
    }
    norm = (
        f"loudnorm=I={target_i}:TP={target_peak}:LRA={target_range}:"
        "linear=true:print_format=json:"
    )
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
            "-metadata",
            f"comment={credit_text}",
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
            f"title={score['title']}",
            "-metadata",
            f"comment={credit_text}",
            str(out / "score.mp3"),
        ]
    )

    def metadata_value(value):
        for character in ["\\", "=", ";", "#", "\n"]:
            value = value.replace(character, "\\" + character)
        return value

    metadata = [
        ";FFMETADATA1",
        f"title={metadata_value(score['title'])}",
        f"comment={metadata_value(credit_text)}",
    ]
    for i, chapter in enumerate(chapters):
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else duration
        metadata += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(chapter['start'] * 1000)}",
            f"END={round(end * 1000)}",
            f"title={metadata_value(chapter['title'])}",
        ]
    meta = out / "chapters.ffmeta"
    meta.write_text("\n".join(metadata) + "\n")
    video = out / (
        "neural-choreography-classical.mp4"
        if recorded
        else (
            "neural-choreography-piano.mp4"
            if complete
            else "neural-choreography-score-preview.mp4"
        )
    )
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
        raise ValueError("Scored film media metadata check failed")

    # Check late movements and the closing aperture as well as the intro.
    comparisons = []
    times = [17.0, 27.8, 107.0]
    if complete:
        times += [250.75, 398.25, 524.65, 604.65, 675.0, duration - 1 / 60]
    for time in times:

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
    measured = loudness(args.ffmpeg, video, target_i, target_peak, target_range)
    if (
        abs(float(measured["input_i"]) - target_i) > 0.7
        or float(measured["input_tp"]) > target_peak + 0.5
    ):
        raise ValueError("Encoded soundtrack loudness/headroom check failed")
    poster = args.video.parent / "review/frame-04.png"
    shutil.copy2(poster, out / "poster.png")
    (out / "index.html").write_text(
        page(
            chapters,
            score["title"],
            credit,
            duration,
            video.name,
            score.get("movements"),
            recorded,
        )
    )
    report = dict(
        status=score["status"],
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
        mastering_targets=dict(
            integrated_lufs=target_i,
            true_peak_dbfs=target_peak,
            loudness_range_lu=target_range,
        ),
        artistic_review=score["artistic_review"],
        music_movements=score.get("movements", []),
        sample_credits=credit,
        recording_credits=score.get("recordings", []),
        edition_license=score.get("edition_license"),
        music_checks=score.get("music_checks", {}),
        source_sha256={
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [
                Path(__file__),
                Path(__file__).with_name(
                    Path(score.get("composer_source", "compose_score.py")).name
                ),
                *(
                    [Path(__file__).with_name("compose_nocturne.py")]
                    if complete and not recorded
                    else []
                ),
            ]
        },
        original_video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        score_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
    )
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if complete:
        (out / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        shutil.copy2(args.timeline, out / "timeline.json")
        shutil.copy2(args.video.parent / "chapters.vtt", out / "chapters.vtt")
        credits_name = "MUSIC-CREDITS.txt" if recorded else "PIANO-SAMPLE-CREDITS.txt"
        (out / credits_name).write_text(credit_text + "\n")
        archive_piano(out, video.name, recorded)
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
