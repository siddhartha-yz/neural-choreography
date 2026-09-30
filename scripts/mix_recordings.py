"""Assemble licensed piano recordings at their original tempo from an edit list."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from scripts.package_score import loudness

RATE = 48000


def decode(ffmpeg, path):
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(RATE),
            "-ac",
            "2",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    edit = json.loads(args.edit.read_text())
    timeline = json.loads(args.timeline.read_text())
    duration = timeline["duration"]
    count = round(duration * RATE)
    mix = np.zeros((count, 2), dtype=np.float32)
    coverage = np.zeros(count, dtype=np.uint8)
    recordings = {}
    credits = []
    source_info = []
    for recording in edit["recordings"]:
        path = args.recordings / recording["filename"]
        pcm = decode(args.ffmpeg, path)
        measured = loudness(args.ffmpeg, path)
        # One constant gain per performance preserves its phrasing and dynamics.
        gain_db = min(
            -21 - float(measured["input_i"]), -3 - float(measured["input_tp"])
        )
        recordings[recording["id"]] = (pcm, 10 ** (gain_db / 20))
        source_info.append(
            dict(
                **recording,
                duration=len(pcm) / RATE,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                gain_db=gain_db,
                measured_loudness=measured,
            )
        )
        credits.append(
            f"{recording['title']} — {recording['composer']}\n"
            f"Recording: {recording['performer']}\n"
            f"Source: {recording['source_url']}\n"
            f"License: {recording['license']} — {recording['license_url']}\n"
        )

    spans = []
    for segment in edit["segments"]:
        pcm, gain = recordings[segment["recording"]]
        start = round(segment["source_start"] * RATE)
        end = round(segment["source_end"] * RATE)
        destination = round(segment["start"] * RATE)
        size = end - start
        stop = destination + size
        fade_in = round(segment.get("fade_in", 0) * RATE)
        fade_out = round(segment.get("fade_out", 0) * RATE)
        if (
            start < 0
            or end > len(pcm)
            or size <= 0
            or destination < 0
            or stop > count
            or fade_in + fade_out > size
        ):
            raise ValueError(f"Invalid source or destination interval: {segment}")
        signal = pcm[start:end].copy() * gain
        if fade_in:
            signal[:fade_in] *= np.sin(np.linspace(0, math.pi / 2, fade_in))[:, None]
        if fade_out:
            signal[-fade_out:] *= np.cos(np.linspace(0, math.pi / 2, fade_out))[:, None]
        mix[destination:stop] += signal
        coverage[destination:stop] += 1
        spans.append(dict(**segment, end=stop / RATE, speed=1.0))
    for silence in edit.get("silences", []):
        start, end = (round(silence[key] * RATE) for key in ("start", "end"))
        if start < 0 or end > count or end <= start or np.any(coverage[start:end]):
            raise ValueError(f"Invalid explicit rest: {silence}")
        coverage[start:end] += 1
    if np.any(coverage == 0) or np.any(coverage > 2):
        raise ValueError(
            "Edit must cover the film with at most two overlapping sources"
        )
    if not np.isfinite(mix).all() or np.max(np.abs(mix)) >= 1:
        raise ValueError("Non-finite or clipped mix")
    if np.max(np.abs(mix[-1])) > 1e-5:
        raise ValueError("End must fade to silence")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out / "score-unmastered.wav"), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(np.rint(mix * 32767).astype("<i2").tobytes())
    credits_text = (
        "Neural Choreography — Classical piano edition\n\n"
        + "\n".join(credits)
        + "\nChanges: excerpts, constant gain, fades, crossfades, loudness mastering "
        "and synchronization with the animation. Original playback speed retained.\n"
        "This mixed soundtrack and scored film are distributed under CC BY-SA 4.0:\n"
        "https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Animation and code: siddhartha-yz / Neural Choreography. "
        "Repository source code remains under its existing MIT license.\n"
    )
    score = dict(
        status="complete_classical_edition",
        title="Neural Choreography · 古典钢琴版",
        duration=count / RATE,
        composer_source="scripts/mix_recordings.py",
        artistic_review="Complete mix awaits user listening review",
        edition_license="CC BY-SA 4.0",
        recordings=source_info,
        movements=edit["movements"],
        segments=spans,
        credits_text=credits_text,
        music_checks=dict(
            samples=count,
            sample_rate=RATE,
            channels=2,
            playback_speed=1.0,
            original_recording_dynamics="constant gain per recording",
            coverage="all samples covered",
            peak=float(np.max(np.abs(mix))),
            final_sample=float(np.max(np.abs(mix[-1]))),
        ),
    )
    (out / "score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "MUSIC-CREDITS.txt").write_text(credits_text)
    shutil.copy2(args.edit, out / "edit.json")
    print(
        json.dumps(
            dict(output=str(out), duration=count / RATE, recordings=len(recordings)),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
