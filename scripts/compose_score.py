"""Original two-minute score sketch for the existing silent film.

No recordings or downloaded music are used. NumPy synthesizes the instruments;
SciPy supplies convolution reverb. This is a listening draft, not the approved
soundtrack of the full film. Run from the repository root with --output DIR.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


SR = 48000
DURATION = 120.0
BEAT = 0.75
SEED = 20260930

# Four bars per 12-second chapter, D minor with open ninths and common tones.
# Low root, upper chord, eight-note thematic answer; MIDI note numbers.
PHRASES = [
    (38, [57, 60, 64, 65], [62, 69, 76, 77, 76, 72, 69, 74]),
    (34, [57, 62, 65, 69], [65, 69, 74, 76, 74, 69, 65, 62]),
    (41, [57, 60, 64, 67], [69, 72, 76, 79, 76, 72, 69, 65]),
    (36, [55, 60, 62, 64], [67, 72, 74, 76, 74, 72, 67, 64]),
    (43, [57, 62, 65, 69], [69, 65, 62, 57, 62, 65, 69, 74]),
    (33, [57, 60, 62, 65], [62, 69, 76, 77, 76, 72, 69, 74]),
    (34, [57, 62, 65, 69], [74, 69, 65, 62, 65, 69, 74, 77]),
    (36, [55, 60, 62, 64], [64, 67, 72, 74, 76, 74, 72, 67]),
    (38, [57, 60, 64, 65], [62, 69, 76, 77, 76, 72, 69, 74]),
    (38, [57, 62, 64, 65], [74, 69, 65, 64, 62, 65, 69, 74]),
]


def frequency(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def envelope(t, length, attack, release):
    return (
        np.sin(np.minimum(t / attack, 1) * np.pi / 2) ** 2
        * np.sin(np.minimum(np.maximum(length - t, 0) / release, 1) * np.pi / 2) ** 2
    )


def instrument(kind, midi, length, velocity, rng):
    t = np.arange(round(length * SR), dtype=np.float64) / SR
    f = frequency(midi)
    phase = rng.uniform(0, 2 * np.pi)
    if kind == "felt":
        # Slightly inharmonic, increasingly damped struck partials.
        y = np.zeros_like(t)
        for h in range(1, 10):
            detune = np.sqrt(1 + 0.00012 * h * h)
            y += (
                h**-1.7
                * np.exp(-t * (0.7 + h * 0.26))
                * np.sin(2 * np.pi * f * h * detune * t + phase / h)
            )
        y *= envelope(t, length, 0.009, 0.6)
    elif kind == "glass":
        modulation = 1.4 * np.exp(-t * 3.5) * np.sin(2 * np.pi * f * 2 * t)
        y = np.sin(2 * np.pi * f * t + modulation + phase) * np.exp(-t * 1.7)
        y += 0.13 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t * 4)
        y *= envelope(t, length, 0.008, 0.35)
    elif kind == "plucked":
        y = np.zeros_like(t)
        for h in range(1, 8):
            y += (
                h**-1.25
                * np.sin(2 * np.pi * f * h * t + phase)
                * np.exp(-t * (2.4 + 0.8 * h))
            )
        y *= envelope(t, length, 0.006, 0.3)
    elif kind == "bowed":
        y = np.zeros_like(t)
        for cents, weight in [(-5, 0.3), (0, 0.4), (5, 0.3)]:
            fc = f * 2 ** (cents / 1200)
            vibrato = 0.17 * np.sin(2 * np.pi * 4.7 * t + phase)
            for h in range(1, 7):
                y += (
                    weight
                    * h**-2.2
                    * np.sin(2 * np.pi * fc * h * t + vibrato * h + phase)
                )
        y *= envelope(t, length, 2.2, 3.0)
        y *= 0.92 + 0.08 * np.sin(2 * np.pi * t / 6 + phase)
    elif kind == "bass":
        y = np.sin(2 * np.pi * f * t + phase)
        y += 0.18 * np.sin(2 * np.pi * 2 * f * t + phase)
        y *= np.exp(-t * 0.3) * envelope(t, length, 0.08, 1.0)
    else:
        raise ValueError(kind)
    return (velocity * y).astype(np.float32)


def room_ir(rng):
    length = 3.6
    t = np.arange(round(SR * length)) / SR
    impulse = np.zeros((len(t), 2), dtype=np.float32)
    # Diffuse tail with a subdued high-frequency response, no direct impulse.
    for channel in range(2):
        noise = rng.normal(size=len(t))
        noise = np.convolve(noise, np.ones(15) / 15, mode="same")
        tail = noise * np.exp(-t * 2.4) * (1 - np.exp(-t * 22))
        tail[: round(0.04 * SR)] = 0
        for delay, gain in [(0.061, 0.45), (0.103, 0.3), (0.173, 0.22)]:
            tail[round((delay + 0.009 * channel) * SR)] += gain
        impulse[:, channel] = tail / np.sqrt(np.sum(tail * tail))
    return impulse


def write_pcm(path, samples):
    if not np.all(np.isfinite(samples)):
        raise ValueError("Nonfinite audio")
    # 32-bit integer PCM keeps the synthesis headroom for the separate master.
    if np.max(np.abs(samples)) >= 1:
        raise ValueError("Unmastered audio exceeded PCM headroom")
    wavfile.write(path, SR, (samples * (2**31 - 1)).astype(np.int32))


def compose(timeline, out):
    chapters = timeline["chapters"][:10]
    if len(chapters) != 10 or any(
        abs(chapter["start"] - i * 12) > 1 / SR for i, chapter in enumerate(chapters)
    ):
        raise ValueError("This sketch requires the existing 0–120 second timeline")
    if timeline["duration"] < DURATION:
        raise ValueError("Film shorter than sketch")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n = round(DURATION * SR)
    stems = {
        name: np.zeros((n, 2), dtype=np.float32)
        for name in ["theme", "harmony", "pulse", "bass", "air"]
    }
    events = []

    def note(stem, kind, onset, midi, length, gain, pan=0):
        if onset >= DURATION:
            return
        onset = max(0, onset)
        sound = instrument(kind, midi, length, gain, rng)
        start = round(onset * SR)
        end = min(start + len(sound), n)
        angle = (pan + 1) * np.pi / 4
        stems[stem][start:end] += sound[: end - start, None] * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        events.append(
            dict(
                stem=stem,
                instrument=kind,
                start=round(onset, 4),
                midi=midi,
                duration=length,
                gain=gain,
                pan=pan,
            )
        )

    for i, (root, chord, melody) in enumerate(PHRASES):
        start = i * 12.0
        density = [0.25, 0.4, 0.72, 0.6, 0.38, 0.68, 0.82, 0.52, 0.68, 0.9][i]
        for voice, pitch in enumerate(chord):
            note(
                "harmony",
                "bowed",
                max(0, start - 1.4 + voice * 0.08),
                pitch,
                14.6,
                0.022 + density * 0.012,
                (voice - 1.5) * 0.3,
            )
        note("bass", "bass", start + 0.1, root, 6.6, 0.078)
        note(
            "bass", "bass", start + 6.1, root + (7 if i in [2, 6, 9] else 0), 6.4, 0.056
        )

        # The motif breathes across each phrase instead of sounding every beat.
        positions = [0.5, 2, 4, 5.5, 8, 10, 12, 14]
        if i == 0:
            positions = [2, 4, 6, 7.5, 9, 11, 13, 14.5]
        for j, (beat, pitch) in enumerate(zip(positions, melody)):
            gain = 0.074 * (0.75 if j % 2 else 1.0)
            if i == 4:
                gain *= 0.7
            note(
                "theme",
                "felt",
                start + beat * BEAT,
                pitch,
                3.8,
                gain,
                0.12 * np.sin(j * 0.9),
            )
        if i in [2, 6, 8, 9]:
            # MLP: two register responses; propagation: answer returns downward.
            answer = melody[::2] if i != 6 else melody[::-2]
            for j, pitch in enumerate(answer):
                note(
                    "theme",
                    "glass",
                    start + (3.0 + j * 3) * BEAT,
                    pitch + 12,
                    2.5,
                    0.019 + density * 0.008,
                    (-1) ** j * 0.48,
                )

        # Changing subdivisions and omissions, rather than a continuous loop.
        spacing = 1.0 if i in [2, 5, 6, 9] else 2.0
        beats = np.arange(0, 16, spacing)
        for j, beat in enumerate(beats):
            if i == 0 and beat < 8:
                continue
            if i == 4 and j % 2:
                continue
            if i == 5 and j in [2, 5, 9, 14]:
                continue  # Dropout produces spaces, not a new percussion track.
            pitch = chord[j % len(chord)]
            if i == 6:
                pitch = chord[(-j - 1) % len(chord)]
            note(
                "pulse",
                "plucked",
                start + beat * BEAT + 0.018,
                pitch,
                1.5,
                0.015 + density * 0.022,
                0.35 * np.sin(j * 1.3 + i),
            )

    # Shape-specific accents, placed within movements rather than at every title.
    for onset, pitches in [
        (17.0, [62, 69, 76]),
        (21.4, [65, 69, 74]),
        (95.0, [62, 65, 69]),
        (107.0, [69, 74, 76]),
    ]:
        for j, pitch in enumerate(pitches):
            note("theme", "glass", onset + j * 0.13, pitch, 3.2, 0.016, (j - 1) * 0.38)

    # A very low, filtered noise swell at two structural arrivals, not a whoosh
    # on every chapter. The noise source is generated here, not a sound library.
    for onset in [22.5, 105.7]:
        length = 3.5
        t = np.arange(round(length * SR)) / SR
        noise = rng.normal(size=len(t)).astype(np.float32)
        noise = np.convolve(noise, np.ones(85) / 85, mode="same")
        noise *= envelope(t, length, 2.4, 1.0) * 0.04
        begin = round(onset * SR)
        end = min(n, begin + len(noise))
        stems["air"][begin:end] += noise[: end - begin, None]

    impulse = room_ir(rng)
    mix = np.zeros((n, 2), dtype=np.float32)
    for name, samples in stems.items():
        wet = {"theme": 0.2, "harmony": 0.28, "pulse": 0.13, "bass": 0.03, "air": 0.2}[
            name
        ]
        for channel in range(2):
            reflection = fftconvolve(samples[:, channel], impulse[:, channel])[:n]
            samples[:, channel] += wet * reflection
        # A sample ending, not the ending of the full-film composition.
        samples[:SR] *= np.linspace(0, 1, SR)[:, None]
        samples[-4 * SR :] *= np.cos(np.linspace(0, np.pi / 2, 4 * SR))[:, None] ** 2
        write_pcm(out / f"stem-{name}.wav", samples)
        mix += samples
    write_pcm(out / "score-unmastered.wav", mix)
    report = dict(
        status="listening_draft",
        title="点还在动 · 声音草稿 01",
        duration=DURATION,
        sample_rate=SR,
        bpm=80,
        key="D minor / open ninths",
        seed=SEED,
        source="original score and numerical instrument synthesis",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artistic_review="pending_listening",
        audio_peak=float(np.abs(mix).max()),
        audio_rms=float(np.sqrt(np.mean(mix**2))),
        cues=[dict(start=c["start"], title=c["title"], bars=4) for c in chapters],
        events=events,
    )
    (out / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "duration": DURATION,
                "notes": len(events),
                "peak": report["audio_peak"],
                "output": str(out),
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose(json.loads(args.timeline.read_text()), args.output.resolve())


if __name__ == "__main__":
    main()
