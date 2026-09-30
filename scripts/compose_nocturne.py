"""Original romantic piano nocturne sketch, not a transcription of Chopin.

Handwritten melody and harmony, 12/8 broken-chord accompaniment, phrase rubato,
ornamented reprise, and harmony-aware sustain pedal. Uses a real piano SoundFont
through FluidSynth's offline C API; never opens an audio device.
"""

import argparse
import ctypes as ct
import ctypes.util
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.io import wavfile


SR = 48000
DURATION = 120.0
SEED = 202609302

# Each bar has four dotted-quarter beats. MIDI pitches, Ab major / C minor.
# Melody entries are (dotted-beat onset, duration, pitch).
A = [
    [
        (0, 1.1, 75),
        (1.2, 0.55, 80),
        (1.8, 0.2, 79),
        (2, 0.9, 77),
        (3, 0.55, 75),
        (3.6, 0.35, 72),
    ],
    [(0, 1.4, 80), (1.5, 0.3, 79), (1.85, 0.15, 77), (2, 0.9, 75), (3, 0.9, 72)],
    [(0, 0.9, 77), (1, 0.6, 73), (1.65, 0.3, 72), (2, 0.95, 70), (3, 0.9, 73)],
    [
        (0, 0.7, 79),
        (0.85, 0.15, 80),
        (1, 0.6, 79),
        (1.65, 0.3, 77),
        (2, 0.6, 75),
        (2.65, 0.3, 73),
        (3, 0.95, 70),
    ],
    [(0, 1.3, 84), (1.5, 0.4, 82), (2, 0.6, 80), (2.65, 0.3, 79), (3, 0.95, 80)],
    [(0, 1.5, 77), (1.7, 0.25, 80), (2, 0.6, 85), (2.7, 0.25, 84), (3, 0.9, 80)],
    [
        (0, 0.9, 79),
        (1, 0.6, 77),
        (1.65, 0.3, 76),
        (2, 0.6, 75),
        (2.65, 0.3, 73),
        (3, 0.95, 79),
    ],
    [(0, 1.8, 80), (2, 0.65, 75), (2.7, 0.25, 72), (3, 0.95, 68)],
]
B = [
    [
        (0, 1.2, 79),
        (1.35, 0.6, 87),
        (2, 0.6, 84),
        (2.7, 0.25, 82),
        (3, 0.6, 80),
        (3.7, 0.25, 79),
    ],
    [
        (0, 0.9, 86),
        (1, 0.25, 87),
        (1.3, 0.25, 86),
        (1.65, 0.3, 84),
        (2, 0.6, 83),
        (2.7, 0.25, 80),
        (3, 0.6, 79),
        (3.7, 0.25, 77),
    ],
    [
        (0, 0.9, 79),
        (1, 0.6, 84),
        (1.7, 0.25, 86),
        (2, 0.6, 87),
        (2.7, 0.25, 86),
        (3, 0.6, 84),
        (3.7, 0.25, 82),
    ],
    [
        (0, 0.9, 84),
        (1, 0.6, 89),
        (1.7, 0.25, 87),
        (2, 0.6, 85),
        (2.7, 0.25, 84),
        (3, 0.6, 82),
        (3.7, 0.25, 80),
    ],
    [
        (0, 0.9, 80),
        (1, 0.6, 85),
        (1.7, 0.25, 89),
        (2, 0.6, 87),
        (2.7, 0.25, 85),
        (3, 0.6, 84),
        (3.7, 0.25, 80),
    ],
    [
        (0, 0.9, 91),
        (1, 0.6, 89),
        (1.7, 0.25, 88),
        (2, 0.6, 85),
        (2.7, 0.25, 84),
        (3, 0.6, 82),
        (3.7, 0.25, 88),
    ],
    [
        (0, 1.3, 91),
        (1.5, 0.2, 89),
        (1.75, 0.2, 87),
        (2, 0.2, 85),
        (2.25, 0.2, 84),
        (2.5, 0.2, 82),
        (2.75, 0.2, 80),
        (3, 0.95, 79),
    ],
    [
        (0, 0.9, 82),
        (1, 0.3, 80),
        (1.4, 0.25, 79),
        (1.7, 0.25, 77),
        (2, 0.6, 75),
        (2.7, 0.25, 73),
        (3, 0.95, 79),
    ],
]
CODA = [
    [(0, 1.6, 80), (1.75, 0.2, 79), (2, 0.6, 77), (2.7, 0.25, 75), (3, 0.95, 72)],
    [(0, 1.2, 77), (1.4, 0.5, 73), (2, 0.6, 72), (2.7, 0.25, 70), (3, 0.95, 73)],
    [
        (0, 0.9, 79),
        (1, 0.6, 77),
        (1.7, 0.25, 75),
        (2, 0.6, 73),
        (2.7, 0.25, 70),
        (3, 0.95, 67),
    ],
    [(0, 3.8, 68)],
]
HARMONY_A = [
    ("Ab", 44, [51, 56, 60]),
    ("Fm7", 41, [56, 60, 63]),
    ("Bbm", 34, [53, 58, 61]),
    ("Eb7", 39, [55, 58, 61]),
    ("Ab/C", 36, [51, 56, 60]),
    ("Dbmaj7", 37, [56, 60, 65]),
    ("Eb7b9", 39, [55, 61, 64]),
    ("Ab", 44, [51, 56, 60]),
]
HARMONY_B = [
    ("Cm", 36, [55, 60, 63]),
    ("G7/B", 35, [53, 55, 59]),
    ("Cm/Eb", 39, [55, 60, 63]),
    ("Fm", 41, [56, 60, 65]),
    ("Db", 37, [56, 61, 65]),
    ("Bbdim7", 34, [55, 61, 64]),
    ("Eb7", 39, [55, 58, 61]),
    ("Eb7", 39, [55, 58, 61]),
]
HARMONY_CODA = [HARMONY_A[1], HARMONY_A[2], HARMONY_A[3], ("Ab", 32, [51, 56, 60])]


def performance():
    rng = np.random.default_rng(SEED)
    melodies = A + B + A + CODA
    harmonies = HARMONY_A + HARMONY_B + HARMONY_A + HARMONY_CODA
    grid = np.linspace(0, 112, 112 * 240 + 1)
    bar = np.minimum((grid[:-1] // 4).astype(int), 27)
    stretches = np.array(
        [
            1.00,
            0.97,
            1.04,
            1.10,
            0.96,
            0.99,
            1.02,
            1.15,
            1.00,
            0.96,
            0.94,
            0.97,
            0.95,
            0.92,
            1.04,
            1.18,
            0.98,
            0.96,
            0.98,
            1.08,
            0.99,
            1.04,
            1.16,
            1.18,
            1.10,
            1.16,
            1.20,
            1.50,
        ]
    )
    phase = grid[:-1] % 4
    pace = stretches[bar] * (1 + 0.035 * np.cos(phase * np.pi / 2))
    pace *= 1 + 0.10 * np.maximum(phase - 3, 0)
    clock = np.r_[0, np.cumsum(pace)]
    clock = 0.3 + clock / clock[-1] * 114.7

    def at(beat):
        return float(np.interp(beat, grid, clock))

    notes = []
    pedals = []
    bars = []

    def note(hand, beat, length, pitch, velocity, delay=0):
        onset = max(0.05, at(beat) + delay)
        end = at(min(112, beat + length)) + delay
        notes.append(
            dict(
                hand=hand,
                start=onset,
                end=max(onset + 0.06, end),
                midi=pitch,
                velocity=int(np.clip(velocity, 20, 100)),
            )
        )

    for i, (melody, harmony) in enumerate(zip(melodies, harmonies)):
        label, root, upper = harmony
        base = i * 4
        level = np.sin(np.pi * (i - 8) / 8) if 8 <= i < 16 else 0
        level = max(0, level)
        bars.append(
            dict(
                bar=i + 1,
                start=at(base),
                chord=label,
                section="A" if i < 8 else "B" if i < 16 else "A′" if i < 24 else "coda",
            )
        )
        # Clear the previous harmony just after striking the new bass.
        for hand in [0, 1]:
            pedals.extend(
                [
                    dict(time=at(base) + 0.028, hand=hand, value=0),
                    dict(time=at(base) + 0.11, hand=hand, value=100),
                ]
            )
        pattern = [
            root,
            upper[0],
            upper[1],
            upper[2],
            upper[1],
            upper[0],
            root + 12,
            upper[0],
            upper[1],
            upper[2],
            upper[1],
            upper[0],
        ]
        if i == 27:
            pattern = [root, upper[0], upper[1], upper[2]]
        for j, pitch in enumerate(pattern):
            beat = base + j / 3
            strength = 45 if j in [0, 6] else 34 if j % 3 == 0 else 30
            strength += level * 7 - (5 if i >= 24 else 0)
            strength += rng.normal(0, 1.4)
            note(
                0, beat, 0.27 if i != 27 else 1.1, pitch, strength, rng.normal(0, 0.005)
            )

        for j, (offset, length, pitch) in enumerate(melody):
            velocity = 65 + level * 13 + 4 * np.sin(j * 0.75)
            if i >= 24:
                velocity -= (i - 23) * 3
            # Melody starts slightly after the accompaniment and leans into
            # long notes. Jitter is small; phrasing is explicitly written.
            delay = 0.022 + (0.018 if length >= 1 else 0) + rng.normal(0, 0.004)
            if 16 <= i < 24 and j == 0 and length > 1:
                # Measured turn on the reprise, preserving the main arrival.
                scale = [68, 70, 72, 73, 75, 77, 79, 80, 82, 84, 85, 87, 89]
                pos = scale.index(pitch)
                turn = [
                    pitch,
                    scale[min(pos + 1, len(scale) - 1)],
                    pitch,
                    scale[max(pos - 1, 0)],
                    pitch,
                ]
                for k, turning_pitch in enumerate(turn):
                    turn_length = 0.17 if k < 4 else length - 0.72
                    note(
                        1,
                        base + offset + k * 0.18,
                        turn_length,
                        turning_pitch,
                        velocity - (3 if k % 2 else 0),
                        delay,
                    )
            else:
                if i in [4, 11, 18] and j == 0:
                    # Quiet lower appoggiatura, resolved into the sung note.
                    grace_start = at(base + offset) - 0.07
                    notes.append(
                        dict(
                            hand=1,
                            start=grace_start,
                            end=grace_start + 0.085,
                            midi=pitch - 1,
                            velocity=48,
                        )
                    )
                note(1, base + offset, length * 0.99, pitch, velocity, delay)
        if i == 27:
            for j, pitch in enumerate([72, 75]):
                note(1, base, 3.8, pitch, 46 - j * 3, 0.085 + 0.025 * j)

    for hand in [0, 1]:
        pedals.append(dict(time=115.25, hand=hand, value=0))
    # Note-offs must precede the next attack of the same key on a given channel.
    for hand in [0, 1]:
        for pitch in {n["midi"] for n in notes if n["hand"] == hand}:
            repeated = sorted(
                [n for n in notes if n["hand"] == hand and n["midi"] == pitch],
                key=lambda n: n["start"],
            )
            for previous, following in zip(repeated, repeated[1:]):
                previous["end"] = min(previous["end"], following["start"] - 0.008)
    if any(
        n["end"] <= n["start"] or n["start"] < 0 or n["end"] > DURATION for n in notes
    ):
        raise ValueError("Invalid piano event")
    return notes, pedals, bars


class Piano:
    def __init__(self, soundfont):
        name = ctypes.util.find_library("fluidsynth")
        if not name:
            raise RuntimeError("Install the FluidSynth shared library first")
        self.lib = ct.CDLL(name)
        signatures = {
            "new_fluid_settings": (ct.c_void_p, []),
            "fluid_settings_setnum": (
                ct.c_int,
                [ct.c_void_p, ct.c_char_p, ct.c_double],
            ),
            "fluid_settings_setint": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
            "new_fluid_synth": (ct.c_void_p, [ct.c_void_p]),
            "fluid_synth_sfload": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
            "fluid_synth_program_select": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 4),
            "fluid_synth_noteon": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 3),
            "fluid_synth_noteoff": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 2),
            "fluid_synth_cc": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 3),
            "fluid_synth_write_float": (
                ct.c_int,
                [
                    ct.c_void_p,
                    ct.c_int,
                    ct.c_void_p,
                    ct.c_int,
                    ct.c_int,
                    ct.c_void_p,
                    ct.c_int,
                    ct.c_int,
                ],
            ),
            "delete_fluid_synth": (None, [ct.c_void_p]),
            "delete_fluid_settings": (None, [ct.c_void_p]),
        }
        for method, (result, arguments) in signatures.items():
            func = getattr(self.lib, method)
            func.restype, func.argtypes = result, arguments
        self.settings = self.lib.new_fluid_settings()
        for key, value in [
            (b"synth.sample-rate", SR),
            (b"synth.gain", 0.55),
            (b"synth.reverb.room-size", 0.55),
            (b"synth.reverb.damp", 0.45),
            (b"synth.reverb.width", 8),
            (b"synth.reverb.level", 0.17),
        ]:
            if self.lib.fluid_settings_setnum(self.settings, key, value) != 0:
                raise RuntimeError(f"Invalid FluidSynth setting: {key}")
        for key, value in [
            (b"synth.chorus.active", 0),
            (b"synth.polyphony", 256),
            (b"synth.cpu-cores", 1),
            (b"synth.threadsafe-api", 0),
        ]:
            if self.lib.fluid_settings_setint(self.settings, key, value) != 0:
                raise RuntimeError(f"Invalid FluidSynth setting: {key}")
        self.synth = self.lib.new_fluid_synth(self.settings)
        if not self.synth:
            raise RuntimeError("Cannot initialize FluidSynth")
        sfid = self.lib.fluid_synth_sfload(self.synth, str(soundfont).encode(), 1)
        if sfid < 0:
            raise RuntimeError("Cannot load piano SoundFont")
        for hand in [0, 1]:
            if self.lib.fluid_synth_program_select(self.synth, hand, sfid, 0, 0) != 0:
                raise RuntimeError("Piano preset 0 missing in SoundFont")
            self.lib.fluid_synth_cc(self.synth, hand, 10, 61 if hand == 0 else 67)

    def render(self, notes, pedals, duration=DURATION):
        events = []
        for note in notes:
            channel, pitch = note["hand"], note["midi"]
            events += [
                (
                    round(note["start"] * SR),
                    2,
                    "noteon",
                    (channel, pitch, note["velocity"]),
                ),
                (round(note["end"] * SR), 0, "noteoff", (channel, pitch)),
            ]
        events += [
            (round(p["time"] * SR), 1, "cc", (p["hand"], 64, p["value"]))
            for p in pedals
        ]
        n = round(duration * SR)
        events.append((n, 3, None, ()))
        audio = np.zeros((n, 2), dtype=np.float32)
        position = 0
        for sample, _, message, values in sorted(events):
            if sample > position:
                block = audio[position:sample]
                if (
                    self.lib.fluid_synth_write_float(
                        self.synth,
                        len(block),
                        block.ctypes.data,
                        0,
                        2,
                        block.ctypes.data,
                        1,
                        2,
                    )
                    != 0
                ):
                    raise RuntimeError("Piano rendering failed")
                position = sample
            if message:
                if (
                    getattr(self.lib, f"fluid_synth_{message}")(self.synth, *values)
                    != 0
                ):
                    raise RuntimeError(f"Piano MIDI event failed: {message}")
        # Short taper after the natural pedal/reverb release, no looping.
        audio[-SR:] *= np.linspace(1, 0, SR)[:, None]
        return audio

    def close(self):
        self.lib.delete_fluid_synth(self.synth)
        self.lib.delete_fluid_settings(self.settings)


def midi_file(notes, pedals):
    def vlq(value):
        result = [value & 127]
        while value := value >> 7:
            result.insert(0, (value & 127) | 128)
        return bytes(result)

    def chunk(events):
        data = bytearray()
        previous = 0
        for tick, _, message in sorted(events):
            data += vlq(tick - previous) + message
            previous = tick
        data += b"\x00\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(data)) + data

    tracks = [chunk([(0, 0, b"\xff\x51\x03\x07\xa1\x20")])]
    for hand in [0, 1]:
        events = [(0, 0, bytes([0xC0 + hand, 0]))]
        for note in [n for n in notes if n["hand"] == hand]:
            events += [
                (
                    round(note["start"] * 960),
                    2,
                    bytes([0x90 + hand, note["midi"], note["velocity"]]),
                ),
                (round(note["end"] * 960), 0, bytes([0x80 + hand, note["midi"], 0])),
            ]
        events += [
            (round(p["time"] * 960), 1, bytes([0xB0 + hand, 64, p["value"]]))
            for p in pedals
            if p["hand"] == hand
        ]
        tracks.append(chunk(events))
    return b"MThd" + struct.pack(">IHHH", 6, 1, 3, 480) + b"".join(tracks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.soundfont.is_file():
        raise FileNotFoundError(args.soundfont)
    notes, pedals, bars = performance()
    piano = Piano(args.soundfont.resolve())
    try:
        audio = piano.render(notes, pedals)
    finally:
        piano.close()
    peak = float(np.abs(audio).max())
    if not np.all(np.isfinite(audio)) or peak <= 0.001 or peak >= 1:
        raise ValueError(f"Invalid piano rendering: peak={peak}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        out / "score-unmastered.wav", SR, (audio * (2**31 - 1)).astype(np.int32)
    )
    (out / "nocturne.mid").write_bytes(midi_file(notes, pedals))
    score = dict(
        status="listening_draft",
        title="点还在动 · 夜曲草稿 02",
        duration=DURATION,
        sample_rate=SR,
        key="A-flat major / C minor",
        meter="12/8",
        seed=SEED,
        form="A–B–A′–coda, 28 bars",
        timing="phrase rubato, not chapter-locked",
        source="original melody and harmony, sampled solo grand piano",
        artistic_review="pending_listening",
        audio_peak=peak,
        composer_source="scripts/compose_nocturne.py",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        soundfont_sha256=hashlib.sha256(args.soundfont.read_bytes()).hexdigest(),
        soundfont=args.soundfont.name,
        bars=bars,
        notes=notes,
        pedal=pedals,
    )
    if "SalamanderGrandPiano" in args.soundfont.name:
        score["sample_credits"] = dict(
            work="Salamander Grand Piano V3+20200602",
            author="Alexander Holm",
            sf2_conversion="Roberto / FreePats",
            source_url="https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html",
            license="CC BY 3.0",
            license_url="https://creativecommons.org/licenses/by/3.0/",
            use="Unmodified SoundFont rendered into this original piano composition",
        )
    (out / "score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"notes": len(notes), "bars": len(bars), "peak": peak, "output": str(out)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
