"""Through-composed solo piano score for the 681.05-second Zanim film.

Preserves the approved sketch's first 24 performed bars, then develops new
themes, tonal regions, textures and dynamics. All notes are authored here;
there is no repeated audio clip or transcription of an existing nocturne.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from scripts.compose_nocturne import (
    A,
    B,
    CODA,
    HARMONY_A,
    HARMONY_B,
    Piano,
    SR,
    midi_file,
    performance,
)


# New subdominant theme, Db major. Longer opening intervals, falling answers.
C = [
    [
        (0, 0.8, 77),
        (1, 0.6, 80),
        (1.7, 0.25, 82),
        (2, 0.6, 84),
        (2.7, 0.25, 85),
        (3, 0.9, 80),
    ],
    [(0, 1.1, 75), (1.3, 0.55, 78), (2, 0.6, 82), (2.7, 0.25, 80), (3, 0.9, 75)],
    [
        (0, 1.2, 85),
        (1.4, 0.5, 89),
        (2, 0.6, 87),
        (2.7, 0.25, 85),
        (3, 0.65, 84),
        (3.7, 0.2, 80),
    ],
    [
        (0, 1.2, 87),
        (1.4, 0.5, 84),
        (2, 0.6, 82),
        (2.7, 0.25, 80),
        (3, 0.65, 78),
        (3.7, 0.2, 77),
    ],
    [(0, 0.9, 75), (1, 0.6, 80), (1.7, 0.25, 85), (2, 0.9, 82), (3, 0.9, 80)],
    [(0, 1.2, 82), (1.4, 0.5, 85), (2, 0.6, 84), (2.7, 0.25, 80), (3, 0.9, 78)],
    [
        (0, 0.9, 77),
        (1, 0.6, 75),
        (1.7, 0.25, 72),
        (2, 0.6, 70),
        (2.7, 0.25, 78),
        (3, 0.9, 75),
    ],
    [(0, 1.8, 73), (2, 0.65, 68), (2.7, 0.2, 65), (3, 0.9, 73)],
]
HC = [
    ("Db", 37, [56, 61, 65]),
    ("Ab7", 44, [54, 60, 63]),
    ("Bbm", 34, [53, 58, 61]),
    ("Fm7", 41, [56, 60, 63]),
    ("Gb", 42, [58, 61, 66]),
    ("Ebm7", 39, [54, 58, 61]),
    ("Ab7", 44, [54, 60, 63]),
    ("Db", 37, [56, 61, 65]),
]
# A lower, restrained F-minor subject, different from the opening melody.
M = [
    [(0, 1.4, 72), (1.6, 0.3, 73), (2, 0.8, 75), (3, 0.9, 77)],
    [(0, 0.9, 76), (1, 0.6, 79), (1.7, 0.2, 82), (2, 0.9, 79), (3, 0.9, 76)],
    [
        (0, 0.9, 77),
        (1, 0.6, 80),
        (1.7, 0.2, 79),
        (2, 0.6, 77),
        (2.7, 0.2, 75),
        (3, 0.9, 72),
    ],
    [(0, 1.4, 73), (1.6, 0.3, 77), (2, 0.9, 82), (3, 0.9, 80)],
    [(0, 0.9, 80), (1, 0.6, 77), (1.7, 0.2, 75), (2, 0.9, 73), (3, 0.9, 72)],
    [(0, 1.4, 71), (1.6, 0.3, 73), (2, 0.6, 76), (2.7, 0.2, 79), (3, 0.9, 82)],
    [
        (0, 0.9, 80),
        (1, 0.6, 79),
        (1.7, 0.2, 77),
        (2, 0.6, 76),
        (2.7, 0.2, 73),
        (3, 0.9, 72),
    ],
    [(0, 1.8, 77), (2, 0.6, 72), (2.7, 0.2, 68), (3, 0.9, 77)],
]
HM = [
    ("Fm", 41, [48, 53, 56]),
    ("C7/E", 40, [55, 58, 60]),
    ("Fm/Ab", 44, [48, 53, 56]),
    ("Bbm", 34, [53, 58, 61]),
    ("Db", 37, [56, 61, 65]),
    ("Gdim7", 43, [53, 56, 59]),
    ("C7", 36, [52, 58, 60]),
    ("Fm", 41, [48, 53, 56]),
]
# Brighter Eb-major subject: quicker triplet replies and a different contour.
E = [
    [(0, 0.6, 75), (0.7, 0.55, 79), (1.4, 0.5, 82), (2, 1.2, 87), (3.3, 0.6, 86)],
    [
        (0, 0.6, 82),
        (0.7, 0.55, 79),
        (1.4, 0.5, 77),
        (2, 0.6, 74),
        (2.7, 0.2, 77),
        (3, 0.9, 79),
    ],
    [(0, 0.6, 79), (0.7, 0.55, 84), (1.4, 0.5, 87), (2, 1.2, 86), (3.3, 0.6, 84)],
    [
        (0, 0.6, 84),
        (0.7, 0.55, 80),
        (1.4, 0.5, 79),
        (2, 0.6, 77),
        (2.7, 0.2, 75),
        (3, 0.9, 72),
    ],
    [(0, 0.6, 77), (0.7, 0.55, 80), (1.4, 0.5, 84), (2, 1.2, 87), (3.3, 0.6, 84)],
    [
        (0, 0.6, 86),
        (0.7, 0.55, 84),
        (1.4, 0.5, 82),
        (2, 0.6, 80),
        (2.7, 0.2, 77),
        (3, 0.9, 74),
    ],
    [(0, 0.6, 79), (0.7, 0.55, 82), (1.4, 0.5, 87), (2, 1.2, 89), (3.3, 0.6, 87)],
    [
        (0, 0.6, 86),
        (0.7, 0.55, 82),
        (1.4, 0.5, 80),
        (2, 0.6, 77),
        (2.7, 0.2, 74),
        (3, 0.9, 75),
    ],
]
HE = [
    ("Eb", 39, [55, 58, 63]),
    ("Gm/Bb", 34, [55, 58, 62]),
    ("Cm", 36, [55, 60, 63]),
    ("Ab", 44, [51, 56, 60]),
    ("Fm7", 41, [56, 60, 63]),
    ("Bb7", 34, [53, 56, 62]),
    ("Eb/G", 43, [58, 63, 67]),
    ("Bb7", 34, [53, 56, 62]),
]


def transpose(melodies, harmonies, steps):
    def bass(pitch):
        pitch += steps
        while pitch < 32:
            pitch += 12
        while pitch > 47:
            pitch -= 12
        return pitch

    return (
        [[(t, d, p + steps) for t, d, p in bar] for bar in melodies],
        [
            (f"{name} ({steps:+d})", bass(root), [p + steps for p in chord])
            for name, root, chord in harmonies
        ],
    )


def complete_performance(timeline):
    duration = timeline["duration"]
    if abs(duration - 681.05) > 1 / SR or len(timeline["chapters"]) != 49:
        raise ValueError("This authored score requires the complete 681.05-second film")
    starts = {c["title"].split(" · ")[-1]: c["start"] for c in timeline["chapters"]}
    rng = np.random.default_rng(20261001)
    old_notes, old_pedals, old_bars = performance()
    bridge = old_bars[24]["start"]
    notes = [dict(n) for n in old_notes if n["start"] < bridge]
    pedals = [dict(p) for p in old_pedals if p["time"] < bridge]
    bars = [dict(b) for b in old_bars[:24]]
    movements = [
        dict(
            start=0,
            end=120,
            title="夜曲主题",
            key="Ab major → Db major",
            texture="cantabile, original 24-bar performance retained",
        )
    ]

    def add(
        start,
        end,
        melodies,
        harmonies,
        name,
        key,
        touch=65,
        texture="arpeggio",
        ornament=False,
    ):
        if len(melodies) != len(harmonies):
            raise ValueError("Melody/harmony bar count mismatch")
        count = len(melodies)
        grid = np.linspace(0, count * 4, count * 960 + 1)
        phase = grid[:-1] % 4
        b = (grid[:-1] // 4).astype(int)
        # Written phrase breathing; the last bar is broader, inner bars press on.
        pace = 1 + 0.05 * np.cos(phase * np.pi / 2)
        pace *= np.choose(b % 8, [1.04, 0.98, 0.96, 1.10, 0.96, 0.94, 0.98, 1.14])
        pace *= 1 + 0.09 * np.maximum(phase - 3, 0)
        pace[b == count - 1] *= 1.12
        clock = np.r_[0, np.cumsum(pace)]
        clock = start + clock / clock[-1] * (end - start)

        def at(beat):
            return float(np.interp(beat, grid, clock))

        def note(hand, beat, length, pitch, velocity, delay=0):
            onset = at(beat) + delay
            finish = min(end + 0.02, at(min(count * 4, beat + length)) + delay)
            notes.append(
                dict(
                    hand=hand,
                    start=onset,
                    end=max(onset + 0.04, finish),
                    midi=int(pitch),
                    velocity=int(np.clip(velocity, 22, 94)),
                )
            )

        for i, (melody, (chord_name, root, upper)) in enumerate(
            zip(melodies, harmonies)
        ):
            base = 4 * i
            swell = np.sin(np.pi * i / max(1, count - 1))
            strength = touch + (7 if texture == "running" else 3) * swell
            bars.append(
                dict(
                    bar=len(bars) + 1,
                    start=at(base),
                    chord=chord_name,
                    section=name,
                    tonal_region=key,
                    texture=texture,
                )
            )
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
            if texture == "wave":
                pattern = [
                    root,
                    upper[0],
                    upper[2],
                    upper[1],
                    upper[2],
                    upper[0],
                    root + 12,
                    upper[1],
                    upper[2],
                    upper[0],
                    upper[2],
                    upper[1],
                ]
            if texture == "running":
                pattern = [
                    root,
                    upper[0],
                    upper[1],
                    upper[2],
                    upper[1],
                    upper[0],
                    upper[1],
                    upper[2],
                    root + 12,
                    upper[0],
                    upper[1],
                    upper[2],
                    upper[1],
                    upper[0],
                    upper[1],
                    upper[2],
                ]
            if texture == "sparse":
                pattern = [root, upper[0], upper[2], upper[1], upper[0], upper[2]]
            if texture == "closing" and i == count - 1:
                pattern = [root, upper[0], upper[1], upper[2]]
            spread = (
                4 / len(pattern) if texture != "closing" or i < count - 1 else 1 / 3
            )
            for j, pitch in enumerate(pattern):
                v = 41 if j == 0 else 33 if j % 3 == 0 else 29
                v += (touch - 65) * 0.45 + swell * 3 + rng.normal(0, 1.2)
                note(
                    0, base + j * spread, spread * 0.82, pitch, v, rng.normal(0, 0.004)
                )
            for j, (offset, length, pitch) in enumerate(melody):
                v = strength + 3 * np.sin(j * 0.8)
                delay = 0.024 + (0.012 if length > 1 else 0) + rng.normal(0, 0.003)
                if ornament and j == 0 and length >= 0.9 and i % 4 == 0:
                    for k, p in enumerate([pitch, pitch + 2, pitch, pitch - 1, pitch]):
                        note(
                            1,
                            base + offset + 0.13 * k,
                            0.12 if k < 4 else length - 0.55,
                            p,
                            v - (4 if k % 2 else 0),
                            delay,
                        )
                else:
                    note(1, base + offset, length * 0.98, pitch, v, delay)
                if texture == "running" and i % 8 >= 4 and j == 0:
                    note(
                        1,
                        base + offset,
                        min(1.1, length),
                        pitch - 12,
                        v - 12,
                        delay + 0.015,
                    )
            if texture == "answer" and i % 2 == 0:
                # A quiet inner answer, using the actual harmony rather than
                # arbitrarily doubling the melody at a fixed interval.
                for j, pitch in enumerate(upper[:2]):
                    note(1, base + 2.3 + j * 0.6, 0.48, pitch + 12, 39, 0.03)

    # Rewrite the sketch's closed coda as a dominant bridge into Db major.
    bridge_m = CODA[:3] + [
        [(0, 1.4, 80), (1.6, 0.3, 78), (2, 0.9, 75), (3, 0.9, 72)],
        [(0, 1.4, 75), (1.6, 0.3, 80), (2, 0.9, 78), (3, 0.9, 75)],
    ]
    bridge_h = HARMONY_A[1:4] + [HC[1], HC[1]]
    add(bridge, 120, bridge_m, bridge_h, "转向降D", "Ab → Db", touch=60)

    a_db, ha_db = transpose(A, HARMONY_A, 5)
    low_c = [[(t, d, p - 12) for t, d, p in bar] for bar in C]
    ii_m = C + a_db + low_c + C
    ii_h = HC + ha_db + HC + HC
    ii_m[-1] = [(0, 1.4, 76), (1.6, 0.3, 79), (2, 0.9, 82), (3, 0.9, 76)]
    ii_h[-1] = ("C7", 36, [52, 58, 60])
    add(
        120,
        starts["08.4"],
        ii_m,
        ii_h,
        "空间舒展",
        "Db major",
        touch=67,
        texture="wave",
        ornament=True,
    )
    movements.append(
        dict(
            start=120,
            end=starts["08.4"],
            title="空间舒展",
            key="Db major",
            texture="wave",
        )
    )

    b_fm, hb_fm = transpose(B, HARMONY_B, -7)
    iii_m = M + b_fm + M + [M[0], M[6]]
    iii_h = HM + hb_fm + HM + [HM[0], ("Bb7", 34, [53, 56, 62])]
    iii_m[-1] = [(0, 0.9, 77), (1, 0.6, 80), (1.7, 0.2, 82), (2, 0.9, 80), (3, 0.9, 74)]
    add(
        starts["08.4"],
        starts["10.1"],
        iii_m,
        iii_h,
        "往复与记忆",
        "F minor",
        touch=63,
        texture="sparse",
    )
    movements.append(
        dict(
            start=starts["08.4"],
            end=starts["10.1"],
            title="往复与记忆",
            key="F minor",
            texture="sparse",
        )
    )

    a_eb, ha_eb = transpose(A, HARMONY_A, -5)
    iv_m, iv_h = E + a_eb + E, HE + ha_eb + HE
    iv_m[-1] = [(0, 0.9, 79), (1, 0.6, 77), (1.7, 0.2, 74), (2, 0.9, 71), (3, 0.9, 79)]
    iv_h[-1] = ("G7", 43, [53, 59, 62])
    add(
        starts["10.1"],
        starts["11.2"],
        iv_m,
        iv_h,
        "光线交织",
        "Eb major",
        touch=70,
        texture="answer",
        ornament=True,
    )
    movements.append(
        dict(
            start=starts["10.1"],
            end=starts["11.2"],
            title="光线交织",
            key="Eb major",
            texture="inner answers",
        )
    )

    m_cm, hm_cm = transpose(M, HM, -5)
    v_m, v_h = B + m_cm + B + m_cm, HARMONY_B + hm_cm + HARMONY_B + hm_cm
    v_m += [
        [(0, 0.9, 79), (1, 0.6, 84), (1.7, 0.2, 87), (2, 0.9, 84), (3, 0.9, 79)],
        [(0, 0.9, 80), (1, 0.6, 83), (1.7, 0.2, 87), (2, 0.9, 83), (3, 0.9, 80)],
        [(0, 0.9, 78), (1, 0.6, 81), (1.7, 0.2, 87), (2, 0.9, 83), (3, 0.9, 78)],
        [(0, 1.4, 75), (1.6, 0.3, 78), (2, 0.9, 83), (3, 0.9, 75)],
    ]
    v_h += [
        ("Cm", 36, [55, 60, 63]),
        ("G#m", 44, [51, 56, 59]),
        ("B7", 35, [54, 57, 63]),
        ("B7", 35, [54, 57, 63]),
    ]
    add(
        starts["11.2"],
        starts["13.3"],
        v_m,
        v_h,
        "推进与高潮",
        "C minor → E major",
        touch=78,
        texture="running",
        ornament=True,
    )
    movements.append(
        dict(
            start=starts["11.2"],
            end=starts["13.3"],
            title="推进与高潮",
            key="C minor → E major",
            texture="four-against-three, octave accents",
        )
    )

    c_e, hc_e = transpose(C, HC, 3)
    e_e, he_e = transpose(E, HE, 1)
    vi_m, vi_h = c_e + e_e, hc_e + he_e
    vi_m += [
        [(0, 1.4, 83), (1.6, 0.3, 81), (2, 0.9, 78), (3, 0.9, 75)],
        [(0, 1.4, 75), (1.6, 0.3, 73), (2, 0.9, 70), (3, 0.9, 79)],
    ]
    vi_h += [("B7", 35, [54, 57, 63]), HARMONY_A[3]]
    add(
        starts["13.3"],
        starts["14.1"],
        vi_m,
        vi_h,
        "明亮的远景",
        "E major → Ab major",
        touch=69,
        texture="wave",
    )
    movements.append(
        dict(
            start=starts["13.3"],
            end=starts["14.1"],
            title="明亮的远景",
            key="E major → Ab major",
            texture="wide arpeggios",
        )
    )

    add(starts["14.1"], starts["15.5"], A, HARMONY_A, "主题归来", "Ab major", touch=61)
    closing_m = [CODA[0], CODA[1], A[7], A[5], CODA[2], CODA[3]]
    closing_h = [
        HARMONY_A[1],
        HARMONY_A[2],
        HARMONY_A[4],
        HARMONY_A[5],
        HARMONY_A[3],
        ("Ab", 32, [51, 56, 60]),
    ]
    add(
        starts["15.5"],
        duration - 2.8,
        closing_m,
        closing_h,
        "归环与尾声",
        "Ab major",
        touch=53,
        texture="closing",
    )
    final_bar = bars[-1]["start"]
    notes += [
        dict(
            hand=1,
            start=final_bar + 0.08 + j * 0.035,
            end=duration - 2.9,
            midi=pitch,
            velocity=43 - j * 3,
        )
        for j, pitch in enumerate([72, 75])
    ]
    for hand in [0, 1]:
        pedals.append(dict(time=duration - 2.65, hand=hand, value=0))
    movements.append(
        dict(
            start=starts["14.1"],
            end=duration,
            title="主题归来与收束",
            key="Ab major",
            texture="cantabile → closing chord",
        )
    )

    # Protect repeated keys and section handovers from overlapping note-offs.
    for hand in [0, 1]:
        for pitch in {n["midi"] for n in notes if n["hand"] == hand}:
            repeated = sorted(
                [n for n in notes if n["hand"] == hand and n["midi"] == pitch],
                key=lambda n: n["start"],
            )
            for a, b in zip(repeated, repeated[1:]):
                a["end"] = min(a["end"], b["start"] - 0.008)
    if any(
        not (0 <= n["start"] < n["end"] <= duration and 21 <= n["midi"] <= 108)
        for n in notes
    ):
        raise ValueError("Invalid full-film piano events")
    return notes, pedals, bars, movements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timeline = json.loads(args.timeline.read_text())
    notes, pedals, bars, movements = complete_performance(timeline)
    piano = Piano(args.soundfont.resolve())
    try:
        audio = piano.render(notes, pedals, duration=timeline["duration"])
    finally:
        piano.close()
    peak = float(np.abs(audio).max())
    if not np.all(np.isfinite(audio)) or not 0.001 < peak < 1:
        raise ValueError(f"Invalid full score PCM: {peak}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        out / "score-unmastered.wav", SR, (audio * (2**31 - 1)).astype(np.int32)
    )
    (out / "nocturne.mid").write_bytes(midi_file(notes, pedals))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    score = dict(
        status="complete_piano_edition",
        title="点还在动 · 钢琴夜曲完整版",
        duration=timeline["duration"],
        sample_rate=SR,
        meter="12/8 with 4:3 development",
        source="original full-length solo piano composition, real piano samples",
        composer_source="scripts/compose_film_nocturne.py",
        source_sha256=digest(Path(__file__)),
        piano_source_sha256=digest(Path(__file__).with_name("compose_nocturne.py")),
        soundfont_sha256=digest(args.soundfont),
        soundfont=args.soundfont.name,
        artistic_review="preview_direction_selected; full_film_listening_pending",
        preview_performance_preserved_until=bars[24]["start"],
        audio_peak=peak,
        bars=bars,
        notes=notes,
        pedal=pedals,
        movements=movements,
        sample_credits=dict(
            work="Salamander Grand Piano V3+20200602",
            author="Alexander Holm",
            sf2_conversion="Roberto / FreePats",
            source_url="https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html",
            license="CC BY 3.0",
            license_url="https://creativecommons.org/licenses/by/3.0/",
            use="Unmodified SoundFont rendered into this original piano composition",
        ),
    )
    (out / "score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                duration=score["duration"], bars=len(bars), notes=len(notes), peak=peak
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
