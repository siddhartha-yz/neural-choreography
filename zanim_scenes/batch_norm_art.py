"""Batch statistics as collective centering, scaling and affine motion."""

import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import GOLD, ease
from zanim_scenes.style import label
from zanim_scenes.models import ep_07_5 as model
from zanim_scenes.models import ep_07_7 as dense

PREFIX_SECONDS = 12.0


def stages():
    x = np.stack(
        [
            model.BATCH_X * 0.4 - 2,
            model.BATCH_X * 0.85 + 1,
            model.BATCH_X * 1.3 - 1,
            model.BATCH_X * 0.65 + 2,
        ]
    )
    centered = x - x.mean(axis=1, keepdims=True)
    standardized = centered / np.sqrt(np.mean(centered**2, axis=1, keepdims=True))
    return x, centered, standardized, model.GAMMA * standardized + model.BETA


def append(s, chapter, palette):
    chapter("批量归一化 · 07.5", "移到中心，统一尺度，再缩放平移。")
    objects = []
    dots = []
    guides = []
    x, centered, standardized, output = stages()

    def xf(xy, size=1):
        return Transform2D.translation(*map(float, xy)) @ Transform2D.scaling(
            max(0.001, float(size))
        )

    def xy(ch, value):
        return np.array([float(value) * 1.2, (1.5 - ch) * 1.3])

    for ch in range(4):
        y = (1.5 - ch) * 1.3
        guide = s.add(
            Line(
                (-7.7, y), (7.7, y), style=Style.outline(palette[ch], 0.012), z_index=0
            )
        )
        guide.opacity(to=0.18, duration=0)
        objects.append(guide)
        for j in range(5):
            dot = s.add(
                Circle(
                    0.13,
                    style=Style.solid(palette[ch]),
                    transform=xf(xy(ch, x[ch, j])),
                    z_index=5,
                )
            )
            dots.append(dot)
            objects.append(dot)
        marker = s.add(
            Circle(
                0.25,
                style=Style.outline(palette[ch], 0.023),
                transform=xf(xy(ch, x[ch].mean())),
                z_index=6,
            )
        )
        objects.append(marker)
        guides.append(marker)
    zero = s.add(Line((0, -3), (0, 3), style=Style.outline(GOLD, 0.012), z_index=1))
    zero.opacity(to=0.3, duration=0)
    objects.append(zero)
    caption = label(s, "0", 0, 3.45, 0.24, GOLD)
    objects.append(caption)
    s.wait(0.85)
    for initial, target in (
        (x, centered),
        (centered, standardized),
        (standardized, output),
    ):
        with s.parallel():
            for ch in range(4):
                for j in range(5):
                    dots[ch * 5 + j].transform_function(
                        lambda u, ch=ch, j=j, initial=initial, target=target: xf(
                            xy(
                                ch,
                                initial[ch, j]
                                + ease(u) * (target[ch, j] - initial[ch, j]),
                            )
                        ),
                        duration=2.1,
                        easing=Easing.LINEAR,
                    )
                guides[ch].transform_function(
                    lambda u, ch=ch, initial=initial, target=target: xf(
                        xy(
                            ch,
                            initial[ch].mean()
                            + ease(u) * (target[ch].mean() - initial[ch].mean()),
                        )
                    ),
                    duration=2.1,
                    easing=Easing.LINEAR,
                )
        s.wait(0.3)
    # Gather into the next chapter's two-channel input motif (artistic handoff).
    incoming = []
    for ch in range(2):
        for cell in range(4):
            r, c = divmod(cell, 2)
            target = np.array(
                [-6 + (c - 0.5) * 0.32 + (ch - 0.5) * 0.95, (0.5 - r) * 0.45]
            )
            item = s.add(
                Circle(
                    0.08 + 0.09 * dense.INPUT_X.reshape(2, 4)[ch, cell],
                    style=Style.solid(GOLD),
                    transform=xf(xy(ch, output[ch, cell])),
                    z_index=7,
                )
            )
            item.opacity(to=0, duration=0)
            incoming.append((item, xy(ch, output[ch, cell]), target))
    with s.parallel():
        for obj in objects:
            obj.fade_out(duration=1.2)
        for item, a, b in incoming:
            item.fade_in(duration=1.2)
            item.transform_function(
                lambda u, a=a, b=b: xf(a + ease(u) * (b - a)),
                duration=3.15,
                easing=Easing.LINEAR,
            )
    for obj in objects:
        obj.remove()
    return [item for item, _, _ in incoming]
