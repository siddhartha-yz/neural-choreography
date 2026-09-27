"""Identity bypass and additive correction, followed by a DenseNet handoff."""

import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import BLUE, GOLD, PINK, ease
from zanim_scenes.style import label
from zanim_scenes.models import ep_07_6 as residual
from zanim_scenes.models import ep_07_7 as dense

PREFIX_SECONDS = 12.0


def batch():
    x = dense.INPUT_X.reshape(2, 4)
    f = (
        residual.WEIGHTS_2 @ np.maximum(residual.WEIGHTS_1 @ x + residual.BIAS_1, 0)
        + residual.BIAS_2
    )
    return x, f, np.maximum(x + f, 0)


def append(s, chapter, palette):
    chapter("残差网络 · 07.6", "原样直达，修正相加。")
    objects = []

    def dot(xy, r, color):
        item = s.add(
            Circle(
                r,
                style=Style.solid(color),
                transform=Transform2D.translation(*map(float, xy)),
                z_index=7,
            )
        )
        objects.append(item)
        return item

    def line(a, b, color, alpha=0.45, width=0.025):
        item = s.add(
            Line(tuple(a), tuple(b), style=Style.outline(color, width), z_index=2)
        )
        item.opacity(to=alpha, duration=0)
        objects.append(item)
        return item

    def text(t, x, y, c):
        objects.append(label(s, t, x, y, 0.24, c))

    def xf(xy, scale=1):
        return Transform2D.translation(*map(float, xy)) @ Transform2D.scaling(
            max(0.001, float(scale))
        )

    def route(a, b, u, lift=0):
        return (1 - u) * a + u * b + np.array([0, lift * math.sin(math.pi * u)])

    def trace(a, b, color, lift=0):
        for k in range(40):
            line(
                route(a, b, k / 40, lift),
                route(a, b, (k + 1) / 40, lift),
                color,
                0.34,
                0.018,
            )

    x, f, y = batch()
    start = np.array([-6.0, 0])
    processing = np.array([-1.0, -2.0])
    join = np.array([4.0, 0])
    trace(start, join, BLUE, 2.3)
    trace(start, processing, GOLD, -0.5)
    trace(processing, join, GOLD, -0.5)
    for cx in (-2.1, 0.1):
        for a, b in (
            ((cx - 0.45, -2.65), (cx + 0.45, -2.65)),
            ((cx + 0.45, -2.65), (cx + 0.45, -1.35)),
            ((cx + 0.45, -1.35), (cx - 0.45, -1.35)),
            ((cx - 0.45, -1.35), (cx - 0.45, -2.65)),
        ):
            line(a, b, GOLD, 0.32)
    text("直连", -0.7, 3, BLUE)
    text("修正", -1, -3.2, GOLD)
    text("+", 4, 1.0, PINK)
    identities = []
    updates = []
    offsets = []
    for channel in range(2):
        for cell in range(4):
            r, c = divmod(cell, 2)
            offset = np.array(
                [(c - 0.5) * 0.32 + (channel - 0.5) * 0.95, (0.5 - r) * 0.45]
            )
            offsets.append(offset)
            identities.append(
                dot(start + offset, 0.08 + 0.09 * x[channel, cell], palette[channel])
            )
            updates.append(dot(start + offset, 0.08 + 0.09 * x[channel, cell], GOLD))
    s.wait(0.85)
    with s.parallel():
        for i, item in enumerate(identities):
            item.transform_function(
                lambda u, i=i: xf(route(start, join, ease(u), 2.3) + offsets[i]),
                duration=2.8,
                easing=Easing.LINEAR,
            )
        for i, item in enumerate(updates):
            item.transform_function(
                lambda u, i=i: xf(route(start, processing, ease(u), -0.5) + offsets[i]),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    with s.parallel():
        for i, item in enumerate(updates):
            scale = (0.025 + 0.6 * abs(f.flat[i])) / (0.08 + 0.09 * x.flat[i])
            item.transform_function(
                lambda u, i=i, scale=scale: xf(
                    processing + offsets[i], 1 + ease(u) * (scale - 1)
                ),
                duration=0.65,
            )
    with s.parallel():
        for i, item in enumerate(updates):
            scale = (0.025 + 0.6 * abs(f.flat[i])) / (0.08 + 0.09 * x.flat[i])
            item.transform_function(
                lambda u, i=i, scale=scale: xf(
                    route(processing, join, ease(u), -0.5) + offsets[i], scale
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    with s.parallel():
        for i, item in enumerate(identities):
            scale = (0.08 + 0.09 * y.flat[i]) / (0.08 + 0.09 * x.flat[i])
            item.transform_function(
                lambda u, i=i, scale=scale: xf(
                    join + offsets[i], 1 + ease(u) * (scale - 1)
                ),
                duration=0.8,
            )
        for item in updates:
            item.fade_out(duration=0.8)
    s.wait(0.7)
    # Match the next chapter's exact opening layout. This is a visual handoff.
    handoff = []
    centers = (-5.7, -1.9, 1.9, 5.7)

    def p(ch, r, c):
        return np.array(
            [
                centers[ch] + (c - 0.5) + (r - 0.5) * 0.32,
                0.25 - (r - 0.5) + (c - 0.5) * 0.12,
            ]
        )

    for ch in range(4):
        corners = [
            p(ch, r, c) for r, c in ((-0.5, -0.5), (-0.5, 1.5), (1.5, 1.5), (1.5, -0.5))
        ]
        for k in range(4):
            item = s.add(
                Line(
                    tuple(corners[k]),
                    tuple(corners[(k + 1) % 4]),
                    style=Style.outline(palette[ch], 0.02),
                    z_index=2,
                )
            )
            item.opacity(to=0, duration=0)
            handoff.append((item, 0.7 if ch < 2 else 0.14))
    with s.parallel():
        for item in objects:
            if item not in identities:
                item.fade_out(duration=0.7)
        for item, alpha in handoff:
            item.opacity(to=alpha, duration=2.6)
        for i, item in enumerate(identities):
            ch = i // 4
            r, c = divmod(i % 4, 2)
            target = p(ch, r, c)
            before = (0.08 + 0.09 * y.flat[i]) / (0.08 + 0.09 * x.flat[i])
            after = (0.10 + 0.12 * x.flat[i]) / (0.08 + 0.09 * x.flat[i])
            item.transform_function(
                lambda u, i=i, target=target, before=before, after=after: xf(
                    join + offsets[i] + ease(u) * (target - join - offsets[i]),
                    before + ease(u) * (after - before),
                ),
                duration=2.6,
                easing=Easing.LINEAR,
            )
    for item in objects:
        if item not in identities:
            item.remove()
    return identities + [item for item, _ in handoff]
