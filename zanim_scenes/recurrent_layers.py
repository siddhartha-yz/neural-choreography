"""Stacked and bidirectional recurrent motifs; the shared topology stays visible."""

import math
import numpy as np
from zanim import Circle, Line, Style, Color, Transform2D, Easing
from zanim_scenes.style import label
from zanim_scenes.gated_art import N, pose, ease, BLUE, GOLD, PINK
from zanim_scenes.continuous import smooth
from zanim_scenes.models import ep_08_4 as model


def inputs(phase=0.0):
    lane = np.arange(N)[:, None] * 2 * np.pi / N
    t = np.arange(13)[None, :]
    return np.stack(
        (1.2 * np.sin(lane + 0.5 * t + phase), np.cos(1.5 * lane - 0.4 * t + phase)),
        axis=-1,
    )


def stacked(phase=0.0, tokens=None):
    x = inputs(phase) if tokens is None else np.asarray(tokens)
    h = np.zeros((3, N, 13, 2))
    for t in range(13):
        for layer in range(3):
            current = x[:, t] if layer == 0 else h[layer - 1, :, t]
            previous = np.zeros((N, 2)) if t == 0 else h[layer, :, t - 1]
            h[layer, :, t] = np.tanh(
                current @ model.WEIGHT_XH.T
                + previous @ model.WEIGHT_HH.T
                + model.BIAS.ravel()
            )
    return x, h


def bidirectional(phase=0.0, tokens=None):
    x = inputs(phase) if tokens is None else np.asarray(tokens)
    forward = np.zeros_like(x)
    backward = np.zeros_like(x)
    previous = np.zeros((N, 2))
    for t in range(13):
        previous = np.tanh(
            x[:, t] @ model.WEIGHT_XH.T
            + previous @ model.WEIGHT_HH.T
            + model.BIAS.ravel()
        )
        forward[:, t] = previous
    previous = np.zeros((N, 2))
    for t in range(12, -1, -1):
        previous = np.tanh(
            x[:, t] @ model.WEIGHT_XH.T
            + previous @ model.WEIGHT_HH.T
            + model.BIAS.ravel()
        )
        backward[:, t] = previous
    return x, forward, backward, np.concatenate((forward, backward), axis=-1)


def geometry(kind, phase):
    lane = (np.arange(N) - (N - 1) / 2) * 0.012
    x = np.broadcast_to(np.linspace(-7.8, 7.8, 13), (N, 13))
    if kind == "deep":
        _, states = stacked(phase)
        heights = (-2.4, 0.0, 2.4)
    else:
        _, f, b, joined = bidirectional(phase)
        states = (f, b, joined[:, :, :2])
        heights = (1.8, -0.25, -2.7)
    return [
        smooth(
            np.stack(
                (x, base + 0.56 * h[:, :, 0] + 0.14 * h[:, :, 1] + lane[:, None]),
                axis=-1,
            )
        )
        for base, h in zip(heights, states)
    ]


def append(s, paths, edges, particles, clock, chapter):
    """Choreograph causal wavefronts through explicit recurrent cells."""
    colors = (BLUE, GOLD, PINK)
    xs = np.linspace(-7.2, 7.2, 13)
    white = Color(225, 230, 232)
    decorations = []

    def circle(x, y, radius, color, filled=False, alpha=1):
        item = s.add(
            Circle(
                radius,
                style=Style.solid(color) if filled else Style.outline(color, 0.025),
                transform=Transform2D.translation(float(x), float(y)),
                z_index=8,
            )
        )
        item.opacity(to=alpha, duration=0)
        decorations.append(item)
        return item

    def segment(a, b, color, width=0.022, alpha=0.4):
        item = s.add(
            Line(tuple(a), tuple(b), style=Style.outline(color, width), z_index=4)
        )
        item.opacity(to=alpha, duration=0)
        decorations.append(item)
        return item

    def arrow(a, b, color, alpha=0.4):
        a, b = np.array(a), np.array(b)
        segment(a, b, color, alpha=alpha)
        v = (b - a) / np.linalg.norm(b - a)
        n = np.array([-v[1], v[0]])
        segment(b, b - 0.13 * v + 0.065 * n, color, alpha=alpha)
        segment(b, b - 0.13 * v - 0.065 * n, color, alpha=alpha)

    def clear():
        with s.parallel():
            for item in decorations:
                item.fade_out(duration=0.35)
        for item in decorations:
            item.remove()
        decorations.clear()

    def layout(kind):
        base = (-2.4, 0, 2.4) if kind == "deep" else (2.3, -2.3, 0)
        _, hh = stacked()
        if kind == "bi":
            _, f, b, _ = bidirectional()
            hh = (f, b, np.zeros_like(f))
        return [
            smooth(
                np.stack(
                    (
                        np.broadcast_to(xs, (N, 13)),
                        y + 0.12 * h[:, :, 0] + (np.arange(N)[:, None] - 13.5) * 0.009,
                    ),
                    axis=-1,
                )
            )
            for y, h in zip(base, hh)
        ]

    def morph(kind):
        nonlocal paths
        target = layout(kind)
        with s.parallel():
            for b, rows in enumerate(edges):
                for i, row in enumerate(rows):
                    for j, item in enumerate(row):
                        a0, a1 = paths[b][i, j], paths[b][i, j + 1]
                        b0, b1 = target[b][i, j], target[b][i, j + 1]
                        item.transform_function(
                            lambda u, a0=a0, a1=a1, b0=b0, b1=b1: pose(
                                a0 + ease(u) * (b0 - a0), a1 + ease(u) * (b1 - a1)
                            ),
                            duration=2.6,
                            easing=Easing.LINEAR,
                        )
                        item.opacity(
                            to=0.07 if kind == "deep" or b < 2 else 0, duration=2.6
                        )
            for group in particles:
                for _, _, dot in group:
                    dot.fade_out(duration=0.35)
        paths = target

    def network(kind):
        ys = (-2.4, 0, 2.4) if kind == "deep" else (2.3, -2.3)
        nodes = []
        for b, y in enumerate(ys):
            color = colors[b]
            for t, x in enumerate(xs):
                ring = circle(x, y, 0.27, color, alpha=0.8)
                core = circle(x, y, 0.19, color, filled=True)
                nodes.append((b, t, x, y, ring, core))
                if t < 12:
                    a, bx = (
                        (x + 0.34, xs[t + 1] - 0.34)
                        if kind == "deep" or b == 0
                        else (xs[t + 1] - 0.34, x + 0.34)
                    )
                    arrow((a, y), (bx, y), color, alpha=0.7)
        if kind == "deep":
            for x in xs:
                for b in range(2):
                    arrow(
                        (x, ys[b] + 0.34), (x, ys[b + 1] - 0.34), colors[b], alpha=0.35
                    )
            for b, y in enumerate(ys):
                decorations.append(label(s, f"第 {b + 1} 层", -8.1, y, 0.22, colors[b]))
        else:
            for x in xs:
                circle(x, 0, 0.07, white, filled=True)
                arrow((x, 0.15), (x, 1.9), white, alpha=0.2)
                arrow((x, -0.15), (x, -1.9), white, alpha=0.2)
            decorations.append(label(s, "同一段输入", 0, 0.58, 0.21, white))
            decorations.append(label(s, "正向 →", -7.2, 3.12, 0.23, BLUE))
            decorations.append(label(s, "← 反向", 7.2, -3.12, 0.23, GOLD))
        return nodes

    def animate(kind, nodes, seconds):
        # A single coherent front per layer; upper layers lag the lower ones.
        pulse_items = []
        beams = []
        for b in range(3 if kind == "deep" else 2):
            y = (-2.4, 0, 2.4)[b] if kind == "deep" else (2.3, -2.3)[b]
            for k in range(7):
                dot = circle(-20, 0, 0.075, colors[b], filled=True)
                pulse_items.append((b, k, y, dot))
        if kind == "deep":
            for b in range(2):
                for t, x in enumerate(xs):
                    for k in range(3):
                        dot = circle(-20, 0, 0.055, colors[b], filled=True)
                        beams.append((b, t, x, k, dot))
        else:
            for b in range(2):
                for t, x in enumerate(xs):
                    for k in range(3):
                        dot = circle(-20, 0, 0.05, colors[b], filled=True)
                        beams.append((b, t, x, k, dot))
        outputs = []
        if kind == "bi":
            for t, x in enumerate(xs):
                for b in range(2):
                    for c in range(2):
                        dot = circle(-20, 0, 0.08, colors[b], filled=True)
                        outputs.append((b, t, c, x, dot))
        _, _, _, joined = bidirectional()

        def front(u, b):
            p = 16 * u - 1
            return p - b * 0.72 if kind == "deep" else (p if b == 0 else 12 - p)

        def arrived(u, b, t):
            f = front(u, b)
            return f - t if kind == "deep" or b == 0 else t - f

        def transform(x, y, size):
            return Transform2D.translation(float(x), float(y)) @ Transform2D.scaling(
                max(0.001, float(size))
            )

        def apply(item, fn):
            item.set_transform(to=fn(0))
            item.transform_function(fn, duration=seconds, easing=Easing.LINEAR)

        with s.parallel():
            for b, t, x, y, ring, core in nodes:
                apply(
                    ring,
                    lambda u, b=b, t=t, x=x, y=y: transform(
                        x, y, 1 + 0.35 * math.exp(-(((arrived(u, b, t)) / 0.28) ** 2))
                    ),
                )
                apply(
                    core,
                    lambda u, b=b, t=t, x=x, y=y: transform(
                        x,
                        y,
                        0.04
                        + 0.65 * ease(arrived(u, b, t) / 0.3)
                        + 0.3 * math.exp(-(((arrived(u, b, t)) / 0.23) ** 2)),
                    ),
                )
            for b, k, y, dot in pulse_items:

                def moving(u, b=b, k=k, y=y):
                    f = front(u, b) + (k - 3) * 0.055
                    x = -7.2 + 1.2 * np.clip(f, 0, 12)
                    return transform(
                        x, y + (k - 3) * 0.033, 1 if 0 <= f <= 12 else 0.001
                    )

                apply(dot, moving)
            for b, t, x, k, dot in beams:

                def climbing(u, b=b, t=t, x=x, k=k):
                    if kind == "deep":
                        q = (16 * u - 1 - t - b * 0.72 - 0.13 - k * 0.035) / 0.55
                        y = -2.4 + b * 2.4 + 2.4 * np.clip(q, 0, 1)
                    else:
                        # Input enters both chains when that directional front arrives.
                        q = (arrived(u, b, t) + 0.45 - k * 0.03) / 0.45
                        y = (2.3 if b == 0 else -2.3) * np.clip(q, 0, 1)
                    return transform(
                        x, y, math.sin(math.pi * q) if 0 < q < 1 else 0.001
                    )

                apply(dot, climbing)
            for b, t, c, x, dot in outputs:

                def collecting(u, b=b, t=t, c=c, x=x):
                    age = arrived(u, b, t)
                    q = ease((age - 0.35) / 1.1)
                    y = (2.3 if b == 0 else -2.3) * (1 - q)
                    size = (0.55 + 0.45 * abs(joined[8, t, b * 2 + c])) * ease(
                        age / 0.25
                    )
                    return transform(x + (b * 2 + c - 1.5) * 0.14 * q, y, size)

                apply(dot, collecting)
        # Let the complete pattern breathe before the next transformation.
        s.wait(0.6)

    chapter("深度循环神经网络 · 09.3", "同一时刻，逐层递推。")
    morph("deep")
    nodes = network("deep")
    animate("deep", nodes, 9.6)
    chapter("双向循环神经网络 · 09.4", "同一序列，从两端读入。")
    clear()
    morph("bi")
    nodes = network("bi")
    animate("bi", nodes, 10.4)
    from zanim_scenes.sequence_art import append as append_sequence

    append_sequence(s, decorations, edges, chapter)
