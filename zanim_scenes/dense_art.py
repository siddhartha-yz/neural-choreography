"""Preserved channel maps, dense reuse, then a geometric handoff to recurrence."""

import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import pose, ease
from zanim_scenes.models import ep_07_7 as model

PREFIX_SECONDS = 12.0


def append(s, chapter, rings, palette):
    chapter("稠密连接网络 · 07.7", "旧特征保留，新特征逐层拼接。")
    objects = []
    cells = []
    centers = (-5.7, -1.9, 1.9, 5.7)
    values = model.STACK_2

    # Oblique 2×2 feature planes. Each color keeps its channel identity.
    def point(channel, r, c):
        return np.array(
            [
                centers[channel] + (c - 0.5) * 1.0 + (r - 0.5) * 0.32,
                0.25 - (r - 0.5) * 1.0 + (c - 0.5) * 0.12,
            ]
        )

    def line(a, b, color, alpha=0.45, width=0.02):
        obj = s.add(
            Line(tuple(a), tuple(b), style=Style.outline(color, width), z_index=2)
        )
        obj.opacity(to=alpha, duration=0)
        objects.append(obj)
        return obj

    for channel in range(4):
        color = palette[channel]
        for r in range(2):
            for c in range(2):
                xy = point(channel, r, c)
                circle = s.add(
                    Circle(
                        0.10 + 0.12 * float(values[channel, r, c]),
                        style=Style.solid(color),
                        transform=Transform2D.translation(*map(float, xy)),
                        z_index=6,
                    )
                )
                circle.opacity(to=1 if channel < 2 else 0, duration=0)
                objects.append(circle)
                cells.append((channel, r, c, circle))
        corners = [
            point(channel, r, c)
            for r, c in ((-0.5, -0.5), (-0.5, 1.5), (1.5, 1.5), (1.5, -0.5))
        ]
        for i in range(4):
            line(corners[i], corners[(i + 1) % 4], color, 0.7 if channel < 2 else 0.14)
    s.wait(1)
    for target in (2, 3):
        travelers = []
        for source in range(target):
            for r in range(2):
                for c in range(2):
                    a, b = point(source, r, c), point(target, r, c)

                    def route(u, a=a, b=b):
                        return (
                            (1 - u) * a
                            + u * b
                            + np.array([0, 2.1 * math.sin(math.pi * u)])
                        )

                    previous = route(0)
                    for k in range(1, 25):
                        nxt = route(k / 24)
                        line(previous, nxt, palette[source], 0.17, 0.014)
                        previous = nxt
                    for k in range(3):
                        obj = s.add(
                            Circle(
                                0.052,
                                style=Style.solid(palette[source]),
                                transform=Transform2D.translation(-30, 0),
                                z_index=7,
                            )
                        )
                        objects.append(obj)
                        travelers.append((obj, route, k))
        with s.parallel():
            for obj, route, k in travelers:

                def moving(u, route=route, k=k):
                    q = float(np.clip((u - k * 0.06) / 0.82, 0, 1))
                    return Transform2D.translation(
                        *map(float, route(ease(q)))
                    ) @ Transform2D.scaling(max(0.001, math.sin(math.pi * q)))

                obj.transform_function(moving, duration=2.4, easing=Easing.LINEAR)
        with s.parallel():
            for ch, r, c, obj in cells:
                if ch == target:
                    obj.fade_in(duration=0.5)
    s.wait(1.2)
    # Channel-colored contours widen into the exact opening geometry of 08.4.
    handoff = []
    angle = np.linspace(0, 2 * np.pi, rings.shape[1])
    for i in range(len(rings)):
        ch = i % 4
        start = np.stack(
            (
                centers[ch] + (0.65 + i * 0.006) * np.cos(angle),
                0.25 + (0.9 + i * 0.009) * np.sin(angle),
            ),
            axis=-1,
        )
        for j in range(len(angle) - 1):
            obj = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(palette[ch], 0.013),
                    transform=pose(start[j], start[j + 1]),
                    z_index=1,
                )
            )
            obj.opacity(to=0, duration=0)
            handoff.append((obj, start[j], start[j + 1], rings[i, j], rings[i, j + 1]))
    with s.parallel():
        for obj in objects:
            obj.fade_out(duration=0.8)
        for obj, a, b, c, d in handoff:
            obj.opacity(to=0.46, duration=0.8)
            obj.transform_function(
                lambda u, a=a, b=b, c=c, d=d: pose(
                    a + ease(u) * (c - a), b + ease(u) * (d - b)
                ),
                duration=3.2,
                easing=Easing.LINEAR,
            )
    for obj in objects:
        obj.remove()
    return [row[0] for row in handoff]
