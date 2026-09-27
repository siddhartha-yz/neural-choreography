"""Parallel receptive fields, channel concatenation, then normalization."""

import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.style import label
from zanim_scenes.models import ep_07_4 as model
from zanim_scenes.batch_norm_art import stages

PREFIX_SECONDS = 12.0


def append(s, chapter, palette):
    chapter("多分支网络 · 07.4", "同一输入，并行提取，按通道拼接。")
    objects = []
    dots = []
    frames = []
    centers = [np.array([-1.0, y]) for y in (2.8, 0.95, -0.95, -2.8)]

    def xf(xy, scale=1):
        return Transform2D.translation(*map(float, xy)) @ Transform2D.scaling(
            max(0.001, float(scale))
        )

    def offset(k, step=0.32):
        r, c = divmod(k, 3)
        return np.array([(c - 1) * step, (1 - r) * step])

    def dot(xy, r, color):
        item = s.add(Circle(r, style=Style.solid(color), transform=xf(xy), z_index=6))
        objects.append(item)
        return item

    def line(a, b, color, alpha=0.25):
        item = s.add(
            Line(tuple(a), tuple(b), style=Style.outline(color, 0.016), z_index=2)
        )
        item.opacity(to=alpha, duration=0)
        objects.append(item)
        return item

    start = np.array([-6.2, 0])
    for k, v in enumerate(model.INPUT_X.flat):
        dot(start + offset(k, 0.45), 0.05 + 0.10 * math.sqrt(v / 8), palette[0])
    labels = ("1 × 1", "3 × 3", "5 × 5", "最大池化")
    for ch, center in enumerate(centers):
        line(start, center, palette[ch])
        line(center, np.array([5.0, 0]), palette[ch])
        objects.append(label(s, labels[ch], 1.0, center[1], 0.22, palette[ch]))
        group = []
        for k, v in enumerate(model.INPUT_X.flat):
            item = dot(
                start + offset(k, 0.45), 0.05 + 0.10 * math.sqrt(v / 8), palette[ch]
            )
            group.append(item)
        dots.append(group)
    s.wait(0.65)
    with s.parallel():
        for ch, group in enumerate(dots):
            for k, item in enumerate(group):
                a = start + offset(k, 0.45)
                b = centers[ch] + offset(k)
                item.transform_function(
                    lambda u, a=a, b=b: xf(a + ease(u) * (b - a)),
                    duration=2.1,
                    easing=Easing.LINEAR,
                )
    for ch, size in enumerate((1, 3, 5, 3)):
        half = size * 0.16
        corners = [
            np.array([a, b])
            for a, b in ((-half, -half), (half, -half), (half, half), (-half, half))
        ]
        for j in range(4):
            item = s.add(
                Line((0, 0), (1, 0), style=Style.outline(palette[ch], 0.025), z_index=7)
            )
            objects.append(item)
            frames.append((ch, corners[j], corners[(j + 1) % 4], item))
    values = np.stack((model.Y1, model.Y3, model.Y5, model.YP)).reshape(4, 9)
    with s.parallel():
        for ch, a, b, item in frames:

            def frame(u, ch=ch, a=a, b=b):
                t = min(8.0, u * 9)
                j = min(int(t), 7)
                q = t - j
                center = centers[ch] + offset(j) + (offset(j + 1) - offset(j)) * ease(q)
                return pose(center + a, center + b)

            item.set_transform(to=frame(0))
            item.transform_function(frame, duration=3.6, easing=Easing.LINEAR)
        for ch, group in enumerate(dots):
            for k, item in enumerate(group):
                initial = 0.05 + 0.10 * math.sqrt(model.INPUT_X.flat[k] / 8)
                final = 0.04 + 0.14 * math.sqrt(values[ch, k] / np.max(values))
                item.transform_function(
                    lambda u, ch=ch, k=k, initial=initial, final=final: xf(
                        centers[ch] + offset(k),
                        1 + ease(u * 9 - k) * (final / initial - 1),
                    ),
                    duration=3.6,
                    easing=Easing.LINEAR,
                )
    targets = {}
    with s.parallel():
        for _, _, _, item in frames:
            item.fade_out(duration=0.4)
        for ch, group in enumerate(dots):
            for k, item in enumerate(group):
                a = centers[ch] + offset(k)
                b = np.array([4.7 + (ch - 1.5) * 0.25, (ch - 1.5) * 0.33]) + offset(
                    k, 0.55
                )
                targets[ch, k] = b
                scale = (0.04 + 0.14 * math.sqrt(values[ch, k] / np.max(values))) / (
                    0.05 + 0.10 * math.sqrt(model.INPUT_X.flat[k] / 8)
                )
                item.transform_function(
                    lambda u, a=a, b=b, scale=scale: xf(a + ease(u) * (b - a), scale),
                    duration=2.2,
                    easing=Easing.LINEAR,
                )
    s.wait(0.5)
    initial_batch = stages()[0]
    handoff = []
    with s.parallel():
        for obj in objects:
            if all(obj not in group for group in dots):
                obj.fade_out(duration=0.7)
        for ch, group in enumerate(dots):
            for k, item in enumerate(group):
                if k >= 5:
                    item.fade_out(duration=0.7)
                    continue
                handoff.append(item)
                a = targets[ch, k]
                b = np.array([initial_batch[ch, k] * 1.2, (1.5 - ch) * 1.3])
                base = 0.05 + 0.10 * math.sqrt(model.INPUT_X.flat[k] / 8)
                initial = (
                    0.04 + 0.14 * math.sqrt(values[ch, k] / np.max(values))
                ) / base
                final = 0.13 / base
                item.transform_function(
                    lambda u, a=a, b=b, initial=initial, final=final: xf(
                        a + ease(u) * (b - a), initial + ease(u) * (final - initial)
                    ),
                    duration=2.15,
                    easing=Easing.LINEAR,
                )
    for obj in objects:
        if obj not in handoff:
            obj.remove()
    return handoff
