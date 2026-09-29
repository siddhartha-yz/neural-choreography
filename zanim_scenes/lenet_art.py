"""LeNet-style forward procession using the original untrained chapter arrays."""

import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.models import ep_06_6 as model
from zanim_scenes.feature_suite import project
from zanim_scenes.vgg_art import geometry as vgg_geometry, EDGES_8
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.style import label

PREFIX_SECONDS = 12.0
ARRAYS = (model.INPUT_X, model.CONV_1, model.POOL_1, model.CONV_2, model.POOL_2)


def geometry(u):
    result = []
    for i, data in enumerate(ARRAYS):
        n = data.shape[0]
        r, c = np.indices(data.shape)
        x = (c.ravel() - (n - 1) / 2) * 0.27
        y = ((n - 1) / 2 - r.ravel()) * 0.4
        xyz = np.stack(
            (x + (-6, -3.5, -0.8, 1.6, 3.65)[i], y, -x * 0.7 + data.ravel() * 0.14),
            axis=-1,
        )
        result.append(project(xyz, 2 + u * 5))
    logits = model.OUTPUT_O.ravel()
    result.append(
        project(
            np.stack((np.full(3, 6), np.array([1.3, 0, -1.3]), logits * 0.2), axis=-1),
            2 + u * 5,
        )
    )
    return result


def connections():
    """Exact receptive-field edges, followed by the original 1→3 dense map."""
    edges = []
    for stage, k, stride, weights in [
        (1, 3, 1, model.KERNEL_1),
        (2, 2, 2, np.full((2, 2), 0.25)),
        (3, 2, 1, model.KERNEL_2),
        (4, 2, 2, np.full((2, 2), 0.25)),
    ]:
        n = ARRAYS[stage].shape[0]
        previous = ARRAYS[stage - 1].shape[0]
        for r in range(n):
            for c in range(n):
                for dr in range(k):
                    for dc in range(k):
                        edges.append(
                            (
                                stage - 1,
                                (r * stride + dr) * previous + c * stride + dc,
                                stage,
                                r * n + c,
                                float(weights[dr, dc]),
                            )
                        )
    edges.extend((4, 0, 5, i, float(model.WEIGHTS[i, 0])) for i in range(3))
    return edges


def append(s, chapter, palette, incoming=()):
    chapter("LeNet · 卷积神经网络 · 06.6", "局部提取，逐级汇聚，连接输出。")
    for item in incoming:
        item.remove()
    objects = []
    queue = []

    class Track:
        def __init__(self, item):
            self.item = item

        def __getattr__(self, name):
            def schedule(*args, **kwargs):
                queue.append(lambda: getattr(self.item, name)(*args, **kwargs))

            return schedule

        def remove(self):
            self.item.remove()

    def play():
        with s.parallel():
            for action in queue:
                action()
        queue.clear()

    def line(color, width=0.013):
        item = Track(
            s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(color, width),
                    transform=Transform2D.translation(-30, 0),
                    z_index=2,
                )
            )
        )
        objects.append(item)
        return item

    def dot(color, r=0.045):
        item = Track(
            s.add(
                Circle(
                    r,
                    style=Style.solid(color),
                    transform=Transform2D.translation(-30, 0),
                    z_index=5,
                )
            )
        )
        objects.append(item)
        return item

    def point(p, scale=1):
        return Transform2D.translation(*map(float, p)) @ Transform2D.scaling(
            max(0.001, scale)
        )

    from functools import lru_cache

    @lru_cache(maxsize=32)
    def frame(u):
        return geometry(u)

    arrivals = (0, 0.6, 2.1, 3.6, 5.1, 6.5)
    for stage, data in enumerate(ARRAYS):
        n = data.shape[0]
        color = palette[stage % 4]
        edges = [(r * n + c, r * n + c + 1) for r in range(n) for c in range(n - 1)] + [
            (r * n + c, (r + 1) * n + c) for r in range(n - 1) for c in range(n)
        ]
        for a, b in edges:
            item = line(color)
            item.opacity(to=0.32 if incoming and stage == 0 else 0, duration=0)
            item.opacity(to=0.32, duration=0.6, at=arrivals[stage])
            item.transform_function(
                lambda u, stage=stage, a=a, b=b: pose(
                    frame(u)[stage][a], frame(u)[stage][b]
                ),
                duration=8.4,
                easing=Easing.LINEAR,
            )
        for j, value in enumerate(data.ravel()):
            item = dot(color, 0.033 + 0.025 * min(2, abs(float(value))))
            item.opacity(to=1 if incoming and stage == 0 else 0, duration=0)
            item.opacity(
                to=1, duration=0.6, at=arrivals[stage] + j / max(1, n * n) * 0.5
            )
            item.transform_function(
                lambda u, stage=stage, j=j: point(frame(u)[stage][j]),
                duration=8.4,
                easing=Easing.LINEAR,
            )
    for j, value in enumerate(model.OUTPUT_O.ravel()):
        item = dot(palette[j], 0.09 + 0.055 * min(2, abs(float(value))))
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.6, at=6.8 + j * 0.12)
        item.transform_function(
            lambda u, j=j: point(frame(u)[5][j]), duration=8.4, easing=Easing.LINEAR
        )
    for src, a, dst, b, weight in connections():
        color = palette[dst % 4] if weight >= 0 else palette[2]
        item = line(color, 0.007)
        item.opacity(to=0, duration=0)
        item.opacity(to=0.10, duration=0.6, at=arrivals[dst])
        item.transform_function(
            lambda u, src=src, a=a, dst=dst, b=b: pose(
                frame(u)[src][a], frame(u)[dst][b]
            ),
            duration=8.4,
            easing=Easing.LINEAR,
        )
        item = dot(color, 0.024 + 0.014 * min(1, abs(weight)))

        def transfer(u, src=src, a=a, dst=dst, b=b):
            q = float(
                np.clip(
                    (u * 8.4 - arrivals[dst] - 0.3 * b / max(1, len(frame(u)[dst])))
                    / 1.1,
                    0,
                    1,
                )
            )
            return point(
                frame(u)[src][a] + ease(q) * (frame(u)[dst][b] - frame(u)[src][a]),
                math.sin(math.pi * q),
            )

        item.transform_function(transfer, duration=8.4, easing=Easing.LINEAR)
    notes = []
    for text, x in [
        ("卷积", -3.5),
        ("平均汇聚", -0.8),
        ("卷积", 1.6),
        ("平均汇聚", 3.65),
        ("全连接", 6),
    ]:
        item = Track(label(s, text, x, -2.4, 0.18, palette[0]))
        notes.append(item)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.7, at=0.3)
    play()
    target = vgg_geometry(0)[0][0]
    source = geometry(1)[5]
    handoff = []
    for item in objects + notes:
        item.fade_out(duration=0.8)
    for a, b in EDGES_8:
        item = line(palette[0], 0.014)
        handoff.append(item)
        item.opacity(to=0, duration=0)
        item.opacity(to=0.32, duration=1)
        oa, ob = source[a % 3], source[b % 3]
        item.transform_function(
            lambda u, a=a, b=b, oa=oa, ob=ob: pose(
                oa + ease(u) * (target[a] - oa), ob + ease(u) * (target[b] - ob)
            ),
            duration=2.8,
            easing=Easing.LINEAR,
        )
    for j in range(64):
        item = dot(palette[0], 0.045)
        handoff.append(item)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=1)
        o = source[j % 3]
        item.transform_function(
            lambda u, j=j, o=o: point(o + ease(u) * (target[j] - o)),
            duration=2.8,
            easing=Easing.LINEAR,
        )
    play()
    for item in objects + notes:
        if item not in handoff:
            item.remove()
    return handoff
