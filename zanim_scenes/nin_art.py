"""NiN: pointwise channel mixing, spatial averaging, then an artistic handoff.

The 9×9 lattice interpolates the original 2×2 maps; it is not extra neurons.
Camera, travelling wave and inter-chapter morph are compositional devices.
"""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.models import ep_07_3 as model
from zanim_scenes.feature_suite import G, UV, EDGES, sampled, surface, project, state
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.style import label

PREFIX_SECONDS = 12.0
MOVEMENT = 8.4


def geometry(u):
    """Projected input/output sheets and their exact per-channel mean anchors."""
    fan = ease((u - 0.34) / 0.22)
    gather = ease((u - 0.65) / 0.30)
    inputs = np.array(
        [
            project(
                surface(
                    ch,
                    np.array([-4.3, (ch - 0.5) * 0.5, (ch - 0.5) * 1.3]),
                    0.4,
                    sampled(model.INPUT_X[ch]),
                    1.65,
                ),
                u * 8.4,
            )
            for ch in range(2)
        ]
    )
    outputs = []
    anchors = []
    for ch in range(3):
        old = np.array([3.5 + (ch - 1) * 0.35, (ch - 1) * 0.45, (ch - 1) * 1.3])
        expanded = np.array([(ch - 1) * 4.5, 0, 0])
        center = old + fan * (expanded - old)
        pts = surface(ch, center, 0.4, sampled(model.FEATURES[ch]), 1.65)
        # The common value baseline keeps output heights and means comparable.
        pts[:, 2] += model.LOGITS[ch] * 0.22
        anchor = center + np.array([0, 0, model.LOGITS[ch] * 0.22])
        radius = np.linalg.norm(UV, axis=1)
        swirl = 0.6 * np.sin(math.pi * gather) * radius
        angle = gather * math.pi * 0.7
        delta = pts - anchor
        x, y = delta[:, 0].copy(), delta[:, 1].copy()
        delta[:, 0] = x * math.cos(angle) - y * math.sin(angle)
        delta[:, 1] = x * math.sin(angle) + y * math.cos(angle)
        delta[:, 2] += swirl
        outputs.append(project(anchor + (1 - gather) * delta, u * 8.4))
        anchors.append(project(anchor[None, :], u * 8.4)[0])
    return inputs, np.array(outputs), np.array(anchors)


def append(s, chapter, palette):
    chapter("网络中的网络 · 07.3", "逐点混合通道，再汇聚整张特征。")
    objects = []
    animations = []

    class Track:
        # Build all objects at a lifetime boundary, then schedule their tracks.
        def __init__(self, item):
            self.item = item

        def __getattr__(self, name):
            def schedule(*args, **kwargs):
                animations.append(lambda: getattr(self.item, name)(*args, **kwargs))

            return schedule

        def remove(self):
            self.item.remove()

    def play():
        with s.parallel():
            for animation in animations:
                animation()
        animations.clear()

    def line(color, width=0.013, alpha=0.3):
        item = s.add(
            Line(
                (0, 0),
                (1, 0),
                style=Style.outline(color, width),
                transform=Transform2D.translation(-30, 0),
                z_index=2,
            )
        )
        item = Track(item)
        item.opacity(to=alpha, duration=0)
        objects.append(item)
        return item

    def dot(color, radius=0.037):
        item = s.add(
            Circle(
                radius,
                style=Style.solid(color),
                transform=Transform2D.translation(-30, 0),
                z_index=5,
            )
        )
        item = Track(item)
        objects.append(item)
        return item

    def at_point(p, scale=1):
        return Transform2D.translation(*map(float, p)) @ Transform2D.scaling(
            max(0.001, scale)
        )

    @lru_cache(maxsize=32)
    def frame(u):
        return geometry(u)

    notes = [
        label(s, "1 × 1", 0, 2.5, 0.25, palette[2]),
        label(s, "全局平均", 0, 2.5, 0.25, palette[2]),
    ]
    notes = [Track(item) for item in notes]
    notes[1].opacity(to=0, duration=0)
    notes[0].fade_out(duration=0.6, at=2.8)
    notes[1].fade_in(duration=0.6, at=4.6)
    notes[1].fade_out(duration=0.6, at=7.8)
    for ch in range(2):
        for a, b in EDGES:
            item = line(palette[ch], alpha=0.4)
            item.transform_function(
                lambda u, ch=ch, a=a, b=b: pose(frame(u)[0][ch, a], frame(u)[0][ch, b]),
                duration=MOVEMENT,
                easing=Easing.LINEAR,
            )
            item.opacity(to=0, duration=0)
            item.opacity(to=0.4, duration=0.6)
            item.fade_out(duration=0.8, at=2.8)
        for j in range(G * G):
            item = dot(palette[ch])
            item.transform_function(
                lambda u, ch=ch, j=j: at_point(frame(u)[0][ch, j]),
                duration=MOVEMENT,
                easing=Easing.LINEAR,
            )
            item.opacity(to=0, duration=0)
            item.fade_in(duration=0.6)
            item.fade_out(duration=0.8, at=2.8)
    for ch in range(3):
        for a, b in EDGES:
            item = line(palette[ch], alpha=0.3)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.3, duration=1.3, at=0.8)
            item.fade_out(duration=1.3, at=5.7)
            item.transform_function(
                lambda u, ch=ch, a=a, b=b: pose(frame(u)[1][ch, a], frame(u)[1][ch, b]),
                duration=MOVEMENT,
                easing=Easing.LINEAR,
            )
        for j in range(G * G):
            item = dot(palette[ch])
            item.opacity(to=0, duration=0)
            item.fade_in(duration=0.5, at=0.65 + (UV[j, 0] + 1) * 0.65)
            item.fade_out(duration=0.4, at=7.6)
            item.transform_function(
                lambda u, ch=ch, j=j: at_point(frame(u)[1][ch, j]),
                duration=MOVEMENT,
                easing=Easing.LINEAR,
            )
        mean = dot(palette[ch], 0.09 + model.LOGITS[ch] * 0.018)
        mean.opacity(to=0, duration=0)
        mean.fade_in(duration=0.7, at=7.0)
        mean.transform_function(
            lambda u, ch=ch: at_point(frame(u)[2][ch]),
            duration=MOVEMENT,
            easing=Easing.LINEAR,
        )
    # Every connection joins the SAME spatial coordinate across channels.
    # Zero weights are omitted; a travelling wave activates many sites.
    for ch in range(3):
        for source in range(2):
            if model.WEIGHT_1X1[ch, source] == 0:
                continue
            for j in (0, 4, 8, 36, 40, 44, 72, 76, 80):
                item = line(palette[ch], 0.009, 0.18)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.18, duration=0.6, at=0.6)
                item.fade_out(duration=0.6, at=2.8)
                item.transform_function(
                    lambda u, ch=ch, source=source, j=j: pose(
                        frame(u)[0][source, j], frame(u)[1][ch, j]
                    ),
                    duration=MOVEMENT,
                    easing=Easing.LINEAR,
                )
                item = dot(palette[ch], 0.05)

                def transfer(u, ch=ch, source=source, j=j):
                    q = float(
                        np.clip(
                            (u * MOVEMENT - 0.55 - (UV[j, 0] + 1) * 0.55) / 1.15,
                            0,
                            1,
                        )
                    )
                    a = frame(u)[0][source, j]
                    b = frame(u)[1][ch, j]
                    return at_point(a + ease(q) * (b - a), math.sin(math.pi * q))

                item.transform_function(
                    transfer, duration=MOVEMENT, easing=Easing.LINEAR
                )
    play()
    # Match every mesh and dot of the following chapter at the boundary.
    xyz, _ = state("inception", 0)
    target = np.array([project(p, 0) for p in xyz])
    anchors = geometry(1)[2]
    handoff = []
    for item in objects:
        item.fade_out(duration=0.7)
    for ch in range(4):
        origin = anchors[ch % 3]
        for a, b in EDGES:
            item = line(palette[ch])
            handoff.append(item)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.3, duration=1.0)
            item.transform_function(
                lambda u, ch=ch, a=a, b=b, origin=origin: pose(
                    origin + ease(u) * (target[ch, a] - origin),
                    origin + ease(u) * (target[ch, b] - origin),
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
        for j in range(G * G):
            item = dot(palette[ch])
            handoff.append(item)
            item.opacity(to=0, duration=0)
            item.fade_in(duration=1.0)
            item.transform_function(
                lambda u, ch=ch, j=j, origin=origin: at_point(
                    origin + ease(u) * (target[ch, j] - origin)
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    play()
    for item in objects + notes:
        if item not in handoff:
            item.remove()
    return handoff
