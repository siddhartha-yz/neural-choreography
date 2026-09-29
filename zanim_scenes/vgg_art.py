"""Two original VGG blocks: shared-size convolution and spatial max pooling."""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.models import ep_07_2 as model
from zanim_scenes.feature_suite import EDGES, project
from zanim_scenes.nin_art import geometry as nin_geometry
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.style import label

PREFIX_SECONDS = 12.0
ARRAYS = (model.INPUT_X, model.CONV_1, model.POOL_1, model.CONV_2, model.POOL_2)
EDGES_8 = [(r * 8 + c, r * 8 + c + 1) for r in range(8) for c in range(7)] + [
    (r * 8 + c, (r + 1) * 8 + c) for r in range(7) for c in range(8)
]


def stage(index):
    """Replicate pooled sites solely for continuous four-to-one trajectories."""
    data = ARRAYS[index]
    n, _, channels = data.shape
    r, c = np.indices((8, 8))
    r = r.ravel() // (8 // n)
    c = c.ravel() // (8 // n)
    out = []
    for ch in range(4):
        lane = ch % channels
        x = (c / (n - 1) - 0.5) * (n * 0.45)
        y = (0.5 - r / (n - 1)) * (n * 0.45)
        values = data[r, c, lane]
        center = (lane - (channels - 1) / 2) * 2.1
        out.append(
            np.stack(
                (
                    x * 0.9 + center,
                    y,
                    -0.4 * x
                    + (values - values.mean()) * 0.14
                    + (lane - (channels - 1) / 2) * 0.6,
                ),
                axis=-1,
            )
        )
    return np.array(out), np.array([float(ch < channels) for ch in range(4)])


def geometry(u):
    # Conv, pool, conv, pool: repeated blocks with the same visual rhythm.
    t = u * 4
    i = min(int(t), 3)
    v = ease(t - i)
    a, va = stage(i)
    b, vb = stage(i + 1)
    xyz = a + (b - a) * v
    points = np.array([project(p, 1 + u * 7) for p in xyz])
    return points, va + (vb - va) * v


def append(s, chapter, palette, incoming=()):
    chapter("VGG · 使用块的网络 · 07.2", "卷积保留尺寸，池化收束空间。")
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

    def line(color, width=0.014):
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

    @lru_cache(maxsize=32)
    def frame(u):
        return geometry(u)

    for ch in range(4):
        for a, b in EDGES_8:
            item = line(palette[ch])
            item.opacity(to=0.32, duration=0)
            if ch:
                item.opacity(to=0, duration=0)
                item.opacity(to=0.32, duration=1.5, at=0 if ch == 1 else 4.2)
            else:
                item.opacity(to=0.32 if incoming else 0, duration=0)
                item.opacity(to=0.32, duration=0.5)
            item.transform_function(
                lambda u, ch=ch, a=a, b=b: pose(frame(u)[0][ch, a], frame(u)[0][ch, b]),
                duration=8.4,
                easing=Easing.LINEAR,
            )
        for j in range(64):
            item = dot(palette[ch])
            item.opacity(to=1 if incoming and ch == 0 else 0, duration=0)
            item.opacity(
                to=1, duration=0.5 if ch == 0 else 1.5, at=0 if ch < 2 else 4.2
            )
            item.transform_function(
                lambda u, ch=ch, j=j: point(frame(u)[0][ch, j], frame(u)[1][ch]),
                duration=8.4,
                easing=Easing.LINEAR,
            )
    # Matching sweep marks repeat for both convolution blocks.
    for start in (0, 4.2):
        for k in range(5):
            item = dot(palette[2], 0.065)

            def sweep(u, start=start, k=k):
                t = u * 8.4
                q = float(np.clip((t - start) / 2.1, 0, 1))
                pp, _ = frame(u)
                j = min(63, int(q * 63))
                return point(
                    pp[0, j] + np.array([0, (k - 2) * 0.07]), math.sin(math.pi * q)
                )

            item.transform_function(sweep, duration=8.4, easing=Easing.LINEAR)
    for start, n, count in ((0, 8, 2), (4.2, 4, 4)):
        stride = 8 // n
        for ch in range(count):
            for edge in range(4):
                item = line(palette[ch], 0.028)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.75, duration=0.25, at=start + 0.15)
                item.fade_out(duration=0.3, at=start + 1.8)

                def window(u, start=start, n=n, stride=stride, ch=ch, edge=edge):
                    q = float(np.clip((u * 8.4 - start) / 2.1, 0, 1))
                    # Slide over adjacent valid 3×3 patches; interpolate the
                    # outline so the scanner does not jump between sites.
                    travel = q * (n - 3)
                    lo = min(int(travel), n - 3)
                    hi = min(lo + 1, n - 3)
                    blend = travel - lo
                    pp, _ = frame(u)
                    corners = []
                    for dr, dc in ((0, 0), (0, 2), (2, 2), (2, 0)):
                        a = (lo + dr) * stride * 8 + (lo + dc) * stride
                        b = (hi + dr) * stride * 8 + (hi + dc) * stride
                        corners.append(pp[ch, a] + blend * (pp[ch, b] - pp[ch, a]))
                    return pose(corners[edge], corners[(edge + 1) % 4])

                item.transform_function(window, duration=8.4, easing=Easing.LINEAR)
    notes = []
    for text, start, end in [
        ("3 × 3", 0, 2.1),
        ("2 × 2 · 池化", 2.1, 4.2),
        ("3 × 3", 4.2, 6.3),
        ("2 × 2 · 池化", 6.3, 8.4),
    ]:
        item = Track(label(s, text, 0, 3, 0.24, palette[2]))
        notes.append(item)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.3, at=start)
        item.fade_out(duration=0.3, at=end - 0.3)
    play()
    # Morph the final four maps into the two input sheets of the next example.
    # This is an artistic chapter transition, not a numerical connection.
    target = nin_geometry(0)[0]
    source = geometry(1)[0]
    handoff = []
    for item in objects:
        item.fade_out(duration=0.7)
    for ch in range(2):

        def origin(j, ch=ch):
            r, c = divmod(j, 9)
            return source[ch * 2 + (c >= 4), min(7, r) * 8 + min(7, c)]

        for a, b in EDGES:
            item = line(palette[ch], 0.013)
            handoff.append(item)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.4, duration=0.8)
            oa, ob = origin(a), origin(b)
            item.transform_function(
                lambda u, ch=ch, a=a, b=b, oa=oa, ob=ob: pose(
                    oa + ease(u) * (target[ch, a] - oa),
                    ob + ease(u) * (target[ch, b] - ob),
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
        for j in range(81):
            item = dot(palette[ch], 0.037)
            handoff.append(item)
            item.opacity(to=0, duration=0)
            item.fade_in(duration=0.8)
            o = origin(j)
            item.transform_function(
                lambda u, ch=ch, j=j, o=o: point(o + ease(u) * (target[ch, j] - o)),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    play()
    for item in objects + notes:
        if item not in handoff:
            item.remove()
    return handoff
