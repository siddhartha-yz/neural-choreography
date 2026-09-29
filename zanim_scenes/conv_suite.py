"""Four connected convolution studies, using the original chapter operators.

All inter-chapter morphs and depth arrangements are artistic, not data flow
between the independent numerical examples.
"""

import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing, Color
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.feature_suite import project
from zanim_scenes.models import ep_06_2 as conv, ep_06_3 as stride, ep_06_5 as pool
from zanim_scenes import multi_channel as channels
from zanim_scenes.style import label

KINDS = ("conv", "stride", "channels", "pool")
TITLES = (
    "卷积层 · 06.2",
    "填充与步幅 · 06.3",
    "多输入多输出通道 · 06.4",
    "汇聚层 · 06.5",
)
SUBTITLES = (
    "局部相乘，汇入一个输出。",
    "边缘补零，间隔采样。",
    "分别计算，跨通道相加。",
    "最大保留强响应，平均汇入全部。",
)
PREFIX_SECONDS = 48.0
DURATION = 8.4
GOLD = Color(244, 205, 110)
GRAY = Color(115, 134, 152)


def plane(data, center, spacing=0.65, height=0.08):
    data = np.asarray(data)
    r, c = np.indices(data.shape)
    x = (c.ravel() - (data.shape[1] - 1) / 2) * spacing
    y = ((data.shape[0] - 1) / 2 - r.ravel()) * spacing
    return np.stack(
        (
            center[0] + x * 0.92,
            center[1] + y,
            center[2] - x * 0.48 + data.ravel() * height,
        ),
        axis=-1,
    )


def edges(shape):
    h, w = shape
    return [(r * w + c, r * w + c + 1) for r in range(h) for c in range(w - 1)] + [
        (r * w + c, (r + 1) * w + c) for r in range(h - 1) for c in range(w)
    ]


def seeds(kind):
    if kind == "conv":
        return [(conv.INPUT_X, (-3.7, 0, 0), 1.0, 0.075, 0)]
    if kind == "stride":
        return [(stride.INPUT_X, (-3.7, 0, 0), 0.9, 0.075, 0)]
    if kind == "channels":
        return [
            (
                np.array(channels.X[ch]),
                (-4.5, (0.5 - ch) * 2.5, ch * 0.35),
                0.75,
                0.07,
                ch,
            )
            for ch in range(2)
        ]
    return [
        (pool.INPUT, (x, 0, 0), 0.85, 0.095, ch) for ch, x in enumerate((-3.3, 3.3))
    ]


def radii(data):
    return np.minimum(0.16, 0.05 + 0.025 * np.sqrt(np.abs(np.asarray(data).ravel())))


class Stage:
    """Create objects at scene boundaries, then submit parallel tracks."""

    def __init__(self, s, palette):
        self.s = s
        self.palette = palette
        self.objects = []
        self.queue = []

    class Track:
        def __init__(self, owner, item):
            self.owner = owner
            self.item = item

        def __getattr__(self, name):
            def action(*args, **kwargs):
                self.owner.queue.append(
                    lambda: getattr(self.item, name)(*args, **kwargs)
                )

            return action

        def remove(self):
            self.item.remove()

    def add(self, item):
        t = self.Track(self, self.s.add(item))
        self.objects.append(t)
        return t

    def line(self, color, width=0.016):
        return self.add(
            Line(
                (0, 0),
                (1, 0),
                style=Style.outline(color, width),
                transform=Transform2D.translation(-30, 0),
                z_index=2,
            )
        )

    def dot(self, color, r=0.05):
        return self.add(
            Circle(
                float(r),
                style=Style.solid(color),
                transform=Transform2D.translation(-30, 0),
                z_index=5,
            )
        )

    def note(self, text, x, y, start=0, end=8.4, color=None):
        t = self.Track(self, label(self.s, text, x, y, 0.23, color or self.palette[2]))
        self.objects.append(t)
        t.opacity(to=0, duration=0)
        t.fade_in(duration=0.4, at=start)
        t.fade_out(duration=0.4, at=end - 0.4)

    def play(self):
        with self.s.parallel():
            for f in self.queue:
                f()
        self.queue.clear()

    def map(self, data, xyz, color, start=0, incoming=False, height=None):
        def pp(u):
            return project(xyz, 0 if incoming is None else u * 3)

        for a, b in edges(np.asarray(data).shape):
            item = self.line(color)
            item.opacity(to=0.32 if incoming else 0, duration=0)
            item.opacity(to=0.32, duration=0.5, at=start)
            item.transform_function(
                lambda u, a=a, b=b: pose(pp(u)[a], pp(u)[b]),
                duration=DURATION,
                easing=Easing.LINEAR,
            )
        for j, r in enumerate(radii(data)):
            item = self.dot(color, r)
            item.opacity(to=1 if incoming else 0, duration=0)
            item.opacity(to=1, duration=0.5, at=start)
            item.transform_function(
                lambda u, j=j: point(pp(u)[j]), duration=DURATION, easing=Easing.LINEAR
            )
        return pp

    def flow(self, a, b, start, duration, color, r=0.045, arch=0.4):
        item = self.dot(color, r)

        def travel(u):
            q = float(np.clip((u * DURATION - start) / duration, 0, 1))
            aa = a(u)
            bb = b(u)
            xy = aa + ease(q) * (bb - aa) + np.array([0, arch * math.sin(math.pi * q)])
            return point(xy, math.sin(math.pi * q))

        item.transform_function(travel, duration=DURATION, easing=Easing.LINEAR)


def point(p, scale=1):
    return Transform2D.translation(*map(float, p)) @ Transform2D.scaling(
        max(0.001, float(scale))
    )


def convolution(b, incoming):
    d, c, sp, h, ch = seeds("conv")[0]
    xyz = plane(d, c, sp, h)
    source = b.map(d, xyz, b.palette[ch], incoming=incoming)
    out = plane(conv.OUTPUT, (3.7, 0, 0), 1.2, 0.012)

    def output(u):
        return project(out, u * 3)

    for j, r in enumerate(radii(conv.OUTPUT) / 2):
        item = b.dot(b.palette[2], r)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.5, at=1.1 + j * 1.8)
        item.transform_function(
            lambda u, j=j: point(output(u)[j]), duration=DURATION, easing=Easing.LINEAR
        )
    for a, z in edges((2, 2)):
        item = b.line(b.palette[2])
        item.opacity(to=0, duration=0)
        item.opacity(to=0.32, duration=1, at=6)
        item.transform_function(
            lambda u, a=a, z=z: pose(output(u)[a], output(u)[z]),
            duration=DURATION,
            easing=Easing.LINEAR,
        )

    def window(u, edge):
        t = float(np.clip((u * DURATION - 0.2) / 1.8, 0, 3.999))
        j = int(t)
        f = ease(min(1, (t - j) * 4))

        def corners(k):
            r, c = divmod(k, 2)
            return [
                source(u)[(r + dr) * 4 + c + dc]
                for dr, dc in ((0, 0), (0, 2), (2, 2), (2, 0))
            ]

        a = np.array(corners(max(0, j - 1)))
        z = np.array(corners(j))
        p = a + f * (z - a)
        return pose(p[edge], p[(edge + 1) % 4])

    for e in range(4):
        item = b.line(GOLD, 0.035)
        item.transform_function(
            lambda u, e=e: window(u, e), duration=DURATION, easing=Easing.LINEAR
        )
    for j in range(4):
        r, c = divmod(j, 2)
        for dr in range(3):
            for dc in range(3):
                contribution = conv.INPUT_X[r + dr, c + dc] * conv.KERNEL[dr, dc]
                if contribution == 0:
                    continue
                k = (r + dr) * 4 + c + dc
                b.flow(
                    lambda u, k=k: source(u)[k],
                    lambda u, j=j: output(u)[j],
                    0.45 + j * 1.8,
                    0.9,
                    GOLD,
                    0.035 + 0.007 * math.sqrt(contribution),
                    (dr - 1) * 0.25,
                )
    b.note("3 × 3", -3.7, 2.6, color=GOLD)
    b.note("Σ", 1.2, 0, color=GOLD)
    return lambda: output(1)


def padding(b, incoming):
    d, c, sp, h, ch = seeds("stride")[0]
    b.map(d, plane(d, c, sp, h), b.palette[0], incoming=incoming)
    padded = np.pad(d, 1)
    padxyz = plane(padded, c, sp, h)

    def pad(u):
        return project(padxyz, u * 3)

    for r in range(5):
        for col in range(5):
            if 0 < r < 4 and 0 < col < 4:
                continue
            j = r * 5 + col
            item = b.dot(GRAY, 0.055)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.7, duration=0.7, at=2)
            item.transform_function(
                lambda u, j=j: point(pad(u)[j]), duration=DURATION, easing=Easing.LINEAR
            )
    full = plane(stride.Y_P1_S1, (3.7, 0, 0), 0.85, 0.01)
    sparse = plane(stride.Y_P1_S2, (3.7, 0, 0), 1.4, 0.01)

    def coordinates(u):
        pp = project(full, u * 3)
        ss = project(sparse, u * 3)
        q = ease((u * DURATION - 5.7) / 1.4)
        for r in (0, 2):
            for c in (0, 2):
                j = r * 4 + c
                pp[j] += q * (ss[(r // 2) * 2 + c // 2] - pp[j])
        return pp

    for j in range(16):
        r, c = divmod(j, 4)
        inner = r in (1, 2) and c in (1, 2)
        kept = r % 2 == 0 and c % 2 == 0
        item = b.dot(b.palette[2], radii(stride.Y_P1_S1)[j] * 0.52)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.6, at=0.7 if inner else 2.6)
        if not kept:
            item.fade_out(duration=0.7, at=5.7)
        item.transform_function(
            lambda u, j=j: point(coordinates(u)[j]),
            duration=DURATION,
            easing=Easing.LINEAR,
        )
    for start, origins, step in [
        (0, [(0, 0), (0, 1), (1, 0), (1, 1)], 1),
        (2.1, [(r, c) for r in range(-1, 3) for c in range(-1, 3)], 1),
        (5.5, [(r, c) for r in (-1, 1) for c in (-1, 1)], 2),
    ]:
        length = 2.0 if start == 0 else 3.0 if start == 2.1 else 2.7
        for e in range(4):
            item = b.line(GOLD, 0.032)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.8, duration=0.2, at=start)
            item.fade_out(duration=0.2, at=start + length - 0.2)

            def window(u, e=e, start=start, origins=origins, length=length):
                t = float(np.clip((u * DURATION - start) / length, 0, 0.99999)) * len(
                    origins
                )
                j = int(t)
                q = ease(t - j)
                r, c = origins[j]
                pr, pc = origins[max(0, j - 1)]
                r = pr + q * (r - pr)
                c = pc + q * (c - pc)
                p = np.array(
                    [
                        [
                            -3.7 + (c + dc - 1) * 0.9 * 0.92,
                            (1 - r - dr) * 0.9,
                            -(c + dc - 1) * 0.9 * 0.48,
                        ]
                        for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0))
                    ]
                )
                p = project(p, u * 3)
                return pose(p[e], p[(e + 1) % 4])

            item.transform_function(window, duration=DURATION, easing=Easing.LINEAR)
        for i, (r, c) in enumerate(origins):
            dest = (r + 1) * 4 + c + 1
            b.flow(
                lambda u, r=r, c=c: pad(u)[(r + 1) * 5 + c + 1],
                lambda u, dest=dest: coordinates(u)[dest],
                start + i * length / len(origins),
                0.5,
                GOLD,
                0.035,
                0,
            )
    b.note("补零", -3.7, 2.7, start=2, end=5.5, color=GRAY)
    b.note("步幅 1", 0, -2.7, end=5.5)
    b.note("步幅 2", 0, -2.7, start=5.5)
    return lambda: coordinates(1)[[0, 2, 8, 10]]


def multichannel(b, incoming):
    inputs = [
        b.map(d, plane(d, c, sp, h), b.palette[ch], incoming=incoming)
        for d, c, sp, h, ch in seeds("channels")
    ]
    result = channels.outputs()
    outpoints = []
    for out, (parts, total) in enumerate(result):
        center = (4.7, (0.5 - out) * 2.8, 0)
        target = plane(total, center, 0.9, 0.006)

        def end(u, target=target):
            return project(target, u * 3)

        outpoints.append(end)
        for ch in range(2):
            part = np.array(parts[ch])
            partialxyz = plane(
                part, (-0.6, (0.5 - out) * 2.8, (ch - 0.5) * 1.6), 0.85, 0.007
            )
            partial = b.map(part, partialxyz, b.palette[ch], start=1.6 + out * 1.4)
            for j in range(4):
                r, c = divmod(j, 2)
                for dr in range(2):
                    for dc in range(2):
                        k = (r + dr) * 3 + c + dc
                        w = channels.K[ch][dr][dc] + out
                        if w == 0:
                            continue
                        b.flow(
                            lambda u, k=k, ch=ch: inputs[ch](u)[k],
                            lambda u, j=j, partial=partial: partial(u)[j],
                            0.5 + out * 1.4 + j * 0.2,
                            0.9,
                            b.palette[ch],
                            0.04,
                            (ch - 0.5) * 0.4,
                        )
                b.flow(
                    lambda u, j=j, partial=partial: partial(u)[j],
                    lambda u, j=j, end=end: end(u)[j],
                    4.7 + out * 0.6 + j * 0.12,
                    1.0,
                    b.palette[out + 2],
                    0.07,
                    (ch - 0.5) * 0.6,
                )
        b.map(np.array(total), target, b.palette[out + 2], start=5.6 + out * 0.6)
        b.note("+", 2.1, (0.5 - out) * 2.8, start=4.3, color=b.palette[out + 2])
    b.note("两组核 · 两个输出通道", 0, 3.6)
    return lambda: np.concatenate([p(1) for p in outpoints])


def pooling(b, incoming):
    ends = []
    for branch, (d, c, sp, h, ch) in enumerate(seeds("pool")):
        xyz = plane(d, c, sp, h)
        target = plane(pool.MAX_OUT if branch == 0 else pool.AVG_OUT, c, 1.3, h)

        def end(u, target=target):
            return project(target, u * 3)

        ends.append(end)

        def begin(u, xyz=xyz):
            return project(xyz, u * 3)

        b.map(d, xyz, b.palette[ch], incoming=incoming)
        # Persistent source fades before the moving contributions take over.
        for item in b.objects[-(16 + len(edges((4, 4)))) :]:
            item.fade_out(duration=0.5, at=2.7)
        for record in pool.WINDOWS:
            r, c0 = record["origin"]
            j = record["out_row"] * 2 + record["out_col"]
            indices = [
                r * 4 + c0,
                r * 4 + c0 + 1,
                (r + 1) * 4 + c0 + 1,
                (r + 1) * 4 + c0,
            ]
            for e in range(4):
                item = b.line(GOLD, 0.022)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.55, duration=0.5, at=1.3)
                item.fade_out(duration=0.5, at=3.3)

                def outline(u, e=e, indices=indices, begin=begin):
                    pp = begin(u)[indices]
                    center = pp.mean(axis=0)
                    pp = center + 1.18 * (pp - center)
                    return pose(pp[e], pp[(e + 1) % 4])

                item.transform_function(
                    outline, duration=DURATION, easing=Easing.LINEAR
                )
            if branch == 0:
                wr, wc = record["winner"]
                item = b.add(
                    Circle(
                        0.18,
                        style=Style.outline(GOLD, 0.02),
                        transform=Transform2D.translation(-30, 0),
                        z_index=6,
                    )
                )
                item.opacity(to=0, duration=0)
                item.fade_in(duration=0.5, at=2)
                item.fade_out(duration=0.5, at=3.3)
                item.transform_function(
                    lambda u, k=wr * 4 + wc, begin=begin: point(begin(u)[k]),
                    duration=DURATION,
                    easing=Easing.LINEAR,
                )

            for dr in range(2):
                for dc in range(2):
                    k = (r + dr) * 4 + c0 + dc
                    winner = (r + dr, c0 + dc) == record["winner"]
                    item = b.dot(b.palette[ch], radii(d)[k])
                    item.opacity(to=0, duration=0)
                    item.fade_in(duration=0.5, at=2.7)
                    if branch == 0 and not winner:
                        item.fade_out(duration=0.8, at=3.2)

                    def moving(u, k=k, j=j, branch=branch, begin=begin, end=end):
                        q = ease((u * DURATION - 3.3) / 2.5)
                        a = begin(u)[k]
                        z = end(u)[j]
                        return point(
                            a
                            + q * (z - a)
                            + np.array(
                                [0, 0.3 * math.sin(math.pi * q) * (1 if k % 2 else -1)]
                            )
                        )

                    item.transform_function(
                        moving, duration=DURATION, easing=Easing.LINEAR
                    )
                    item.fade_out(duration=0.4, at=6)
            item = b.dot(
                b.palette[ch], radii(pool.MAX_OUT if branch == 0 else pool.AVG_OUT)[j]
            )
            item.opacity(to=0, duration=0)
            item.fade_in(duration=0.4, at=5.8)
            item.transform_function(
                lambda u, j=j, end=end: point(end(u)[j]),
                duration=DURATION,
                easing=Easing.LINEAR,
            )
        b.note(
            "最大汇聚" if branch == 0 else "平均汇聚", c[0], 2.9, color=b.palette[ch]
        )
    return lambda: np.concatenate([p(1) for p in ends])


def handoff(b, source, next_kind, palette):
    from zanim_scenes.lenet_art import geometry as lenet_geometry
    from zanim_scenes.models import ep_06_6 as lenet

    if next_kind == "lenet":
        specs = [
            (
                lenet.INPUT_X,
                lenet_geometry(0)[0],
                0,
                0.013,
                0.32,
                0.033 + 0.025 * np.minimum(2, np.abs(lenet.INPUT_X.ravel())),
            )
        ]
    else:
        specs = [
            (d, project(plane(d, c, sp, h), 0), ch, 0.016, 0.32, radii(d))
            for d, c, sp, h, ch in seeds(next_kind)
        ]
    old = list(b.objects)
    for item in old:
        item.fade_out(duration=0.8)
    carry = []
    for d, target, ch, width, alpha, rr in specs:
        for a, z in edges(np.asarray(d).shape):
            item = b.line(palette[ch], width)
            carry.append(item)
            item.opacity(to=0, duration=0)
            item.opacity(to=alpha, duration=1)
            aa, zz = source[a % len(source)], source[z % len(source)]
            item.transform_function(
                lambda u, a=a, z=z, aa=aa, zz=zz, target=target: pose(
                    aa + ease(u) * (target[a] - aa), zz + ease(u) * (target[z] - zz)
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
        for j, r in enumerate(rr):
            item = b.dot(palette[ch], r)
            carry.append(item)
            item.opacity(to=0, duration=0)
            item.fade_in(duration=1)
            a = source[j % len(source)]
            item.transform_function(
                lambda u, j=j, a=a, target=target: point(a + ease(u) * (target[j] - a)),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    b.play()
    for item in old:
        item.remove()
    return carry


def append(s, chapter, palette, kind, incoming=()):
    i = KINDS.index(kind)
    chapter(TITLES[i], SUBTITLES[i])
    for item in incoming:
        item.remove()
    b = Stage(s, palette)
    finish = (convolution, padding, multichannel, pooling)[i](b, bool(incoming))
    b.play()
    return handoff(b, finish(), KINDS[i + 1] if i < 3 else "lenet", palette)
