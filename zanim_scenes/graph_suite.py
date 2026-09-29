"""Three connected graph studies; original models determine values and gradients."""

import math
import numpy as np
from zanim import Circle, Style, Transform2D, Easing
from zanim_scenes.conv_suite import Stage, point, handoff as conv_handoff, GOLD
from zanim_scenes.feature_suite import project
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.models import ep_04_7 as back, ep_04_8 as stable, ep_05_1 as blocks

KINDS = ("backprop", "stability", "blocks")
PREFIX_SECONDS = 36.0
DURATION = 8.4
TITLES = ("正向与反向传播 · 04.7", "数值稳定性与初始化 · 04.8", "层和块 · 05.1")
SUBTITLES = (
    "沿图向前，沿原路回传。",
    "局部导数连乘，决定回传强度。",
    "层嵌入块，块组成网络。",
)


def projected(xy, u=0):
    xy = np.asarray(xy)
    xyz = np.c_[xy, 0.3 * np.sin(xy[:, 1])]
    return project(xyz, u * 2)


def initial(kind, palette):
    if kind == "stability":
        xy = np.array([[x, y] for y in (1.6, -1.6) for x in (-5, -1.7, 1.7, 5)])
        values = np.r_[stable.SAT_H, stable.GOOD_H]
        return projected(xy), 0.11 + 0.12 * values, [palette[0]] * 4 + [palette[3]] * 4
    values = back.INPUT_X if kind == "backprop" else blocks.INPUT_X
    return (
        projected([[-6, 1.1], [-6, -1.1]]),
        0.1 + 0.06 * np.abs(values.ravel()),
        [palette[0]] * 2,
    )


def nodes(b, xy, values, color, start=0, incoming=False):
    def loc(u):
        return projected(xy, u)

    for i, v in enumerate(np.asarray(values).ravel()):
        item = b.dot(color, 0.1 + 0.06 * min(3, abs(float(v))))
        item.opacity(to=1 if incoming else 0, duration=0)
        item.opacity(to=1, duration=0.5, at=start)
        item.transform_function(
            lambda u, i=i: point(loc(u)[i]), duration=DURATION, easing=Easing.LINEAR
        )
    return loc


def wire(b, a, z, color, arch=0, width=0.016, start=0):
    for k in range(14):

        def at(u, q):
            return (
                a(u) + q * (z(u) - a(u)) + np.array([0, arch * math.sin(math.pi * q)])
            )

        item = b.line(color, width)
        item.opacity(to=0, duration=0)
        item.opacity(to=0.32, duration=0.6, at=start)
        item.transform_function(
            lambda u, k=k, at=at: pose(at(u, k / 14), at(u, (k + 1) / 14)),
            duration=DURATION,
            easing=Easing.LINEAR,
        )


def stream(b, a, z, start, duration, color, strength=1, arch=0):
    if abs(strength) < 1e-12:
        return
    for k in range(5):
        b.flow(
            a,
            z,
            start + k * 0.08,
            duration,
            color,
            0.035 + 0.04 * min(2, abs(strength)) ** 0.5,
            arch,
        )


def backprop(b, incoming):
    g = back.GRAPH
    inp = nodes(
        b, [[-6, 1.1], [-6, -1.1]], back.INPUT_X, b.palette[0], incoming=incoming
    )
    hid = nodes(b, [[-1.4, 1.5], [-1.4, -1.5]], g["h"], b.palette[2], start=1)
    out = nodes(b, [[3, 0]], [g["yhat"]], b.palette[3], start=2.7)
    loss = nodes(b, [[6, 0]], [g["loss"]], GOLD, start=3.6)
    for i in range(2):
        for j in range(2):

            def a(u, i=i):
                return inp(u)[i]

            def z(u, j=j):
                return hid(u)[j]

            arch = (i - j) * 0.25
            wire(
                b,
                a,
                z,
                b.palette[0],
                arch,
                width=0.013 + 0.008 * abs(back.WEIGHTS_1[j, i]),
            )
            stream(b, a, z, 0.55, 1.3, b.palette[0], back.WEIGHTS_1[j, i], arch)
            stream(
                b,
                z,
                a,
                6.0,
                1.35,
                GOLD,
                float(g["dL_dz"][j, 0] * back.WEIGHTS_1[j, i]),
                arch,
            )
    for j in range(2):

        def a(u, j=j):
            return hid(u)[j]

        def z(u):
            return out(u)[0]

        wire(b, a, z, b.palette[2], start=0.8)
        stream(b, a, z, 2, 1.2, b.palette[2], float(g["h"][j, 0]))
        stream(b, z, a, 5, 1.1, GOLD, float(g["dL_dh"][j, 0]))
    wire(b, lambda u: out(u)[0], lambda u: loss(u)[0], GOLD, start=2)
    stream(
        b, lambda u: out(u)[0], lambda u: loss(u)[0], 3.3, 0.7, b.palette[3], g["yhat"]
    )
    stream(b, lambda u: loss(u)[0], lambda u: out(u)[0], 4.2, 0.7, GOLD, g["dL_dyhat"])
    # The zero ReLU derivative visibly blocks the lower return route.
    for j in range(2):
        item = b.add(
            Circle(
                0.3,
                style=Style.outline(b.palette[2], 0.025),
                transform=Transform2D.translation(-30, 0),
                z_index=6,
            )
        )
        item.opacity(to=0, duration=0)
        item.opacity(to=0.8, duration=0.5, at=1.2)
        item.transform_function(
            lambda u, j=j: point(hid(u)[j]), duration=DURATION, easing=Easing.LINEAR
        )
    b.note("ReLU", -1.4, 2.65)
    b.note("ℓ", 6, 1, color=GOLD)
    b.note("正向", 0, -3, start=0, end=4)
    b.note("反向", 0, -3, start=4, color=GOLD)
    return lambda: np.concatenate([inp(1), hid(1), out(1)])


def stability(b, incoming):
    allpoints = []
    for row, mix in enumerate((0.0, 1.0)):
        z, h, grad, _ = stable.network_at(mix)
        y = 1.6 - row * 3.2
        xs = (-5, -1.7, 1.7, 5)

        def loc(u, y=y):
            return projected([[x, y] for x in xs], u)

        allpoints.append(loc)
        for i in range(4):
            item = b.dot(b.palette[0 if row == 0 else 3], 0.11 + 0.12 * h[i])
            item.opacity(to=1 if incoming else 0, duration=0)
            item.opacity(to=1, duration=0.5)
            item.transform_function(
                lambda u, i=i, loc=loc: point(loc(u)[i]),
                duration=DURATION,
                easing=Easing.LINEAR,
            )
            # A narrowed gate depicts the local sigmoid slope.
            slope = float(stable.sigmoid_prime(np.array([z[i]]))[0])
            for side in (-1, 1):
                item = b.line(b.palette[row * 3], 0.018)

                def gate(u, i=i, side=side, loc=loc, slope=slope):
                    center = loc(u)[i]
                    return pose(
                        center + [-0.42, side * 0.52],
                        center + [0.42, side * (0.035 + 1.8 * slope)],
                    )

                item.opacity(to=0, duration=0)
                item.opacity(to=0.65, duration=0.6)
                item.transform_function(gate, duration=DURATION, easing=Easing.LINEAR)
        for i in range(3):

            def a(u, i=i, loc=loc):
                return loc(u)[i]

            def zpos(u, i=i, loc=loc):
                return loc(u)[i + 1]

            wire(b, a, zpos, b.palette[row * 3])
            stream(b, a, zpos, 0.3 + i * 0.85, 0.85, b.palette[row * 3], float(h[i]))
            for k in range(9):
                item = b.dot(GOLD, 0.15)

                def reverse(u, i=i, k=k, loc=loc, grad=grad):
                    q = float(
                        np.clip(
                            (u * DURATION - 3.6 - (2 - i) * 1.05 - k * 0.055) / 1.05,
                            0,
                            1,
                        )
                    )
                    a = loc(u)[i + 1]
                    z = loc(u)[i]
                    # Monotone fourth-root display gain makes tiny gradients
                    # visible; it does not imply their true numeric ratio.
                    gain = ((1 - q) * grad[i + 1] + q * grad[i]) ** 0.25
                    return point(a + ease(q) * (z - a), gain * math.sin(math.pi * q))

                item.transform_function(
                    reverse, duration=DURATION, easing=Easing.LINEAR
                )
    b.note("饱和初始化", -6.9, 2.9, color=b.palette[0])
    b.note("较小权重", -6.9, -2.9, color=b.palette[3])
    return lambda: np.concatenate([p(1) for p in allpoints])


def nested(b, incoming):
    inp = nodes(
        b, [[-6, 1.1], [-6, -1.1]], blocks.INPUT_X, b.palette[0], incoming=incoming
    )
    pre = nodes(
        b,
        [[-2, 1.25], [-2, 0], [-2, -1.25]],
        blocks.PREACTIVATIONS,
        b.palette[1],
        start=0.8,
    )
    hidden = nodes(
        b, [[0.4, 1.25], [0.4, 0], [0.4, -1.25]], blocks.HIDDEN, b.palette[2], start=2.5
    )
    out = nodes(b, [[4.3, 1], [4.3, -1]], blocks.OUTPUTS, b.palette[3], start=4.6)
    for x0, x1, y, color, start in [
        (-4.1, 5.4, 3.1, b.palette[0], 0),
        (-3.5, 1.5, 2.4, b.palette[2], 0.3),
        (-2.8, -1.1, 1.8, b.palette[1], 0.6),
        (-0.4, 1.1, 1.8, b.palette[2], 0.8),
        (3.4, 5.1, 1.8, b.palette[3], 1),
    ]:
        corners = np.array([[x0, -y], [x1, -y], [x1, y], [x0, y]])
        for e in range(4):
            item = b.line(color, 0.016)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.45, duration=0.7, at=start)
            item.transform_function(
                lambda u, e=e, corners=corners: pose(
                    projected(corners, u)[e], projected(corners, u)[(e + 1) % 4]
                ),
                duration=DURATION,
                easing=Easing.LINEAR,
            )
    for i in range(2):
        for j in range(3):

            def a(u, i=i):
                return inp(u)[i]

            def z(u, j=j):
                return pre(u)[j]

            wire(b, a, z, b.palette[0])
            stream(b, a, z, 0.6, 1.3, b.palette[0], blocks.WEIGHTS_1[j, i])
    for j in range(3):

        def a(u, j=j):
            return pre(u)[j]

        def z(u, j=j):
            return hidden(u)[j]

        wire(b, a, z, b.palette[2])
        stream(b, a, z, 2.1, 1.2, b.palette[2], blocks.PREACTIVATIONS[j, 0])
        for i in range(2):

            def a(u, j=j):
                return hidden(u)[j]

            def z(u, i=i):
                return out(u)[i]

            wire(b, a, z, b.palette[3])
            stream(
                b,
                a,
                z,
                3.7,
                1.5,
                b.palette[3],
                blocks.HIDDEN[j, 0] * blocks.WEIGHTS_2[i, j],
            )
    b.note("Sequential", 0.5, 3.5)
    b.note("块", -1, 2.65)
    b.note("ReLU", 0.4, -2.1)
    return lambda: out(1)


def append(s, chapter, palette, kind, incoming=()):
    i = KINDS.index(kind)
    chapter(TITLES[i], SUBTITLES[i])
    for item in incoming:
        item.remove()
    b = Stage(s, palette)
    end = (backprop, stability, nested)[i](b, bool(incoming))
    b.play()
    source = end()
    if i == 2:
        return conv_handoff(b, source, "conv", palette)
    target, rr, colors = initial(KINDS[i + 1], palette)
    old = list(b.objects)
    carry = []
    for item in old:
        item.fade_out(duration=0.8)
    for j, r in enumerate(rr):
        item = b.dot(colors[j], r)
        carry.append(item)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=1)
        a = source[j % len(source)]
        item.transform_function(
            lambda u, j=j, a=a: point(a + ease(u) * (target[j] - a)),
            duration=2.8,
            easing=Easing.LINEAR,
        )
    b.play()
    for item in old:
        item.remove()
    return carry
