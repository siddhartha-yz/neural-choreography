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
    "输入触发执行，结果返回调用处。",
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
        _, h, grad, grad_x = stable.network_at(mix)
        y = 1.6 - row * 3.2
        xy = np.array([[x, y] for x in (-5, -1.7, 1.7, 5)])

        def loc(u, xy=xy):
            return projected(xy, u)

        allpoints.append(loc)
        for i in range(4):
            item = b.dot(b.palette[row * 3], 0.11 + 0.12 * h[i])
            item.opacity(to=1 if incoming else 0, duration=0)
            item.opacity(to=1, duration=0.4)
            item.opacity(to=0.22, duration=0.7, at=0.7)
            item.transform_function(
                lambda u, i=i, loc=loc: point(loc(u)[i]),
                duration=DURATION,
                easing=Easing.LINEAR,
            )
        path = np.vstack(([-6.8, y], xy))
        amplitudes = np.r_[np.linalg.norm(grad_x), grad]
        # Identical incoming bundles. Width is sqrt(|gradient|) on both rows.
        # A whole sheet travels through the chain instead of isolated dots.
        for lane in range(33):
            for tail in range(6):
                item = b.dot(GOLD, 0.038)

                def packet(u, lane=lane, tail=tail, path=path, amplitudes=amplitudes):
                    t = u * DURATION
                    progress = np.clip((t - 1.0 - tail * 0.075) / 4.8, 0, 1) * 4
                    step = min(int(progress), 3)
                    q = progress - step
                    i = 4 - step
                    j = i - 1
                    gain = math.exp(
                        (1 - q) * math.log(amplitudes[i]) + q * math.log(amplitudes[j])
                    )
                    center = path[i] + ease(q) * (path[j] - path[i])
                    center[1] += (lane / 16 - 1) * 1.1 * math.sqrt(gain)
                    pp = projected(center[None, :], u)[0]
                    visible = ease((t - 0.6) / 0.35) * (1 - ease((t - 6.3) / 0.5))
                    return point(pp, visible * (0.18 + 0.82 * math.sqrt(gain)))

                item.transform_function(packet, duration=DURATION, easing=Easing.LINEAR)
        for i in range(4):
            center = xy[i]
            for side in (-1, 1):
                item = b.line(b.palette[row * 3], 0.024)

                def gate(u, i=i, side=side, center=center, grad=grad):
                    # Common area scale; contraction remains directly comparable.
                    width = 1.1 * math.sqrt(grad[i])
                    pts = np.array(
                        [center + [-0.15, side * width], center + [0.15, side * width]]
                    )
                    pp = projected(pts, u)
                    return pose(pp[0], pp[1])

                item.opacity(to=0, duration=0)
                item.opacity(to=0.75, duration=0.5)
                item.transform_function(gate, duration=DURATION, easing=Easing.LINEAR)
        # Same explicit linear magnification for both input-gradient norms.
        radius = 600 * float(np.linalg.norm(grad_x))
        item = b.add(
            Circle(
                radius,
                style=Style.outline(GOLD, 0.035),
                transform=Transform2D.translation(-30, 0),
                z_index=6,
            )
        )
        item.opacity(to=0, duration=0)
        item.opacity(to=1, duration=0.6, at=5.8)
        item.transform_function(
            lambda u, y=y: point(projected(np.array([[-6.8, y]]), u)[0]),
            duration=DURATION,
            easing=Easing.LINEAR,
        )
        for k in range(24):
            item = b.dot(GOLD, 0.023)

            def residual(u, k=k, y=y, radius=radius):
                theta = k * math.tau / 24 + (u * DURATION - 6) * 0.5
                r = radius * ease((u * DURATION - 5.8) / 0.6)
                pp = projected(
                    np.array([[-6.8 + r * math.cos(theta), y + r * math.sin(theta)]]), u
                )[0]
                return point(pp, ease((u * DURATION - 5.8) / 0.6))

            item.transform_function(residual, duration=DURATION, easing=Easing.LINEAR)
    b.note("饱和初始化", 0, 3.3, color=b.palette[0])
    b.note("较小权重", 0, -3.3, color=b.palette[3])
    b.note("输入端 · 同倍率 ×600", -5.5, -0.05, start=5.8, color=GOLD)
    return lambda: np.concatenate([p(1) for p in allpoints])


CALL_INPUTS = np.array([[1.0, 0.5], [-0.5, 1.2], [0.2, -1.0]])


def block_calls():
    z = CALL_INPUTS @ blocks.WEIGHTS_1.T + blocks.BIAS_1.ravel()
    h = np.maximum(z, 0)
    return z, h, h @ blocks.WEIGHTS_2.T + blocks.BIAS_2.ravel()


def nested(b, incoming):
    from zanim_scenes.block_invocation_art import draw

    return draw(b, incoming)


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
