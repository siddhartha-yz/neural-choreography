"""SGD ribbon, shared softmax allocation and a piecewise-linear MLP lattice.

Numerical knots/operators are original chapter models. Curves, depth copies,
travel and interpolation between knots are compositional devices. Lattice
vertices are input samples, not neurons; chapter morphs do not feed models.
"""

from functools import lru_cache
import math
import numpy as np
from zanim import Easing, Circle, Style, Transform2D
from zanim_scenes.conv_suite import Stage, point, GOLD
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.feature_suite import project
from zanim_scenes.models import ep_03_1 as linear, ep_03_4 as soft, ep_04_1 as mlp

KINDS = ("linear", "softmax", "mlp")
PREFIX_SECONDS = 36.0
SECONDS = 8.4
SAMPLES = 65
LANES = 13
GX, GY = 19, 13
XX, YY = np.meshgrid(np.linspace(-3.0, 3.0, GX), np.linspace(-2.0, 2.0, GY))
INPUTS = np.stack((XX.ravel(), YY.ravel()), -1) + mlp.INPUT_X.ravel()
EDGES = [(r * GX + j, r * GX + j + 1) for r in range(GY) for j in range(GX - 1)] + [
    (r * GX + j, (r + 1) * GX + j) for r in range(GY - 1) for j in range(GX)
]
TITLES = ("线性回归 · 03.1", "Softmax 回归 · 03.4", "多层感知机 · 04.1")
SUBTITLES = (
    "小批量误差推动同一条拟合曲带。",
    "同除总量，归一为一个整体。",
    "全连接汇入，激活后再分发。",
)


def sgd_parameters(q):
    v = np.clip(q, 0, 1) * (len(linear.TRAINING_STATES) - 1)
    j = min(int(v), len(linear.TRAINING_STATES) - 2)
    w0, b0 = linear.TRAINING_STATES[j]
    w1, b1 = linear.TRAINING_STATES[j + 1]
    blend = ease(v - j)
    return (
        float((w0 + blend * (w1 - w0)).item()),
        float((b0 + blend * (b1 - b0)).item()),
        j,
    )


def regression_xyz(x, y, lane=LANES // 2):
    return np.stack(
        (
            np.asarray(x) * 2.2,
            np.asarray(y) * 0.7 - 0.15,
            np.full_like(np.asarray(x), (lane - (LANES - 1) / 2) * 0.12),
        ),
        -1,
    )


def soft_state(t):
    phase = 2 * math.pi * ease((t - 0.6) / 7.2)
    x = soft.INPUT_X.ravel() + np.array(
        [0.9 * math.sin(phase), 0.9 * (math.cos(phase) - 1)]
    )
    logits = soft.WEIGHTS @ x + soft.BIAS.ravel()
    probs = soft.softmax(logits)
    angles = math.pi + 2 * math.pi * np.r_[0, np.cumsum(probs)]
    return x, logits, probs, angles


def soft_inputs(u=0):
    return project(np.array([[-7.5, 0.45, 0], [-7.5, -0.45, 0]]), u * 2)


def ring_points(angles, lane, u):
    radius = 2.45 + (lane - 1.5) * 0.09
    return project(
        np.stack(
            (
                3.7 + radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, (lane - 1.5) * 0.15),
            ),
            -1,
        ),
        u * 2,
    )


def lattice_values():
    pre = INPUTS @ mlp.WEIGHTS_1.T + mlp.BIAS_1.ravel()
    hidden = mlp.relu(pre)
    outputs = hidden @ mlp.WEIGHTS_2.T + mlp.BIAS_2.ravel()
    return pre, hidden, outputs


def lattice_projection(values, stage, u=0):
    origins = (mlp.INPUT_X.ravel(), mlp.PREACTIVATIONS.ravel(), mlp.OUTPUTS.ravel())
    local = (values - origins[stage]) * 0.7
    xyz = np.c_[
        local + np.array([(-5.7, -0.1, 5.2)[stage], 0.1]),
        np.full(len(values), (stage - 1) * 0.25),
    ]
    return project(xyz, u * 2)


def initial(kind, palette):
    if kind == "softmax":
        return soft_inputs(), np.array([0.12, 0.085]), [palette[0]] * 2, ()
    from zanim_scenes.classification_art import network_nodes

    points = network_nodes(0)
    return points, np.array([0.12, 0.085]), [palette[0]] * 2, ()


def linear_draw(b, incoming):
    axis = np.linspace(-2.7, 2.45, SAMPLES)

    @lru_cache(maxsize=64)
    def state(u):
        w, bias, batch = sgd_parameters(np.clip((u * SECONDS - 0.65) / 7.0, 0, 1))
        y = w * axis + bias
        return (
            np.array(
                [project(regression_xyz(axis, y, lane), u * 3) for lane in range(LANES)]
            ),
            w,
            bias,
            batch,
        )

    for lane in range(LANES):
        for j in range(SAMPLES - 1):
            item = b.line(b.palette[0], 0.028 if lane == LANES // 2 else 0.013)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.9 if lane == LANES // 2 else 0.35, duration=0.5)
            item.transform_function(
                lambda u, lane=lane, j=j: pose(
                    state(u)[0][lane, j], state(u)[0][lane, j + 1]
                ),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
        for phase in range(2):
            for tail in range(4):
                item = b.dot(b.palette[0], 0.036)
                item.opacity(to=0, duration=0)
                item.fade_in(duration=0.5)

                def running(u, lane=lane, phase=phase, tail=tail):
                    q = (u * 1.25 + phase / 2 + lane * 0.015 - tail * 0.006) % 1
                    v = q * (SAMPLES - 1)
                    j = min(int(v), SAMPLES - 2)
                    pp = state(u)[0][lane]
                    return point(
                        pp[j] + (v - j) * (pp[j + 1] - pp[j]), math.sin(math.pi * q)
                    )

                item.transform_function(running, duration=SECONDS, easing=Easing.LINEAR)
    for j, (x, y) in enumerate(zip(linear.FEATURES.ravel(), linear.TARGETS.ravel())):
        raw = regression_xyz(np.array([x]), np.array([y]))
        item = b.dot(GOLD, 0.068)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.5)
        item.transform_function(
            lambda u, raw=raw: point(project(raw, u * 3)[0]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
        item = b.add(
            Circle(
                0.15,
                style=Style.outline(GOLD, 0.025),
                transform=Transform2D.translation(-30, 0),
                z_index=6,
            )
        )
        item.opacity(to=0, duration=0)
        item.opacity(to=0.8, duration=0.5)

        def selected(u, j=j, raw=raw):
            batch = state(u)[3]
            return point(
                project(raw, u * 3)[0], 1 if j in linear.MINIBATCHES[batch] else 0
            )

        item.transform_function(selected, duration=SECONDS, easing=Easing.LINEAR)
        item = b.line(GOLD, 0.017)
        item.opacity(to=0, duration=0)
        item.opacity(to=0.5, duration=0.5)

        def residual(u, x=x, y=y):
            _, w, bias, _ = state(u)
            return pose(
                *project(
                    regression_xyz(np.array([x, x]), np.array([y, w * x + bias])), u * 3
                )
            )

        item.transform_function(residual, duration=SECONDS, easing=Easing.LINEAR)
        for tail in range(5):
            item = b.dot(GOLD, 0.035)

            def correction(u, j=j, x=x, y=y, tail=tail):
                _, w, bias, batch = state(u)
                q = (u * SECONDS - 0.65) * 9 / 7.0
                q = float(np.clip((q - math.floor(q) - tail * 0.035) / 0.8, 0, 1))
                pp = project(
                    regression_xyz(np.array([x, x]), np.array([y, w * x + bias])), u * 3
                )
                gain = (
                    math.sin(math.pi * q)
                    if j in linear.MINIBATCHES[batch] and u * SECONDS > 0.65
                    else 0
                )
                return point(pp[0] + ease(q) * (pp[1] - pp[0]), gain)

            item.transform_function(correction, duration=SECONDS, easing=Easing.LINEAR)
    return lambda: state(1)[0].reshape(-1, 2)


def softmax_draw(b, incoming):
    colors = (b.palette[0], GOLD, b.palette[2])

    @lru_cache(maxsize=64)
    def state(u):
        return soft_state(u * SECONDS)

    @lru_cache(maxsize=64)
    def wheel(u):
        a = state(u)[3]
        return np.array(
            [
                [
                    ring_points(np.linspace(a[ch], a[ch + 1], SAMPLES), lane, u)
                    for lane in range(4)
                ]
                for ch in range(3)
            ]
        )

    for i in range(2):
        item = b.dot(b.palette[0], (0.12, 0.085)[i])
        item.opacity(to=1 if incoming else 0, duration=0)
        item.opacity(to=1, duration=0.45)
        item.transform_function(
            lambda u, i=i: point(soft_inputs(u)[i]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
    for ch, color in enumerate(colors):

        def port(u, ch=ch):
            return project(np.array([[-5.3, (1 - ch) * 2.0, 0]]), u * 2)[0]

        item = b.dot(color, 0.14)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.6)
        item.transform_function(
            lambda u, ch=ch, port=port: point(
                port(u), 0.6 + 0.3 * abs(state(u)[1][ch])
            ),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
        for inp in range(2):
            item = b.line(color, 0.013)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.3, duration=0.6)
            item.transform_function(
                lambda u, inp=inp, port=port: pose(soft_inputs(u)[inp], port(u)),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
            for tail in range(5):
                item = b.dot(color, 0.035)

                def input_flow(u, inp=inp, ch=ch, tail=tail, port=port):
                    q = (u * 3 - tail * 0.025) % 1
                    a = soft_inputs(u)[inp]
                    z = port(u)
                    value = abs(soft.WEIGHTS[ch, inp] * state(u)[0][inp])
                    return point(
                        a + ease(q) * (z - a),
                        min(1.5, value)
                        * math.sin(math.pi * q)
                        * ease(u * SECONDS / 0.6),
                    )

                item.transform_function(
                    input_flow, duration=SECONDS, easing=Easing.LINEAR
                )
        for lane in range(4):
            for j in range(SAMPLES - 1):
                item = b.line(color, 0.019 if lane != 1 else 0.032)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.7, duration=0.65)

                def arc(u, ch=ch, lane=lane, j=j):
                    return pose(wheel(u)[ch, lane, j], wheel(u)[ch, lane, j + 1])

                item.transform_function(arc, duration=SECONDS, easing=Easing.LINEAR)
        for lane in range(4):
            for pulse in range(2):
                for tail in range(5):
                    item = b.dot(color, 0.037)

                    def orbit(u, ch=ch, lane=lane, pulse=pulse, tail=tail):
                        q = (u * 1.6 + pulse / 2 - tail * 0.01) % 1
                        v = q * (SAMPLES - 1)
                        j = min(int(v), SAMPLES - 2)
                        pp = wheel(u)[ch, lane]
                        return point(
                            pp[j] + (v - j) * (pp[j + 1] - pp[j]),
                            math.sin(math.pi * q) * ease(u * SECONDS / 0.65),
                        )

                    item.transform_function(
                        orbit, duration=SECONDS, easing=Easing.LINEAR
                    )
        for lane in range(9):

            def feed(u, q, ch=ch, lane=lane, port=port):
                a = state(u)[3]
                middle = (a[ch] + a[ch + 1]) / 2
                start = port(u) + np.array([0, (lane - 4) * 0.065])
                end = ring_points(np.array([middle]), 1.5, u)[0]
                pp = start + q * (end - start)
                pp[1] += math.sin(math.pi * q) * (
                    (1 - ch) * 0.5 + (lane - 4) * state(u)[2][ch] * 0.16
                )
                return pp

            for j in range(20):
                item = b.line(color, 0.012)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.22, duration=0.65)
                item.transform_function(
                    lambda u, j=j, feed=feed: pose(
                        feed(u, j / 20), feed(u, (j + 1) / 20)
                    ),
                    duration=SECONDS,
                    easing=Easing.LINEAR,
                )
            for pulse in range(2):
                for tail in range(4):
                    item = b.dot(color, 0.035)

                    def traveling(u, ch=ch, pulse=pulse, tail=tail, feed=feed):
                        q = (u * 2 + pulse / 2 - tail * 0.009) % 1
                        return point(
                            feed(u, ease(q)),
                            math.sin(math.pi * q)
                            * math.sqrt(state(u)[2][ch])
                            * ease(u * SECONDS / 0.6)
                            * 2,
                        )

                    item.transform_function(
                        traveling, duration=SECONDS, easing=Easing.LINEAR
                    )
    b.note("Softmax", 3.7, 0, color=b.palette[2])

    def end():
        aa = state(1)[3]
        return np.concatenate(
            [
                ring_points(np.linspace(aa[ch], aa[ch + 1], SAMPLES), 1.5, 1)
                for ch in range(3)
            ]
        )

    return end


def mlp_draw(b, incoming):
    pre, hidden, outputs = lattice_values()

    @lru_cache(maxsize=64)
    def state(u):
        t = u * SECONDS
        first = lattice_projection(INPUTS, 0, u)
        clipped = pre + ease((t - 2.2) / 2.2) * (hidden - pre)
        mid_target = lattice_projection(clipped, 1, u)
        middle = first + ease((t - 0.3) / 1.65) * (mid_target - first)
        target = lattice_projection(outputs, 2, u)
        last = middle + ease((t - 4.65) / 2.0) * (target - middle)
        return np.array([first, middle, last])

    for stage in range(3):
        color = b.palette[stage]
        for a, z in EDGES:
            item = b.line(color, 0.015)
            item.opacity(to=0.32 if incoming and stage == 0 else 0, duration=0)
            item.opacity(to=0.32, duration=0.65, at=(0, 0.25, 4.65)[stage])
            item.transform_function(
                lambda u, stage=stage, a=a, z=z: pose(
                    state(u)[stage, a], state(u)[stage, z]
                ),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
        for depth in (-1, 1):
            for a, z in EDGES:
                item = b.line(color, 0.011)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.17, duration=0.65, at=(0, 0.25, 4.65)[stage])

                def copied(u, stage=stage, a=a, z=z, depth=depth):
                    offset = project(np.array([[0, 0, depth * 0.42]]), u * 2)[0]
                    return pose(
                        state(u)[stage, a] + offset, state(u)[stage, z] + offset
                    )

                item.transform_function(copied, duration=SECONDS, easing=Easing.LINEAR)
        for row in (0, 4, 8, 12):
            for tail in range(5):
                item = b.dot(color, 0.037)
                item.opacity(to=0, duration=0)
                item.opacity(to=1, duration=0.65, at=(0, 0.25, 4.65)[stage])

                def scan(u, stage=stage, row=row, tail=tail):
                    q = (u * 1.5 + row * 0.031 - tail * 0.009) % 1
                    v = q * (GX - 1)
                    j = min(int(v), GX - 2)
                    pp = state(u)[stage, row * GX : (row + 1) * GX]
                    return point(
                        pp[j] + (v - j) * (pp[j + 1] - pp[j]), math.sin(math.pi * q)
                    )

                item.transform_function(scan, duration=SECONDS, easing=Easing.LINEAR)
        for j in range(len(INPUTS)):
            item = b.dot(color, 0.032)
            item.opacity(to=1 if incoming and stage == 0 else 0, duration=0)
            item.opacity(to=1, duration=0.65, at=(0, 0.25, 4.65)[stage])
            item.transform_function(
                lambda u, stage=stage, j=j: point(state(u)[stage, j]),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
        # Original input is exactly the center vertex of the deterministic grid.
        j = (GY // 2) * GX + GX // 2
        item = b.add(
            Circle(
                0.11,
                style=Style.outline(GOLD, 0.035),
                transform=Transform2D.translation(-30, 0),
                z_index=7,
            )
        )
        item.opacity(to=0, duration=0)
        item.opacity(to=1, duration=0.6, at=(0, 0.25, 4.65)[stage])
        item.transform_function(
            lambda u, stage=stage, j=j: point(state(u)[stage, j]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
    # The two zero thresholds remain fixed while negative coordinates collapse.
    for axis in range(2):
        vv = (
            np.array([[0, -4.5], [0, 3.0]])
            if axis == 0
            else np.array([[-3.5, 0], [6.0, 0]])
        )
        item = b.line(GOLD, 0.021)
        item.opacity(to=0, duration=0)
        item.opacity(to=0.65, duration=0.4, at=2.05)
        item.fade_out(duration=0.4, at=4.45)
        item.transform_function(
            lambda u, vv=vv: pose(*lattice_projection(vv, 1, u)),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
    for stage in (0, 1):
        for j in range(0, len(INPUTS), 11):
            for tail in range(4):
                item = b.dot(b.palette[stage], 0.036)

                def flowing(u, stage=stage, j=j, tail=tail):
                    t = u * SECONDS
                    start = (0.45, 4.8)[stage]
                    q = float(
                        np.clip((t - start - tail * 0.028 - (j % 7) * 0.04) / 1.3, 0, 1)
                    )
                    aa = state(u)[stage, j]
                    zz = state(u)[stage + 1, j]
                    return point(
                        aa
                        + ease(q) * (zz - aa)
                        + np.array([0, 0.2 * math.sin(math.pi * q)]),
                        math.sin(math.pi * q),
                    )

                item.transform_function(flowing, duration=SECONDS, easing=Easing.LINEAR)
    b.note("输入", -5.7, 3.0, color=b.palette[0])
    b.note("ReLU", -0.1, 3.0, start=2.0, color=GOLD)
    b.note("输出", 5.2, 3.0, start=4.65)
    return lambda: state(1)[2]


def append(s, chapter, palette, kind, incoming=()):
    from zanim_scenes.classification_art import softmax, mlp

    i = KINDS.index(kind)
    chapter(TITLES[i], SUBTITLES[i])
    for item in incoming:
        item.remove()
    b = Stage(s, palette)
    end = (linear_draw, softmax, mlp)[i](b, bool(incoming))
    b.play()
    source = end()
    old = list(b.objects)
    for item in old:
        item.fade_out(duration=0.8)
    carry = []
    if i == 2:
        from zanim_scenes.regularization_suite import initial as fitting_initial

        target, _, _ = fitting_initial("capacity", palette)
        # The two outputs unfold into the following pair of fitting ribbons.
        # This is a compositional fan-out, not a numerical model connection.
        curves = np.array(
            [
                np.stack(
                    [
                        np.full(SAMPLES, source[row % len(source), axis])
                        + (np.linspace(-1, 1, SAMPLES) * 0.45 if axis == 0 else 0)
                        for axis in range(2)
                    ],
                    -1,
                )
                for row in range(GY)
            ]
        )
        for band in range(2):
            for lane in range(LANES):
                for j in range(SAMPLES - 1):
                    index = (band * LANES + lane) * SAMPLES + j
                    item = b.line(
                        palette[(0, 2)[band]], 0.028 if lane == LANES // 2 else 0.013
                    )
                    carry.append(item)
                    item.opacity(to=0, duration=0)
                    item.opacity(to=0.9 if lane == LANES // 2 else 0.35, duration=1)
                    a, z = curves[lane, j : j + 2]
                    aa, zz = target[index : index + 2]
                    item.transform_function(
                        lambda u, a=a, z=z, aa=aa, zz=zz: pose(
                            a + ease(u) * (aa - a), z + ease(u) * (zz - z)
                        ),
                        duration=2.8,
                        easing=Easing.LINEAR,
                    )
    else:
        target, rr, colors, edges = initial(KINDS[i + 1], palette)
        starts = source[np.linspace(0, len(source) - 1, len(target)).astype(int)]
        for a, z in edges:
            item = b.line(palette[0], 0.015)
            carry.append(item)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.32, duration=1)
            item.transform_function(
                lambda u, a=a, z=z: pose(
                    starts[a] + ease(u) * (target[a] - starts[a]),
                    starts[z] + ease(u) * (target[z] - starts[z]),
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
        for j, r in enumerate(rr):
            item = b.dot(colors[j], r)
            carry.append(item)
            item.opacity(to=0, duration=0)
            item.fade_in(duration=1)
            item.transform_function(
                lambda u, j=j: point(starts[j] + ease(u) * (target[j] - starts[j])),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    b.play()
    for item in old:
        item.remove()
    return carry
