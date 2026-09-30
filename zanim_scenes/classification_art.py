"""Explicit shared normalization and a classic 2–4–2 fully connected MLP.

Exp(z-max(z)) is visualized in common units, then EVERY segment is scaled by
one shared denominator. The MLP is fixed, untrained; its first two hidden
units reuse 04.1 and the two extra units are inactive for the original input.
"""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Style, Transform2D, Easing
from zanim_scenes.conv_suite import point, GOLD
from zanim_scenes.feature_suite import project
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.models import ep_04_1 as original

SECONDS = 8.4
SAMPLES = 65
LANES = 13
WEIGHT_1 = np.vstack((original.WEIGHTS_1, [[0.6, -0.8], [-0.75, 0.9]]))
BIAS_1 = np.r_[original.BIAS_1.ravel(), -0.15, 0.05]
WEIGHT_2 = np.c_[original.WEIGHTS_2, [[0.35, -0.4], [0.45, 0.55]]]
BIAS_2 = original.BIAS_2.ravel()
CALLS = np.array([[0.8, 0.6], [-1.5, -0.6], [1.5, -1.2], [-0.4, 1.5], [0.8, 0.6]])
STARTS = np.array([0.15, 1.75, 3.35, 4.95, 6.55])


def normalization_values(t):
    from zanim_scenes.foundation_art import soft_state

    # Hold the example through the common-division move, then vary the input.
    sample_time = max(0, t - 4.0) * SECONDS / 4.4
    x, logits, probabilities, angles = soft_state(sample_time)
    masses = np.exp(logits - np.max(logits))
    denominator = float(masses.sum())
    division = ease((t - 2.6) / 1.4)
    scale = (1 - division) + division / denominator
    return x, masses, denominator, scale, probabilities, angles


def normalized_shape(u):
    t = u * SECONDS
    _, masses, total, scale, _, angles = normalization_values(t)
    stack = ease((t - 1.1) / 1.4)
    roll = ease((t - 5.1) / 1.3)
    cumulative = np.r_[0, np.cumsum(masses)]
    out = []
    for ch in range(3):
        copies = []
        q = np.linspace(0, 1, SAMPLES)
        for lane in range(LANES):
            depth = (lane - 6) * 0.09
            left = -3 * total * scale + stack * 6 * cumulative[ch] * scale
            xx = left + 6 * masses[ch] * scale * q
            yy = np.full(SAMPLES, (1 - stack) * (1 - ch) * 1.8 + (lane - 6) * 0.035)
            bar = np.stack((xx, yy, np.full(SAMPLES, depth)), -1)
            theta = angles[ch] + q * (angles[ch + 1] - angles[ch])
            radius = 2.35 + (lane - 6) * 0.045
            ring = np.stack(
                (
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    np.full(SAMPLES, depth),
                ),
                -1,
            )
            copies.append(project(bar + roll * (ring - bar), u * 2))
        out.append(copies)
    return np.array(out)


def softmax(b, incoming):
    from zanim_scenes.foundation_art import soft_inputs

    colors = (b.palette[0], GOLD, b.palette[2])

    @lru_cache(maxsize=64)
    def frame(u):
        return normalized_shape(u)

    for j in range(2):
        item = b.dot(b.palette[0], (0.12, 0.085)[j])
        item.opacity(to=1 if incoming else 0, duration=0)
        item.opacity(to=1, duration=0.4)
        item.fade_out(duration=0.4, at=0.85)
        item.transform_function(
            lambda u, j=j: point(soft_inputs(u)[j]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
    for ch, color in enumerate(colors):
        for lane in range(LANES):
            for j in range(SAMPLES - 1):
                item = b.line(color, 0.027 if lane == 6 else 0.012)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.9 if lane == 6 else 0.35, duration=0.6)
                item.transform_function(
                    lambda u, ch=ch, lane=lane, j=j: pose(
                        frame(u)[ch, lane, j], frame(u)[ch, lane, j + 1]
                    ),
                    duration=SECONDS,
                    easing=Easing.LINEAR,
                )
            for pulse in range(2):
                for tail in range(4):
                    item = b.dot(color, 0.037)

                    def flowing(u, ch=ch, lane=lane, pulse=pulse, tail=tail):
                        q = (u * 1.6 + pulse / 2 + lane * 0.012 - tail * 0.008) % 1
                        v = q * (SAMPLES - 1)
                        j = min(int(v), SAMPLES - 2)
                        pp = frame(u)[ch, lane]
                        return point(
                            pp[j] + (v - j) * (pp[j + 1] - pp[j]),
                            math.sin(math.pi * q) * ease(u * SECONDS / 0.6),
                        )

                    item.transform_function(
                        flowing, duration=SECONDS, easing=Easing.LINEAR
                    )
        # The incoming sample leaves into the three positive response bands.
        for inp in range(2):
            for tail in range(5):
                item = b.dot(color, 0.035)

                def enter(u, inp=inp, ch=ch, tail=tail):
                    q = float(np.clip((u * SECONDS - 0.15 - tail * 0.025) / 0.65, 0, 1))
                    a = soft_inputs(u)[inp]
                    z = frame(u)[ch, 6, 0]
                    return point(a + ease(q) * (z - a), math.sin(math.pi * q))

                item.transform_function(enter, duration=SECONDS, easing=Easing.LINEAR)
    # A SINGLE shared brace spans the total and contracts to one unit.
    for edge in range(3):
        item = b.line(GOLD, 0.024)
        item.opacity(to=0, duration=0)
        item.opacity(to=0.75, duration=0.4, at=1.7)
        item.fade_out(duration=0.4, at=4.9)

        def brace(u, edge=edge):
            _, _, total, scale, _, _ = normalization_values(u * SECONDS)
            h = 3 * total * scale
            pp = project(np.array([[-h, -0.55, 0], [h, -0.55, 0]]), u * 2)
            if edge == 0:
                return pose(*pp)
            end = pp[edge - 1]
            return pose(end, end + np.array([0, 0.22]))

        item.transform_function(brace, duration=SECONDS, easing=Easing.LINEAR)
    b.note("exp(z−c)", 0, 3.3, end=1.65)
    # Fixed unit ruler stays visible while the total above contracts.
    ruler = b.line(b.palette[3], 0.018)
    ruler.opacity(to=0, duration=0)
    ruler.opacity(to=0.65, duration=0.4, at=1.7)
    ruler.fade_out(duration=0.4, at=5.0)
    ruler.transform_function(
        lambda u: pose(*project(np.array([[-3, -1.6, 0], [3, -1.6, 0]]), u * 2)),
        duration=SECONDS,
        easing=Easing.LINEAR,
    )
    b.note("1", 0, -2.05, start=1.7, end=5.4, color=b.palette[3])
    b.note("合成总量", 0, 3.3, start=1.65, end=2.75)
    b.note("÷ 同一个总量", 0, 3.3, start=2.75, end=4.35, color=GOLD)
    b.note("总量", 0, -1.1, start=1.7, end=3.4, color=GOLD)
    b.note("1", 0, -1.1, start=4.0, end=5.1, color=GOLD)
    b.note("Σp = 1", 0, 0, start=6.5, color=GOLD)
    return lambda: frame(1).reshape(-1, 2)


def network_values(x):
    pre = np.asarray(x) @ WEIGHT_1.T + BIAS_1
    hidden = np.maximum(pre, 0)
    out = hidden @ WEIGHT_2.T + BIAS_2
    return pre, hidden, out


def network_nodes(stage, u=0):
    count = (2, 4, 2)[stage]
    ys = np.array([1.4, -1.4]) if count == 2 else np.linspace(2.4, -2.4, 4)
    xyz = np.c_[
        np.full(count, (-5.6, 0, 5.6)[stage]), ys, np.full(count, (stage - 1) * 0.35)
    ]
    return project(xyz, u * 2)


def network_state(t):
    call = int(np.clip(np.searchsorted(STARTS, t, side="right") - 1, 0, 4))
    local = max(0, t - STARTS[call])
    blend = ease(local / 0.22)
    x = CALLS[max(0, call - 1)] + blend * (CALLS[call] - CALLS[max(0, call - 1)])
    return call, local, x, *network_values(x)


def connection(stage, src, dst, u):
    a = network_nodes(stage, u)[src]
    z = network_nodes(stage + 1, u)[dst]
    direction = (z - a) / np.linalg.norm(z - a)
    return a + 0.39 * direction, z - 0.37 * direction


def mlp(b, incoming):
    @lru_cache(maxsize=64)
    def frame(u):
        return network_state(u * SECONDS)

    for stage, count in enumerate((2, 4, 2)):
        color = b.palette[(0, 2, 3)[stage]]
        for j in range(count):
            item = b.add(
                Circle(
                    0.36 if stage == 1 else 0.38,
                    style=Style.outline(color, 0.034),
                    transform=Transform2D.translation(-30, 0),
                    z_index=5,
                )
            )
            item.opacity(to=0, duration=0)
            item.opacity(to=0.85, duration=0.6)
            item.transform_function(
                lambda u, stage=stage, j=j: point(network_nodes(stage, u)[j]),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
            r = (0.12, 0.085)[j] if stage == 0 else 0.17
            item = b.dot(color, r)
            item.opacity(to=1 if incoming and stage == 0 else 0, duration=0)
            item.opacity(to=1, duration=0.6)

            def activation(u, stage=stage, j=j):
                _, t, x, pre, h, out = frame(u)
                if stage == 0:
                    gain = math.sqrt(abs(x[j] / original.INPUT_X[j, 0]))
                elif stage == 1:
                    clipped = pre[j] + ease((t - 0.65) / 0.22) * (h[j] - pre[j])
                    gain = math.sqrt(abs(clipped)) * ease((t - 0.45) / 0.18)
                else:
                    gain = math.sqrt(abs(out[j])) * ease((t - 1.12) / 0.22)
                return point(network_nodes(stage, u)[j], gain)

            item.transform_function(activation, duration=SECONDS, easing=Easing.LINEAR)
    for stage, weights in enumerate((WEIGHT_1, WEIGHT_2)):
        for dst in range(weights.shape[0]):
            for src in range(weights.shape[1]):
                w = weights[dst, src]
                color = b.palette[2] if w < 0 else b.palette[0]
                item = b.line(color, 0.016 + 0.012 * abs(w))
                item.opacity(to=0, duration=0)
                item.opacity(to=0.42, duration=0.6)
                item.transform_function(
                    lambda u, stage=stage, src=src, dst=dst: pose(
                        *connection(stage, src, dst, u)
                    ),
                    duration=SECONDS,
                    easing=Easing.LINEAR,
                )
                for tail in range(7):
                    item = b.dot(color, 0.043)

                    def pulse(u, stage=stage, src=src, dst=dst, w=w, tail=tail):
                        call, t, x, _, h, _ = frame(u)
                        q = float(
                            np.clip(
                                (t - (0.23, 0.93)[stage] - tail * 0.025) / 0.4, 0, 1
                            )
                        )
                        contribution = (x if stage == 0 else h)[src] * w
                        a, z = connection(stage, src, dst, u)
                        return point(
                            a + ease(q) * (z - a),
                            math.sin(math.pi * q) * math.sqrt(abs(contribution)),
                        )

                    item.transform_function(
                        pulse, duration=SECONDS, easing=Easing.LINEAR
                    )
    # Brief annular responses keep each layer's computation legible.
    for stage, count in ((1, 4), (2, 2)):
        for j in range(count):
            item = b.add(
                Circle(
                    0.46,
                    style=Style.outline(GOLD, 0.025),
                    transform=Transform2D.translation(-30, 0),
                    z_index=6,
                )
            )

            def response(u, stage=stage, j=j):
                _, t, _, _, h, out = frame(u)
                q = float(np.clip((t - (0.87, 1.18)[stage - 1]) / 0.28, 0, 1))
                active = (h if stage == 1 else out)[j]
                return point(
                    network_nodes(stage, u)[j],
                    math.sin(math.pi * q) if abs(active) > 1e-12 else 0,
                )

            item.transform_function(response, duration=SECONDS, easing=Easing.LINEAR)
    b.note("输入层", -5.6, 3.5, color=b.palette[0])
    b.note("隐层 · ReLU", 0, 3.5, color=b.palette[2])
    b.note("输出层", 5.6, 3.5, color=b.palette[3])
    return lambda: network_nodes(2, 1)
