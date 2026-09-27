"""Continuous motion study: recurrent flow -> adjoint return -> gated merge.

State/gradient endpoints use a fixed RNN. Curves, projection, travel speed and
inter-chapter morphs are compositional devices, not extra model time steps.
"""

from contextlib import nullcontext
import math
import numpy as np
from zanim import Scene, Canvas, Circle, Line, Color, Style, Transform2D, Easing
from zanim_scenes.style import label, rect
from zanim_scenes.models import ep_08_4 as model

BLUE = Color(88, 196, 221)
GOLD = Color(255, 220, 112)
PALETTE = (BLUE, Color(104, 154, 228), Color(150, 140, 222), Color(81, 203, 183))
COUNT = 28
STEPS = 13
SAMPLES = 49


def trajectories(offset=0.0):
    phase = np.arange(COUNT) * (2 * np.pi / COUNT)
    t = np.arange(STEPS - 1)
    tokens = np.stack(
        (
            1.25 * np.sin(phase[:, None] + t[None, :] * 0.57 + offset),
            1.15 * np.cos(phase[:, None] * 2 - t[None, :] * 0.39 + offset),
        ),
        axis=-1,
    )
    h = np.zeros((COUNT, STEPS, 2))
    h[:, 0, 0] = 0.8 * np.sin(phase)
    h[:, 0, 1] = 0.8 * np.cos(phase)
    for j in range(STEPS - 1):
        h[:, j + 1] = np.tanh(
            tokens[:, j] @ model.WEIGHT_XH.T
            + h[:, j] @ model.WEIGHT_HH.T
            + model.BIAS.ravel()
        )
    grad = np.zeros_like(h)
    grad[:, -1] = h[:, -1]  # d(1/2 ||h_T||^2)/dh_T
    for j in range(STEPS - 2, -1, -1):
        grad[:, j] = (grad[:, j + 1] * (1 - h[:, j + 1] ** 2)) @ model.WEIGHT_HH
    return tokens, h, grad


def smooth(points):
    """Catmull-Rom display curves through every actual model knot."""
    result = []
    for j in range(STEPS - 1):
        p0 = points[:, max(0, j - 1)]
        p1 = points[:, j]
        p2 = points[:, j + 1]
        p3 = points[:, min(STEPS - 1, j + 2)]
        for u in (0.0, 0.25, 0.5, 0.75):
            result.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * u
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u
                )
            )
    result.append(points[:, -1])
    return np.stack(result, axis=1)


def curves(h):
    x = np.linspace(-7.85, 7.85, STEPS)
    y = h[:, :, 0] * 2.45 + h[:, :, 1] * 0.7 - 0.15
    return smooth(np.stack((np.broadcast_to(x, y.shape), y), axis=-1))


def pose(a, b):
    v = b - a
    return (
        Transform2D.translation(float(a[0]), float(a[1]))
        @ Transform2D.rotation(math.atan2(v[1], v[0]))
        @ Transform2D.scaling(max(float(np.linalg.norm(v)), 1e-7), 1)
    )


def along(path, q):
    value = float(np.clip(q, 0, 1)) * (len(path) - 1)
    j = min(int(value), len(path) - 2)
    return path[j] + (value - j) * (path[j + 1] - path[j])


def gate_paths(h, phase=0.0):
    lanes = np.arange(COUNT)
    old = h[:, -1]
    candidate = np.tanh(
        np.stack(
            (np.sin(lanes * 0.31 + phase) * 1.6, np.cos(lanes * 0.43 - phase) * 1.4),
            axis=-1,
        )
    )
    z = 1 / (1 + np.exp(-2.2 * np.sin(lanes * 0.27 + phase)))
    mixed = z[:, None] * old + (1 - z[:, None]) * candidate
    project = lambda a: a[:, 0] * 2.25 + a[:, 1] * 0.55
    ya, yb, ym = map(project, (old, candidate, mixed))
    q = np.linspace(0, 1, SAMPLES)
    blend = q * q * (3 - 2 * q)
    arch = np.sin(np.pi * q) ** 2
    a = (
        (1 - blend[None, :]) * ya[:, None]
        + blend[None, :] * ym[:, None]
        + 1.0 * arch[None, :]
    )
    b = (
        (1 - blend[None, :]) * yb[:, None]
        + blend[None, :] * ym[:, None]
        - 1.0 * arch[None, :]
    )
    x = np.broadcast_to(np.linspace(-7.85, 7.85, SAMPLES), a.shape)
    return np.stack((x, a - 0.15), -1), np.stack((x, b - 0.15), -1), z


def memory_states(phase=0.0):
    """Constructed LSTM gates; true cell/output recurrences, no training claim."""
    lane = np.arange(COUNT)[:, None] * 0.24
    t = np.arange(STEPS - 1)[None, :] * 0.49
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    forget = 0.72 + 0.25 * sigmoid(1.5 * np.sin(lane + t + phase))
    incoming = sigmoid(2 * np.cos(lane - t * 0.7 + phase))
    output = sigmoid(1.7 * np.sin(lane + t * 0.8 - phase))
    candidate = np.tanh(1.3 * np.sin(lane - t + phase))
    cell = np.zeros((COUNT, STEPS))
    hidden = np.zeros_like(cell)
    cell[:, 0] = 0.4 * np.sin(lane[:, 0])
    hidden[:, 0] = np.tanh(cell[:, 0])
    for j in range(STEPS - 1):
        cell[:, j + 1] = forget[:, j] * cell[:, j] + incoming[:, j] * candidate[:, j]
        hidden[:, j + 1] = output[:, j] * np.tanh(cell[:, j + 1])
    return cell, hidden, forget, incoming, output, candidate


def memory_paths(phase=0.0):
    cell, hidden, *_ = memory_states(phase)
    lane = (np.arange(COUNT) - (COUNT - 1) / 2) * 0.066
    x = np.broadcast_to(np.linspace(-7.85, 7.85, STEPS), cell.shape)
    memory = np.stack((x, lane[:, None] + 0.46 * cell + 1.05), axis=-1)
    emitted = np.stack((x + 0.4, lane[:, None] + 0.78 * hidden - 1.95), axis=-1)
    return smooth(memory), smooth(emitted)


def build(width=1920, height=1080, fps=60):
    s = Scene(canvas=Canvas(width, height, width / 19.2), fps=fps)
    rect(s, 0, 0, 19.2, 10.8, Color(0, 0, 0), -100)
    header = []

    def chapter(title, subtitle):
        nonlocal header
        if header:
            with s.parallel():
                for item in header:
                    item.fade_out(duration=0.35)
            for item in header:
                item.remove()
        header = [
            label(s, title, -8.15, 4.55, 0.29, Color(217, 226, 233), left=True),
            label(s, subtitle, 5.9, -4.55, 0.20, Color(141, 153, 164)),
        ]
        for item in header:
            item.opacity(to=0, duration=0)
        with s.parallel():
            for item in header:
                item.fade_in(duration=0.45)

    # A nested aperture opens into the same connected temporal strands.
    angle = np.linspace(0, 2 * np.pi, SAMPLES)
    rings = np.array(
        [
            np.stack(
                (
                    (1.4 + i * 0.07) * 1.8 * np.cos(angle + i * 0.065),
                    (1.4 + i * 0.07) * np.sin(angle + i * 0.065) - 0.15,
                ),
                -1,
            )
            for i in range(COUNT)
        ]
    )
    from zanim_scenes.inception_art import (
        append as prepend_inception,
        PREFIX_SECONDS as INCEPTION_SECONDS,
    )

    inception_handoff = prepend_inception(s, chapter, PALETTE)
    with s.parallel():
        for item in header:
            item.fade_out(duration=0.35)
    for item in header + inception_handoff:
        item.remove()
    header = []
    assert abs(s.duration - INCEPTION_SECONDS) < 1e-8
    bn_start = s.duration
    from zanim_scenes.batch_norm_art import (
        append as prepend_bn,
        PREFIX_SECONDS as BN_SECONDS,
    )

    bn_handoff = prepend_bn(s, chapter, PALETTE)
    with s.parallel():
        for item in header:
            item.fade_out(duration=0.35)
    for item in header + bn_handoff:
        item.remove()
    header = []
    assert abs(s.duration - bn_start - BN_SECONDS) < 1e-8
    residual_start = s.duration
    from zanim_scenes.residual_art import (
        append as prepend_residual,
        PREFIX_SECONDS as RESIDUAL_SECONDS,
    )

    residual_handoff = prepend_residual(s, chapter, PALETTE)
    with s.parallel():
        for item in header:
            item.fade_out(duration=0.35)
    for item in header + residual_handoff:
        item.remove()
    header = []
    assert abs(s.duration - residual_start - RESIDUAL_SECONDS) < 1e-8
    dense_start = s.duration
    from zanim_scenes.dense_art import append as prepend_dense, PREFIX_SECONDS

    handoff = prepend_dense(s, chapter, rings, PALETTE)
    with s.parallel():
        for item in header:
            item.fade_out(duration=0.35)
    for item in header:
        item.remove()
    header = []
    for item in handoff:
        item.remove()
    assert abs(s.duration - dense_start - PREFIX_SECONDS) < 1e-8
    current = rings.copy()
    edges = []
    for i in range(COUNT):
        color = PALETTE[i % 4]
        row = []
        for j in range(SAMPLES - 1):
            obj = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(color, 0.013),
                    transform=pose(current[i, j], current[i, j + 1]),
                    z_index=1,
                )
            )
            obj.opacity(to=0.46, duration=0)
            row.append(obj)
        edges.append(row)

    def particles(colors):
        result = []
        for i in range(COUNT):
            for k in range(2):
                c = colors[i % len(colors)]
                dot = s.add(
                    Circle(
                        0.055 if i % 7 == 0 else 0.037, style=Style.solid(c), z_index=5
                    )
                )
                tail = s.add(
                    Line((0, 0), (1, 0), style=Style.outline(c, 0.027), z_index=4)
                )
                dot.set_transform(to=Transform2D.translation(-30, 0))
                tail.set_transform(to=Transform2D.translation(-30, 0))
                result.append((i, k, dot, tail))
        return result

    forward = particles(PALETTE)
    phases = {}

    def animate(
        target,
        seconds,
        group,
        edge_group=None,
        direction=1,
        gradient=None,
        weights=None,
        start=None,
        batch=True,
    ):
        nonlocal current
        initial = current.copy() if start is None else start.copy()
        end = target.copy()
        phase = phases.get(id(group), 0.0)
        travel = seconds * 0.115
        selected_edges = edges if edge_group is None else edge_group

        def location(u, i, k, tail=False):
            q = (phase + i * 0.043 + k * 0.5 + direction * u * travel) % 1
            if tail:
                q = float(np.clip(q - direction * 0.048, 0, 1))
            blend = u * u * (3 - 2 * u)
            path = initial[i] + blend * (end[i] - initial[i])
            return along(path, q), q

        with s.parallel() if batch else nullcontext():
            for i, row in enumerate(selected_edges):
                for j, obj in enumerate(row):
                    obj.transform_function(
                        lambda u, i=i, j=j, initial=initial, end=end: pose(
                            initial[i, j] + u * (end[i, j] - initial[i, j]),
                            initial[i, j + 1] + u * (end[i, j + 1] - initial[i, j + 1]),
                        ),
                        duration=seconds,
                    )
            for i, k, dot, tail in group:

                def dot_pose(u, i=i, k=k):
                    p, q = location(u, i, k)
                    size = min(1, q * 18, (1 - q) * 18)
                    if gradient is not None:
                        n = np.linalg.norm(gradient[i], axis=-1)
                        g = float(np.interp(q, np.linspace(0, 1, STEPS), n))
                        size *= 2.4 * math.sqrt(g / max(float(np.max(n)), 1e-8))
                    if weights is not None:
                        size *= 0.35 + 1.1 * weights[i]
                    return Transform2D.translation(
                        *map(float, p)
                    ) @ Transform2D.scaling(max(size, 0.001))

                def tail_pose(u, i=i, k=k):
                    a, q = location(u, i, k, True)
                    b, _ = location(u, i, k)
                    if gradient is not None:
                        n = np.linalg.norm(gradient[i], axis=-1)
                        g = float(np.interp(q, np.linspace(0, 1, STEPS), n))
                        a = b + (a - b) * math.sqrt(g / max(float(np.max(n)), 1e-8))
                    return pose(a, b)

                dot.set_transform(to=dot_pose(0))
                tail.set_transform(to=tail_pose(0))
                dot.transform_function(dot_pose, duration=seconds, easing=Easing.LINEAR)
                tail.transform_function(
                    tail_pose, duration=seconds, easing=Easing.LINEAR
                )
        current = end
        phases[id(group)] = (phase + direction * travel) % 1

    chapter("循环神经网络 · 08.4", "状态沿时间交织。")
    _, h, g = trajectories()
    animate(rings, 2.0, forward)
    animate(curves(h), 4.5, forward)
    _, h, g = trajectories(0.7)
    animate(curves(h), 3.8, forward)
    chapter("通过时间反向传播 · 08.7", "梯度沿原路回传。")
    with s.parallel():
        for _, _, dot, tail in forward:
            dot.fade_out(duration=0.3)
            tail.fade_out(duration=0.3)
    for _, _, dot, tail in forward:
        dot.remove()
        tail.remove()
    backward = particles((GOLD, Color(245, 164, 100)))
    animate(current, 6.8, backward, direction=-1, gradient=g)
    with s.parallel():
        for _, _, dot, tail in backward:
            dot.fade_out(duration=0.3)
            tail.fade_out(duration=0.3)
    for _, _, dot, tail in backward:
        dot.remove()
        tail.remove()
    from zanim_scenes.gated_art import append

    append(s, current, edges, chapter)
    return s
