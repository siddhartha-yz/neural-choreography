"""Recognizable gated topology, with continuous particle choreography."""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Line, Color, Style, Transform2D, Easing
from zanim_scenes.style import label
from zanim_scenes.models import ep_08_4 as model

BLUE = Color(88, 196, 221)
GOLD = Color(255, 220, 112)
PINK = Color(202, 152, 215)
N = 28
K = 49
Q = np.linspace(0, 1, K)


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def values(kind, phase, overrides=None):
    overrides = overrides or {}
    lane = np.arange(N) * 2 * np.pi / N
    old = np.stack((0.8 * np.sin(lane), 0.7 * np.cos(lane)), axis=-1)
    x = np.stack((np.sin(lane * 0.7 + 0.5), np.cos(lane * 1.3)), axis=-1)
    if kind == "gru":
        r = 0.5 + 0.44 * math.sin(phase + 0.8)
        z = 0.5 + 0.44 * math.cos(phase)
        r = overrides.get("r", r)
        z = overrides.get("z", z)
        candidate = np.tanh(
            x @ model.WEIGHT_XH.T + (r * old) @ model.WEIGHT_HH.T + model.BIAS.ravel()
        )
        mixed = z * old + (1 - z) * candidate
        return dict(old=old, x=x, candidate=candidate, mixed=mixed, gates=(z, 1 - z, r))
    f = 0.5 + 0.44 * math.cos(phase)
    i = 0.5 + 0.44 * math.sin(phase + 0.4)
    o = 0.5 + 0.44 * math.sin(phase - 1.1)
    f, i, o = (overrides.get("f", f), overrides.get("i", i), overrides.get("o", o))
    previous = 1.3 * old
    candidate = np.tanh(
        x @ model.WEIGHT_XH.T + old @ model.WEIGHT_HH.T + model.BIAS.ravel()
    )
    cell = f * previous + i * candidate
    hidden = o * np.tanh(cell)
    return dict(
        old=previous, candidate=candidate, cell=cell, hidden=hidden, gates=(f, i, o)
    )


def frame(kind, phase):
    d = values(kind, phase)
    q = Q[None, :]
    lane = (np.arange(N) - (N - 1) / 2) * 0.018
    offset = lane[:, None]

    def path(x, y):
        return np.stack((np.broadcast_to(x, (N, K)), y + np.zeros((N, K))), axis=-1)

    if kind == "gru":
        join = ease((q - 0.57) / 0.28)
        upper = 1.65 * (1 - join)
        lower = -2.15 * (1 - join)
        old = d["old"][:, 0, None]
        candidate = d["candidate"][:, 0, None]
        mixed = d["mixed"][:, 0, None]
        top = path(
            -7.8 + 15.6 * q, upper + 0.44 * ((1 - join) * old + join * mixed) + offset
        )
        source = (1 - ease((q - 0.24) / 0.16)) * d["x"][:, 0, None] + ease(
            (q - 0.24) / 0.16
        ) * candidate
        bottom = path(
            -7.8 + 15.6 * q,
            lower + 0.44 * ((1 - join) * source + join * mixed) + offset,
        )
        reset = path(
            -5 + 3.2 * q,
            1.65
            - 3.8 * ease(q)
            + 0.44 * ((1 - ease(q)) * old + ease(q) * candidate)
            + offset,
        )
        return [top, bottom, reset], d
    mix = ease((q - 0.40) / 0.16)
    previous = d["old"][:, 0, None]
    cell = d["cell"][:, 0, None]
    candidate = d["candidate"][:, 0, None]
    f, i, o = d["gates"]
    retained = (1 - ease((q - 0.20) / 0.08)) * previous + ease(
        (q - 0.20) / 0.08
    ) * f * previous
    top = path(
        -7.8 + 15.6 * q, 1.65 + 0.40 * ((1 - mix) * retained + mix * cell) + offset
    )
    # Writing joins the memory highway; its last segment is the same c_t.
    wx = -6.2 + 14 * q
    join = ease((q - 0.27) / 0.22)
    write = path(
        wx, -2.7 + 4.35 * join + 0.40 * ((1 - join) * candidate + join * cell) + offset
    )
    # The output branch taps the updated memory, leaving the highway intact.
    output = path(
        2 + 5.8 * q,
        1.65
        - 3.45 * ease(q / 0.72)
        + 0.40 * ((1 - ease(q)) * cell + ease(q) * d["hidden"][:, 0, None])
        + offset,
    )
    return [top, write, output], d


def pose(a, b):
    v = b - a
    return (
        Transform2D.translation(*map(float, a))
        @ Transform2D.rotation(math.atan2(v[1], v[0]))
        @ Transform2D.scaling(max(1e-7, float(np.linalg.norm(v))), 1)
    )


def at(path, q):
    t = np.clip(q, 0, 1) * (K - 1)
    j = min(int(t), K - 2)
    return path[j] + (t - j) * (path[j + 1] - path[j])


def append(s, start, existing, chapter):
    colors = (BLUE, GOLD, PINK)
    paths = [start.copy() for _ in range(3)]
    edges = [existing]
    for color in colors[1:]:
        rows = []
        for i in range(N):
            row = []
            for j in range(K - 1):
                item = s.add(
                    Line(
                        (0, 0),
                        (1, 0),
                        style=Style.outline(color, 0.014),
                        transform=pose(start[i, j], start[i, j + 1]),
                        z_index=2,
                    )
                )
                item.opacity(to=0.30, duration=0)
                row.append(item)
            rows.append(row)
        edges.append(rows)
    particles = []
    for branch, color in enumerate(colors):
        group = []
        for i in range(N):
            for k in range(2):
                dot = s.add(
                    Circle(
                        0.047,
                        style=Style.solid(color),
                        transform=Transform2D.translation(-30, 0),
                        z_index=6,
                    )
                )
                group.append((i, k, dot))
        particles.append(group)
    gates = []
    for color in colors:
        blades = []
        for sign in (-1, 1):
            blade = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(color, 0.06),
                    transform=Transform2D.translation(-30, 0),
                    z_index=8,
                )
            )
            blades.append((sign, blade))
        gates.append(blades)
    labels = []
    clock = 0.0

    def annotations(kind):
        nonlocal labels
        if labels:
            with s.parallel():
                for item in labels:
                    item.fade_out(duration=0.25)
            for item in labels:
                item.remove()
        specs = (
            [
                ("hₜ₋₁", -7.5, 2.8, BLUE),
                ("xₜ", -7.5, -3.15, GOLD),
                ("候选", -1.8, -3.15, GOLD),
                ("hₜ", 7.6, 0.95, BLUE),
                ("z", 1.1, 2.9, BLUE),
                ("1−z", 1.1, -3.3, GOLD),
                ("重置 r", -4.5, -0.6, PINK),
            ]
            if kind == "gru"
            else [
                ("cₜ₋₁", -7.5, 2.9, BLUE),
                ("cₜ", 7.6, 2.9, BLUE),
                ("候选写入", -6.1, -3.6, GOLD),
                ("hₜ", 7.7, -2.9, PINK),
                ("遗忘 f", -4.0, 3.2, BLUE),
                ("写入 i", -1.4, -3.5, GOLD),
                ("输出 o", 5.4, -2.8, PINK),
                ("tanh", 3.1, 0.45, PINK),
            ]
        )
        labels = [label(s, t, x, y, 0.24, c) for t, x, y, c in specs]
        for item in labels:
            item.opacity(to=0, duration=0)
        with s.parallel():
            for item in labels:
                item.fade_in(duration=0.3)

    def gate_centers(kind):
        return (
            [(1.1, 1.65), (1.1, -2.15), (-3.4, -0.3)]
            if kind == "gru"
            else [(-4, 1.65), (-1.44, -1.66), (5.4, -1.48)]
        )

    def run(kind, phase0, phase1, duration, morph=False):
        nonlocal paths, clock
        initial = [p.copy() for p in paths]
        start_clock = clock

        @lru_cache(maxsize=16)
        def state(u):
            target, data = frame(kind, phase0 + (phase1 - phase0) * u)
            if morph:
                w = ease(u)
                target = [a + w * (b - a) for a, b in zip(initial, target)]
            return target, data

        def blade_pose(u, branch, sign):
            _, data = state(u)
            x, y = gate_centers(kind)[branch]
            gap = 0.09 + 0.78 * data["gates"][branch]
            return pose(np.array([x, y + sign * gap]), np.array([x, y + sign * 1.05]))

        with s.parallel():
            for branch, rows in enumerate(edges):
                for i, row in enumerate(rows):
                    for j, item in enumerate(row):
                        item.transform_function(
                            lambda u, b=branch, i=i, j=j: pose(
                                state(u)[0][b][i, j], state(u)[0][b][i, j + 1]
                            ),
                            duration=duration,
                            easing=Easing.LINEAR,
                        )
            for branch, group in enumerate(particles):
                for i, k, dot in group:

                    def motion(u, b=branch, i=i, k=k):
                        pp, data = state(u)
                        q = (
                            start_clock + i * 0.037 + k * 0.5 + u * duration * 0.115
                        ) % 1
                        xy = at(pp[b][i], q)
                        gain = 1.0
                        threshold = (
                            (0.57, 0.57, 0.5)[b]
                            if kind == "gru"
                            else (0.245, 0.34, 0.58)[b]
                        )
                        if q > threshold:
                            gain = math.sqrt(data["gates"][b])
                        # Old/write particles persist on c_t; output alone is gated by o.
                        size = max(0.001, min(1, q * 20, (1 - q) * 20) * gain)
                        return Transform2D.translation(
                            *map(float, xy)
                        ) @ Transform2D.scaling(size)

                    dot.set_transform(to=motion(0))
                    dot.transform_function(
                        motion, duration=duration, easing=Easing.LINEAR
                    )
            for branch, blades in enumerate(gates):
                for sign, blade in blades:
                    blade.set_transform(to=blade_pose(0, branch, sign))
                    blade.transform_function(
                        lambda u, b=branch, sg=sign: blade_pose(u, b, sg),
                        duration=duration,
                        easing=Easing.LINEAR,
                    )
        paths = state(1)[0]
        clock = (clock + duration * 0.115) % 1

    chapter("门控循环单元 · 09.1", "旧状态与候选，此消彼长。")
    run("gru", 0, 0, 3.0, morph=True)
    annotations("gru")
    run("gru", 0, 2 * math.pi, 9.0)
    chapter("长短期记忆网络 · 09.2", "主干延续，侧路写入与读出。")
    with s.parallel():
        for item in labels:
            item.fade_out(duration=0.3)
        for blades in gates:
            for _, blade in blades:
                blade.fade_out(duration=0.3)
    for item in labels:
        item.remove()
    labels = []
    run("lstm", 0, 0, 3.8, morph=True)
    annotations("lstm")
    with s.parallel():
        for blades in gates:
            for _, blade in blades:
                blade.fade_in(duration=0.3)
    run("lstm", 0, 2 * math.pi, 10.0)

    with s.parallel():
        for item in labels:
            item.fade_out(duration=0.3)
        for blades in gates:
            for _, blade in blades:
                blade.fade_out(duration=0.3)
    for item in labels:
        item.remove()
    for blades in gates:
        for _, blade in blades:
            blade.remove()
    from zanim_scenes.recurrent_layers import append as append_layers

    append_layers(s, paths, edges, particles, clock, chapter)
