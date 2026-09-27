"""Encoding, autoregressive generation, and a width-two search frontier."""

import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import BLUE, GOLD, PINK, ease
from zanim_scenes.style import label
from zanim_scenes.models import ep_09_6 as encoder
from zanim_scenes.models import ep_09_8 as search


def beam_steps(depth=4, width=2):
    frontier = [((), 0.0)]
    result = []
    for _ in range(depth):
        candidates = []
        for prefix, score in frontier:
            probabilities = search.P_STEP1 if not prefix else search.P_STEP2[prefix[-1]]
            candidates.extend(
                (prefix + (token,), score + math.log(float(p)))
                for token, p in enumerate(probabilities)
            )
        candidates.sort(key=lambda item: (-item[1], item[0]))
        frontier = candidates[:width]
        result.append((candidates, frontier))
    return result


def append(s, old, edges, chapter):
    chapter("编码器—解码器 · 09.6", "收进状态，再逐个生成。")
    with s.parallel():
        for item in old:
            item.fade_out(duration=0.7)
        for group in edges:
            for row in group:
                for item in row:
                    item.fade_out(duration=0.7)
    for item in old:
        item.remove()
    objects = []

    def dot(x, y, r, color, solid=True):
        item = s.add(
            Circle(
                r,
                style=Style.solid(color) if solid else Style.outline(color, 0.025),
                transform=Transform2D.translation(float(x), float(y)),
                z_index=8,
            )
        )
        objects.append(item)
        return item

    def line(a, b, color, alpha=0.45, width=0.025):
        item = s.add(
            Line(tuple(a), tuple(b), style=Style.outline(color, width), z_index=3)
        )
        item.opacity(to=alpha, duration=0)
        objects.append(item)
        return item

    def arrow(a, b, color):
        a, b = np.array(a), np.array(b)
        line(a, b, color)
        v = (b - a) / np.linalg.norm(b - a)
        n = np.array([-v[1], v[0]])
        line(b, b - 0.14 * v + 0.07 * n, color)
        line(b, b - 0.14 * v - 0.07 * n, color)

    def move(item, a, b, seconds=0.5, size=1):
        item.transform_function(
            lambda u: (
                Transform2D.translation(
                    *map(float, np.array(a) + ease(u) * (np.array(b) - a))
                )
                @ Transform2D.scaling(size)
            ),
            duration=seconds,
            easing=Easing.LINEAR,
        )

    def annotation(text, x, y, color):
        objects.append(label(s, text, x, y, 0.24, color))

    ex = np.linspace(-7.3, -1.8, 7)
    dx = np.linspace(1.8, 7.3, 5)
    tokens = tuple(encoder.SOURCE[t % 3] * (0.8 + 0.06 * t) for t in range(7))
    states = encoder.encode(tokens)
    outputs, decoded = encoder.decode(states[-1], 5)
    annotation("编码", -4.5, -1, BLUE)
    annotation("状态", 0, -1, PINK)
    annotation("解码", 4.5, -1, GOLD)
    nodes = []
    for xs, color in ((ex, BLUE), (dx, GOLD)):
        for j, x in enumerate(xs):
            nodes.append(dot(x, 0, 0.25, color, False))
            if j + 1 < len(xs):
                arrow((x + 0.31, 0), (xs[j + 1] - 0.31, 0), color)
    dot(0, 0, 0.43, PINK, False)
    arrow((ex[-1] + 0.31, 0), (-0.5, 0), BLUE)
    arrow((0.5, 0), (dx[0] - 0.31, 0), PINK)
    sources = []
    for j, x in enumerate(ex):
        for c in range(2):
            sources.append(
                (
                    j,
                    c,
                    dot(
                        x + (c - 0.5) * 0.13,
                        2.3,
                        0.085 + 0.055 * abs(float(tokens[j][c, 0])),
                        BLUE,
                    ),
                )
            )
    carrier = dot(ex[0], 0, 0.14, BLUE)
    # Consume a token, update the state, then send that state to the next cell.
    for j, x in enumerate(ex):
        with s.parallel():
            for t, c, item in sources:
                if t == j:
                    move(
                        item,
                        (x + (c - 0.5) * 0.13, 2.3),
                        (x + (c - 0.5) * 0.13, 0),
                        0.36,
                    )
        with s.parallel():
            nodes[j].transform_function(
                lambda u, x=x: (
                    Transform2D.translation(float(x), 0)
                    @ Transform2D.scaling(1 + 0.3 * math.sin(math.pi * u))
                ),
                duration=0.25,
            )
            carrier.set_transform(
                to=Transform2D.translation(float(x), 0)
                @ Transform2D.scaling(0.7 + 0.3 * float(np.linalg.norm(states[j + 1])))
            )
        if j + 1 < len(ex):
            move(carrier, (x, 0), (ex[j + 1], 0), 0.28)
    move(carrier, (ex[-1], 0), (0, 0), 0.7)
    context = dot(0, 0, 0.2, PINK)
    context.opacity(to=0, duration=0)
    with s.parallel():
        carrier.fade_out(duration=0.25)
        context.fade_in(duration=0.25)
    move(context, (0, 0), (dx[0], 0), 0.65)
    feedback = None
    for j, x in enumerate(dx):
        if feedback is not None:
            # The preceding generated output is an input to the next decoder step.
            move(feedback, (dx[j - 1], 2.3), (x, 0), 0.55)
        with s.parallel():
            nodes[7 + j].transform_function(
                lambda u, x=x: (
                    Transform2D.translation(float(x), 0)
                    @ Transform2D.scaling(1 + 0.3 * math.sin(math.pi * u))
                ),
                duration=0.3,
            )
        emitted = [
            dot(
                x + (c - 0.5) * 0.16,
                0,
                0.08 + 0.045 * abs(float(outputs[j][c, 0])),
                GOLD,
            )
            for c in range(2)
        ]
        with s.parallel():
            for c, item in enumerate(emitted):
                move(item, (x + (c - 0.5) * 0.16, 0), (x + (c - 0.5) * 0.16, 2.3), 0.48)
        if j + 1 < len(dx):
            feedback = dot(x, 2.3, 0.075, GOLD)
            line((x, 2.3), (dx[j + 1], 0), GOLD, 0.18)
            move(context, (x, 0), (dx[j + 1], 0), 0.3)
    s.wait(0.7)
    chapter("束搜索 · 09.8", "分叉，保留，再生长。")
    with s.parallel():
        for item in objects:
            item.fade_out(duration=0.6)
    for item in objects:
        item.remove()
    objects.clear()
    annotation("保留 2 条", -6.5, 3.3, GOLD)
    positions = {(): np.array([-7.4, 0.0])}
    dot(-7.4, 0, 0.18, GOLD)
    surviving_edges = {(): []}
    for step, (candidates, kept) in enumerate(beam_steps()):
        kept_paths = {p for p, _ in kept}
        order = sorted(
            candidates, key=lambda item: (positions[item[0][:-1]][1], item[0][-1])
        )
        ys = np.linspace(-2.65, 2.65, len(order))
        current = []
        x = -3.7 + step * 3.7
        for (path, score), y in zip(order, ys):
            origin = positions[path[:-1]]
            target = np.array([x, y])
            positions[path] = target
            branch = line(origin, target, BLUE, 0.3)
            branch.opacity(to=0, duration=0)
            orb = dot(*origin, 0.12, BLUE)
            current.append((path, score, origin, target, branch, orb))
        with s.parallel():
            for path, score, origin, target, branch, orb in current:
                branch.fade_in(duration=0.8)
                move(orb, origin, target, 0.8)
        with s.parallel():
            for path, score, origin, target, branch, orb in current:
                if path in kept_paths:
                    branch.opacity(to=0.65, duration=0.45)
                    orb.transform_function(
                        lambda u, target=target: (
                            Transform2D.translation(*map(float, target))
                            @ Transform2D.scaling(1 + 0.55 * ease(u))
                        ),
                        duration=0.45,
                    )
                else:
                    branch.opacity(to=0.07, duration=0.45)
                    orb.fade_out(duration=0.45)
        for path, score, origin, target, branch, orb in current:
            if path in kept_paths:
                dot(*target, 0.23, GOLD, False)
                surviving_edges[path] = surviving_edges[path[:-1]] + [(origin, target)]
        s.wait(0.6)
    winner = beam_steps()[-1][1][0][0]
    highlights = [line(a, b, GOLD, 0.95, 0.045) for a, b in surviving_edges[winner]]
    for highlight in highlights:
        highlight.opacity(to=0, duration=0)
    with s.parallel():
        for highlight in highlights:
            highlight.fade_in(duration=0.8)
    s.wait(1.2)
    from zanim_scenes.attention_art import append as append_attention

    append_attention(s, objects, chapter)
