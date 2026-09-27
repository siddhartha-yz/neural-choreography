"""Query-driven routing and Gaussian attention pooling on a continuous stage."""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import BLUE, GOLD, PINK, ease, pose
from zanim_scenes.style import label
from zanim_scenes.models import ep_10_1 as attention
from zanim_scenes.models import ep_10_2 as pooling


def selection(u):
    phase = min(float(u) * 3, 2.999999)
    j = int(phase)
    q = (1 - ease(phase - j)) * attention.QUERIES[j] + ease(
        phase - j
    ) * attention.QUERIES[(j + 1) % 3]
    weights = attention.softmax_rows((q @ attention.KEYS.T)[None, :])[0]
    return q, weights, float(weights @ attention.VALUES[:, 0])


def bezier(a, b, q, bend):
    return (1 - q) * a + q * b + np.array([bend * math.sin(math.pi * q), 0])


def append(s, old, chapter):
    objects = []

    def clear(items):
        with s.parallel():
            for item in items:
                item.fade_out(duration=0.6)
        for item in items:
            item.remove()
        items.clear()

    def dot(x, y, r, color, filled=True):
        item = s.add(
            Circle(
                r,
                style=Style.solid(color) if filled else Style.outline(color, 0.025),
                transform=Transform2D.translation(float(x), float(y)),
                z_index=7,
            )
        )
        objects.append(item)
        return item

    def line(a, b, color, width=0.02):
        item = s.add(
            Line(
                (0, 0),
                (1, 0),
                style=Style.outline(color, width),
                transform=pose(np.array(a), np.array(b)),
                z_index=2,
            )
        )
        objects.append(item)
        return item

    def text(t, x, y, c):
        objects.append(label(s, t, x, y, 0.23, c))

    def xf(xy, size=1):
        return Transform2D.translation(*map(float, xy)) @ Transform2D.scaling(
            max(0.001, float(size))
        )

    def apply(item, fn, duration):
        item.set_transform(to=fn(0))
        item.transform_function(fn, duration=duration, easing=Easing.LINEAR)

    chapter("注意力线索 · 10.1", "查询改变，信息重新汇集。")
    clear(old)
    keys = [np.array([x, 2.25]) for x in (-5, 0, 5)]
    query = np.array([-6, -0.15])
    output = np.array([0, -2.65])
    text("键 / 值", -7.3, 2.25, BLUE)
    text("查询", -7.3, -0.15, PINK)
    text("汇合", 1.1, -2.65, GOLD)
    qring = dot(*query, 0.38, PINK, False)
    qcomponents = [dot(-0.13, 0.0, 0.09, PINK), dot(0.13, 0.0, 0.09, PINK)]
    out = dot(*output, 0.28, GOLD)
    halos = []
    rays = []
    flows = []
    for j, a in enumerate(keys):
        dot(*a, 0.22, BLUE, False)
        halos.append(dot(*a, 0.36, BLUE, False))
        for c in range(2):
            dot(
                a[0] + (c - 0.5) * 0.17,
                a[1],
                0.055 + 0.035 * abs(attention.KEYS[j, c]),
                BLUE,
            )
        rays.append(line(query, a, PINK, 0.012))
        for k in range(3):
            d = dot(*query, 0.045, PINK)
            flows.append(("query", j, k, d))
        for k in range(10):
            d = dot(*a, 0.055, GOLD)
            flows.append(("value", j, k, d))
        line(a, output, GOLD, 0.014).opacity(to=0.28, duration=0)

    @lru_cache(maxsize=32)
    def data(u):
        return selection(u)

    with s.parallel():
        apply(qring, lambda u: xf(query, 1 + 0.12 * math.sin(u * 6 * math.pi)), 9.6)
        for c, item in enumerate(qcomponents):
            apply(
                item,
                lambda u, c=c: xf(
                    query + np.array([(c - 0.5) * 0.24, 0.1 * data(u)[0][c]]),
                    0.6 + 0.4 * abs(data(u)[0][c]),
                ),
                9.6,
            )
        apply(out, lambda u: xf(output, 0.7 + 0.55 * data(u)[2]), 9.6)
        for j, item in enumerate(halos):
            apply(item, lambda u, j=j: xf(keys[j], 0.65 + 1.6 * data(u)[1][j]), 9.6)
        for j, item in enumerate(rays):
            item.transform_function(
                lambda u, j=j: (
                    pose(query, keys[j])
                    @ Transform2D.scaling(1, 0.25 + 3 * data(u)[1][j])
                ),
                duration=9.6,
                easing=Easing.LINEAR,
            )
        for kind, j, k, item in flows:

            def flowing(u, kind=kind, j=j, k=k):
                p = (u * 3 + k / (3 if kind == "query" else 10)) % 1
                a, b = (query, keys[j]) if kind == "query" else (keys[j], output)
                size = (
                    0.35 if kind == "query" else math.sqrt(data(u)[1][j]) * 2
                ) * math.sin(math.pi * p)
                return xf((1 - p) * a + p * b, size)

            apply(item, flowing, 9.6)
    chapter("注意力汇聚 · 10.2", "靠近查询的值，贡献更多。")
    clear(objects)
    xs = np.linspace(-7.2, 7.2, 9)
    anchors = [np.array([x, 1.8]) for x in xs]
    text("查询", -8, 3.3, PINK)
    text("值", -8, 1.8, BLUE)
    text("汇聚", -8, -2.4, GOLD)
    marker = dot(xs[0], 3.3, 0.2, PINK, False)
    output_dot = dot(0, -2.4, 0.25, GOLD)
    halos = [dot(*a, 0.3, BLUE, False) for a in anchors]
    for j, a in enumerate(anchors):
        dot(*a, 0.08 + 0.035 * pooling.VALUES[j], BLUE)

    @lru_cache(maxsize=32)
    def state(u):
        q = 0.4 + 4 * (0.5 - 0.5 * math.cos(2 * math.pi * u))
        w = pooling.attention_weights(q)
        target = np.array([(q - 2.4) * 2.4, -2.4])
        return q, w, target, pooling.nw_predict(q)

    strands = []
    moving = []
    for j, a in enumerate(anchors):
        for lane in (-1, 0, 1):
            bend = (j - 4) * 0.17 + lane * 0.16
            for k in range(20):
                obj = line(
                    bezier(a, np.array([0, -2.4]), k / 20, bend),
                    bezier(a, np.array([0, -2.4]), (k + 1) / 20, bend),
                    GOLD,
                    0.012,
                )
                obj.opacity(to=0.5, duration=0)
                strands.append((j, k, bend, obj))
            for k in range(4):
                moving.append((j, k, bend, dot(*a, 0.05, GOLD)))
    with s.parallel():
        apply(marker, lambda u: xf(np.array([(state(u)[0] - 2.4) * 3.6, 3.3])), 10.8)
        apply(output_dot, lambda u: xf(state(u)[2], 0.6 + 0.3 * state(u)[3]), 10.8)
        for j, item in enumerate(halos):
            apply(
                item,
                lambda u, j=j: xf(anchors[j], 0.5 + 2.5 * math.sqrt(state(u)[1][j])),
                10.8,
            )
        for j, k, bend, item in strands:
            item.transform_function(
                lambda u, j=j, k=k, bend=bend: (
                    pose(
                        bezier(anchors[j], state(u)[2], k / 20, bend),
                        bezier(anchors[j], state(u)[2], (k + 1) / 20, bend),
                    )
                    @ Transform2D.scaling(1, 0.2 + 4 * state(u)[1][j])
                ),
                duration=10.8,
                easing=Easing.LINEAR,
            )
        for j, k, bend, item in moving:

            def flowing(u, j=j, k=k, bend=bend):
                q = (u * 4 + k / 4) % 1
                return xf(
                    bezier(anchors[j], state(u)[2], q, bend),
                    3 * math.sqrt(state(u)[1][j]) * math.sin(math.pi * q),
                )

            apply(item, flowing, 10.8)
    s.wait(0.8)
