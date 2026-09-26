"""The recurrent map acts on an entire state plane, not merely a plotted point."""

import math
import numpy as np
from zanim import Scene, Canvas, Color, Line, Circle, Polygon, Style, Transform2D
from zanim_scenes.style import label, rect
from zanim_scenes.rnn_geometry import formula
from zanim_scenes.models import ep_08_4 as d

BLACK = Color(0, 0, 0)
BLUE = Color(88, 196, 221)
GREEN = Color(131, 193, 103)
YELLOW = Color(255, 231, 99)
WHITE = Color(235, 235, 235)
GREY = Color(125, 132, 145)
GRID = Color(28, 37, 48)
SCALE = 2.5
OFFSET = np.array([0.0, -0.95])


def pos(v):
    return np.asarray(v) * SCALE + OFFSET


def pose(a, b):
    a, b = pos(a), pos(b)
    delta = b - a
    return (
        Transform2D.translation(*map(float, a))
        @ Transform2D.rotation(math.atan2(delta[1], delta[0]))
        @ Transform2D.scaling(max(float(np.linalg.norm(delta)), 1e-8), 1)
    )


def segment(s, a, b, color=GRID, thickness=0.012, z=0):
    return s.add(
        Line(
            (0, 0),
            (1, 0),
            style=Style.outline(color, thickness),
            transform=pose(a, b),
            z_index=z,
        )
    )


def stages(points, token):
    memory = np.asarray(points) @ d.WEIGHT_HH.T
    shifted = memory + (d.WEIGHT_XH @ token + d.BIAS).ravel()
    return memory, shifted, np.tanh(shifted)


def build(width=1920, height=1080, fps=60):
    s = Scene(canvas=Canvas(width, height, width / 19.2), fps=fps)
    rect(s, 0, 0, 19.2, 10.8, BLACK, -100)
    label(s, "循环神经网络 · 08.4", -8.55, 4.75, 0.23, GREY, left=True)
    # A single compact equation stays out of the geometric stage.
    formula(s, "h_t =", -4.5, 3.98, 0.4, YELLOW)
    equation = [
        formula(s, "W_h h_(t-1)", -0.85, 3.98, 0.4, BLUE),
        formula(s, "+ W_x x_t + b", 2.05, 3.98, 0.4, GREEN),
    ]
    formula(s, 'tanh "("', -3.05, 3.98, 0.4, WHITE)
    formula(s, '")"', 4.25, 3.98, 0.4, WHITE)
    # The expression reads as an operation strip, with a marker identifying it.
    for item in equation:
        item.opacity(to=0.65, duration=0)
    backdrop = []
    for value in np.arange(-3, 3.01, 0.5):
        backdrop.append(segment(s, (value, -1.3), (value, 1.65)))
    for value in np.arange(-1, 1.51, 0.5):
        backdrop.append(segment(s, (-3.35, value), (3.35, value)))
    backdrop.append(segment(s, (-3.35, 0), (3.35, 0), Color(66, 72, 80), 0.015))
    backdrop.append(segment(s, (0, -1.3), (0, 1.65), Color(66, 72, 80), 0.015))

    axis = np.linspace(-1, 1, 13)
    points = np.array([(x, y) for y in axis for x in axis])
    n = len(axis)
    pairs = [(j * n + i, j * n + i + 1) for j in range(n) for i in range(n - 1)]
    pairs += [(j * n + i, (j + 1) * n + i) for j in range(n - 1) for i in range(n)]
    edges = [
        segment(s, points[a], points[b], Color(58, 143, 168), 0.018, 2)
        for a, b in pairs
    ]
    dots = [
        s.add(
            Circle(
                0.026,
                style=Style.solid(BLUE),
                transform=Transform2D.translation(*map(float, pos(p))),
                z_index=4,
            )
        )
        for p in points
    ]
    selected = (n * n) // 2
    # Two faint halos preserve the selected state's identity through every map.
    halos = [
        s.add(
            Circle(
                radius,
                style=Style.solid(color),
                transform=Transform2D.translation(*map(float, pos(points[selected]))),
                z_index=z,
            )
        )
        for radius, color, z in [
            (0.19, Color(48, 43, 16), 5),
            (0.115, Color(119, 105, 35), 6),
            (0.063, YELLOW, 7),
        ]
    ]
    alternate = (n - 3) * n + (n - 3)
    pink = s.add(
        Circle(
            0.065,
            style=Style.solid(Color(232, 143, 194)),
            transform=Transform2D.translation(*map(float, pos(points[alternate]))),
            z_index=8,
        )
    )
    caption = None
    counter = None

    def say(text, color=WHITE):
        nonlocal caption
        if caption is not None:
            caption.fade_out(duration=0.15)
            caption.remove()
        caption = label(s, text, 0, -4.7, 0.31, color)
        caption.opacity(to=0, duration=0)
        caption.fade_in(duration=0.2)

    def warp(target, seconds):
        nonlocal points
        start = points.copy()
        end = target.copy()
        with s.parallel():
            for edge, (a, b) in zip(edges, pairs):
                edge.transform_function(
                    lambda u, a=a, b=b, start=start, end=end: pose(
                        start[a] + u * (end[a] - start[a]),
                        start[b] + u * (end[b] - start[b]),
                    ),
                    duration=seconds,
                )
            for dot, p in zip(dots, end):
                dot.move(to=tuple(pos(p)), duration=seconds)
            for halo in halos:
                halo.move(to=tuple(pos(end[selected])), duration=seconds)
            pink.move(to=tuple(pos(end[alternate])), duration=seconds)
        points = end

    say("每一个点，都是一种可能的记忆。")
    s.wait(1.4)
    say("黄与粉：两种不同的过去。", YELLOW)
    s.wait(0.9)
    for t, token in enumerate(d.TOKENS, 1):
        if counter is not None:
            counter.remove()
        counter = formula(s, f"x_{t}", 8.2, 4.05, 0.33, GREEN)
        memory, shifted, bounded = stages(points, token)
        say("旧记忆：一起旋转、压缩。", BLUE)
        equation[0].opacity(to=1, duration=0)
        equation[1].opacity(to=0.35, duration=0)
        warp(memory, 1.65 if t == 1 else 0.7)
        s.wait(0.3)
        say("新输入：推动整个状态空间。", GREEN)
        equation[0].opacity(to=0.35, duration=0)
        equation[1].opacity(to=1, duration=0)
        shift = (d.WEIGHT_XH @ token + d.BIAS).ravel()
        start = points.mean(axis=0)
        arrow = segment(s, start, start + shift, GREEN, 0.035, 3)
        tip = s.add(
            Polygon(
                ((0, 0), (-0.16, 0.07), (-0.16, -0.07)),
                style=Style.solid(GREEN),
                z_index=6,
                transform=Transform2D.translation(*map(float, pos(start + shift)))
                @ Transform2D.rotation(math.atan2(shift[1], shift[0])),
            )
        )
        warp(shifted, 1.55 if t == 1 else 0.7)
        arrow.fade_out(duration=0.2)
        arrow.remove()
        tip.remove()
        say("tanh：越靠外，压得越紧。", YELLOW)
        equation[1].opacity(to=0.35, duration=0)
        warp(bounded, 1.8 if t == 1 else 0.9)
        s.wait(0.55)
    # Same inputs, different initial states: visibly converging histories.
    say("三次输入相同，记忆差异越来越小。")
    s.wait(1.4)
    # A declared display-only magnification; the recurrence is already complete.
    magnification = min(12.0, 2.6 / float(np.ptp(points, axis=0).max()))
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    counter.remove()
    label(s, f"局部放大 ×{magnification:.1f}", 7.4, 4.05, 0.24, GREY)
    with s.parallel():
        for item in backdrop:
            item.fade_out(duration=0.5)
    warp((points - center) * magnification, 1.65)
    segment(s, points[selected], points[alternate], YELLOW, 0.018, 4)
    for item in equation:
        item.opacity(to=1, duration=0.15)
    say("尚未完全重合：过去仍留下痕迹。")
    s.wait(2.8)
    return s
