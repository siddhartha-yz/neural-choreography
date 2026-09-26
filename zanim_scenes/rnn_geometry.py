"""RNN explained through vector addition and a two-dimensional state plane."""

import math
import numpy as np
from zanim import Scene, Canvas, Color, Line, Polygon, Circle, Math, Style, Transform2D
from zanim_scenes.style import label, rect
from zanim_scenes.mechanisms import remove
from zanim_scenes.models import ep_08_4 as d
from zanim_scenes.rnn_motion import ensemble_states

BLACK = Color(0, 0, 0)
WHITE = Color(238, 238, 238)
BLUE = Color(88, 196, 221)
YELLOW = Color(255, 231, 99)
GREEN = Color(131, 193, 103)
GREY = Color(150, 155, 165)
GRID = Color(28, 47, 65)
ORIGIN = np.array([-3.45, -0.35])
SCALE = 2.15


def p(v):
    return ORIGIN + np.asarray(v, dtype=float) * SCALE


def formula(s, source, x, y, height=0.43, color=WHITE):
    obj = Math(source, color=color, z_index=20)
    obj.transform = Transform2D.scaling(height / obj.bounds().height)
    obj.move_to((x, y))
    return s.add(obj)


def line(s, a, b, color=GRID, width=0.015):
    return s.add(Line(tuple(a), tuple(b), style=Style.outline(color, width)))


def point(s, v, color=YELLOW, r=0.065):
    return s.add(
        Circle(
            r,
            style=Style.solid(color),
            transform=Transform2D.translation(*map(float, p(v))),
            z_index=10,
        )
    )


def vector(s, origin, start, end, color, duration=1.0, mover=None):
    """Animate an arrow's endpoint; arrowhead size stays fixed during a morph."""
    origin = np.asarray(origin, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    shaft = s.add(Line((0, 0), (1, 0), style=Style.outline(color, 0.036), z_index=5))
    tip = s.add(
        Polygon(
            ((0, 0), (-0.15, 0.07), (-0.15, -0.07)), style=Style.solid(color), z_index=6
        )
    )

    def shaft_pose(u):
        v = (start + (end - start) * u) * SCALE
        length = max(float(np.linalg.norm(v)), 1e-6)
        return (
            Transform2D.translation(*map(float, p(origin)))
            @ Transform2D.rotation(math.atan2(v[1], v[0]))
            @ Transform2D.scaling(length, 1)
        )

    def tip_pose(u):
        v = start + (end - start) * u
        return (
            Transform2D.translation(*map(float, p(origin + v)))
            @ Transform2D.rotation(math.atan2(v[1], v[0]))
            @ Transform2D.scaling(min(1, float(np.linalg.norm(v)) * 10))
        )

    shaft.set_transform(to=shaft_pose(0))
    tip.set_transform(to=tip_pose(0))
    with s.parallel():
        shaft.transform_function(shaft_pose, duration=duration)
        tip.transform_function(tip_pose, duration=duration)
        if mover is not None:
            mover.move(to=tuple(p(origin + end)), duration=duration)
    return [shaft, tip]


def parts(t):
    old = d.HIDDEN_STATES[t - 1].ravel()
    memory = (d.WEIGHT_HH @ old).ravel()
    incoming = (d.WEIGHT_XH @ d.TOKENS[t - 1] + d.BIAS).ravel()
    return memory, incoming, memory + incoming, d.HIDDEN_STATES[t].ravel()


def build(width=1920, height=1080, fps=60):
    s = Scene(canvas=Canvas(width, height, width / 19.2), fps=fps)
    rect(s, 0, 0, 19.2, 10.8, BLACK, -100)
    label(s, "循环神经网络 · 08.4", -8.45, 4.72, 0.22, GREY, left=True)
    label(s, "记忆，是怎样被更新的？", 0, 3.9, 0.43, WHITE)
    # The equation is split into consistently colored mathematical terms.
    formula(s, "h_t", -3.1, 2.9, 0.46, YELLOW)
    formula(s, '= tanh "("', -1.65, 2.9, 0.46)
    formula(s, "W_h h_(t-1)", 0.25, 2.9, 0.46, BLUE)
    formula(s, "+", 1.95, 2.9, 0.4)
    formula(s, "W_x x_t + b", 3.65, 2.9, 0.46, GREEN)
    formula(s, '")"', 5.18, 2.9, 0.46)
    for tick in np.arange(-1.5, 1.26, 0.5):
        line(s, p((tick, -1.5)), p((tick, 1.25)))
        line(s, p((-1.5, tick)), p((1.5, tick)))
    line(s, p((-1.6, 0)), p((1.65, 0)), GREY, 0.019)
    line(s, p((0, -1.55)), p((0, 1.3)), GREY, 0.019)
    for tick in (-1, 1):
        label(s, str(tick), float(p((tick, 0))[0]), -0.68, 0.18, GREY)
        label(s, str(tick), -3.8, float(p((0, tick))[1]), 0.18, GREY)
    formula(s, "h^((1))", 0.3, -0.35, 0.28, GREY)
    formula(s, "h^((2))", -3.82, 2.34, 0.28, GREY)
    label(s, "状态空间", -3.45, -4.05, 0.23, GREY)
    label(s, "点的位置，就是两个状态分量的数值。", -3.45, -4.52, 0.2, GREY)
    panel = []
    arrows = []
    history = []

    def explain(step, heading, math_source, note, color=WHITE):
        nonlocal panel
        if panel:
            with s.parallel():
                for item in panel:
                    item.fade_out(duration=0.2)
            remove(panel)
        panel = [
            label(s, f"时间步 {step}", 4.5, 1.68, 0.22, GREY),
            label(s, heading, 4.5, 0.92, 0.32, color),
            formula(s, math_source, 4.5, -0.1, 0.42, color),
            label(s, note, 4.5, -1.15, 0.21, GREY),
        ]
        for obj in panel:
            obj.opacity(to=0, duration=0)
        with s.parallel():
            for obj in panel:
                obj.fade_in(duration=0.3)

    moving = point(s, [0, 0])
    s.wait(0.8)
    for t in (1, 2, 3):
        remove(arrows)
        arrows = []
        memory, incoming, total, new = parts(t)
        explain(
            t,
            "先变换上一刻的记忆",
            "W_h h_(t-1)",
            "蓝色向量：历史状态经过线性变换。",
            BLUE,
        )
        moving.move(to=tuple(p(memory)), duration=0.85)
        if np.linalg.norm(memory) > 1e-8:
            arrows += vector(s, [0, 0], [0, 0], memory, BLUE, 0.7)
        else:
            panel.append(formula(s, "h_0 = vec(0,0)", 4.5, -2, 0.36, BLUE))
            s.wait(0.7)
        explain(
            t,
            "把当前输入，接到向量末端",
            "+ W_x x_t + b",
            "绿色向量：新输入与偏置的贡献。",
            GREEN,
        )
        arrows += vector(s, memory, [0, 0], incoming, GREEN, 1.2, mover=moving)
        explain(
            t,
            "最后，用 tanh 更新状态",
            "h_t = tanh(z_t)",
            "对两个分量分别作用，结果落在 −1 到 1。",
            YELLOW,
        )
        remove(arrows)
        arrows = vector(s, [0, 0], total, new, YELLOW, 1.2, mover=moving)
        history.append(point(s, new, YELLOW, 0.047))
        if t > 1:
            history.append(
                line(s, p(d.HIDDEN_STATES[t - 1].ravel()), p(new), YELLOW, 0.025)
            )
        panel.append(
            formula(
                s, f"h_{t} = vec({new[0]:.3f}, {new[1]:.3f})", 4.5, -2.15, 0.50, YELLOW
            )
        )
        s.wait(0.8)
    remove(arrows)
    explain(
        3,
        "同一套规则，不同的记忆",
        "h_t = f(x_t, h_(t-1))",
        "48 条独立序列使用同一组权重。",
        WHITE,
    )
    _, states = ensemble_states()
    dots = [point(s, [0, 0], BLUE, 0.025) for _ in range(47)]
    for t in (1, 2, 3):
        with s.parallel():
            for i, obj in enumerate(dots, 1):
                obj.move(to=tuple(p(states[i, t])), duration=0.8)
        for i in range(1, 48):
            line(s, p(states[i, t - 1]), p(states[i, t]), Color(38, 83, 104), 0.013)
        s.wait(0.12)
    # Restate the selected path clearly over the ensemble, using the same data.
    for t in (1, 2, 3):
        line(s, p(states[0, t - 1]), p(states[0, t]), YELLOW, 0.036)
        point(s, states[0, t], YELLOW, 0.065)
    label(s, "黄线：刚才追踪的序列", 4.5, -2.3, 0.23, YELLOW)
    label(s, "蓝线：另外 47 条独立序列", 4.5, -2.85, 0.21, BLUE)
    label(s, "轨迹交叉不表示状态交换。", 4.5, -3.55, 0.2, GREY)
    s.wait(2.5)
    return s
