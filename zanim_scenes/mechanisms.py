"""Native 2D scenes for mechanisms best explained on a number line or graph."""

import importlib
import numpy as np
from zanim import Circle, Line, Polyline, Style, Transform2D
from scripts.episode_info import EPISODES
from zanim_scenes.style import (
    BLUE,
    CORAL,
    GOLD,
    MUTED,
    TEAL,
    INK,
    label,
    matrix,
    rect,
    new_scene,
    intro,
    header,
)

EPISODE_IDS = ("07.5", "07.6", "11.2")


def model(ep):
    return importlib.import_module("zanim_scenes.models.ep_" + ep.replace(".", "_"))


def line(s, a, b, color=MUTED, width=0.02):
    return s.add(Line(a, b, style=Style.outline(color, width)))


def dot(s, x, y, color=TEAL):
    return s.add(
        Circle(
            0.10,
            style=Style.solid(color),
            transform=Transform2D.translation(x, y),
            z_index=10,
        )
    )


def remove(items):
    for item in items:
        item.remove()


def vector(s, values, x, y, gap=0.58):
    values = np.asarray(values).reshape(-1)
    items = []
    for i, value in enumerate(values):
        yy = y + ((len(values) - 1) / 2 - i) * gap
        items += [rect(s, x, yy, 1.35, gap - 0.07), label(s, f"{value:g}", x, yy, 0.23)]
    return items


def start(ep, width, height, fps, subtitle):
    title, explanation = EPISODES[ep]
    s = new_scene(width, height, fps)
    intro(s, title, ep, explanation)
    header(s, title, ep, subtitle)
    return s


def batch_norm(width, height, fps):
    d = model("07.5")
    s = start("07.5", width, height, fps, "同一批样本，先统一尺度，再学习缩放与平移。")
    px = lambda v: 2.7 + (v - 1) * 1.4
    line(s, (px(-2.5), 0), (px(4.2), 0))
    for tick in range(-2, 5):
        line(s, (px(tick), -0.09), (px(tick), 0.09))
        label(s, str(tick), px(tick), -0.42, 0.2, MUTED)
    label(s, "每个点对应同一个样本；金色点持续跟随。", 2.7, -2.3, 0.22, MUTED)
    dots = [
        dot(s, px(v), 0, GOLD if i == d.TRAVELER_INDEX else TEAL)
        for i, v in enumerate(d.BATCH_X)
    ]
    stages = [
        ("原始批次", d.BATCH_X, "x", f"μ = {d.BATCH_MU:.2f} · σ = {d.BATCH_SIGMA:.3f}"),
        ("减去均值", d.BATCH_X - d.BATCH_MU, "x − μ", "把批次中心移到 0"),
        ("除以标准差", d.X_HAT, "x̂ = (x − μ) / σ", "均值 0，方差 1"),
        ("缩放与平移", d.Y_OUT, "y = 1.5x̂ + 0.4", "γ = 1.5 · β = 0.4"),
    ]
    panel = []
    numbers = []
    for index, (name, values, formula, note) in enumerate(stages):
        remove(panel + numbers)
        panel = [
            label(s, name, -6.2, 2.4, 0.37, TEAL),
            label(s, formula, -6.2, 1.45, 0.31),
            label(s, note, -6.2, 0.6, 0.23, MUTED),
        ]
        panel += vector(s, np.round(values, 3), -6.2, -1.45)
        if index:
            with s.parallel():
                for point, value in zip(dots, values):
                    point.move(to=(px(value), 0), duration=1.5)
        numbers = [
            label(
                s, f"{v:.2f}", px(v), 0.6, 0.2, GOLD if i == d.TRAVELER_INDEX else TEAL
            )
            for i, v in enumerate(values)
        ]
        s.wait(3)
    label(
        s,
        "规范化用当前批次统计量；这里展示的是训练时的一步。",
        -8.3,
        -4.5,
        0.25,
        left=True,
    )
    s.wait(4)
    return s


def residual(width, height, fps):
    d = model("07.6")
    s = start(
        "07.6", width, height, fps, "主路计算 F(x)，旁路直接保留 x，汇合时逐项相加。"
    )
    for x, text in [(-1.5, "输入 x"), (1.4, "F(x)"), (6.8, "ReLU")]:
        rect(s, x, 1.4, 2.0, 1.15)
        label(s, text, x, 1.4, 0.29)
    s.add(
        Circle(
            0.36,
            style=Style.outline(GOLD, 0.035),
            transform=Transform2D.translation(4.25, 1.4),
        )
    )
    label(s, "+", 4.25, 1.4, 0.3, GOLD)
    for a, b in [
        ((-0.5, 1.4), (0.4, 1.4)),
        ((2.4, 1.4), (3.89, 1.4)),
        ((4.61, 1.4), (5.8, 1.4)),
    ]:
        line(s, a, b, TEAL, 0.035)
    for a, b in [
        ((-1.5, 0.82), (-1.5, -1.3)),
        ((-1.5, -1.3), (4.25, -1.3)),
        ((4.25, -1.3), (4.25, 1.04)),
    ]:
        line(s, a, b, GOLD, 0.035)
    label(s, "恒等旁路：x 不变", 1.2, -1.85, 0.26, GOLD)
    label(s, "F(x) + x 后再经过 ReLU", 3.8, -2.8, 0.23, MUTED)
    panel = []

    def show(name, data, note):
        nonlocal panel
        remove(panel)
        panel = [
            label(s, name, -6.4, 2.3, 0.34, TEAL),
            label(s, note, -6.4, 1.45, 0.22, MUTED),
        ]
        panel += vector(s, np.round(data, 4), -6.4, 0.15, 0.8)

    show("输入向量 x", d.INPUT_X, "固定权重，展示一次前向计算")
    main = dot(s, -1.5, 1.4)
    shortcut = dot(s, -1.5, 1.4, GOLD)
    s.wait(2)
    with s.parallel():
        main.move(to=(1.4, 1.4), duration=1.5)
        shortcut.move(to=(-1.5, -1.3), duration=1.5)
    main.remove()
    shortcut.move(to=(4.25, -1.3), duration=1.5)
    show("主路输出 F(x)", d.RESIDUAL_F, "仿射 → ReLU → 仿射")
    s.wait(4)
    main = dot(s, 2.4, 1.4)
    with s.parallel():
        main.move(to=(4.25, 1.4), duration=1.5)
        shortcut.move(to=(4.25, 1.4), duration=1.5)
    main.remove()
    shortcut.remove()
    show("逐项相加：F(x) + x", d.SUMMED, "维度保持不变，不是通道拼接")
    s.wait(4)
    main = dot(s, 4.6, 1.4)
    main.move(to=(6.8, 1.4), duration=1.5)
    main.remove()
    show("残差块输出 y", d.OUTPUT_Y, "y = ReLU(F(x) + x)")
    label(
        s, "残差块学习需要补充的变化，旁路保留原始输入。", -8.3, -4.5, 0.26, left=True
    )
    s.wait(5)
    return s


def convexity(width, height, fps):
    d = model("11.2")
    s = start("11.2", width, height, fps, "比较曲线上的点，与端点连线上的点。")
    xy = lambda x, y: (3.2 + x * 1.7, -0.2 + y * 1.1)
    line(s, xy(-2.15, 0), xy(2.15, 0))
    line(s, xy(0, -1.3), xy(0, 2.2))
    for x in (-2, -1, 1, 2):
        label(s, str(x), *xy(x, -0.3), size=0.19, color=MUTED)
    label(s, "横轴 x", 7.1, -0.55, 0.19, MUTED)
    label(s, "金色：端点连线　青绿：函数曲线", 3.2, -2.5, 0.22, MUTED)
    panel = []
    chart = []
    for name, fn, a, b, fz, chord, positive in [
        ("凸函数", d.convex_f, d.F_A, d.F_B, d.F_Z, d.F_CHORD, True),
        ("非凸函数", d.nonconvex_g, d.G_A, d.G_B, d.G_Z, d.G_CHORD, False),
    ]:
        remove(panel + chart)
        panel = [
            label(s, name, -6.3, 2.3, 0.36, TEAL),
            label(
                s, "f(x) = 0.5x²" if positive else "g(x) = cos(πx)", -6.3, 1.45, 0.31
            ),
            label(s, "λ = 0.4 · z = 0.4a + 0.6b", -6.3, 0.65, 0.23, MUTED),
        ]
        chart = [
            s.add(
                Polyline(
                    [xy(float(x), float(fn(x))) for x in d.X_GRID],
                    style=Style.outline(TEAL, 0.035),
                )
            ),
            line(s, xy(d.XA, a), xy(d.XB, b), GOLD, 0.035),
        ]
        chart += [dot(s, *xy(d.XA, a), BLUE), dot(s, *xy(d.XB, b), BLUE)]
        point = dot(s, *xy(d.XB, b), GOLD)
        chart.append(point)
        s.wait(1)
        point.move(to=xy(d.Z, chord), duration=2)
        chart += [
            dot(s, *xy(d.Z, fz), TEAL),
            line(s, xy(d.Z, chord), xy(d.Z, fz), CORAL, 0.04),
        ]
        sign = "≤" if positive else ">"
        panel += [
            label(s, f"曲线值 {fz:.3f}", -6.3, -0.35, 0.27, TEAL),
            label(s, f"{sign} 连线值 {chord:.3f}", -6.3, -1.1, 0.27, GOLD),
            label(
                s,
                "这组端点满足凸性不等式" if positive else "找到反例：曲线高于连线",
                -6.3,
                -2,
                0.23,
                MUTED,
            ),
        ]
        s.wait(5)
    label(
        s,
        "凸性要求：对任意两个端点、任意 λ∈[0,1]，曲线都不高于连线。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(4)
    return s


def build(ep, width=1920, height=1080, fps=60):
    return {"07.5": batch_norm, "07.6": residual, "11.2": convexity}[ep](
        width, height, fps
    )
