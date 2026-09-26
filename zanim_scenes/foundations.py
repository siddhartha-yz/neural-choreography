"""Native plots and computational paths for the introductory chapters."""

import importlib
import numpy as np
from zanim import Polyline, Style
from zanim_scenes.style import BLUE, CORAL, GOLD, MUTED, TEAL, label, rect
from zanim_scenes.mechanisms import start, remove, dot, line, vector

EPISODE_IDS = ("03.1", "03.4", "04.1", "04.4", "04.5", "04.6", "04.7", "04.8", "05.1")


def model(ep):
    return importlib.import_module("zanim_scenes.models.ep_" + ep.replace(".", "_"))


def curve(s, points, color=TEAL, width=0.035):
    return s.add(
        Polyline(
            [tuple(map(float, p)) for p in points], style=Style.outline(color, width)
        )
    )


def axes(s, xlim, ylim):
    def project(x, y):
        return (
            -1.6 + (float(x) - xlim[0]) / (xlim[1] - xlim[0]) * 9.1,
            -2.65 + (float(y) - ylim[0]) / (ylim[1] - ylim[0]) * 5.1,
        )

    y0 = max(ylim[0], min(0, ylim[1]))
    x0 = max(xlim[0], min(0, xlim[1]))
    line(s, project(xlim[0], y0), project(xlim[1], y0), MUTED)
    line(s, project(x0, ylim[0]), project(x0, ylim[1]), MUTED)
    for x in np.linspace(*xlim, 5):
        px, _ = project(x, y0)
        label(s, f"{x:g}", px, -3.03, 0.17, MUTED)
    for y in np.linspace(*ylim, 5):
        _, py = project(x0, y)
        label(s, f"{y:g}", -2.03, py, 0.17, MUTED)
    label(s, "x", 7.8, -3.03, 0.2, MUTED)
    label(s, "y", -2.03, 2.78, 0.2, MUTED)
    return project


def regression(ep, width, height, fps):
    d = model(ep)
    subtitle = {
        "03.1": "移动拟合直线，观察小批量梯度下降如何减少误差。",
        "04.4": "训练误差很低，并不意味着新样本预测更准确。",
        "04.5": "增大 L2 惩罚，让高次多项式的系数收缩。",
    }[ep]
    s = start(ep, width, height, fps, subtitle)
    if ep == "03.1":
        xs, ys = d.FEATURES.ravel(), d.TARGETS.ravel()
        stages = [
            (
                f"SGD 第 {i} 步",
                lambda x, w=w, b=b: float(w.item()) * x + float(b.item()),
                f"MSE = {d.mean_squared_loss(w, b):.3f}",
                f"w = {w.item():.3f} · b = {b.item():.3f}",
            )
            for i, (w, b) in enumerate(d.TRAINING_STATES)
        ]
        ylim = (-3, 4)
        hold = None
        wait = 1.25
    elif ep == "04.4":
        xs, ys = d.TRAIN_X, d.TRAIN_Y
        ylim = (-2, 2)
        hold = (d.TRAVELER_X, d.TRAVELER_Y)
        wait = 5
        stages = [
            (
                "一次拟合 · 容量偏低",
                lambda x: np.polyval(d.LOW_COEFFS, x),
                f"训练 MSE = {d.LOW_TRAIN_LOSS:.3f}",
                f"留出点误差² = {d.LOW_VAL_LOSS:.3f}",
            ),
            (
                "六次拟合 · 追随噪声",
                lambda x: np.polyval(d.HIGH_COEFFS, x),
                f"训练 MSE = {d.HIGH_TRAIN_LOSS:.3f}",
                f"留出点误差² = {d.HIGH_VAL_LOSS:.3f}",
            ),
            ("真实生成曲线", d.true_function, "训练点包含噪声", "金色留出点未参与拟合"),
        ]
    else:
        xs, ys = d.FEATURES, d.TARGETS
        ylim = (-2, 2)
        hold = (d.X_HOLD, d.Y_HOLD)
        wait = 3
        stages = []
        for unit in (0, 0.3, 0.6, 1):
            penalty = d.displayed_lambda(unit)
            w = d.ridge_weights(penalty)
            stages.append(
                (
                    f"λ = {penalty:.4f}",
                    lambda x, w=w: d.polynomial_value(x, w),
                    f"系数范数 = {d.weight_norm(w):.3f}",
                    f"留出残差 = {d.held_out_residual(w):+.3f}",
                )
            )
    p = axes(s, (-3, 3), ylim)
    for x, y in zip(xs, ys):
        dot(s, *p(x, y), BLUE)
    if hold:
        dot(s, *p(*hold), GOLD)
    label(
        s,
        "蓝色：训练样本" + (" · 金色：留出样本" if hold else ""),
        2.8,
        -3.65,
        0.22,
        MUTED,
    )
    panel = []
    previous = None
    trace_x = hold[0] if hold else float(xs[3])
    marker = dot(s, *p(trace_x, float(np.clip(stages[0][1](trace_x), *ylim))), CORAL)
    for index, (name, fn, metric, note) in enumerate(stages):
        remove(panel)
        panel = [
            label(s, name, -6.2, 2.4, 0.30, TEAL),
            label(s, metric, -6.2, 1.5, 0.27),
            label(s, note, -6.2, 0.7, 0.21, MUTED),
        ]
        # Do not connect out-of-range polynomial branches across the viewport.
        groups = []
        group = []
        for x in np.linspace(-2.7, 2.7, 181):
            y = fn(float(x))
            if ylim[0] <= y <= ylim[1]:
                group.append(p(x, y))
            elif group:
                groups.append(group)
                group = []
        if group:
            groups.append(group)
        if previous:
            remove(previous)
        previous = [
            curve(s, g, TEAL if index < len(stages) - 1 else GOLD)
            for g in groups
            if len(g) > 1
        ]
        if hold:
            y = fn(hold[0])
            panel.append(
                line(s, p(*hold), p(hold[0], float(np.clip(y, *ylim))), CORAL, 0.025)
            )
        else:
            batch = d.MINIBATCHES[min(index, len(d.MINIBATCHES) - 1)]
            panel += [dot(s, *p(xs[i], ys[i]), GOLD) for i in batch]
        marker.move(to=p(trace_x, float(np.clip(fn(trace_x), *ylim))), duration=0.8)
        s.wait(wait)
    label(
        s,
        {
            "03.1": "每次更新只用一小批样本；数字来自实际 SGD 迭代。",
            "04.4": "这里比较一个留出点；它不能代表完整的泛化评估。",
            "04.5": "截距不受惩罚；本例直接求解正则化拟合。",
        }[ep],
        -8.3,
        -4.5,
        0.24,
        left=True,
    )
    s.wait(max(4, 22 - s.duration))
    return s


def pipeline(ep, width, height, fps):
    d = model(ep)
    s = start(ep, width, height, fps, "沿着计算路径，跟随同一个输入逐步变成输出。")
    if ep == "03.4":
        stages = [
            ("输入 x", d.INPUT_X, "两个输入特征"),
            ("线性得分", d.LOGITS, "z = Wx + b"),
            ("类别概率", d.PROBABILITIES, "p = softmax(z)"),
        ]
        conclusion = (
            "最大概率对应类别 " + str(d.MAX_CLASS + 1) + "；各类别概率之和为 1。"
        )
    else:
        stages = [
            ("输入 x", d.INPUT_X, "输入层"),
            ("线性变换", d.PREACTIVATIONS, "z = W₁x + b₁"),
            ("ReLU", d.HIDDEN, "h = max(z, 0)"),
            ("输出", d.OUTPUTS, "o = W₂h + b₂"),
        ]
        conclusion = "负的预激活被 ReLU 截为 0；这里仅做前向计算。"
    xs = np.linspace(-1.5, 7.0, len(stages))
    panel = []
    columns = []
    if ep == "05.1":
        curve(
            s,
            [(-2.5, -2.35), (8, -2.35), (8, 2.55), (-2.5, 2.55), (-2.5, -2.35)],
            BLUE,
            0.018,
        )
        label(s, "Sequential：按顺序组合这些层", 2.75, 2.65, 0.25, BLUE)
    for i, (name, values, formula) in enumerate(stages):
        if i:
            line(s, (float(xs[i - 1]) + 0.72, 0), (float(xs[i]) - 0.72, 0), BLUE)
        label(s, name, float(xs[i]), 1.5, 0.25, TEAL)
        rect(s, float(xs[i]), 0, 1.6, 2.1)
    for i, (name, values, formula) in enumerate(stages):
        remove(panel)
        panel = [
            label(s, name, -6.2, 2.4, 0.34, TEAL),
            label(s, formula, -6.2, 1.4, 0.27),
        ]
        panel += vector(s, np.round(values, 3), -6.2, -0.1)
        if i:
            traveler = dot(s, float(xs[i - 1]), 0, GOLD)
            traveler.move(to=(float(xs[i]), 0), duration=1.2)
            traveler.remove()
        columns += vector(s, np.round(values, 3), float(xs[i]), 0)
        s.wait(3.8)
    if ep == "03.4":
        label(s, "指数变换 → 归一化 → 三类概率", 2.7, -2.2, 0.25, MUTED)
    label(s, conclusion, -8.3, -4.5, 0.24, left=True)
    s.wait(max(4, 23 - s.duration))
    return s


def dropout(width, height, fps):
    d = model("04.6")
    s = start(
        "04.6", width, height, fps, "训练时随机屏蔽单元；推理时使用保留概率缩放。"
    )
    xs = np.linspace(-1.4, 6.6, 4)
    label(s, "经典 dropout：p = 0.5", 2.6, 2.5, 0.29, TEAL)
    label(s, "本例采用训练不缩放、推理乘 (1 − p) 的约定。", 2.6, -3.3, 0.2, MUTED)
    panel = []
    nodes = []
    steps = (
        [("原始激活", d.HIDDEN.ravel(), "四个 ReLU 单元")]
        + [
            (f"训练掩码 {i + 1}", v, "掩码 " + "".join(str(int(x)) for x in d.MASKS[i]))
            for i, v in enumerate(d.H_TRAIN)
        ]
        + [("推理", d.H_INFER, "所有单元保留，激活乘 0.5")]
    )
    for name, values, note in steps:
        remove(panel + nodes)
        nodes = []
        panel = [
            label(s, name, -6.2, 2.4, 0.33, TEAL),
            label(s, note, -6.2, 1.4, 0.22, MUTED),
        ]
        panel += vector(s, np.round(values, 3), -6.2, -0.35)
        for i, (x, v) in enumerate(zip(xs, values)):
            color = TEAL if v else MUTED
            nodes += [
                rect(s, float(x), 0, 1.5, 1.5),
                label(s, f"{v:.2f}", float(x), 0, 0.35, color),
                label(s, f"h{i + 1}", float(x), -1.2, 0.23, MUTED),
            ]
            if v == 0:
                nodes.append(
                    line(s, (float(x) - 0.5, -0.5), (float(x) + 0.5, 0.5), CORAL, 0.04)
                )
            else:
                traveler = dot(s, float(x), 1.4, TEAL)
                traveler.move(to=(float(x), 0.8), duration=0.2)
                traveler.remove()
        s.wait(2.5)
    label(s, "随机屏蔽会改变每次的激活；推理保持确定性。", -8.3, -4.5, 0.25, left=True)
    s.wait(4)
    return s


def backward(width, height, fps):
    d = model("04.7")
    g = d.GRAPH
    s = start("04.7", width, height, fps, "前向得到损失；反向沿同一路径应用链式法则。")
    xs = (-1.6, 0.6, 2.8, 5.0, 7.2)
    names = ("x", "z", "h = ReLU(z)", "预测 ŷ", "损失 L")
    vals = (d.INPUT_X, g["z"], g["h"], g["yhat"], g["loss"])
    grads = (g["dL_dx"], g["dL_dz"], g["dL_dh"], g["dL_dyhat"], 1.0)
    for i, x in enumerate(xs):
        label(s, names[i], x, 1.65, 0.23, TEAL)
        vector(s, np.round(np.atleast_1d(vals[i]), 3) + 0.0, x, 0.6)
        if i:
            line(s, (xs[i - 1] + 0.7, 0.6), (x - 0.7, 0.6), BLUE)
    panel = [
        label(s, "前向计算", -6.2, 2.4, 0.32, TEAL),
        label(s, "L = (ŷ − y)²", -6.2, 1.4, 0.3),
        label(s, f"y = {d.TARGET_Y:.2f}", -6.2, 0.5, 0.27, MUTED),
    ]
    for i in range(1, 5):
        p = dot(s, xs[i - 1], 0.6, GOLD)
        p.move(to=(xs[i], 0.6), duration=1)
        p.remove()
    s.wait(2)
    remove(panel)
    panel = [
        label(s, "反向求导", -6.2, 2.4, 0.32, CORAL),
        label(s, "局部梯度沿路径相乘", -6.2, 1.4, 0.26),
        label(s, "ReLU 的负区间阻断梯度", -6.2, 0.4, 0.22, MUTED),
    ]
    for i in reversed(range(5)):
        label(
            s,
            "∂L / ∂"
            + ("ŷ" if i == 3 else ("L" if i == 4 else ("h" if i == 2 else names[i]))),
            xs[i],
            -1.2,
            0.20,
            CORAL,
        )
        vector(s, np.round(np.atleast_1d(grads[i]), 3) + 0.0, xs[i], -2.15)
        if i:
            p = dot(s, xs[i], 0.6, CORAL)
            p.move(to=(xs[i - 1], 0.6), duration=1)
            p.remove()
        s.wait(1)
    label(
        s, "显示的是同一个计算图的实际数值；参数未更新。", -8.3, -4.5, 0.25, left=True
    )
    s.wait(6)
    return s


def saturation(width, height, fps):
    d = model("04.8")
    s = start(
        "04.8", width, height, fps, "Sigmoid 接近饱和时，局部导数变小，梯度逐层衰减。"
    )
    xs = np.linspace(-1.2, 7.0, 4)
    panel = []
    numbers = []
    marks = []
    for i, x in enumerate(xs):
        label(s, f"第 {i + 1} 层", float(x), 2.5, 0.24, TEAL)
        if i:
            line(s, (float(xs[i - 1]) + 0.75, 0), (float(x) - 0.75, 0), BLUE)
    for name, z, h, grad in [
        ("饱和的参数", d.SAT_Z, d.SAT_H, d.SAT_GRAD_H),
        ("较温和的参数", d.GOOD_Z, d.GOOD_H, d.GOOD_GRAD_H),
    ]:
        remove(panel + numbers + marks)
        numbers = []
        marks = []
        panel = [
            label(s, name, -6.2, 2.4, 0.32, TEAL),
            label(s, "σ′(z) = σ(z)(1 − σ(z))", -6.2, 1.4, 0.23),
            label(s, "最右端传入梯度为 1", -6.2, 0.4, 0.23, MUTED),
        ]
        for i, x in enumerate(xs):
            numbers += [
                label(s, f"z = {z[i]:.2f}", float(x), 1.5, 0.23, MUTED),
                label(s, f"h = {h[i]:.3f}", float(x), 0.6, 0.25),
            ]
        s.wait(3)
        for i in reversed(range(4)):
            # Linear length scale; exact small gradients remain readable as text.
            marks += [
                rect(s, float(xs[i]), -0.9, 1.5, 0.05, BLUE),
                rect(
                    s,
                    float(xs[i]) - 0.75 + float(grad[i]) * 0.75,
                    -0.9,
                    max(float(grad[i]) * 1.5, 0.003),
                    0.15,
                    CORAL,
                ),
                label(s, f"{grad[i]:.5f}", float(xs[i]), -1.6, 0.22, CORAL),
            ]
            s.wait(1)
        label_handle = label(s, "柱长：|∂L/∂h| · 同一线性尺度", 2.8, -2.6, 0.22, MUTED)
        numbers.append(label_handle)
        s.wait(2)
    label(
        s,
        "比较两组固定参数，不是训练过程；多层乘积仍可能使梯度变小。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(4)
    return s


def build(ep, width=1920, height=1080, fps=60):
    if ep in ("03.1", "04.4", "04.5"):
        return regression(ep, width, height, fps)
    if ep in ("03.4", "04.1", "05.1"):
        return pipeline(ep, width, height, fps)
    return {"04.6": dropout, "04.7": backward, "04.8": saturation}[ep](
        width, height, fps
    )
