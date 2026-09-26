"""Recurrent states, gates, backward paths and beam search in native Zanim."""

import importlib
import numpy as np
from zanim import Box3D, Transform3D, Vec3
from zanim_scenes.style import BLUE, CORAL, GOLD, MUTED, PANEL, TEAL, Color, label, rect
from zanim_scenes.mechanisms import start, remove, dot, line, vector
from zanim_scenes.attention import project

EPISODE_IDS = ("08.4", "08.7", "09.1", "09.2", "09.3", "09.4", "09.6", "09.8")


def model(ep):
    return importlib.import_module("zanim_scenes.models.ep_" + ep.replace(".", "_"))


def fmt(values):
    return "(" + ", ".join(f"{v:.3f}" for v in np.asarray(values).ravel()) + ")"


def bars(s, values, x=0.0, color=TEAL, scale=1.8):
    values = np.asarray(values).ravel()
    objects = []
    moves = []
    for i, value in enumerate(values):
        z = (i - (len(values) - 1) / 2) * 0.85
        height = max(abs(float(value)) * scale, 0.003)
        base = s.add(
            Box3D(
                Vec3(0.48, 0.035, 0.60),
                color=BLUE,
                transform=Transform3D.translation(x, 0, z),
            )
        )
        bar = s.add(
            Box3D(
                Vec3(0.4, height, 0.5),
                color=color,
                transform=Transform3D.translation(x, 0, z)
                @ Transform3D.scaling(1, 0.001, 1),
            )
        )
        moves.append((bar, Transform3D.translation(x, float(value) * scale / 2, z)))
        objects += [base, bar]
    with s.parallel():
        for bar, pose in moves:
            bar.transform(to=pose, duration=0.8)
    return objects


def node(s, x, y, name, value=None, color=TEAL):
    objects = [rect(s, x, y, 1.25, 0.76), label(s, name, x, y, 0.22, color)]
    if value is not None:
        objects.append(label(s, fmt(value), x, y + 0.64, 0.19, color))
    return objects


def rnn(width, height, fps):
    from zanim_scenes.rnn_flow import build

    return build(width, height, fps)


def bptt(width, height, fps):
    d = model("08.7")
    s = start("08.7", width, height, fps, "先沿时间正向计算，再从末端误差反向求梯度。")
    xs = np.linspace(-1.7, 7.1, 6)
    for i, x in enumerate(xs):
        node(s, float(x), 0.6, f"h{i + 1}")
        label(s, f"x = {d.INPUTS[i, 0]:.2f}", float(x), -1.0, 0.2, MUTED)
        line(s, (float(x), -0.6), (float(x), 0.2), BLUE)
        if i:
            line(s, (float(xs[i - 1]) + 0.64, 0.6), (float(x) - 0.64, 0.6))
    panel = [label(s, "正向：逐步计算状态", -6.2, 2.4, 0.31, TEAL)]
    for i, x in enumerate(xs):
        traveler = dot(s, float(x), -0.6)
        traveler.move(to=(float(x), 0.6), duration=0.55)
        traveler.remove()
        label(s, f"{d.HIDDEN[i + 1, 0]:.3f}", float(x), 1.25, 0.2, TEAL)
        s.wait(0.5)
    remove(panel)
    panel = [
        label(s, "反向：从最后一步开始", -6.2, 2.4, 0.3, CORAL),
        label(s, "L = ½(h₆ − y)²", -6.2, 1.5, 0.3),
        label(s, f"y = {d.TARGET.item():.2f}", -6.2, 0.85, 0.24, MUTED),
    ]
    gradients = []
    for i in range(5, -1, -1):
        gradients.append(
            label(s, f"{d.GRADIENT_H[i, 0]:.4f}", float(xs[i]), -0.25, 0.18, CORAL)
        )
        if i:
            traveler = dot(s, float(xs[i]), 0.6, CORAL)
            traveler.move(to=(float(xs[i - 1]), 0.6), duration=0.8)
            traveler.remove()
        s.wait(0.45)
    s.wait(1)
    remove(panel + gradients)
    panel = [
        label(s, "截断反传：最近 3 步", -6.2, 2.4, 0.29, GOLD),
        label(s, "在截断边界停止更早的梯度传播", -6.2, 1.6, 0.21, MUTED),
    ]
    cutoff = float((xs[2] + xs[3]) / 2)
    line(s, (cutoff, -1.5), (cutoff, 1.9), GOLD, 0.04)
    for i in (5, 4, 3):
        label(s, f"∂L/∂h = {d.GRADIENT_H[i, 0]:.4f}", float(xs[i]), -2.0, 0.17, CORAL)
        s.wait(1)
    label(
        s, "截断减少反传范围；完整反传会经过更早的时间步。", -8.3, -4.5, 0.25, left=True
    )
    s.wait(5)
    return s


def gru(width, height, fps):
    d = model("09.1")
    s = start("09.1", width, height, fps, "聚焦更新门：旧状态与候选状态按比例混合。")
    label(s, "候选状态已给定；本集展示 GRU 的更新门。", 2.5, -3.2, 0.21, MUTED)
    label(s, "柱高表示分量数值，负值向下。", 2.5, -3.75, 0.2, MUTED)
    for x, title, values in [
        (-7.7, "旧状态", d.H_PREV),
        (-6.2, "门 z", d.Z),
        (-4.7, "候选状态", d.H_CANDIDATE),
    ]:
        label(s, title, x, 2.3, 0.22, MUTED)
        vector(s, np.round(values, 3), x, 0.85)
    old = d.Z * d.H_PREV
    new = (1 - d.Z) * d.H_CANDIDATE
    status = []
    for x, title, values, color in [
        (-1.5, "z ⊙ h旧", old, TEAL),
        (0.0, "(1 − z) ⊙ h候选", new, CORAL),
        (1.5, "h新 = 两项相加", d.H_NEXT, GOLD),
    ]:
        remove(status)
        status = [label(s, title, -6.2, -1.5, 0.29, color)]
        status += vector(s, np.round(values, 3), -6.2, -2.7, 0.5)
        bars(s, values, x=x, color=color)
        px, py = project(s, (x, 0, 1.2))
        label(
            s,
            "旧贡献" if x < 0 else "新贡献" if x == 0 else "新状态",
            px,
            -2.65,
            0.20,
            color,
        )
        s.wait(5)
    label(s, "逐分量控制：z 越大，保留的旧状态比例越高。", -8.3, -4.5, 0.25, left=True)
    s.wait(4)
    return s


def lstm(width, height, fps):
    d = model("09.2")
    s = start("09.2", width, height, fps, "记忆 C 和输出 H 是两条不同的状态。")
    label(s, "前后两根柱对应两个分量；负值向下。", 2.6, -3.2, 0.22, MUTED)
    label(s, "遗忘门 F · 输入门 I · 输出门 O", 2.6, -3.75, 0.22, MUTED)
    panel = []
    objects = []
    previous = d.CELL_0
    for time, step in enumerate((d.STEP1, d.STEP2), 1):
        for name, values, note in [
            ("旧记忆 C", previous, "上一时刻保留的记忆"),
            ("遗忘后的记忆", step["F"] * previous, "F = " + fmt(step["F"])),
            ("写入的新信息", step["I"] * step["C_tilde"], "I = " + fmt(step["I"])),
            ("更新记忆 C", step["C"], "C = F ⊙ C旧 + I ⊙ C候选"),
            ("生成输出 H", step["H"], "H = O ⊙ tanh(C)"),
        ]:
            remove(panel + objects)
            panel = [
                label(s, f"时刻 {time} · {name}", -6.2, 2.4, 0.3, TEAL),
                label(s, note, -6.2, 1.55, 0.21, MUTED),
            ]
            panel += vector(s, np.round(values, 3), -6.2, 0.35)
            if name == "生成输出 H":
                panel.append(label(s, "O = " + fmt(step["O"]), -6.2, -1.0, 0.23, GOLD))
            objects = bars(s, values, color=GOLD if name == "生成输出 H" else TEAL)
            s.wait(1.6)
        previous = step["C"]
    label(s, "先更新记忆，再由输出门选择暴露多少记忆。", -8.3, -4.5, 0.26, left=True)
    s.wait(4)
    return s


def lattice(ep, width, height, fps):
    d = model(ep)
    deep = ep == "09.3"
    s = start(ep, width, height, fps, "沿时间传递状态，同时看清不同计算方向。")
    count = 3 if deep else 4
    xs = np.linspace(-1.4, 6.6, count)
    row_y = (-0.6, 1.3)
    state_rows = (
        [np.asarray(d.LAYER_1[1:]).ravel(), np.asarray(d.LAYER_2[1:]).ravel()]
        if deep
        else [d.FORWARD_H.ravel(), d.BACKWARD_H.ravel()]
    )
    for layer in range(2):
        for i, x in enumerate(xs):
            node(s, float(x), row_y[layer], f"t{i + 1}")
            if i:
                line(
                    s,
                    (float(xs[i - 1]) + 0.63, row_y[layer]),
                    (float(x) - 0.63, row_y[layer]),
                    BLUE,
                )
    for i, x in enumerate(xs):
        if deep:
            line(s, (float(x), -0.2), (float(x), 0.9), TEAL)
        label(s, f"x{i + 1}", float(x), -2.25, 0.22, MUTED)
        line(s, (float(x), -1.9), (float(x), -0.99), BLUE)
    label(s, "上层：第 2 层" if deep else "上排：从右向左", 2.6, 2.6, 0.23, CORAL)
    label(s, "下层：第 1 层" if deep else "下排：从左向右", 2.6, -2.9, 0.23, TEAL)
    panel = []
    order = (
        [(layer, i) for i in range(count) for layer in (0, 1)]
        if deep
        else [(0, i) for i in range(count)] + [(1, i) for i in reversed(range(count))]
    )
    for layer, i in order:
        remove(panel)
        name = (
            f"时间步 {i + 1} · 第 {layer + 1} 层"
            if deep
            else ("正向状态" if layer == 0 else "反向状态") + f" · 位置 {i + 1}"
        )
        value = state_rows[layer][i]
        panel = [
            label(s, name, -6.2, 2.4, 0.29, TEAL),
            label(s, f"h = {value:.4f}", -6.2, 1.3, 0.35),
            label(s, "先接收当前层的历史，再结合当前输入。", -6.2, 0.3, 0.19, MUTED),
        ]
        if deep and layer == 1:
            origin = (float(xs[i]), row_y[0])
        else:
            origin = (
                float(
                    xs[max(0, i - 1) if layer == 0 or deep else min(count - 1, i + 1)]
                ),
                row_y[layer],
            )
        traveler = dot(s, *origin, color=TEAL if layer == 0 else CORAL)
        traveler.move(to=(float(xs[i]), row_y[layer]), duration=0.7)
        traveler.remove()
        rect(s, float(xs[i]), row_y[layer] + 0.62, 1.2, 0.36, PANEL, z=15)
        label(
            s,
            f"{value:.3f}",
            float(xs[i]),
            row_y[layer] + 0.62,
            0.21,
            TEAL if layer == 0 else CORAL,
        )
        s.wait(1.5)
    if not deep:
        remove(panel)
        panel = [label(s, "位置 3：拼接两个方向", -6.2, 2.4, 0.29, TEAL)]
        panel += vector(s, np.round(d.CONCAT_H, 3), -6.2, 1.0)
        panel.append(label(s, "两边的信息共同描述这个位置。", -6.2, -0.1, 0.22, MUTED))
    label(
        s,
        "深度增加层数，时间方向的循环仍然保留。"
        if deep
        else "双向模型会使用右侧上下文；两方向状态在同一位置拼接。",
        -8.3,
        -4.5,
        0.24,
        left=True,
    )
    s.wait(5)
    return s


def encoder_decoder(width, height, fps):
    d = model("09.6")
    s = start(
        "09.6", width, height, fps, "编码器把序列压成状态，解码器从状态开始逐步生成。"
    )
    encoder_x = (-1.5, 0.5, 2.5)
    for i, x in enumerate(encoder_x):
        node(s, x, 1.3, f"编码 {i + 1}")
        if i:
            line(s, (x - 1.35, 1.3), (x - 0.63, 1.3), TEAL)
    node(s, 4.5, 1.3, "上下文 c", color=GOLD)
    line(s, (3.13, 1.3), (3.87, 1.3), GOLD)
    node(s, 4.5, -1.4, "解码 1")
    node(s, 6.8, -1.4, "解码 2")
    line(s, (4.5, 0.9), (4.5, -1.0), GOLD)
    line(s, (5.13, -1.4), (6.17, -1.4), TEAL)
    label(s, "上一输出作为下一输入", 5.65, -2.3, 0.21, MUTED)
    panel = []
    for i, x in enumerate(encoder_x):
        remove(panel)
        panel = [
            label(s, f"读取输入 {i + 1}", -6.2, 2.4, 0.31, TEAL),
            label(s, "编码状态", -6.2, 1.5, 0.23, MUTED),
        ]
        panel += vector(s, np.round(d.ENCODER_STATES[i + 1], 3), -6.2, 0.5)
        traveler = dot(s, x - 1.2, 1.3)
        traveler.move(to=(x, 1.3), duration=0.6)
        traveler.remove()
        s.wait(2)
    remove(panel)
    panel = [label(s, "传递最后的编码状态", -6.2, 2.4, 0.3, GOLD)]
    panel += vector(s, np.round(d.CONTEXT, 3), -6.2, 1.1)
    traveler = dot(s, 2.5, 1.3, GOLD)
    traveler.move(to=(4.5, 1.3), duration=0.8)
    traveler.move(to=(4.5, -1.4), duration=0.8)
    traveler.remove()
    s.wait(2)
    for i, x in enumerate((4.5, 6.8)):
        remove(panel)
        panel = [
            label(s, f"生成输出 {i + 1}", -6.2, 2.4, 0.31, TEAL),
            label(
                s, "BOS = 0" if i == 0 else "输入为上一时刻输出", -6.2, 1.6, 0.23, MUTED
            ),
        ]
        panel += vector(s, np.round(d.OUTPUTS[i], 3), -6.2, 0.5)
        if i:
            traveler = dot(s, 4.5, -1.4)
            traveler.move(to=(x, -1.4), duration=0.8)
            traveler.remove()
        s.wait(3)
    label(
        s,
        "示例输出是连续向量，用于展示自回归传递，不是文字翻译。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(4)
    return s


def beam(width, height, fps):
    d = model("09.8")
    s = start(
        "09.8", width, height, fps, "每一步保留高分候选，最终比较整条路径的概率。"
    )
    panel = []
    tree = []
    dim = Color(53, 69, 87)
    for k in (1, 2):
        remove(panel + tree)
        tree = []
        keep = getattr(d, f"K{k}_T1_KEEP")
        winners = getattr(d, f"K{k}_T2_KEEP")
        panel = [
            label(s, f"束宽 k = {k}", -6.2, 2.4, 0.38, TEAL),
            label(s, "保留 " + ", ".join(d.VOCAB[i] for i in keep), -6.2, 1.5, 0.3),
            label(s, "整条路径概率 = 两步概率相乘", -6.2, 0.6, 0.21, MUTED),
        ]
        tree += node(s, -1.6, 0, "起点")
        for i, name in enumerate(d.VOCAB):
            y = 1.8 - i * 1.8
            color = TEAL if i in keep else dim
            tree += [
                line(s, (-0.97, 0), (0.8, y), color),
                label(s, name, 1.15, y, 0.27, color),
                label(s, f"{d.P_STEP1[i]:.3f}", 1.8, y, 0.2, color),
            ]
        s.wait(3)
        for i in range(3):
            for j in range(3):
                row = i * 3 + j
                yy = 2.4 - row * 0.6
                color = GOLD if (i, j) in winners else TEAL if i in keep else dim
                tree += [
                    line(s, (2.2, 1.8 - i * 1.8), (4.7, yy), color, 0.02),
                    label(s, d.VOCAB[i] + d.VOCAB[j], 5.15, yy, 0.23, color),
                    label(s, f"{d.JOINT[i, j]:.3f}", 6.5, yy, 0.23, color),
                ]
        best = winners[0]
        sequence = "".join(d.VOCAB[i] for i in best)
        panel += [
            label(s, f"最佳：{sequence}", -6.2, -0.5, 0.35, GOLD),
            label(s, f"路径概率 {d.JOINT[best]:.3f}", -6.2, -1.25, 0.27, GOLD),
        ]
        s.wait(5)
    label(
        s,
        "本例保留两条候选找到更高分路径；有限束宽仍不保证全局最优。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(4)
    return s


def build(ep, width=1920, height=1080, fps=60):
    if ep in ("09.3", "09.4"):
        return lattice(ep, width, height, fps)
    return {
        "08.4": rnn,
        "08.7": bptt,
        "09.1": gru,
        "09.2": lstm,
        "09.6": encoder_decoder,
        "09.8": beam,
    }[ep](width, height, fps)
