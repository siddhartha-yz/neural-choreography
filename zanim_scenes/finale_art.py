"""Continuous final movements: attention, optimization, vision and language.

Geometry is authored directly in Zanim. Original numerical chapter models
supply weights, paths and boxes. Surface extrusion, repetitions in depth,
pulses, camera motion and inter-chapter morphs are composition, not training.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing, Color
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.feature_suite import project
from zanim_scenes.models import (
    ep_10_3 as scoring,
    ep_10_4 as bahdanau,
    ep_10_5 as heads,
    ep_10_6 as self_attention,
    ep_11_2 as convex,
    ep_11_3 as gd,
    ep_11_4 as sgd,
    ep_11_6 as momentum,
    ep_11_7 as ada,
    ep_11_8 as rms,
    ep_11_10 as adam,
    ep_11_11 as scheduler,
    ep_13_3 as boxes,
    ep_13_4 as anchors,
    ep_13_5 as scales,
    ep_14_1 as embedding,
    ep_14_7 as analogy,
    ep_15_5 as inference,
)
from zanim_scenes.transposed_conv import contributions
from zanim_scenes.style import label

COLORS = (
    Color(88, 196, 221),
    Color(244, 205, 110),
    Color(178, 146, 233),
    Color(105, 217, 180),
)
MOVEMENTS = (
    ("10.3", "注意力评分", "不同评分，改变同一组信息的去向。", 14),
    ("10.4", "Bahdanau 注意力", "每次生成，都重新回望输入。", 14),
    ("10.5", "多头注意力", "分头汇聚，拼接后共同输出。", 16),
    ("10.6", "自注意力与位置编码", "同一序列彼此关联，波纹留下位置。", 16),
    ("11.2", "凸性", "连线与曲面的关系，决定地形。", 16),
    ("11.3", "梯度下降", "步长改变，下降路径随之改变。", 14),
    ("11.4", "随机梯度下降", "样本轮换，下降带着扰动。", 14),
    ("11.6", "动量法", "速度保留，让摆动继续向前。", 16),
    ("11.7", "AdaGrad", "累积越多，该方向的步幅越小。", 14),
    ("11.8", "RMSProp", "滑动记忆，持续调整各轴步幅。", 14),
    ("11.10", "Adam", "方向记忆与尺度记忆共同迈步。", 16),
    ("11.11", "学习率调度", "先展开步幅，再缓缓收紧。", 14),
    ("13.3", "目标检测与边界框", "角点与中心，描述同一个边界。", 14),
    ("13.4", "锚框", "候选铺满网格，重叠决定匹配。", 16),
    ("13.5", "多尺度检测", "不同尺度，保留不同的视野。", 16),
    ("13.10", "转置卷积", "单点铺开，重叠贡献相加。", 16),
    ("14.1", "词嵌入", "上下文牵引，向量关系逐步改变。", 16),
    ("14.7", "词向量类比", "同一个位移，移到另一组关系。", 14),
    ("15.5", "自然语言推断", "两列词元对齐，比较后汇合。", 18),
)

LEGENDS = {
    "10.3": (("加性评分", -4, 3.5, 0), ("缩放点积", -4, -3.2, 1)),
    "10.4": (("编码序列", 0, 3.5, 0), ("解码查询", 0, -3.5, 2)),
    "10.5": (("头 1", -6, 3.5, 0), ("头 2", -6, -3.2, 1), ("拼接", 5.4, 2.7, 3)),
    "10.6": (("位置编码", -5.8, 3.5, 1),),
    "11.2": (("凸", -4, -3.5, 0), ("非凸", 4, -3.5, 1)),
    "11.3": (("小步", -4, -3.4, 1), ("大步", -1, -3.4, 2)),
    "11.4": (("全量", -4, -3.4, 1), ("单样本", -1, -3.4, 2)),
    "11.6": (("GD", -4, -3.4, 1), ("动量", -1, -3.4, 2)),
    "11.7": (("GD", -4, -3.4, 1), ("AdaGrad", -1, -3.4, 2), ("累计平方", 4.7, -2.8, 2)),
    "11.8": (
        ("AdaGrad", -4, -3.4, 1),
        ("RMSProp", -1, -3.4, 2),
        ("平方记忆", 4.7, -2.8, 2),
    ),
    "11.10": (
        ("SGD", -4, -3.4, 1),
        ("Adam", -1, -3.4, 2),
        ("一阶 / 二阶", 4.7, -2.8, 3),
    ),
    "11.11": (("学习率", 4.7, 2, 3),),
    "13.3": (("宽", 0, -3.1, 1), ("高", 4, 0, 2)),
    "13.4": (("匹配", 0, -3.6, 1),),
    "13.5": (("细尺度", -3.7, -3.5, 0), ("粗尺度", 3.7, -3.5, 2)),
    "13.10": (("2 × 2", -5, 2.8, 0), ("3 × 3", 3.4, 2.8, 3), ("+", 3.4, -3, 1)),
    "14.1": (("中心词", -2.7, -3.5, 1), ("上下文", 2.7, -3.5, 3)),
    "14.7": (
        ("man", -3, -2.8, 0),
        ("woman", 2.7, -2.8, 1),
        ("king", -3, 2.8, 0),
        ("queen", 2.7, 2.8, 3),
    ),
    "15.5": (("前提", -5.8, 3.5, 0), ("假设", 0.2, 3.5, 1), ("对齐", 3.5, 3.5, 2)),
}


def xyz(p):
    p = np.asarray(p, dtype=float)
    return np.r_[p, 0.0] if len(p) == 2 else p


@dataclass
class Drawing:
    dots: list = field(default_factory=list)
    lines: list = field(default_factory=list)

    def dot(self, p, color=0, radius=0.08):
        self.dots.append((xyz(p), color, max(0.0, float(radius))))

    def line(self, a, b, color=0, width=0.6):
        self.lines.append((xyz(a), xyz(b), color, max(0.0, float(width))))

    def curve(self, pp, color=0, width=0.6):
        for a, b in zip(pp[:-1], pp[1:]):
            self.line(a, b, color, width)

    def ring(self, center, radius, color=0, width=0.7, n=40):
        q = np.linspace(0, 2 * np.pi, n + 1)
        self.curve(
            xyz(center)
            + np.c_[radius * np.cos(q), radius * np.sin(q), np.zeros(n + 1)],
            color,
            width,
        )

    def flow(self, a, b, color, phase, weight=1.0, count=5, bend=0.0):
        a, b = xyz(a), xyz(b)
        self.line(a, b, color, 0.25 + 0.7 * weight)
        for k in range(count):
            q = (phase + k / count) % 1
            p = a + q * (b - a) + np.array([0.0, bend * math.sin(np.pi * q), 0.0])
            self.dot(p, color, 0.055 * math.sqrt(max(weight, 0)) * math.sin(np.pi * q))

    def box(self, corners, color=0, width=0.7):
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            self.line(a, b, color, width)

    def packed(self, time):
        dot_groups, line_groups = [], []
        for color in range(4):
            dd = [(p, r) for p, c, r in self.dots if c == color]
            ll = [(a, b, w) for a, b, c, w in self.lines if c == color]
            pp = project(np.array([p for p, _ in dd]), time) if dd else np.empty((0, 2))
            dot_groups.append(np.c_[pp, [r for _, r in dd]])
            if ll:
                pp = project(
                    np.array([[a, b] for a, b, _ in ll]).reshape(-1, 3), time
                ).reshape(-1, 4)
                line_groups.append(np.c_[pp, [w for _, _, w in ll]])
            else:
                line_groups.append(np.empty((0, 5)))
        return dot_groups, line_groups


def interpolate(path, progress):
    q = float(np.clip(progress, 0, 1)) * (len(path) - 1)
    j = min(int(q), len(path) - 2)
    return path[j] + ease(q - j) * (path[j + 1] - path[j])


def attention_picture(kind, t):
    d = Drawing()
    phase = t * 0.24
    if kind == "10.3":
        q = scoring.QUERY + 0.7 * np.array([math.sin(t * 0.5), math.cos(t * 0.5) - 1])
        weights = (
            scoring.softmax(
                np.tanh(scoring.KEYS @ scoring.W_K.T + scoring.W_Q @ q) @ scoring.W_V
            ),
            scoring.softmax(scoring.KEYS @ q * scoring.SCALE),
        )
        for branch in range(2):
            y = 1.8 - 3.6 * branch
            source, output = (-6.4, y, 0), (6.1, y, 0)
            d.ring(source, 0.32, 2)
            for j in range(3):
                key = (0, y + (1 - j) * 0.9, (branch - 0.5) * 0.6)
                d.dot(key, 0, 0.08 + 0.18 * weights[branch][j])
                d.flow(
                    source, key, 2, phase + j * 0.1, float(weights[branch][j]), bend=0.2
                )
                d.flow(
                    key,
                    output,
                    branch,
                    phase + 0.4 + j * 0.1,
                    float(weights[branch][j]),
                    bend=-0.2,
                )
            value = weights[branch] @ scoring.VALUES
            d.dot(output, branch, 0.13 + 0.08 * np.linalg.norm(value))
    elif kind == "10.4":
        q = bahdanau.QUERY_S0 + (0.5 - 0.5 * math.cos(t * 0.65)) * (
            bahdanau.QUERY_S1 - bahdanau.QUERY_S0
        )
        _, weights, context = bahdanau.bahdanau(q)
        decoder = (4.8 * math.sin(t * 0.28), -2.1, 0.4)
        for j in range(3):
            p = (-4.8 + 4.8 * j, 2.0, -0.2)
            d.ring(p, 0.24 + 0.42 * weights[j], 0)
            d.dot(p, 0, 0.13)
            d.flow(p, decoder, 1, phase + 0.11 * j, float(weights[j]), 9, bend=0.45)
            if j:
                d.flow((-4.8 + 4.8 * (j - 1), 2, -0.2), p, 0, phase, 0.35, 3)
        d.ring(decoder, 0.4, 2)
        d.dot(decoder, 1, 0.12 + 0.12 * np.linalg.norm(context))
        d.flow(decoder, (6.5, -2.1, 0.4), 3, phase, 0.8, 7)
    elif kind == "10.5":
        for h in range(2):
            y = 1.5 - 3 * h
            output = (3.2, y, h * 0.5)
            for j in range(3):
                p = (-5.8 + j * 2.2, y + 1.0, h * 0.5)
                w = float(heads.HEAD_WEIGHTS[h][j])
                d.ring(p, 0.18 + 0.25 * w, h)
                d.flow(p, output, h, phase + 0.12 * j, w, 8, bend=-0.35)
            d.dot(output, h, 0.13 + 0.12 * abs(heads.HEAD_OUTPUTS[h]))
            concat = (5.5, 0.22 - 0.44 * h, 0)
            d.flow(output, concat, h, phase, 0.8, 5)
            d.dot(concat, h, 0.14)
            for j in range(2):
                d.flow(
                    concat,
                    (7.2, 0.8 - 1.6 * j, 0),
                    3 if heads.WEIGHT_O[j, h] > 0 else 2,
                    phase + 0.2,
                    abs(float(heads.WEIGHT_O[j, h])),
                    4,
                )
        for j in range(2):
            d.dot((7.2, 0.8 - 1.6 * j, 0), 3, 0.13 + 0.12 * abs(heads.YHAT[j]))
    else:
        n = 4
        p = np.array([[-5.4 + j * 3.6, 1.6, 0] for j in range(n)])
        out = p.copy()
        out[:, 1] = -1.6
        for i in range(n):
            d.ring(p[i], 0.25, i % 4)
            d.dot(
                out[i], i % 4, 0.13 + 0.06 * np.linalg.norm(self_attention.OUTPUTS[i])
            )
            for j in range(n):
                d.flow(
                    p[j],
                    out[i],
                    i % 4,
                    phase + 0.13 * j,
                    float(self_attention.WEIGHTS[i, j]),
                    5,
                )
            # Four sinusoidal components of standard positional encoding.
            for component in range(4):
                frequency = 10000.0 ** (-2 * (component // 2) / 4)
                q = np.linspace(0, 1, 13)
                value = (
                    np.sin((i + q) * frequency)
                    if component % 2 == 0
                    else np.cos((i + q) * frequency)
                )
                pp = np.c_[
                    p[i, 0] + q * 2.3 - 1.15,
                    2.9 + 0.22 * value,
                    np.full(13, (component - 1.5) * 0.24),
                ]
                d.curve(pp, component, 0.5)
    return d


OPTIMIZER_PATHS = {
    "11.3": (gd.PATH_GOOD, gd.PATH_LARGE),
    "11.4": (sgd.PATH_FULL, sgd.PATH_SGD),
    "11.6": (momentum.PATH_GD, momentum.PATH_MOM),
    "11.7": (ada.PATH_GD, ada.PATH_ADA),
    "11.8": (rms.PATH_ADA, rms.PATH_RMS),
    "11.10": (adam.PATH_SGD, adam.PATH_ADAM),
    "11.11": (scheduler.PATH,),
}


def loss(kind, p):
    return (1.0 if kind in ("11.3", "11.4", "11.11") else 0.1) * p[..., 0] ** 2 + 2 * p[
        ..., 1
    ] ** 2


def loss_point(kind, p):
    return np.array([p[0] * 1.18, -2.4 + loss(kind, p) * 0.13, p[1] * 1.4])


def optimizer_memory(kind, branch, progress):
    if kind == "11.7" and branch == 1:
        return interpolate(ada.S_ADA, progress)
    if kind == "11.8":
        return interpolate((rms.S_ADA, rms.S_RMS)[branch], progress)
    if kind == "11.10" and branch == 1:
        # Reconstruct the two bias-corrected moments used by the original Adam.
        first, second, history = np.zeros(2), np.zeros(2), [np.zeros((2, 2))]
        for step, g in enumerate(adam.GRADS_ADAM, 1):
            first = adam.BETA1 * first + (1 - adam.BETA1) * g
            second = adam.BETA2 * second + (1 - adam.BETA2) * g * g
            history.append(
                np.array(
                    [first / (1 - adam.BETA1**step), second / (1 - adam.BETA2**step)]
                )
            )
        return interpolate(np.array(history), progress)
    return None


def optimization_picture(kind, t, duration):
    d = Drawing()
    progress = ease((t - 0.8) / (duration - 2.2))
    if kind == "11.2":
        x = np.linspace(-2, 2, 49)
        for side in range(2):
            center = (-4, 4)[side]
            fn = (convex.convex_f, convex.nonconvex_g)[side]
            for depth in np.linspace(-0.7, 0.7, 7):
                d.curve(
                    np.c_[
                        center + x * 1.55, fn(x) * 1.35 - 0.8, np.full(len(x), depth)
                    ],
                    side,
                    0.42,
                )
            a, b = convex.XA, convex.XB
            ya, yb = float(fn(a)), float(fn(b))
            d.line(
                (center + a * 1.55, ya * 1.35 - 0.8, 0),
                (center + b * 1.55, yb * 1.35 - 0.8, 0),
                1,
                1.5,
            )
            q = 0.5 - 0.5 * math.cos(t * 0.7)
            xx = (1 - q) * a + q * b
            y_chord = (1 - q) * ya + q * yb
            p = (center + xx * 1.55, y_chord * 1.35 - 0.8, 0)
            surface = (center + xx * 1.55, float(fn(xx)) * 1.35 - 0.8, 0)
            d.dot(p, 1, 0.13)
            d.dot(surface, side, 0.08)
            d.line(p, surface, 3 if y_chord >= fn(xx) else 2, 1.1)
        return d
    xx = np.linspace(-5.2, 1.5, 29)
    zz = np.linspace(-2.3, 2.3, 17)
    for y in zz:
        pp = np.c_[xx, np.full(len(xx), y)]
        d.curve([loss_point(kind, p) for p in pp], 0, 0.28)
    for x in xx[::2]:
        pp = np.c_[np.full(len(zz), x), zz]
        d.curve([loss_point(kind, p) for p in pp], 0, 0.28)
    for branch, path in enumerate(OPTIMIZER_PATHS[kind]):
        # Two close ribbons are depth copies of the same numerical trajectory.
        for replica in range(3):
            offset = np.array([0, 0, (replica - 1) * 0.08])
            for j in range(len(path) - 1):
                reveal = ease((progress * (len(path) - 1) - j) * 3)
                a, b = (
                    loss_point(kind, path[j]) + offset,
                    loss_point(kind, path[j + 1]) + offset,
                )
                d.line(a, a + reveal * (b - a), 1 + branch, 1.4 * reveal)
            current = interpolate(path, progress)
            p = loss_point(kind, current) + offset
            d.dot(p, 1 + branch, 0.13 if replica == 1 else 0.055)
        j = min(int(progress * (len(path) - 1)), len(path) - 2)
        a, b = loss_point(kind, path[j]), loss_point(kind, path[j + 1])
        d.flow(a, b, 1 + branch, t * 0.65, 0.7, 6)
        memory = optimizer_memory(kind, branch, progress)
        if memory is not None:
            quantities = memory if memory.ndim == 2 else memory[None, :]
            for row in range(len(quantities)):
                center = np.array([4.7, -0.5 + row * 1.7, 0])
                for axis in range(2):
                    value = quantities[row, axis]
                    r = 0.17 + 0.75 * abs(value) / (1 + abs(value))
                    d.line(
                        center,
                        center
                        + np.array(
                            [r * 2 if axis == 0 else 0, r * 2 if axis == 1 else 0, 0]
                        ),
                        2 + row,
                        1.5,
                    )
                    d.ring(center, r, 2 + row, 0.7, 24)
        if kind == "11.6" and branch == 1:
            j = min(int(progress * (len(path) - 1)), len(path) - 2)
            velocity = (path[j] - path[j + 1]) / momentum.ETA
            origin = (4.8, 0, 0)
            d.line(origin, xyz(origin) + np.r_[velocity * 0.6, 0], 2, 1.5)
            d.ring(origin, 0.35 + 0.18 * np.linalg.norm(velocity), 2, 0.8, 30)
    if kind == "11.11":
        steps = np.linspace(0, 10, 65)
        curve = np.array(
            [[2.7 + x * 0.43, -2.9 + scheduler.eta_of(x) * 15, 0] for x in steps]
        )
        d.curve(curve, 3, 1.2)
        k = progress * 10
        d.dot((2.7 + k * 0.43, -2.9 + scheduler.eta_of(k) * 15, 0), 3, 0.14)
    return d


def corners(box, center=(0, 0, 0), scale=6):
    x1, y1, x2, y2 = box
    pp = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    pp = (pp - 0.5) * np.array([scale, -scale])
    return np.c_[pp, np.zeros(4)] + np.array(center)


def vision_picture(kind, t, duration):
    d = Drawing()
    if kind == "13.3":
        q = ease(t / 3)
        outline = boxes.object_boundary(65)
        local = (outline - boxes.OBJECT_CENTER) * 3
        angle = 0.1 * math.sin(t * 0.5)
        rot = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        # Object/box share a visual camera motion; their coordinates do not drift.
        for depth in np.linspace(-0.4, 0.4, 5):
            d.curve(np.c_[local @ rot.T, np.full(len(local), depth)], 3, 0.55)
        frame = (boxes.CORNER[0] - np.tile(boxes.OBJECT_CENTER, 2)) * np.array(
            [3, 3, 3, 3]
        )
        cc = np.array(
            [
                [frame[0], frame[1], 0],
                [frame[2], frame[1], 0],
                [frame[2], frame[3], 0],
                [frame[0], frame[3], 0],
            ]
        )
        for i, p in enumerate(cc):
            other = cc[(i + 1) % 4]
            a = p + (1 - q) * np.array([(-1) ** i * 2, (-1) ** (i // 2) * 1, 0])
            d.line(a, a + q * (other - a), 0, 1.2)
            d.dot(a, 1, 0.1)
            if t > 4:
                d.flow((0, 0, 0), p, 1, t * 0.18, 0.5, 5)
        d.ring((0, 0, 0), 0.2, 1)
        d.line((frame[0], -2.5, 0), (frame[2], -2.5, 0), 1, 1.2)
        d.line((3.5, frame[1], 0), (3.5, frame[3], 0), 2, 1.2)
    elif kind == "13.4":
        truth = corners(anchors.GROUND_TRUTH)
        d.box(truth, 3, 1.8)
        for r in range(6):
            for c in range(6):
                cell = ((c + 0.5) / 6 - 0.5) * 6, (0.5 - (r + 0.5) / 6) * 6, 0
                d.dot(cell, 0, 0.04)
                raw = anchors.multibox_prior_cell(
                    6, r, c, anchors.SIZES, anchors.RATIOS
                )
                overlaps = anchors.box_iou(raw, anchors.GROUND_TRUTH[None, :])[:, 0]
                for k, box in enumerate(raw):
                    phase = ease((t - (r * 6 + c) * 0.045) / 2)
                    pp = corners(box) * phase + (1 - phase) * np.array(cell)
                    color = 1 if (r, c, k) == (*anchors.CELL, anchors.WINNER) else k * 2
                    d.box(pp, color, 0.16 + 0.8 * overlaps[k])
        hit = anchors.INTER_BOXES[anchors.WINNER]
        # Hatch only the actual intersection; no invented numeric coverage.
        for x in np.linspace(hit[0], hit[2], 11):
            pp = corners([x, hit[1], x, hit[3]])
            d.line(pp[0], pp[2], 1, 0.6 * ease((t - 4) / 2))
    elif kind == "13.5":
        for level, (n, size, center, color) in enumerate(
            (
                (4, scales.S_FINE, (-3.7, 0, -0.6), 0),
                (2, scales.S_COARSE, (3.7, 0, 0.6), 2),
            )
        ):
            aa = scales.square_anchors(n, n, size).reshape(-1, 4)
            overlaps = scales.box_iou(aa, scales.GROUND_TRUTH)
            for i, box in enumerate(aa):
                pp = corners(box, center, 5.5)
                lift = 0.3 * math.sin(t * 0.35 + i * 0.35)
                pp[:, 2] += lift
                d.box(pp, color, 0.35 + overlaps[i])
                p = pp.mean(axis=0)
                d.dot(p, color, 0.05 + 0.14 * overlaps[i])
            truth = corners(scales.GROUND_TRUTH, center, 5.5)
            d.box(truth, 1, 1.5)
            best = int(np.argmax(overlaps))
            d.flow(
                corners(aa[best], center, 5.5).mean(axis=0),
                truth.mean(axis=0),
                1,
                t * 0.35,
                0.8,
                8,
            )
        d.line((-1, -3.1, 0), (1, -3.1, 0), 1, 1.2)
    else:
        q = t / 2.1
        for j, (r, c, patch, _) in enumerate(contributions()):
            center = (-5.6 + c * 1.2, 1.2 - r * 1.2, -0.4)
            d.ring(center, 0.25, j)
            for kr in range(2):
                for kc in range(2):
                    value = patch[kr][kc]
                    target = np.array(
                        [3.4 + (c + kc - 1) * 1.5, 1.2 - (r + kr) * 1.5, j * 0.22]
                    )
                    start = np.array(center)
                    travel = ease((q - j) / 0.85)
                    p = start + travel * (target - start)
                    d.dot(p, j, 0.035 + 0.055 * math.sqrt(value))
                    if value:
                        d.flow(start, target, j, t * 0.23 + j * 0.1, 0.4 * travel, 4)
            # A connected 2x2 footprint makes each spreading kernel visible.
            reveal = ease((q - j) / 0.85)
            footprint = np.array(
                [
                    [3.4 + (c + kc - 1) * 1.5, 1.2 - (r + kr) * 1.5, j * 0.22]
                    for kr, kc in ((0, 0), (0, 1), (1, 1), (1, 0))
                ]
            )
            d.box(footprint, j, 0.8 * reveal)
            # Four differently colored contribution planes retain source identity.
            for kr in range(2):
                for kc in range(2):
                    p = np.array(
                        [3.4 + (c + kc - 1) * 1.5, 1.2 - (r + kr) * 1.5, j * 0.22]
                    )
                    merge = ease((t - 10) / 3)
                    stack = sum(
                        l[2][r + kr - l[0]][c + kc - l[1]]
                        for l in contributions()[:j]
                        if 0 <= r + kr - l[0] < 2 and 0 <= c + kc - l[1] < 2
                    )
                    p[1] += merge * (stack + patch[kr][kc] / 2) * 0.11
                    p[2] *= 1 - merge
                    d.ring(
                        p,
                        0.16 + 0.035 * patch[kr][kc],
                        j,
                        0.6 * ease((q - j) / 0.85),
                        24,
                    )
        total = np.array(contributions()[-1][-1])
        for k in range(3):
            d.line([1.9, 1.2 - k * 1.5, 0], [4.9, 1.2 - k * 1.5, 0], 3, 0.4)
            d.line([1.9 + k * 1.5, 1.2, 0], [1.9 + k * 1.5, -1.8, 0], 3, 0.4)
        for r in range(3):
            for c in range(3):
                p = np.array([3.4 + (c - 1) * 1.5, 1.2 - r * 1.5, 0])
                d.line(
                    p,
                    p + np.array([0, total[r, c] * 0.11, 0]),
                    3,
                    2 * ease((t - 10) / 3),
                )
    return d


def language_picture(kind, t, duration):
    d = Drawing()
    if kind == "14.1":
        state = embedding.mix_state(
            ease((t - 0.6) / (duration - 2)) * embedding.SGD_STEPS
        )
        center = state["v"]
        for replica in range(5):
            offset = np.array([-0.4, 0, (replica - 2) * 0.35])
            a = np.r_[center * 2.2, 0] + offset
            d.line(offset, a, 1, 0.8)
            d.dot(a, 1, 0.1 if replica == 2 else 0.045)
            for j, p in enumerate(state["U"]):
                b = np.r_[p * 2.2, 0] + offset
                d.line(offset, b, 3 if j == embedding.CONTEXT else 0, 0.6)
                d.dot(b, 3 if j == embedding.CONTEXT else 0, 0.08)
                d.flow(
                    a,
                    b,
                    3 if j == embedding.CONTEXT else 2,
                    t * 0.25 + replica * 0.1,
                    float(state["p"][j]),
                    4,
                )
        for j, state0 in enumerate(embedding.STATES):
            d.dot(np.r_[state0["U"][embedding.CONTEXT] * 2.2, 0], 3, 0.035)
    elif kind == "14.7":
        vectors = np.array(analogy.TOKENS)
        origin = np.array([1.1, 0.9])
        for replica in range(5):
            offset = np.array([0, 0, (replica - 2) * 0.28])
            points = np.c_[(vectors - origin) * 3.2, np.zeros(4)] + offset
            a, b, c, q = points
            d.line(a, b, 1, 1.1)
            d.line(a, c, 0, 0.7)
            d.dot(a, 0, 0.07)
            d.dot(b, 1, 0.07)
            d.dot(c, 0, 0.07)
            d.dot(q, 3, 0.08)
            travel = ease((t - 2) / 5)
            displacement = b - a
            moving_a = a + travel * (c - a)
            moving_b = moving_a + displacement
            d.line(moving_a, moving_b, 1, 1.5)
            d.dot(moving_b, 1, 0.11)
            d.line(moving_b, q, 2, 0.8 * travel)
            d.flow(moving_a, moving_b, 1, t * 0.3, 0.8, 6)
            d.line(b, q, 0, 0.5)
    else:
        for i in range(3):
            a = np.array([-5.8, 2.2 - i * 2.2, 0])
            d.ring(a, 0.27, 0)
            d.dot(a, 0, 0.08)
            for j in range(3):
                b = np.array([0.2, 2.2 - j * 2.2, 0.25])
                w = float(inference.WEIGHTS[i, j])
                highlight = 0.35 + 0.65 * math.exp(-(((t * 0.32 % 3) - i) ** 2) * 2)
                d.flow(a, b, 1, t * 0.25 + i * 0.17, w * highlight, 6)
            aligned = np.array([3.5, 2.2 - i * 2.2, 0])
            d.dot(aligned, 2, 0.11 + 0.04 * np.linalg.norm(inference.ALIGNED[i]))
            d.flow(a, aligned, 2, t * 0.25, 0.2, 3)
        for j in range(3):
            p = (0.2, 2.2 - j * 2.2, 0.25)
            d.ring(p, 0.24, 1)
        for j in range(3):
            p = np.array([7, 1.7 - j * 1.7, 0])
            d.ring(p, 0.23 + inference.PROBABILITIES[j] * 0.65, j)
            d.flow((3.5, 0, 0), p, j, t * 0.25, float(inference.PROBABILITIES[j]), 7)
    return d


def picture(kind, t, duration):
    if kind.startswith("10."):
        d = attention_picture(kind, t)
    elif kind.startswith("11."):
        d = optimization_picture(kind, t, duration)
    elif kind.startswith("13."):
        d = vision_picture(kind, t, duration)
    else:
        d = language_picture(kind, t, duration)
    return d.packed(t * 0.6)


def padded(array, count, columns):
    out = np.zeros((count, columns))
    out[: len(array)] = array
    return out


def blend(a, b, q):
    out = []
    for old, new in zip(a, b):
        mixed = []
        for x, y in zip(old, new):
            n = max(len(x), len(y))
            xx, yy = padded(x, n, x.shape[1]), padded(y, n, y.shape[1])
            # New and departing geometry fades at its own position.
            xx[len(x) :, :-1] = yy[len(x) :, :-1]
            yy[len(y) :, :-1] = xx[len(y) :, :-1]
            mixed.append(xx + ease(q) * (yy - xx))
        out.append(mixed)
    return tuple(out)


def append(s, old, chapter):
    # Determine exact bank sizes across the deterministic picture functions.
    maximum_dots = np.zeros(4, dtype=int)
    maximum_lines = np.zeros(4, dtype=int)
    for code, _, _, duration in MOVEMENTS:
        for t in (0.0, duration * 0.25, duration * 0.6, duration):
            dots, lines = picture(code, t, duration)
            maximum_dots = np.maximum(maximum_dots, list(map(len, dots)))
            maximum_lines = np.maximum(maximum_lines, list(map(len, lines)))
    dots = []
    lines = []
    for color in range(4):
        dots.append(
            [
                s.add(
                    Circle(
                        0.1,
                        style=Style.solid(COLORS[color]),
                        transform=Transform2D.translation(-30, 0),
                        z_index=6,
                    )
                )
                for _ in range(maximum_dots[color])
            ]
        )
        lines.append(
            [
                s.add(
                    Line(
                        (0, 0),
                        (1, 0),
                        style=Style.outline(COLORS[color].with_alpha(155), 0.02),
                        transform=Transform2D.translation(-30, 0),
                        z_index=2,
                    )
                )
                for _ in range(maximum_lines[color])
            ]
        )
    bank = [o for group in dots + lines for o in group]
    # Keep the incoming pooling strands through the title handoff. Fade them
    # only while the first pair of scoring fans unfolds, avoiding a blank stage.
    s.wait(0.6)
    previous = None
    for code, title, subtitle, duration in MOVEMENTS:
        chapter(f"{title} · {code}", subtitle)
        annotations = [
            label(s, text, x, y, 0.21, COLORS[color])
            for text, x, y, color in LEGENDS[code]
        ]
        first = picture(code, 0.0, duration)

        @lru_cache(maxsize=16)
        def frame(u, code=code, duration=duration, previous=previous, first=first):
            t = u * duration
            raw = picture(code, max(0, t - 2), duration - 2)
            if t < 2:
                if previous is None:
                    return tuple(
                        [
                            [
                                a * np.r_[np.ones(a.shape[1] - 1), ease(t / 2)]
                                for a in group
                            ]
                            for group in first
                        ]
                    )
                return blend(previous, first, t / 2)
            return raw

        def dot_pose(u, c, j, frame=frame):
            dd = frame(u)[0][c]
            if j >= len(dd) or dd[j, -1] < 1e-7:
                return Transform2D.translation(-30, 0)
            x, y, r = dd[j]
            return Transform2D.translation(float(x), float(y)) @ Transform2D.scaling(
                float(r / 0.1)
            )

        def line_pose(u, c, j, frame=frame):
            ll = frame(u)[1][c]
            if j >= len(ll) or ll[j, -1] < 1e-7:
                return Transform2D.translation(-30, 0)
            ax, ay, bx, by, w = ll[j]
            return pose(np.array([ax, ay]), np.array([bx, by])) @ Transform2D.scaling(
                1, float(w)
            )

        with s.parallel():
            if previous is None:
                for item in old:
                    item.fade_out(duration=2.4)
            for item in annotations:
                item.opacity(to=0, duration=0)
                item.fade_in(duration=0.5, at=0.5)
                item.fade_out(duration=0.5, at=duration - 0.5)
            for c in range(4):
                for j, o in enumerate(dots[c]):
                    o.set_transform(to=dot_pose(0, c, j))
                    o.transform_function(
                        lambda u, c=c, j=j, fn=dot_pose: fn(u, c, j),
                        duration=duration,
                        easing=Easing.LINEAR,
                    )
                for j, o in enumerate(lines[c]):
                    o.set_transform(to=line_pose(0, c, j))
                    o.transform_function(
                        lambda u, c=c, j=j, fn=line_pose: fn(u, c, j),
                        duration=duration,
                        easing=Easing.LINEAR,
                    )
        if previous is None:
            for item in old:
                item.remove()
        previous = frame(1)
        for item in annotations:
            item.remove()
    # Resolve the last alignment into a quiet aperture and retain motion at end.
    final_dots, final_lines = previous
    with s.parallel():
        for group in lines:
            for o in group:
                o.fade_out(duration=4)
        for c, group in enumerate(dots):
            for j, o in enumerate(group):
                data = final_dots[c]
                if j >= len(data):
                    continue
                start = data[j, :2]
                theta = 2 * np.pi * (j / max(1, len(data))) + c * 0.2

                def resolve(u, start=start, theta=theta, c=c):
                    angle = theta + u * 0.7
                    radius = 1.4 + c * 0.18
                    target = np.array(
                        [radius * math.cos(angle), radius * 0.65 * math.sin(angle)]
                    )
                    p = start + ease(u) * (target - start)
                    return Transform2D.translation(
                        *map(float, p)
                    ) @ Transform2D.scaling(0.45)

                o.transform_function(resolve, duration=6, easing=Easing.LINEAR)
    return bank
