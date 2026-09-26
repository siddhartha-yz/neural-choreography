"""Numerical attention weights rendered as native Zanim 3D columns."""

import importlib
from dataclasses import dataclass
import numpy as np
from zanim import Box3D, Transform3D, Vec3
from zanim_scenes.style import BLUE, CORAL, GOLD, MUTED, TEAL, label
from zanim_scenes.mechanisms import start, remove, dot, line, vector

EPISODE_IDS = ("10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "15.5")


def model(ep):
    return importlib.import_module("zanim_scenes.models.ep_" + ep.replace(".", "_"))


@dataclass
class Stage:
    name: str
    note: str
    scores: np.ndarray
    weights: np.ndarray
    values: np.ndarray
    output: np.ndarray


def positional_encoding(length):
    pos = np.arange(length, dtype=float)
    return np.stack((np.sin(pos), np.cos(pos)), axis=1)


def stages(ep):
    d = model(ep)
    result = []
    if ep == "10.1":
        for i in range(3):
            result.append(
                Stage(
                    f"查询 q{i + 1}",
                    "分数 = q · k",
                    d.SCORES[i],
                    d.WEIGHTS[i],
                    d.VALUES,
                    d.OUTPUTS[i],
                )
            )
    elif ep == "10.2":
        for q in (d.QUERY_A, d.QUERY_B):
            w = d.attention_weights(q)
            result.append(
                Stage(
                    f"查询位置 q = {q:g}",
                    "分数 = −(q − k)² / 2",
                    -0.5 * (q - d.KEYS) ** 2,
                    w,
                    d.VALUES,
                    np.atleast_1d(w @ d.VALUES),
                )
            )
    elif ep == "10.3":
        for name, prefix in [("加性评分", "ADDITIVE"), ("缩放点积评分", "DOT")]:
            result.append(
                Stage(
                    name,
                    "同一组查询、键和值，两种评分方式",
                    getattr(d, prefix + "_SCORES"),
                    getattr(d, prefix + "_WEIGHTS"),
                    d.VALUES,
                    getattr(d, prefix + "_YHAT"),
                )
            )
    elif ep == "10.4":
        for i in range(2):
            result.append(
                Stage(
                    f"解码第 {i + 1} 步",
                    "查询改变后，重新计算注意力",
                    getattr(d, f"SCORES_{i}"),
                    getattr(d, f"ALPHA_{i}"),
                    d.ENCODER_H,
                    getattr(d, f"CONTEXT_{i}"),
                )
            )
    elif ep == "10.5":
        for i in range(2):
            result.append(
                Stage(
                    f"注意力头 {i + 1}",
                    "各头分别投影 Q、K、V",
                    d.HEAD_SCORES[i],
                    d.HEAD_WEIGHTS[i],
                    d.HEAD_VALUES[i],
                    np.atleast_1d(d.HEAD_OUTPUTS[i]),
                )
            )
    elif ep == "10.6":
        i = d.TRAVELER
        result.append(
            Stage(
                "自注意力 · 词元 2",
                "查询、键、值都来自同一个序列",
                d.SCORES[i],
                d.WEIGHTS[i],
                d.TOKENS,
                d.OUTPUTS[i],
            )
        )
        tokens = d.TOKENS + positional_encoding(len(d.TOKENS))
        scores = tokens @ tokens.T * d.SCALE
        weights = d.softmax_rows(scores)
        result.append(
            Stage(
                "加入位置编码",
                "P(pos) = (sin pos, cos pos)",
                scores[i],
                weights[i],
                tokens,
                (weights @ tokens)[i],
            )
        )
    else:
        for i in range(len(d.PREMISE)):
            result.append(
                Stage(
                    f"前提词元 {i + 1} → 假设序列",
                    "在另一句话中寻找相关表示",
                    d.SCORES[i],
                    d.WEIGHTS[i],
                    d.HYPOTHESIS,
                    d.ALIGNED[i],
                )
            )
    return result


def project(scene, point):
    state = scene.camera3d.state()
    toarray = lambda v: np.array([v.x, v.y, v.z])
    forward = toarray(state.target) - toarray(state.position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, toarray(state.up))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(point) - toarray(state.target)
    factor = (scene.canvas.height / scene.canvas.unit_size) / state.orthographic_height
    return (float(delta @ right * factor), float(delta @ up * factor))


def short(values):
    a = np.atleast_1d(values).ravel()
    return ", ".join(f"{v:.2f}" for v in a)


def weight_columns(s, weights, z=0.0, color=TEAL, key_names=None):
    count = len(weights)
    spacing = 0.66 if count > 5 else 0.95
    meshes = []
    positions = []
    for i, w in enumerate(weights):
        x = (i - (count - 1) / 2) * spacing
        sbase = s.add(
            Box3D(
                Vec3(spacing * 0.82, 0.055, 0.55),
                color=BLUE,
                transform=Transform3D.translation(x, -0.055, z),
            )
        )
        h = max(float(w) * 2.8, 0.003)
        bar = s.add(
            Box3D(
                Vec3(spacing * 0.7, h, 0.45),
                color=GOLD if i == int(np.argmax(weights)) else color,
                transform=Transform3D.translation(x, 0.001, z)
                @ Transform3D.scaling(1, 0.001, 1),
            )
        )
        meshes.append((bar, Transform3D.translation(x, h / 2, z), sbase))
        positions.append((x, h, z))
    with s.parallel():
        for bar, target, _ in meshes:
            bar.transform(to=target, duration=1.2)
    annotations = []
    for i, (w, pos) in enumerate(zip(weights, positions)):
        px, py = project(s, pos)
        annotations.append(
            label(
                s,
                f"{w:.3f}",
                px,
                py + 0.23,
                0.19,
                GOLD if i == int(np.argmax(weights)) else TEAL,
            )
        )
        px, py = project(s, (pos[0], 0, z))
        annotations.append(
            label(
                s,
                str(key_names[i]) if key_names is not None else f"k{i + 1}",
                px,
                py - 0.3,
                0.18,
                MUTED,
            )
        )
    return [m for bar, _, base in meshes for m in (bar, base)] + annotations, positions


def build(ep, width=1920, height=1080, fps=60):
    s = start(ep, width, height, fps, "先比较匹配程度，再把值按权重汇总。")
    label(s, "柱高 = 注意力权重 α", 2.6, -2.8, 0.25, TEAL)
    label(s, "每组权重相加为 1；金色标记最大权重。", 2.6, -3.4, 0.21, MUTED)
    panel = []
    objects = []
    for stage in stages(ep):
        remove(panel + objects)
        scores = np.asarray(stage.scores).ravel()
        weights = np.asarray(stage.weights).ravel()
        panel = [
            label(s, stage.name, -6.2, 2.5, 0.32, TEAL),
            label(s, stage.note, -6.2, 1.9, 0.2, MUTED),
        ]
        for x, name in [(-7.8, "分数"), (-6.3, "权重"), (-4.7, "值 v")]:
            panel.append(label(s, name, x, 1.25, 0.21, MUTED))
        gap = 0.36 if len(weights) > 5 else 0.58
        for i, (score, w, value) in enumerate(zip(scores, weights, stage.values)):
            yy = 0.7 - i * gap
            panel += [
                label(s, f"{score:.2f}", -7.8, yy, 0.21),
                label(
                    s,
                    f"{w:.3f}",
                    -6.3,
                    yy,
                    0.21,
                    GOLD if i == int(weights.argmax()) else TEAL,
                ),
                label(s, short(value), -4.7, yy, 0.18),
            ]
        s.wait(1.8)
        objects, positions = weight_columns(
            s,
            weights,
            key_names=[f"{k:g}" for k in model(ep).KEYS] if ep == "10.2" else None,
        )
        travelers = [
            dot(s, *project(s, pos), GOLD if i == int(weights.argmax()) else TEAL)
            for i, pos in enumerate(positions)
        ]
        panel.append(label(s, "Σ αᵢvᵢ", 6.6, -1.25, 0.3, TEAL))
        with s.parallel():
            for traveler in travelers:
                traveler.move(to=(6.6, -1.25), duration=0.9)
        remove(travelers)
        panel += [
            label(
                s, "加权输出 = (" + short(stage.output) + ")", -6.2, -2.9, 0.25, CORAL
            ),
            label(s, f"Σ α = {weights.sum():.3f}", -6.2, -3.45, 0.22, MUTED),
        ]
        s.wait(3.1)
    if ep == "10.5":
        d = model(ep)
        remove(panel + objects)
        panel = [
            label(s, "拼接两头，再做输出投影", -6.2, 2.4, 0.29, TEAL),
            label(s, "两头输出", -6.2, 1.6, 0.23, MUTED),
        ]
        panel += vector(s, np.round(d.CONCAT, 3), -6.2, 0.65)
        panel += [
            label(s, "Wₒ × concat", -6.2, -0.5, 0.27),
            label(s, "最终输出 (" + short(d.YHAT) + ")", -6.2, -1.25, 0.26, CORAL),
        ]
        # Two rows in depth explicitly separate the independent heads.
        for i, w in enumerate(d.HEAD_WEIGHTS):
            layer, _ = weight_columns(s, w, z=(i - 0.5) * 1.8, color=(TEAL, CORAL)[i])
            objects += layer
        label(
            s, "前后两排对应两个头，分别学习不同的关系。", -8.3, -4.5, 0.24, left=True
        )
        label(s, "后排：头 1 · 前排：头 2", 2.6, -2.2, 0.22, MUTED)
        s.wait(4)
    elif ep == "15.5":
        d = model(ep)
        remove(panel + objects)
        objects, _ = weight_columns(s, d.TRAVELER_WEIGHTS)
        panel = [
            label(s, "对齐后，交给示例分类器", -6.2, 2.4, 0.29, TEAL),
            label(s, "使用前提词元 2 的对齐表示", -6.2, 1.7, 0.2, MUTED),
        ]
        for i, name in enumerate(("蕴含", "矛盾", "中立")):
            panel.append(
                label(
                    s,
                    f"{name}  {d.PROBABILITIES[i]:.3f}",
                    -6.2,
                    0.8 - i * 0.65,
                    0.29,
                    CORAL,
                )
            )
        label(
            s,
            "固定向量与未训练分类器；这里只演示对齐与汇总机制。",
            -8.3,
            -4.5,
            0.24,
            left=True,
        )
        s.wait(4)
    elif ep == "10.6":
        label(
            s,
            "位置编码补充顺序信息；二维示例中使用正弦与余弦。",
            -8.3,
            -4.5,
            0.24,
            left=True,
        )
    else:
        label(s, "分数 → softmax → 权重 → 加权求和", -8.3, -4.5, 0.25, left=True)
    s.wait(5)
    return s
