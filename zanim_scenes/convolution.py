"""Native Zanim spatial windows and tensor/channel demonstrations."""

import importlib
import numpy as np
from zanim import Box3D, Color, Transform3D, Vec3
from scripts.episode_info import EPISODES
from zanim_scenes.style import (
    BLUE,
    CORAL,
    GOLD,
    MUTED,
    TEAL,
    COLORS,
    label,
    matrix,
    new_scene,
    intro,
    header,
)

EPISODE_IDS = ("06.2", "06.3", "06.5", "06.6", "07.2", "07.3", "07.4", "07.7")


def model(ep):
    return importlib.import_module("zanim_scenes.models.ep_" + ep.replace(".", "_"))


def remove(items):
    for item in items:
        item.remove()


def cell_color(value, scale, base):
    if value < 0:
        base = CORAL
    strength = 0.45 + 0.55 * min(abs(float(value)) / max(scale, 1e-9), 1)
    return Color(*[int(v * strength) for v in (base.r, base.g, base.b)])


def grid(s, values, x=0.0, y=0.0, z=0.0, cell=0.6, color=BLUE, scale=None):
    values = np.asarray(values)
    if scale is None:
        scale = max(float(np.abs(values).max()), 1e-9)
    h, w = values.shape
    items = []
    for r in range(h):
        for c in range(w):
            pos = (x + (c - (w - 1) / 2) * cell, y, z + (r - (h - 1) / 2) * cell)
            item = s.add(
                Box3D(
                    Vec3(cell * 0.9, 0.1, cell * 0.9),
                    color=cell_color(values[r, c], scale, color),
                    transform=Transform3D.translation(*pos),
                )
            )
            items.append((item, pos))
    return items


def plain(handles):
    return [h for h, _ in handles]


def start(ep, width, height, fps, subtitle):
    s = new_scene(width, height, fps)
    title, explanation = EPISODES[ep]
    intro(s, title, ep, explanation)
    header(s, title, ep, subtitle)
    return s


def window_scene(ep, width, height, fps):
    d = model(ep)
    s = start(ep, width, height, fps, "窗口走过哪里，就在输出里写下一个结果。")
    label(s, "输入 / 滑动窗口", 0.6, -2.8, 0.25, TEAL)
    label(s, "输出", 4.9, -2.8, 0.25, CORAL)
    label(s, "平面位置对应行列；亮度表示数值大小。", 2.5, -3.5, 0.21, MUTED)
    phases = (
        [(0, 1, "互相关：逐项相乘后相加")]
        if ep == "06.2"
        else [
            (0, 1, "无填充 · 步幅 1"),
            (1, 1, "边缘补零 · 步幅 1"),
            (1, 2, "边缘补零 · 步幅 2"),
        ]
    )
    left = []
    base = []
    written = []
    footer = []
    for phase_index, (padding, stride, name) in enumerate(phases):
        remove(left + plain(base) + written + footer)
        source = np.pad(d.INPUT_X, padding)
        kernel = d.KERNEL
        kh, kw = kernel.shape
        out = (
            d.corr2d(d.INPUT_X, kernel)
            if ep == "06.2"
            else d.corr2d(d.INPUT_X, kernel, padding=padding, stride=stride)
        )
        base = grid(s, source, x=-0.7, cell=0.58)
        written = []
        left = [
            label(s, name, -6.2, 2.5, 0.28, TEAL),
            label(s, "窗口内的数值", -7.5, 1.85, 0.2, MUTED),
            label(s, "卷积核", -4.9, 1.85, 0.2, MUTED),
        ]
        left += matrix(s, kernel, -4.9, 0.65, 0.52)
        left += [
            label(
                s,
                f"{source.shape[0]} × {source.shape[1]} → {out.shape[0]} × {out.shape[1]}",
                -6.2,
                -1.4,
                0.34,
            ),
            label(
                s, f"填充 p = {padding} · 步幅 s = {stride}", -6.2, -2.05, 0.22, MUTED
            ),
        ]
        if ep == "06.3":
            left += [
                label(s, "输出边长 = ⌊(n − k + 2p) / s⌋ + 1", -6.2, -2.8, 0.21, MUTED)
            ]
        footer = [label(s, name, -8.3, -4.5, 0.26, left=True)]
        s.wait(0.8)
        detail = []
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                remove(detail)
                rr, cc = r * stride, c * stride
                patch = source[rr : rr + kh, cc : cc + kw]
                detail = matrix(s, patch, -7.5, 0.65, 0.52, TEAL)
                detail += [label(s, f"加权和 = {out[r, c]:g}", -6.2, -0.65, 0.3, GOLD)]
                cx = -0.7 + (cc + (kw - 1) / 2 - (source.shape[1] - 1) / 2) * 0.58
                cz = (rr + (kh - 1) / 2 - (source.shape[0] - 1) / 2) * 0.58
                moving = grid(s, np.ones_like(kernel), cx, 0.35, cz, 0.58, TEAL)
                s.wait(1.8 if ep == "06.2" else 0.32)
                pos = (
                    2.8 + (c - (out.shape[1] - 1) / 2) * 0.58,
                    0,
                    (r - (out.shape[0] - 1) / 2) * 0.58,
                )
                result = s.add(
                    Box3D(
                        Vec3(0.50, 0.13, 0.50),
                        color=cell_color(out[r, c], float(np.abs(out).max()), CORAL),
                        transform=Transform3D.translation(cx, 0.4, cz),
                    )
                )
                result.transform(
                    to=Transform3D.translation(*pos),
                    duration=0.5 if ep == "06.2" else 0.22,
                )
                written.append(result)
                remove(plain(moving))
        remove(detail)
        remove(left)
        left = [
            label(s, name, -6.2, 2.5, 0.28, TEAL),
            label(s, "完整输出", -6.2, 1.8, 0.23, MUTED),
        ]
        left += matrix(s, out, -6.2, 0.3, 0.65, CORAL)
        left += [
            label(s, f"填充 p = {padding} · 步幅 s = {stride}", -6.2, -1.6, 0.25, MUTED)
        ]
        s.wait(2)
    s.wait(4 if ep == "06.2" else 2)
    return s


def pooling_scene(width, height, fps):
    ep = "06.5"
    d = model(ep)
    s = start(
        ep, width, height, fps, "同一个窗口：最大汇聚保留最大值，平均汇聚取均值。"
    )
    grid(s, d.INPUT, x=-0.7, cell=0.66)
    label(s, "4 × 4 输入", 0.6, -2.7, 0.25, TEAL)
    label(s, "2 × 2 输出", 4.8, -2.7, 0.25, CORAL)
    label(s, "窗口 2 × 2 · 步幅 2 · 没有可训练参数", 2.5, -3.4, 0.21, MUTED)
    panel = []
    written = []
    for mode, out, color in [("max", d.MAX_OUT, CORAL), ("avg", d.AVG_OUT, GOLD)]:
        remove(panel + written)
        panel = []
        written = []
        name = "最大汇聚" if mode == "max" else "平均汇聚"
        panel = [label(s, name, -6.1, 2.5, 0.38, color)]
        detail = []
        for rec in d.WINDOWS:
            remove(detail)
            patch = rec["patch"]
            value = rec[mode]
            detail = matrix(s, patch, -6.1, 1.2, 0.78, TEAL)
            operation = "取窗口最大值" if mode == "max" else f"({patch.sum():g}) ÷ 4"
            detail += [
                label(s, operation, -6.1, -0.3, 0.26, MUTED),
                label(s, f"= {value:g}", -6.1, -1.2, 0.48, color),
            ]
            r, c = rec["origin"]
            cx = -0.7 + (c - 1) * 0.66
            cz = (r - 1) * 0.66
            highlight = grid(s, np.ones((2, 2)), cx, 0.2, cz, 0.66, TEAL)
            s.wait(1.1)
            rr, cc = rec["out_row"], rec["out_col"]
            result = s.add(
                Box3D(
                    Vec3(0.56, 0.14, 0.56),
                    color=cell_color(value, float(out.max()), color),
                    transform=Transform3D.translation(cx, 0.3, cz),
                )
            )
            result.transform(
                to=Transform3D.translation(
                    2.8 + (cc - 0.5) * 0.66, 0, (rr - 0.5) * 0.66
                ),
                duration=0.65,
            )
            written.append(result)
            remove(plain(highlight))
        remove(detail)
        panel += matrix(s, out, -6.1, 0.4, 0.8, color)
        panel += [label(s, "每个窗口只产生一个输出。", -6.1, -1.5, 0.25, MUTED)]
        s.wait(3)
    label(s, "空间变小；最大值与平均值保留的是不同信息。", -8.3, -4.5, 0.27, left=True)
    s.wait(3)
    return s


def tensor_stages(ep):
    d = model(ep)
    if ep == "06.6":
        return [
            (name, a[..., None] if a.ndim == 2 else a, note)
            for name, a, note in [
                ("输入", d.INPUT_X, "8 × 8 的单通道示例图像"),
                ("卷积 1", d.CONV_1, "3 × 3 核，无填充"),
                ("平均汇聚 1", d.POOL_1, "2 × 2 窗口，步幅 2"),
                ("卷积 2", d.CONV_2, "2 × 2 核，无填充"),
                ("平均汇聚 2 / 展平", d.POOL_2, "1 × 1 特征展平为向量"),
                (
                    "全连接输出",
                    d.OUTPUT_O.reshape(1, 1, 3),
                    "未训练的类别分数，不是预测概率",
                ),
            ]
        ]
    return [
        (name, a, note)
        for name, a, note in [
            ("输入", d.INPUT_X, "空间 8 × 8，通道 1"),
            ("卷积块 1", d.CONV_1, "3 × 3 卷积 + ReLU，保持空间尺寸"),
            ("最大汇聚 1", d.POOL_1, "空间边长减半，通道数不变"),
            ("卷积块 2", d.CONV_2, "再次应用同类块，通道数增至 4"),
            ("最大汇聚 2", d.POOL_2, "空间缩至 2 × 2，保留 4 个通道"),
        ]
    ]


def tensor_scene(ep, width, height, fps):
    s = start(ep, width, height, fps, "空间尺寸与通道数，是两种不同的变化。")
    stages = tensor_stages(ep)
    # All displayed feature maps are actual numerical forward-pass outputs.
    scale = max(float(np.abs(a).max()) for _, a, _ in stages)
    channel_caption = label(s, "每一层平面 = 一个通道", 2.6, -2.9, 0.25, TEAL)
    label(
        s, "格数对应空间尺寸；亮度对应绝对值，珊瑚色表示负值。", 2.6, -3.5, 0.18, MUTED
    )
    label(s, "简化结构演示 · 固定随机权重 · 未训练", -8.3, -4.5, 0.25, left=True)
    old = []
    panel = []
    for index, (name, array, note) in enumerate(stages):
        a = array if array.ndim == 3 else array.reshape(1, 1, -1)
        h, w, channels = a.shape
        remove(panel)
        panel = [
            label(s, name, -6.1, 2.5, 0.36, TEAL),
            label(
                s,
                "输出分数向量：3"
                if ep == "06.6" and name == "全连接输出"
                else f"H × W × C = {h} × {w} × {channels}",
                -6.1,
                1.7,
                0.3,
            ),
            label(s, note, -6.1, 1.0, 0.21, MUTED),
        ]
        dense_output = ep == "06.6" and name == "全连接输出"
        if dense_output:
            channel_caption.remove()
            channel_caption = label(s, "每个色块 = 一个输出分数", 2.6, -2.9, 0.25, TEAL)
        sample = a.reshape(-1, 1) if dense_output else a[:3, :3, 0]
        panel += [
            label(
                s,
                "输出向量（四舍五入）"
                if dense_output
                else "通道 1 · 左上角数值（四舍五入）",
                -6.1,
                0.2,
                0.19,
                MUTED,
            )
        ]
        panel += matrix(s, np.round(sample, 2), -6.1, -1.1, 0.74)
        new = []
        for ch in range(channels):
            new += grid(
                s,
                a[:, :, ch],
                x=0.1,
                y=ch * 0.64 + 0.4,
                cell=0.4,
                color=BLUE,
                scale=scale,
            )
        with s.parallel():
            for item in old:
                item.fade_out(duration=0.5)
            for item, pos in new:
                item.transform(
                    to=Transform3D.translation(pos[0], pos[1] - 0.4, pos[2]),
                    duration=0.7,
                )
        remove(old)
        old = plain(new)
        s.wait(3.4)
    s.wait(3)
    return s


def channels_scene(ep, width, height, fps):
    d = model(ep)
    s = start(ep, width, height, fps, "把通道分开看，再看它们如何组合。")
    label(s, "每层是一张特征图；层间距离仅用于展示。", 2.6, -3.4, 0.21, MUTED)
    panel = []

    def explain(title, note, values):
        nonlocal panel
        remove(panel)
        panel = [
            label(s, title, -6.1, 2.4, 0.33, TEAL),
            label(s, note, -6.1, 1.7, 0.22, MUTED),
        ]
        panel += matrix(s, np.round(values, 2), -6.1, 0.1, 0.7)

    if ep == "07.3":
        inputs = []
        for ch in range(2):
            inputs += grid(
                s, d.INPUT_X[ch], x=-1, y=ch * 0.8, cell=0.65, color=COLORS[ch]
            )
        explain("1 × 1 卷积：混合通道", "左上角输入向量与权重", d.WEIGHT_1X1)
        panel += [label(s, "(1, 0) → (1, 2, 0)", -6.1, -1.65, 0.29, GOLD)]
        s.wait(4)
        outputs = []
        for ch in range(3):
            layer = grid(
                s, d.FEATURES[ch], x=-1, y=ch * 0.8 + 0.25, cell=0.65, color=COLORS[ch]
            )
            with s.parallel():
                for item, pos in layer:
                    item.transform(
                        to=Transform3D.translation(pos[0] + 3, pos[1], pos[2]),
                        duration=1,
                    )
            outputs += layer
            s.wait(1)
        s.wait(2)
        remove(plain(inputs))
        remove(plain(outputs))
        explain(
            "全局平均汇聚", "每张特征图取平均，得到一个分数", d.LOGITS.reshape(-1, 1)
        )
        panel += [label(s, "结果：(4, 5, 3)", -6.1, -1.65, 0.29, GOLD)]
        for ch in range(3):
            layer = grid(
                s, d.FEATURES[ch], x=0, y=ch * 0.8, cell=0.65, color=COLORS[ch]
            )
            s.wait(1)
            with s.parallel():
                for item, _ in layer:
                    item.transform(
                        to=Transform3D.translation(1.5, ch * 0.8, 0), duration=0.8
                    )
            remove(plain(layer))
            grid(s, [[d.LOGITS[ch]]], x=1.5, y=ch * 0.8, cell=0.6, color=COLORS[ch])
        label(s, "空间汇聚成 1 × 1；通道 3 保留下来。", -8.3, -4.5, 0.27, left=True)
        s.wait(5)
    elif ep == "07.4":
        names = ["1 × 1 卷积", "3 × 3 卷积", "5 × 5 卷积", "3 × 3 最大汇聚"]
        maps = [d.Y1, d.Y3, d.Y5, d.YP]
        branches = []
        explain(
            "四条分支，并行读取同一个输入",
            "中心位置的四个结果",
            d.TRAVELER_CHANNELS.reshape(-1, 1),
        )
        for i, values in enumerate(maps):
            branches.append(
                grid(
                    s,
                    values,
                    x=(-1.2 if i % 2 == 0 else 1.2),
                    y=(i // 2) * 1.3,
                    cell=0.45,
                    color=COLORS[i],
                )
            )
        label(s, "蓝：1×1　绿：3×3　珊瑚：5×5　金：汇聚", 2.4, -2.8, 0.18, MUTED)
        s.wait(6)
        explain(
            "沿通道轴拼接", "3 × 3 × 1 → 3 × 3 × 4", d.TRAVELER_CHANNELS.reshape(-1, 1)
        )
        with s.parallel():
            for ch, layer in enumerate(branches):
                for i, (item, _) in enumerate(layer):
                    r, c = divmod(i, 3)
                    item.transform(
                        to=Transform3D.translation(
                            (c - 1) * 0.45, ch * 0.7, (r - 1) * 0.45
                        ),
                        duration=2,
                    )
        label(
            s,
            "空间尺寸保持一致；拼接增加通道，数值不相加。",
            -8.3,
            -4.5,
            0.26,
            left=True,
        )
        s.wait(9)
    else:
        for ch in range(2):
            grid(s, d.INPUT_X[ch], y=ch * 0.7, cell=0.8, color=COLORS[ch])
        explain(
            "从 2 个输入通道开始", "同一位置的通道值", d.INPUT_X[:, 0, 0].reshape(-1, 1)
        )
        s.wait(4)
        for stage, (feature, stack) in enumerate(
            ((d.FEATURE_1, d.STACK_1), (d.FEATURE_2, d.STACK_2))
        ):
            explain(
                f"第 {stage + 1} 层：读取已有全部通道",
                f"拼接后保留 {stack.shape[0]} 个通道",
                stack[:, 0, 0].reshape(-1, 1),
            )
            new = grid(
                s,
                feature[0],
                x=2.2,
                y=(stage + 2) * 0.7,
                cell=0.8,
                color=COLORS[stage + 2],
            )
            s.wait(2)
            with s.parallel():
                for item, pos in new:
                    item.transform(
                        to=Transform3D.translation(pos[0] - 2.2, pos[1], pos[2]),
                        duration=1.5,
                    )
            s.wait(3)
        label(s, "旧通道原样保留；新通道接到后面。", -8.3, -4.5, 0.27, left=True)
        s.wait(5)
    return s


def build(ep, width=1920, height=1080, fps=60):
    if ep in ("06.2", "06.3"):
        return window_scene(ep, width, height, fps)
    if ep == "06.5":
        return pooling_scene(width, height, fps)
    if ep in ("06.6", "07.2"):
        return tensor_scene(ep, width, height, fps)
    return channels_scene(ep, width, height, fps)
