"""Native bounding boxes, multiscale planes and embedding geometry."""

import importlib
import numpy as np
from zanim import Box3D, Transform3D, Vec3
from zanim_scenes.style import BLUE, CORAL, GOLD, MUTED, TEAL, label
from zanim_scenes.mechanisms import start, remove, dot, line, vector
from zanim_scenes.foundations import curve, axes

EPISODE_IDS = ("13.3", "13.4", "13.5", "14.1", "14.7")


def model(ep):
    return importlib.import_module("zanim_scenes.models.ep_" + ep.replace(".", "_"))


def outline(s, box, p, color, width=0.03):
    x1, y1, x2, y2 = box
    return curve(
        s, [p(x1, y1), p(x2, y1), p(x2, y2), p(x1, y2), p(x1, y1)], color, width
    )


def boxes(ep, width, height, fps):
    d = model(ep)
    s = start(ep, width, height, fps, "同一张图像上的几何关系，全部由坐标计算。")
    if ep == "13.3":

        def p(x, y):
            return (-0.6 + float(x) * 2.8, 2.5 - float(y) * 2.8)

        outline(s, (0, 0, 2.6, 2), p, BLUE, 0.015)
        boundary = d.object_boundary(100)
        curve(s, [p(*v) for v in np.vstack((boundary, boundary[0]))], MUTED, 0.06)
        label(s, "图像坐标：原点在左上，x 向右，y 向下", 2.9, -3.65, 0.21, MUTED)
        panel = [
            label(s, "两个角点", -6.2, 2.4, 0.33, TEAL),
            label(s, "(x₁, y₁, x₂, y₂)", -6.2, 1.5, 0.26),
        ]
        panel += vector(s, d.CORNER.ravel(), -6.2, -0.25)
        outline(s, d.CORNER[0], p, TEAL)
        for x, y, name in [(d.X1, d.Y1, "左上角"), (d.X2, d.Y2, "右下角")]:
            dot(s, *p(x, y), GOLD)
            px, py = p(x, y)
            label(s, name, px, py + (0.35 if name == "左上角" else -0.35), 0.21, GOLD)
        s.wait(6)
        remove(panel)
        panel = [
            label(s, "中心与尺寸", -6.2, 2.4, 0.33, TEAL),
            label(s, "(cₓ, cᵧ, w, h)", -6.2, 1.5, 0.26),
        ]
        panel += vector(s, np.round(d.CENTER.ravel(), 3), -6.2, -0.25)
        marker = dot(s, *p(d.X1, d.Y1), CORAL)
        marker.move(to=p(d.CX, d.CY), duration=1.5)
        line(s, p(d.X1, d.CY), p(d.X2, d.CY), CORAL)
        line(s, p(d.CX, d.Y1), p(d.CX, d.Y2), CORAL)
        label(s, f"w = {d.WIDTH:.1f}", *p(d.CX, d.Y1 - 0.14), 0.22, CORAL)
        label(s, f"h = {d.HEIGHT:.1f}", *p(d.X2 + 0.25, d.CY), 0.22, CORAL)
        s.wait(6)
        label(s, "中心是两角的中点；宽高是两角坐标之差。", -8.3, -4.5, 0.24, left=True)
        s.wait(5)
    else:

        def p(x, y):
            return (-0.7 + float(x) * 6.0, 2.5 - float(y) * 6.0)

        for i in range(7):
            line(s, p(i / 6, 0), p(i / 6, 1), BLUE, 0.009)
            line(s, p(0, i / 6), p(1, i / 6), BLUE, 0.009)
        outline(s, d.GROUND_TRUTH, p, GOLD, 0.055)
        label(s, "金色：真实框 · 青色：候选锚框", 2.3, -3.95, 0.21, MUTED)
        panel = []
        current = []
        for i, anchor in enumerate(d.ANCHORS):
            remove(panel + current)
            panel = [
                label(
                    s, f"锚框 {i + 1} · 宽高比 {d.RATIOS[i]:g}", -6.2, 2.4, 0.29, TEAL
                ),
                label(s, "IoU = 交集面积 / 并集面积", -6.2, 1.4, 0.23),
                label(s, f"IoU = {d.IOU[i]:.2f}", -6.2, 0.4, 0.4, GOLD),
            ]
            current = [
                outline(s, anchor, p, TEAL, 0.04),
                outline(s, d.INTER_BOXES[i], p, CORAL, 0.07),
            ]
            cx = (anchor[0] + anchor[2]) / 2
            cy = (anchor[1] + anchor[3]) / 2
            marker = dot(s, *p(cx, cy), GOLD)
            marker.move(to=p(anchor[2], anchor[3]), duration=1)
            marker.remove()
            s.wait(5)
        remove(panel + current)
        outline(s, d.ANCHORS[d.WINNER], p, TEAL, 0.055)
        label(s, "本例选择锚框 1", -6.2, 2.4, 0.31, TEAL)
        label(s, "0.50 > 0.25", -6.2, 1.3, 0.38, GOLD)
        label(s, "位置相同，形状不同，覆盖程度也不同。", -8.3, -4.5, 0.24, left=True)
        s.wait(7)
    return s


def multiscale(width, height, fps):
    d = model("13.5")
    s = start(
        "13.5", width, height, fps, "把两种分辨率分层展开，比较它们对同一小目标的覆盖。"
    )
    panel = []

    def tile(x, z, y, size, color):
        return s.add(
            Box3D(
                Vec3(size, 0.035, size),
                color=color,
                transform=Transform3D.translation(x, y, z),
            )
        )

    def border(box, y, color):
        x1, z1, x2, z2 = (np.asarray(box) - 0.5) * 3.6
        objs = []
        for x, z, w, h in [
            ((x1 + x2) / 2, z1, x2 - x1, 0.025),
            ((x1 + x2) / 2, z2, x2 - x1, 0.025),
            (x1, (z1 + z2) / 2, 0.025, z2 - z1),
            (x2, (z1 + z2) / 2, 0.025, z2 - z1),
        ]:
            objs.append(
                s.add(
                    Box3D(
                        Vec3(float(w), 0.04, float(h)),
                        color=color,
                        transform=Transform3D.translation(float(x), y, float(z)),
                    )
                )
            )
        return objs

    for n, y, color in [(4, -0.7, TEAL), (2, 1.2, BLUE)]:
        for r in range(n):
            for c in range(n):
                tile(
                    ((c + 0.5) / n - 0.5) * 3.6,
                    ((r + 0.5) / n - 0.5) * 3.6,
                    y,
                    3.6 / n - 0.08,
                    color,
                )
        border(d.GROUND_TRUTH, y + 0.05, GOLD)
    label(s, "上层：2 × 2 · 下层：4 × 4", 2.8, -3.25, 0.24, MUTED)
    label(s, "层间距离仅用于展示；金框在两层表示同一目标。", 2.8, -3.85, 0.2, MUTED)
    for name, y, anchors, hit, iou in [
        ("细网格", -0.7, d.FINE_ANCHORS, d.HIT_FINE, d.FINE_IOU),
        ("粗网格", 1.2, d.COARSE_ANCHORS, d.HIT_COARSE, d.COARSE_IOU),
    ]:
        remove(panel)
        panel = [
            label(s, name, -6.2, 2.4, 0.34, TEAL),
            label(s, f"候选框 IoU = {iou[hit]:.2f}", -6.2, 1.4, 0.3, GOLD),
            label(s, "同一小目标，不同锚框尺度", -6.2, 0.45, 0.22, MUTED),
        ]
        border(anchors[hit], y + 0.10, CORAL)
        # Grow a short vertical pointer at the winning cell, then leave the box visible.
        center = (anchors[hit][:2] + anchors[hit][2:]) / 2
        x, z = (center - 0.5) * 3.6
        point = s.add(
            Box3D(
                Vec3(0.12, 0.6, 0.12),
                color=GOLD,
                transform=Transform3D.translation(float(x), y + 0.4, float(z))
                @ Transform3D.scaling(1, 0.001, 1),
            )
        )
        point.transform(
            to=Transform3D.translation(float(x), y + 0.4, float(z)), duration=1
        )
        s.wait(5)
        point.remove()
    label(
        s,
        "细尺度更贴合本例小目标；多尺度让检测器覆盖不同大小的物体。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(7)
    return s


def skipgram(width, height, fps):
    d = model("14.1")
    s = start(
        "14.1",
        width,
        height,
        fps,
        "用一个中心词—上下文词对，观察真实 SGD 如何改变向量。",
    )
    p = axes(s, (-2, 2), (-2, 2))
    panel = []
    edges = []
    labels = []
    colors = (TEAL, CORAL, BLUE, MUTED)
    positions = [
        d.STATES[0]["v"],
        d.STATES[0]["U"][1],
        d.STATES[0]["U"][2],
        d.STATES[0]["U"][3],
    ]
    points = [dot(s, *p(*v), c) for v, c in zip(positions, colors)]
    names = ("中心 v", "上下文 uₒ", "其他 uₙ", "其他 uₖ")
    for index, st in enumerate(d.STATES):
        remove(panel + edges + labels)
        edges = []
        labels = []
        loss = -np.log(st["p"][d.CONTEXT])
        panel = [
            label(s, f"SGD 第 {index} 步", -6.2, 2.4, 0.33, TEAL),
            label(s, f"P(上下文 | 中心) = {st['p'][d.CONTEXT]:.3f}", -6.2, 1.4, 0.24),
            label(s, f"损失 = {loss:.3f}", -6.2, 0.5, 0.3, GOLD),
        ]
        positions = [st["v"], st["U"][1], st["U"][2], st["U"][3]]
        with s.parallel():
            for point, v in zip(points, positions):
                point.move(to=p(*v), duration=0.8)
        for name, v, c in zip(names, positions, colors):
            x, y = p(*v)
            edges.append(line(s, p(0, 0), (x, y), c, 0.025))
            labels.append(label(s, name, x, y + 0.35, 0.19, c))
        s.wait(1.5)
    label(
        s,
        "优化目标是提高正确上下文的概率；这里只训练一个示例词对。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(4)
    return s


def analogy(width, height, fps):
    d = model("14.7")
    s = start(
        "14.7", width, height, fps, "用向量的平移与相加，画出类比关系的几何意义。"
    )
    p = axes(s, (0, 2.5), (0, 2.5))
    for name, v in zip(d.TOKEN_NAMES, d.TOKENS):
        dot(s, *p(*v), BLUE)
        x, y = p(*v)
        label(s, name, x, y - 0.35, 0.23, MUTED)
    label(s, "二维示意词向量，未经训练", 2.8, -3.65, 0.22, MUTED)
    panel = [
        label(s, "关系向量", -6.2, 2.4, 0.32, TEAL),
        label(s, "woman − man", -6.2, 1.4, 0.3),
    ]
    relation = d.WOMAN - d.MAN
    panel += vector(s, np.round(relation, 3), -6.2, 0.1)
    line(s, p(*d.MAN), p(*d.WOMAN), TEAL, 0.04)
    traveler = dot(s, *p(*d.MAN), GOLD)
    traveler.move(to=p(*d.WOMAN), duration=2)
    traveler.remove()
    s.wait(3)
    remove(panel)
    label(s, "平移到 king", -6.2, 2.4, 0.32, TEAL)
    label(s, "king − man + woman", -6.2, 1.4, 0.26)
    vector(s, np.round(d.COMPOSED, 3), -6.2, 0.1)
    line(s, p(*d.KING), p(*d.COMPOSED), TEAL, 0.04)
    traveler = dot(s, *p(*d.KING), GOLD)
    traveler.move(to=p(*d.COMPOSED), duration=2)
    px, py = p(*d.COMPOSED)
    label(s, "组合向量", px, py + 0.35, 0.22, GOLD)
    s.wait(4)
    line(s, p(*d.COMPOSED), p(*d.QUEEN), CORAL, 0.04)
    label(s, f"距 queen = {np.linalg.norm(d.RESIDUAL):.3f}", -6.2, -1.1, 0.26, CORAL)
    label(
        s,
        "类比点接近 queen，但不完全重合；示意关系不保证普遍成立。",
        -8.3,
        -4.5,
        0.23,
        left=True,
    )
    s.wait(7)
    return s


def build(ep, width=1920, height=1080, fps=60):
    if ep in ("13.3", "13.4"):
        return boxes(ep, width, height, fps)
    return {"13.5": multiscale, "14.1": skipgram, "14.7": analogy}[ep](
        width, height, fps
    )
