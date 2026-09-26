"""Actual Zanim 3D contribution accumulation, with exact D2L tensors."""

from zanim import Box3D, Transform3D, Vec3
from zanim_scenes.style import (
    COLORS,
    MUTED,
    TEAL,
    label,
    matrix,
    new_scene,
    intro,
    header,
)

X = [[0, 1], [2, 3]]
K = [[0, 1], [2, 3]]


def contributions():
    layers = []
    total = [[0 for _ in range(3)] for _ in range(3)]
    for r in range(2):
        for c in range(2):
            patch = [[X[r][c] * v for v in row] for row in K]
            for kr in range(2):
                for kc in range(2):
                    total[r + kr][c + kc] += patch[kr][kc]
            layers.append((r, c, patch, [row[:] for row in total]))
    return layers


def build(width=1920, height=1080, fps=60):
    s = new_scene(width, height, fps)
    intro(s, "转置卷积", "13.10", "输入值乘以核，铺到对应位置；重叠处相加。")
    header(s, "转置卷积", "13.10", "把一次乘法，展开成一片贡献。")
    label(s, "输入 X", -7, 2.55, 0.25, TEAL)
    label(s, "卷积核 K", -4.6, 2.55, 0.25, MUTED)
    matrix(s, X, -7, 1.55)
    matrix(s, K, -4.6, 1.55)
    label(s, "2 × 2  →  3 × 3", -5.8, 0.25, 0.25)
    label(s, "步幅 1 · 填充 0", -5.8, -0.3, 0.19, MUTED)
    label(s, "每种颜色 = 一次输入的贡献", 2.6, -3.25, 0.24, MUTED)
    label(s, "色块高度与数值成正比", 2.6, -3.72, 0.20, MUTED)
    for r in range(3):
        for c in range(3):
            s.add(
                Box3D(
                    Vec3(0.91, 0.08, 0.91),
                    color=COLORS[0].with_alpha(90),
                    transform=Transform3D.translation(c - 1, -0.10, r - 1),
                )
            )
    current = [[0] * 3 for _ in range(3)]
    status = label(s, "01 / 取出一个输入值", -8.3, -4.5, 0.28, left=True)
    display = []
    s.wait(1)
    for index, (r, c, patch, total) in enumerate(contributions()):
        status.remove()
        for obj in display:
            obj.remove()
        display = matrix(s, X, -7, 1.55, active=(r, c))
        display += [label(s, f"{X[r][c]} × K", -5.8, -1.2, 0.35, COLORS[index])]
        display += matrix(s, patch, -5.8, -2.4, 0.65, COLORS[index])
        status = label(
            s,
            f"0{index + 1} / 输入 {X[r][c]}：乘以核，向输出位置铺开",
            -8.3,
            -4.5,
            0.28,
            left=True,
        )
        s.wait(0.9)
        blocks = []
        for kr in range(2):
            for kc in range(2):
                v = patch[kr][kc]
                if not v:
                    continue
                rr, cc = r + kr, c + kc
                h = v * 0.22
                target = Transform3D.translation(
                    cc - 1, current[rr][cc] * 0.22 + h / 2, rr - 1
                )
                block = s.add(
                    Box3D(
                        Vec3(0.85, h, 0.85),
                        color=COLORS[index],
                        transform=Transform3D.translation(
                            cc - 1, current[rr][cc] * 0.22 + h / 2 + 0.9, rr - 1
                        ),
                    )
                )
                blocks.append((block, target))
        if blocks:
            with s.parallel():
                for block, target in blocks:
                    block.transform(to=target, duration=1.4)
        else:
            zero = label(s, "0 × K = 0，不增加输出", 2.6, 1.8, 0.25, COLORS[index])
            s.wait(1.4)
            zero.remove()
        current = total
        s.wait(1.4)
    status.remove()
    for obj in display:
        obj.remove()
    label(s, "最终输出 Y", -5.8, -1.15, 0.3, TEAL)
    matrix(s, current, -5.8, -2.45, 0.66)
    label(s, "05 / 重叠处相加：中心格 2 + 2 = 4", -8.3, -4.5, 0.28, left=True)
    label(s, "转置卷积 ≠ 逆卷积", 2.6, 2.5, 0.28, TEAL)
    s.wait(6)
    return s
