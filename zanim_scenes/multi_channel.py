"""Two input channels and two independently computed output channels."""

from zanim import Box3D, Transform3D, Vec3
from zanim_scenes.style import (
    BLUE,
    CORAL,
    GOLD,
    MUTED,
    TEAL,
    label,
    matrix,
    new_scene,
    intro,
    header,
)

X = [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
K = [[[0, 1], [2, 3]], [[1, 2], [3, 4]]]


def partial(channel, kernel):
    return [
        [
            sum(
                channel[r + i][c + j] * kernel[i][j] for i in range(2) for j in range(2)
            )
            for c in range(2)
        ]
        for r in range(2)
    ]


def outputs():
    results = []
    for offset in (0, 1):
        parts = [
            partial(X[ch], [[v + offset for v in row] for row in K[ch]])
            for ch in range(2)
        ]
        results.append(
            (
                parts,
                [[parts[0][r][c] + parts[1][r][c] for c in range(2)] for r in range(2)],
            )
        )
    return results


def tile(s, x, y, z, color, side=0.54, thickness=0.12):
    return s.add(
        Box3D(
            Vec3(side, thickness, side),
            color=color,
            transform=Transform3D.translation(x, y, z),
        )
    )


def build(width=1920, height=1080, fps=60):
    s = new_scene(width, height, fps)
    intro(
        s,
        "多输入多输出通道",
        "06.4",
        "输入通道分别计算后相加，多组卷积核生成多个输出通道。",
    )
    header(s, "多输入多输出通道", "06.4", "一层是一个通道；一组卷积核，生成一层输出。")
    label(s, "输入通道 1", -7.4, 2.5, 0.23, TEAL)
    label(s, "输入通道 2", -4.8, 2.5, 0.23, BLUE)
    matrix(s, X[0], -7.4, 1.22, 0.57)
    matrix(s, X[1], -4.8, 1.22, 0.57)
    label(s, "2 × 3 × 3", -6.1, -0.1, 0.27, MUTED)
    label(s, "输入通道", 0.6, -2.5, 0.25, TEAL)
    label(s, "输出通道", 4.7, -2.5, 0.25, CORAL)
    label(s, "层间距离只用于分开通道，不表示数值大小。", 2.7, -3.55, 0.21, MUTED)
    for ch, color in enumerate((TEAL, BLUE)):
        for r in range(3):
            for c in range(3):
                tile(s, -1.4 + (c - 1) * 0.6, ch * 1.5, (r - 1) * 0.6, color)
    status = label(
        s, "01 / 两个输入通道，各自使用对应的核", -8.3, -4.5, 0.28, left=True
    )
    s.wait(2)
    details = []
    for out, (parts, total) in enumerate(outputs()):
        status.remove()
        for item in details:
            item.remove()
        color = (CORAL, GOLD)[out]
        details = [
            label(s, f"第 {out + 1} 组核 · 左上角窗口", -6.1, -1.05, 0.25, color),
            label(
                s,
                f"{parts[0][0][0]} + {parts[1][0][0]} = {total[0][0]}",
                -6.1,
                -1.7,
                0.4,
                color,
            ),
        ]
        details += matrix(s, total, -6.1, -2.9, 0.67, color)
        status = label(
            s,
            f"0{out + 2} / 对应位置相加 → 输出通道 {out + 1}",
            -8.3,
            -4.5,
            0.28,
            left=True,
        )
        floating = []
        for ch in range(2):
            for r in range(2):
                for c in range(2):
                    item = tile(
                        s,
                        -1.7 + c * 0.6,
                        ch * 1.5 + 0.22,
                        -0.3 + r * 0.6,
                        (TEAL, BLUE)[ch],
                        0.5,
                        0.08,
                    )
                    floating.append(
                        (
                            item,
                            Transform3D.translation(
                                2.1 + c * 0.6,
                                out * 1.5 + ch * 0.4 + 0.2,
                                -0.3 + r * 0.6,
                            ),
                        )
                    )
        with s.parallel():
            for item, target in floating:
                item.transform(to=target, duration=1.8)
        s.wait(1.2)
        with s.parallel():
            for i, (item, _) in enumerate(floating):
                r, c = divmod(i % 4, 2)
                item.transform(
                    to=Transform3D.translation(
                        2.1 + c * 0.6, out * 1.5, -0.3 + r * 0.6
                    ),
                    duration=1,
                )
                item.fade_out(duration=1)
        for item, _ in floating:
            item.remove()
        for r in range(2):
            for c in range(2):
                tile(s, 2.1 + c * 0.6, out * 1.5, -0.3 + r * 0.6, color)
        s.wait(3)
    status.remove()
    label(s, "04 / 两组不同的核，得到两个不同的输出通道", -8.3, -4.5, 0.28, left=True)
    label(s, "2 × 2 × 2", 4.7, -2.95, 0.25, MUTED)
    s.wait(5)
    return s
