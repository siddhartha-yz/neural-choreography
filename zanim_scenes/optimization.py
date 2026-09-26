"""Shared 3D presentation of the original, distinct optimizer trajectories."""

import importlib
import math
from zanim import Surface3D, Transform3D
from scripts.episode_info import EPISODES
from zanim_scenes.mesh import sphere, ribbon
from zanim_scenes.style import (
    BLUE,
    CORAL,
    GOLD,
    MUTED,
    TEAL,
    label,
    new_scene,
    intro,
    header,
)

# Path name, method label, update equation, learning-rate / state description.
CONFIG = {
    "11.3": (
        1.0,
        [
            (
                "PATH_GOOD",
                "梯度下降 · η = 0.1",
                "w ← w − ηg",
                "较小步长，稳步向谷底移动",
            ),
            ("PATH_LARGE", "梯度下降 · η = 0.4", "w ← w − ηg", "较大步长，会跨过谷底"),
        ],
    ),
    "11.4": (
        1.0,
        [
            ("PATH_FULL", "完整梯度", "g = 所有样本梯度的均值", "η = 0.1"),
            ("PATH_SGD", "单样本 SGD", "g = 当前样本的梯度", "η = 0.1 · 路径会有抖动"),
        ],
    ),
    "11.6": (
        0.1,
        [
            ("PATH_GD", "普通梯度下降", "w ← w − ηg", "η = 0.4"),
            (
                "PATH_MOM",
                "动量法",
                "v ← 0.5v + g；w ← w − ηv",
                "η = 0.4 · 累积过去的方向",
            ),
        ],
    ),
    "11.7": (
        0.1,
        [
            ("PATH_GD", "普通梯度下降", "w ← w − ηg", "η = 0.4"),
            ("PATH_ADA", "AdaGrad", "s ← s + g²", "η = 0.4 · 用 1/√(s+ε) 缩放更新"),
        ],
    ),
    "11.8": (
        0.1,
        [
            ("PATH_ADA", "AdaGrad", "s ← s + g²", "η = 0.3 · 累积全部历史"),
            ("PATH_RMS", "RMSProp", "s ← 0.9s + 0.1g²", "η = 0.3 · 用近期梯度调整步长"),
        ],
    ),
    "11.10": (
        0.1,
        [
            ("PATH_SGD", "单样本 SGD", "w ← w − ηg", "η = 0.4"),
            (
                "PATH_ADAM",
                "Adam",
                "一阶矩 + 二阶矩 + 偏置校正",
                "β₁ = 0.9 · β₂ = 0.999 · η = 0.4",
            ),
        ],
    ),
    "11.11": (
        1.0,
        [
            (
                "PATH",
                "预热 → 余弦衰减",
                "w ← w − η(t)g",
                "前 3 步预热，随后逐步减小学习率",
            )
        ],
    ),
}


def model(episode):
    return importlib.import_module(
        "zanim_scenes.models.ep_" + episode.replace(".", "_")
    )


def build(episode, width=1920, height=1080, fps=60):
    coefficient, series = CONFIG[episode]
    data = model(episode)
    title, explanation = EPISODES[episode]
    objective = lambda w: coefficient * w[0] ** 2 + 2 * w[1] ** 2
    height_scale = 0.045 if coefficient == 1 else 0.12

    def world(w, lift=0.0):
        return (w[0] * 0.6 + 1.1, objective(w) * height_scale + lift, -w[1] * 0.75)

    s = new_scene(width, height, fps)
    intro(s, title, episode, explanation)
    header(s, title, episode, "在同一损失曲面上，看清每一步更新。")
    label(s, "损失函数", -7.9, 2.5, 0.24, TEAL, left=True)
    label(
        s,
        "L = " + ("" if coefficient == 1 else "0.1") + "w₁² + 2w₂²",
        -7.9,
        1.85,
        0.34,
        left=True,
    )
    label(s, "高度 = 损失 L", 2.8, -3.4, 0.25, MUTED)
    label(s, "水平面的两个方向 = 参数 w₁、w₂", 2.8, -3.9, 0.21, MUTED)
    if episode in ("11.4", "11.10"):
        label(s, "L 省略了与参数无关的常数项", -7.9, 1.2, 0.18, MUTED, left=True)
    # The pinned 0.7.0rc1 Surface3D maps (x, height, source-y), with Y up.
    s.add(
        Surface3D(
            lambda x, y: objective((x, y)) * height_scale,
            x_range=(-5.5, 3),
            y_range=(-2.6, 2.6),
            resolution=(65, 49),
            color=BLUE.with_alpha(150),
            transform=Transform3D.translation(1.1, 0, 0)
            @ Transform3D.scaling(0.6, 1, 0.75),
        )
    )
    for level in data.CONTOUR_LEVELS:
        points = []
        for i in range(241):
            angle = 2 * math.pi * i / 240
            w = (
                math.sqrt(level / coefficient) * math.cos(angle),
                math.sqrt(level / 2) * math.sin(angle),
            )
            if -5.5 <= w[0] <= 3 and -2.6 <= w[1] <= 2.6:
                points.append(world(w, 0.008))
            else:
                if len(points) > 1:
                    s.add(ribbon(points, BLUE, 0.012))
                points = []
        if len(points) > 1:
            s.add(ribbon(points, BLUE, 0.012))
    minimum = sphere(0.08, GOLD)
    minimum.transform = Transform3D.translation(*world((0, 0), 0.08))
    s.add(minimum)
    for legend_index, (_, name, _, _) in enumerate(series):
        label(
            s, name, 0.8 + legend_index * 4.2, -2.75, 0.2, (TEAL, CORAL)[legend_index]
        )
    total_steps = sum(len(getattr(data, entry[0])) - 1 for entry in series)
    step_duration = 17 / total_steps
    panel = []
    for series_index, (path_name, name, equation, note) in enumerate(series):
        color = (TEAL, CORAL)[series_index]
        pts = getattr(data, path_name)
        for old in panel:
            old.remove()
        panel = [
            label(s, name, -7.9, 0.45, 0.33, color, left=True),
            label(s, equation, -7.9, -0.35, 0.25, left=True),
            label(s, note, -7.9, -1.0, 0.21, MUTED, left=True),
        ]
        footer = label(s, name + "：" + explanation, -8.3, -4.5, 0.25, left=True)
        ball = sphere(0.115, color)
        ball.transform = Transform3D.translation(*world(pts[0], 0.12))
        traveler = s.add(ball)
        stats = []
        s.wait(1)
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            for item in stats:
                item.remove()
            stats = [
                label(s, f"第 {i + 1} 步", -7.9, -1.85, 0.24, MUTED, left=True),
                label(
                    s,
                    f"L : {objective(a):.3f} → {objective(b):.3f}",
                    -7.9,
                    -2.5,
                    0.3,
                    left=True,
                ),
            ]
            if episode == "11.11":
                stats.append(
                    label(
                        s,
                        f"η = {data.eta_of(i):.4f}",
                        -7.9,
                        -3.15,
                        0.29,
                        color,
                        left=True,
                    )
                )
            elif episode == "11.4" and path_name == "PATH_SGD":
                stats.append(
                    label(
                        s,
                        f"本步样本：{int(data.SGD_INDEX[i]) + 1}",
                        -7.9,
                        -3.15,
                        0.25,
                        color,
                        left=True,
                    )
                )
            elif path_name in ("PATH_ADA", "PATH_RMS"):
                state = getattr(data, "S_RMS" if path_name == "PATH_RMS" else "S_ADA")[
                    i + 1
                ]
                stats.append(
                    label(
                        s,
                        f"s = ({state[0]:.2f}, {state[1]:.2f})",
                        -7.9,
                        -3.15,
                        0.23,
                        color,
                        left=True,
                    )
                )

            def motion(t, a=a, b=b):
                w = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                return Transform3D.translation(*world(w, 0.12))

            traveler.transform_function(motion, duration=step_duration * 0.85)
            track = [world(a + (b - a) * j / 20, 0.035) for j in range(21)]
            s.add(ribbon(track, color, 0.028))
            s.wait(step_duration * 0.15)
        s.wait(1)
        for item in stats:
            item.remove()
        footer.remove()
    label(
        s,
        "示例轨迹由实际更新公式计算；不同问题上的表现会不同。",
        -8.3,
        -4.5,
        0.25,
        left=True,
    )
    s.wait(3 if len(series) > 1 else 5)
    return s
