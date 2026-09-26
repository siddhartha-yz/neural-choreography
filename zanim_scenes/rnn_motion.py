"""RNN motion study: carry, combine, update; positions encode the true hidden state."""

import math
from dataclasses import replace
import numpy as np
from zanim import Scene, Canvas, Camera3D, Vec3, Box3D, Transform3D
from zanim_scenes.style import BG, TEAL, CORAL, GOLD, MUTED, INK, Color, label, rect
from zanim_scenes.mesh import sphere, ribbon
from zanim.mesh3d import MeshObject3D, TriangleMesh
from zanim_scenes.mechanisms import remove
from zanim_scenes.models import ep_08_4 as data

DIM = Color(62, 85, 105)
TIMES = (-4.8, -1.6, 1.6, 4.8)


def state_point(t):
    h = data.HIDDEN_STATES[t].ravel()
    return np.array([TIMES[t], float(h[0]) * 1.3, float(h[1]) * 1.3])


def orb(s, point, radius=0.14, color=TEAL):
    obj = sphere(radius, color)
    obj.transform = Transform3D.translation(*map(float, point))
    return s.add(obj)


def rail(s, a, b, color=DIM, width=0.012):
    return s.add(ribbon([a, b], color, width))


def plane(s, x, color=DIM):
    items = []
    for z in (-1.3, 0, 1.3):
        items.append(
            s.add(
                Box3D(
                    Vec3(0.015, 2.6, 0.015),
                    color=color,
                    transform=Transform3D.translation(x, 0, z),
                )
            )
        )
    for y in (-1.3, 0, 1.3):
        items.append(
            s.add(
                Box3D(
                    Vec3(0.015, 0.015, 2.6),
                    color=color,
                    transform=Transform3D.translation(x, y, 0),
                )
            )
        )
    return items


def pulse(s, point, color):
    # A ring lies in the state plane. Expanding it marks one discrete update.
    items = []
    for angle in np.linspace(0, 2 * math.pi, 20, endpoint=False):
        p = np.asarray(point) + np.array(
            [0, 0.18 * math.cos(angle), 0.18 * math.sin(angle)]
        )
        items.append((orb(s, p, 0.026, color), angle))
    with s.parallel():
        for obj, angle in items:
            p = np.asarray(point) + np.array(
                [0, 0.64 * math.cos(angle), 0.64 * math.sin(angle)]
            )
            obj.transform(to=Transform3D.translation(*map(float, p)), duration=0.65)
            obj.fade_out(duration=0.65)
    remove([obj for obj, _ in items])


def fly(s, point, a, b, duration, arc=0.4):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def motion(u):
        p = a + (b - a) * u + np.array([0, math.sin(math.pi * u) * arc, 0])
        return Transform3D.translation(*map(float, p))

    point.transform_function(motion, duration=duration)


def ensemble_states(count=48):
    """Independent seeded sequences; sequence 0 is the original teaching example."""
    rng = np.random.default_rng(84)
    tokens = rng.normal(0, 1.35, (count, 3, 2))
    tokens[0] = np.asarray(data.TOKENS).reshape(3, 2)
    states = np.zeros((count, 4, 2))
    for t in range(3):
        states[:, t + 1] = np.tanh(
            tokens[:, t] @ data.WEIGHT_XH.T
            + states[:, t] @ data.WEIGHT_HH.T
            + data.BIAS.ravel()
        )
    return tokens, states


def ensemble_point(states, i, t):
    return np.array([TIMES[t], *list(states[i, t] * 1.3)])


def strand(s, a, b, color, radius=0.014):
    """Round cross-section keeps paths legible at every camera angle."""
    a, b = np.asarray(a), np.asarray(b)
    direction = b - a
    direction /= np.linalg.norm(direction)
    u = np.cross(direction, [0.0, 1.0, 0.0])
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(direction, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    vertices, normals, indices = [], [], []
    count = 8
    for center in (a, b):
        for angle in np.linspace(0, 2 * math.pi, count, endpoint=False):
            normal = u * math.cos(angle) + v * math.sin(angle)
            vertices.append(Vec3(*map(float, center + normal * radius)))
            normals.append(Vec3(*map(float, normal)))
    for i in range(count):
        j = (i + 1) % count
        indices.extend((i, j, i + count, j, j + count, i + count))
    return s.add(
        MeshObject3D(
            TriangleMesh(tuple(vertices), tuple(normals), tuple(indices)), color=color
        )
    )


def ensemble(s, states):
    palette = (TEAL, Color(94, 174, 249), Color(165, 148, 250), CORAL)
    dots = [
        orb(s, ensemble_point(states, i, 0), 0.07 if i else 0.16, palette[i % 4])
        for i in range(len(states))
    ]
    paths = []
    for t in (1, 2, 3):
        with s.parallel():
            for i, point in enumerate(dots):
                fly(
                    s,
                    point,
                    ensemble_point(states, i, t - 1),
                    ensemble_point(states, i, t),
                    1.05,
                    0,
                )
        for i in range(len(states)):
            paths.append(
                strand(
                    s,
                    ensemble_point(states, i, t - 1),
                    ensemble_point(states, i, t),
                    palette[i % 4],
                    0.014 if i else 0.026,
                )
            )
    return dots, paths


def build(width=1920, height=1080, fps=60):
    camera = Camera3D(
        position=Vec3(7, 4.8, 13), target=Vec3(0, 0.3, 0), orthographic_height=9.0
    )
    s = Scene(canvas=Canvas(width, height, width / 19.2), fps=fps, camera3d=camera)
    rect(s, 0, 0, 19.2, 10.8, BG, -100)
    label(s, "循环神经网络 · 08.4", -8.3, 4.5, 0.24, MUTED, left=True)
    title = label(s, "48 条序列，48 种记忆。", -8.3, 3.7, 0.52, INK, left=True)
    label(
        s,
        "48 条独立序列 · 共用权重 · 状态分别计算，彼此不交换",
        -8.3,
        2.9,
        0.20,
        MUTED,
        left=True,
    )
    label(s, "RNN / STATE IN MOTION", 6.6, 4.5, 0.16, TEAL)
    # Quiet spatial reference. Only the active state has strong contrast.
    for z in (-2.1, 2.1):
        rail(s, (-5.4, -1.85, z), (5.4, -1.85, z), DIM, 0.008)
    for x in TIMES:
        plane(s, x)
    _, batch_states = ensemble_states()
    opening = label(
        s, "同一套 RNN，接收不同的输入。", -8.3, -3.25, 0.34, INK, left=True
    )
    opening_note = label(
        s,
        "每条线连接同一序列的状态；交叉不表示序列间通信。",
        -8.3,
        -3.9,
        0.22,
        MUTED,
        left=True,
    )
    group_dots, group_paths = ensemble(s, batch_states)
    s.wait(0.6)
    with s.parallel():
        for item in group_dots + group_paths:
            item.opacity(to=0.08, duration=0.7)
        title.fade_out(duration=0.5)
        opening.fade_out(duration=0.5)
        opening_note.fade_out(duration=0.5)
        s.camera3d.configure(
            to=replace(
                camera.state(), position=Vec3(5.5, 4.2, 14), orthographic_height=8.4
            ),
            duration=0.7,
        )
    remove([title, opening, opening_note])
    title = label(s, "跟随一条序列，看清每次更新。", -8.3, 3.7, 0.46, INK, left=True)
    history = orb(s, state_point(0), 0.19, TEAL)
    label(
        s,
        "h₀ → h₁ → h₂ → h₃   /   网格纵向、纵深对应两个状态分量",
        0,
        -2.65,
        0.21,
        MUTED,
    )
    label(s, "初始状态 h₀ = (0, 0)", -8.3, 2.35, 0.19, MUTED, left=True)
    panel = []
    phase = []

    def caption(k, headline, subtitle):
        nonlocal panel
        with s.parallel():
            for item in panel:
                item.fade_out(duration=0.18)
        remove(panel)
        panel = [
            label(s, f"0{k}", -8.3, -3.25, 0.42, TEAL, left=True),
            label(s, headline, -7.25, -3.17, 0.32, INK, left=True),
            label(s, subtitle, -7.25, -3.82, 0.22, MUTED, left=True),
        ]
        for item in panel:
            item.opacity(to=0, duration=0)
        with s.parallel():
            for item in panel:
                item.fade_in(duration=0.25)

    for t in (1, 2, 3):
        remove(phase)
        phase = []
        old = state_point(t - 1)
        target = state_point(t)
        merge = np.array([TIMES[t], old[1], old[2]])
        caption(
            t, "把上一刻，带到这一刻。", f"序列 01 · 时间步 {t} · 青色状态 + 珊瑚色输入"
        )
        active = plane(s, TIMES[t], Color(97, 179, 190))
        # Three smaller followers make the direction readable during travel.
        followers = [orb(s, old, 0.04, TEAL) for _ in range(3)]
        with s.parallel():
            fly(s, history, old, merge, 1.25, 0.25)
            for i, obj in enumerate(followers):
                # Each delayed follower follows the same spatial route.
                def trail(u, old=old.copy(), merge=merge.copy()):
                    p = (
                        old
                        + (merge - old) * u
                        + np.array([0, math.sin(math.pi * u) * 0.25, 0])
                    )
                    return Transform3D.translation(*map(float, p))

                obj.transform_function(trail, duration=1.25, at=0.09 * (i + 1))
        remove(followers)
        rail(s, old, merge, Color(56, 104, 110), 0.022)
        caption(
            t,
            "新输入，加入当前计算。",
            f"x{t} = ("
            + ", ".join(f"{v:.2f}" for v in data.TOKENS[t - 1].ravel())
            + ")",
        )
        source = np.array([TIMES[t], 2.1, merge[2] + 0.7])
        incoming = orb(s, source, 0.18, CORAL)
        with s.parallel():
            fly(s, incoming, source, merge, 1.0, 0.35)
            history.transform(
                to=Transform3D.translation(*map(float, merge)),
                duration=1.0,
            )
        incoming.remove()
        pulse(s, merge, CORAL)
        caption(t, "融合之后，状态改变。", "hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)")
        with s.parallel():
            fly(s, history, merge, target, 1.15, 0)
            s.camera3d.configure(
                to=replace(s.camera3d.state(), position=Vec3(5.5 - 0.35 * t, 4.2, 14)),
                duration=1.15,
            )
        rail(s, merge, target, GOLD, 0.025)
        pulse(s, target, GOLD)
        orb(s, target, 0.10, GOLD)
        remove(phase)
        values = data.HIDDEN_STATES[t].ravel()
        phase = [
            label(
                s, f"h{t} = ({values[0]:+.3f}, {values[1]:+.3f})", 6.0, -2.5, 0.3, GOLD
            ),
            label(s, f"第 {t} 步 / 共 3 步", 6.0, -3.2, 0.2, MUTED),
        ]
        # Timeline progress advances continuously, not as another static slide.
        rect(s, -7.6 + (t - 1) * 1.1, -4.55, 0.82, 0.035, TEAL, 10)
        s.wait(1.15)
        remove(active)
    caption(
        3,
        "同一套权重，形成不同的记忆。",
        "重新展开 48 条独立序列，比较各自的状态轨迹。",
    )
    remove(phase)
    with s.parallel():
        title.fade_out(duration=0.4)
        history.fade_out(duration=0.4)
        s.camera3d.configure(
            to=replace(
                s.camera3d.state(), position=Vec3(6.5, 5.5, 14), orthographic_height=9
            ),
            duration=1.2,
        )
        for item in group_paths:
            item.opacity(to=1, duration=1.2)
    title.remove()
    label(s, "不同的输入，走出不同的轨迹。", -8.3, 3.7, 0.46, INK, left=True)
    # Replay each independently calculated trajectory, with its own moving head.
    for i, point in enumerate(group_dots):
        point.transform(
            to=Transform3D.translation(*map(float, ensemble_point(batch_states, i, 0))),
            duration=0,
        )
        point.opacity(to=1, duration=0)
    for t in (1, 2, 3):
        with s.parallel():
            for i, point in enumerate(group_dots):
                fly(
                    s,
                    point,
                    ensemble_point(batch_states, i, t - 1),
                    ensemble_point(batch_states, i, t),
                    1.05,
                    0,
                )
    label(s, "落点来自实际计算；连线与移动展示时间顺序。", 5.2, -4.65, 0.16, MUTED)
    s.wait(1.5)
    return s
