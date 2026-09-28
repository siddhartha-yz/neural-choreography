"""Four continuous geometric movements, with orthographic 3D feature surfaces.

Surface subdivision, camera motion and spatial interpolation are artistic;
chapter model arrays retain their original definitions.
"""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Line, Style, Transform2D, Easing
from zanim_scenes.gated_art import pose, ease
from zanim_scenes.models import ep_07_4 as inception
from zanim_scenes.models import ep_07_7 as dense
from zanim_scenes.batch_norm_art import stages
from zanim_scenes.residual_art import batch
from zanim_scenes.style import label

PREFIX_SECONDS = 48.0
G = 9
U, V = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G))
UV = np.stack((U.ravel(), V.ravel()), axis=-1)
EDGES = [(r * G + c, r * G + c + 1) for r in range(G) for c in range(G - 1)] + [
    (r * G + c, (r + 1) * G + c) for r in range(G - 1) for c in range(G)
]


def sampled(values):
    axis = np.linspace(0, 1, values.shape[0])
    target = np.linspace(0, 1, G)
    rows = np.array([np.interp(target, axis, row) for row in values])
    return np.array([np.interp(target, axis, rows[:, j]) for j in range(G)]).T.ravel()


def project(points, t):
    yaw = 0.30 * math.sin(t * 0.16) + 0.22
    pitch = 0.30 + 0.10 * math.sin(t * 0.13)
    x, y, z = points.T
    a = x * math.cos(yaw) + z * math.sin(yaw)
    b = -x * math.sin(yaw) + z * math.cos(yaw)
    return np.stack((a, y * math.cos(pitch) - b * math.sin(pitch)), axis=-1)


def surface(ch, center, angle, values, scale=1):
    u, v = UV.T
    height = (values - values.mean()) * 0.22
    return np.stack(
        (
            center[0] + scale * u * math.cos(angle) + height * math.sin(angle),
            center[1] + scale * v,
            center[2] - scale * u * math.sin(angle) + height * math.cos(angle),
        ),
        axis=-1,
    )


def state(kind, p):
    """Returns four physical surfaces plus per-channel visibility."""
    inc = [inception.Y1, inception.Y3, inception.Y5, inception.YP]
    out = []
    visible = np.ones(4)
    for ch in range(4):
        if kind == "inception":
            raw = sampled(inc[ch])
            raw = raw / max(1, float(raw.max()))
            spread = ease(p / 0.32)
            stack = ease((p - 0.67) / 0.33)
            flower = np.array(
                [(-4.8, -1.6, 1.6, 4.8)[ch], (1.5, -1.2, 1.5, -1.2)[ch], 0.0]
            )
            book = np.array([(ch - 1.5) * 0.5, 0, (ch - 1.5) * 0.8])
            center = (1 - stack) * spread * flower + stack * book
            angle = (1 - stack) * (
                (ch - 1.5) * 0.27 + 0.25 * math.sin(p * 4)
            ) + stack * 0.65
            points = surface(ch, center, angle, raw, 1.45)
        elif kind == "norm":
            arrays = stages()
            q = min(p * 3, 2.99999)
            j = int(q)
            a = arrays[j][ch]
            b = arrays[j + 1][ch]
            vals = a + ease(q - j) * (b - a)
            curve = np.interp((UV[:, 0] + 1) * 2, np.arange(5), vals)
            theta = UV[:, 1] * 1.2 + p * 0.8
            points = np.stack(
                (
                    curve * 1.35,
                    (1.5 - ch) * 1.6 + 0.42 * np.sin(theta),
                    0.65 * np.cos(theta),
                ),
                axis=-1,
            )
        elif kind == "residual":
            x, f, y = batch()
            channel = ch % 2
            original = sampled(x[channel].reshape(2, 2))
            correction = sampled(f[channel].reshape(2, 2))
            result = sampled(y[channel].reshape(2, 2))
            is_update = ch >= 2
            if not is_update:
                travel = ease(p / 0.65)
                center = np.array(
                    [
                        -5.8 + 10.3 * travel,
                        2.5 * math.sin(math.pi * travel),
                        (channel - 0.5) * 0.7,
                    ]
                )
                field = original + (result - original) * ease((p - 0.88) / 0.1)
            else:
                first = ease(p / 0.35)
                last = ease((p - 0.60) / 0.30)
                center = np.array(
                    [
                        -5.8 + 4.8 * first + 5.5 * last,
                        -2.1 * first * (1 - last),
                        (channel - 0.5) * 0.7,
                    ]
                )
                field = original + (correction - original) * ease((p - 0.35) / 0.20)
                visible[ch] = 1 - ease((p - 0.91) / 0.09)
            points = surface(ch, center, 0.25, field, 0.85)
        else:
            data = sampled(dense.STACK_2[ch])
            visible[ch] = ease((p - 0.53) / 0.2) if ch < 3 else ease((p - 0.73) / 0.08)
            center = np.array(
                [5 + (ch - 1.5) * 0.25, (ch - 1.5) * 0.22, (ch - 1.5) * 0.5]
            )
            points = surface(ch, center, 0.5, data, 1.15)
        points[:, 2] += 0.16 * np.sin(UV[:, 0] * 2.8 + UV[:, 1] * 1.6 + p * 2 * math.pi)
        out.append(points)
    return np.array(out), visible


def append(s, chapter, rings, palette):
    current, _ = state("inception", 0)
    current = np.array([project(p, 0) for p in current])
    mesh = []
    dots = []
    sparks = []
    for ch in range(4):
        lines = []
        cloud = []
        for a, b in EDGES:
            item = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(palette[ch], 0.013),
                    transform=pose(current[ch, a], current[ch, b]),
                    z_index=2,
                )
            )
            item.opacity(to=0.3, duration=0)
            lines.append(item)
        for i in range(G * G):
            item = s.add(
                Circle(
                    0.037,
                    style=Style.solid(palette[ch]),
                    transform=Transform2D.translation(*map(float, current[ch, i])),
                    z_index=5,
                )
            )
            cloud.append(item)
        stream = []
        for k in range(12):
            item = s.add(
                Circle(
                    0.062,
                    style=Style.solid(palette[ch]),
                    transform=Transform2D.translation(-30, 0),
                    z_index=7,
                )
            )
            stream.append(item)
        mesh.append(lines)
        dots.append(cloud)
        sparks.append(stream)
    links = []
    semantic = []

    def note(t, x, y, color):
        semantic.append(label(s, t, x, y, 0.24, color))

    def semantic_line(a, b, color):
        item = s.add(
            Line(
                (0, 0),
                (1, 0),
                style=Style.outline(color, 0.018),
                transform=pose(np.array(a), np.array(b)),
                z_index=1,
            )
        )
        item.opacity(to=0.42, duration=0)
        semantic.append(item)
        return item

    def clear_semantic():
        for item in semantic:
            item.remove()
        semantic.clear()

    def ghost_surface(ch, station, values):
        pts = surface(
            ch,
            np.array(
                [station + (ch - 0.5) * 0.25, (ch - 0.5) * 0.22, (ch - 0.5) * 0.5]
            ),
            0.5,
            sampled(values),
            1.15,
        )
        items = []
        for a, b in EDGES:
            item = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(palette[ch], 0.013),
                    transform=Transform2D.translation(-30, 0),
                    z_index=2,
                )
            )
            item.opacity(to=0.55, duration=0)
            semantic.append(item)
            items.append((a, b, item))
        for i in range(0, G * G, 4):
            item = s.add(
                Circle(
                    0.035,
                    style=Style.solid(palette[ch]),
                    transform=Transform2D.translation(-30, 0),
                    z_index=5,
                )
            )
            semantic.append(item)
            items.append((i, None, item))
        return pts, items

    windows = []
    for ch in range(4):
        for edge in range(4):
            item = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(palette[ch], 0.032),
                    transform=Transform2D.translation(-30, 0),
                    z_index=8,
                )
            )
            item.opacity(to=0.65, duration=0)
            windows.append((ch, edge, item))

    def run(kind, seconds, time, transition=0.20):
        nonlocal current
        start = current.copy()

        @lru_cache(maxsize=32)
        def frame(u):
            xyz, visible = state(kind, u)
            pts = np.array([project(p, time + u * seconds) for p in xyz])
            w = ease(u / transition)
            return start + w * (pts - start), visible

        ghosts = []
        streams = []
        if kind == "residual":
            note("x · 直连", -2.5, 3.5, palette[0])
            note("F(x) · 修正", -1, -3.5, palette[2])
            note("+", 3.2, 0, palette[2])
            note("相加", 5, -1.9, palette[0])
            for ch in range(2):
                ghosts.append((ch, 0, *ghost_surface(ch, -5.8, dense.INPUT_X[ch])))
            for top in (True, False):
                previous = None
                for q in np.linspace(0, 1, 41):
                    xy = np.array(
                        [
                            -5.8 + 10.3 * q,
                            2.5 * math.sin(math.pi * q)
                            if top
                            else -2.1 * math.sin(math.pi * q),
                        ]
                    )
                    if previous is not None:
                        semantic_line(previous, xy, palette[0 if top else 2])
                    previous = xy
        elif kind == "dense":
            note("2 通道", -5.5, -2.3, palette[0])
            note("3 通道", -0.5, -2.3, palette[2])
            note("4 通道", 4.6, -2.3, palette[3])
            note("保留并拼接", 0, 3.0, palette[2])
            for station, count in ((-5.3, 2), (-0.4, 3)):
                for ch in range(count):
                    ghosts.append(
                        (
                            ch,
                            0 if station < -1 else 1,
                            *ghost_surface(ch, station, dense.STACK_2[ch]),
                        )
                    )
            for a, b, count in ((-5.3, -0.4, 2), (-0.4, 5, 3)):
                for ch in range(count):
                    for k in (0, 4, 8):
                        start = np.array([a + 0.9, (k - 4) * 0.18 + (ch - 0.5) * 0.22])
                        end = np.array([b - 0.9, start[1]])
                        semantic_line(start, end, palette[ch])
                        dot = s.add(
                            Circle(
                                0.06,
                                style=Style.solid(palette[ch]),
                                transform=Transform2D.translation(-30, 0),
                                z_index=7,
                            )
                        )
                        semantic.append(dot)
                        streams.append((start, end, ch, k, dot, a))
        with s.parallel():
            for ch, station, pts, items in ghosts:
                for a, b, item in items:

                    def ghost_pose(u, ch=ch, station=station, pts=pts, a=a, b=b):
                        pp = project(pts, time + u * seconds)
                        visible = (
                            1
                            if kind == "residual" or station == 0
                            else ease((u - (0.24 if ch < 2 else 0.40)) / 0.16)
                        )
                        if b is None:
                            return Transform2D.translation(
                                *map(float, pp[a])
                            ) @ Transform2D.scaling(max(0.001, visible))
                        return pose(pp[a], pp[b]) @ Transform2D.scaling(
                            1, max(0.001, visible)
                        )

                    item.transform_function(
                        ghost_pose, duration=seconds, easing=Easing.LINEAR
                    )
            for a, b, ch, k, item, station in streams:

                def transfer(u, a=a, b=b, ch=ch, k=k, station=station):
                    start = 0.12 if station < -1 else 0.48
                    q = float(np.clip((u - start - k * 0.006) / 0.24, 0, 1))
                    xy = a + ease(q) * (b - a)
                    return Transform2D.translation(
                        *map(float, xy)
                    ) @ Transform2D.scaling(max(0.001, math.sin(math.pi * q)))

                item.transform_function(
                    transfer, duration=seconds, easing=Easing.LINEAR
                )
            if kind == "inception":
                for ch, edge, item in windows:

                    def window(u, ch=ch, edge=edge):
                        pp, _ = frame(u)
                        cx = (pp[ch, 40] + pp[ch, 41]) * 0.5
                        a = (pp[ch, 44] - pp[ch, 36]) * 0.5
                        b = (pp[ch, 76] - pp[ch, 4]) * 0.5
                        cx = (
                            cx
                            + 0.35 * math.sin(u * 2 * math.pi) * a
                            + 0.3 * math.cos(u * 2 * math.pi) * b
                        )
                        scale = (0.2, 0.6, 1.0, 0.6)[ch]
                        corners = [
                            cx + scale * (i * a + j * b)
                            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                        ]
                        return pose(corners[edge], corners[(edge + 1) % 4])

                    item.transform_function(
                        window, duration=seconds, easing=Easing.LINEAR
                    )
                    item.opacity(to=0, duration=0.7, at=seconds - 0.7)
            for ch in range(4):
                for (a, b), item in zip(EDGES, mesh[ch]):
                    item.transform_function(
                        lambda u, ch=ch, a=a, b=b: (
                            pose(frame(u)[0][ch, a], frame(u)[0][ch, b])
                            @ Transform2D.scaling(1, max(0.001, frame(u)[1][ch]))
                        ),
                        duration=seconds,
                        easing=Easing.LINEAR,
                    )
                for i, item in enumerate(dots[ch]):
                    item.transform_function(
                        lambda u, ch=ch, i=i: (
                            Transform2D.translation(*map(float, frame(u)[0][ch, i]))
                            @ Transform2D.scaling(max(0.001, frame(u)[1][ch]))
                        ),
                        duration=seconds,
                        easing=Easing.LINEAR,
                    )
                for k, item in enumerate(sparks[ch]):

                    def flowing(u, ch=ch, k=k):
                        q = (u * 2.5 + k / 12) % 1
                        f = q * (G * G - 1)
                        j = min(int(f), G * G - 2)
                        pp, vis = frame(u)
                        xy = pp[ch, j] + (f - j) * (pp[ch, j + 1] - pp[ch, j])
                        return Transform2D.translation(
                            *map(float, xy)
                        ) @ Transform2D.scaling(
                            max(0.001, vis[ch] * math.sin(math.pi * q))
                        )

                    item.transform_function(
                        flowing, duration=seconds, easing=Easing.LINEAR
                    )
            if kind == "dense":
                for a, b, i, line, dot in links:
                    line.transform_function(
                        lambda u, a=a, b=b, i=i: (
                            pose(frame(u)[0][a, i], frame(u)[0][b, i])
                            @ Transform2D.scaling(1, max(0.001, frame(u)[1][b]))
                        ),
                        duration=seconds,
                        easing=Easing.LINEAR,
                    )

                    def crossing(u, a=a, b=b, i=i):
                        q = (u * 4 + i * 0.017) % 1
                        pp, vis = frame(u)
                        xy = pp[a, i] + q * (pp[b, i] - pp[a, i])
                        return Transform2D.translation(
                            *map(float, xy)
                        ) @ Transform2D.scaling(
                            max(0.001, vis[b] * math.sin(math.pi * q))
                        )

                    dot.transform_function(
                        crossing, duration=seconds, easing=Easing.LINEAR
                    )
        current = frame(1)[0]

    chapter("多分支网络 · 07.4", "同源展开，多尺度并行。")
    run("inception", 11.55, 0)
    for _, _, item in windows:
        item.remove()
    chapter("批量归一化 · 07.5", "居中，统一尺度，再缩放平移。")
    run("norm", 11.2, 12)
    chapter("残差网络 · 07.6", "保留直连，叠加修正。")
    run("residual", 11.2, 24)
    clear_semantic()
    chapter("稠密连接网络 · 07.7", "旧特征保留，新层继续生长。")
    run("dense", 8.05, 36)
    # Continue the feature contours into the existing recurrent aperture.
    handoff = []
    angle = np.linspace(0, 2 * math.pi, rings.shape[1])
    for i in range(len(rings)):
        ch = i % 4
        center = current[ch].mean(axis=0)
        initial = center + np.stack((1.1 * np.cos(angle), 1.2 * np.sin(angle)), axis=-1)
        for j in range(len(angle) - 1):
            item = s.add(
                Line(
                    (0, 0),
                    (1, 0),
                    style=Style.outline(palette[ch], 0.013),
                    transform=pose(initial[j], initial[j + 1]),
                    z_index=2,
                )
            )
            item.opacity(to=0, duration=0)
            handoff.append(
                (item, initial[j], initial[j + 1], rings[i, j], rings[i, j + 1])
            )
    with s.parallel():
        for item in semantic:
            item.fade_out(duration=1.2)
        for group in mesh + dots + sparks:
            for item in group:
                item.fade_out(duration=1.2)
        for *_, line, dot in links:
            line.fade_out(duration=1.2)
            dot.fade_out(duration=1.2)
        for item, a, b, c, d in handoff:
            item.opacity(to=0.46, duration=1.2)
            item.transform_function(
                lambda u, a=a, b=b, c=c, d=d: pose(
                    a + ease(u) * (c - a), b + ease(u) * (d - b)
                ),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    for group in mesh + dots + sparks:
        for item in group:
            item.remove()
    for *_, line, dot in links:
        line.remove()
        dot.remove()
    clear_semantic()
    return [item for item, *_ in handoff]
