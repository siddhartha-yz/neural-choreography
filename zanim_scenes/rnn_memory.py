"""A three-dimensional RNN story about information carried across time.

The last act is a genuine counterfactual: set only x1 to zero, replay the same
fixed recurrence, then compare h3 with the original sequence.
"""

import numpy as np
from zanim import Scene, Canvas, Camera3D, Vec3, Transform3D, Color
from zanim_scenes.style import label, rect
from zanim_scenes.mechanisms import remove
from zanim_scenes.rnn_motion import orb, strand, fly, pulse
from zanim_scenes.models import ep_08_4 as d

BG = Color(9, 16, 30)
INK = Color(241, 247, 250)
MUTED = Color(150, 177, 193)
FRAME = Color(53, 98, 122)
TEAL = Color(83, 219, 209)
GOLD = Color(255, 219, 100)
INPUTS = (Color(255, 139, 113), Color(119, 183, 255), Color(190, 157, 255))
XS = (-3.8, 0, 3.8)


def counterfactual():
    actual = np.asarray(d.HIDDEN_STATES).reshape(4, 2)
    hidden = d.HIDDEN_START.copy()
    states = [hidden.ravel().copy()]
    for t, token in enumerate(d.TOKENS):
        hidden = d.rnn_step(np.zeros_like(token) if t == 0 else token, hidden)
        states.append(hidden.ravel().copy())
    return actual, np.asarray(states)


def point(t, h):
    return np.array([XS[t - 1], float(h[0]) * 1.35, float(h[1]) * 1.35])


def frame(s, x, color=FRAME):
    items = []
    for y in (-1.65, 1.65):
        for z in (-1.65, 1.65):
            items.append(strand(s, (x - 0.65, y, z), (x + 0.65, y, z), color, 0.014))
    for xx in (x - 0.65, x + 0.65):
        for y in (-1.65, 1.65):
            items.append(strand(s, (xx, y, -1.65), (xx, y, 1.65), color, 0.014))
        for z in (-1.65, 1.65):
            items.append(strand(s, (xx, -1.65, z), (xx, 1.65, z), color, 0.014))
    return items


def build(width=1920, height=1080, fps=60):
    actual, ghost = counterfactual()
    camera = Camera3D(
        position=Vec3(5.1, 4.3, 12.7), target=Vec3(0, 0.1, 0), orthographic_height=8.7
    )
    s = Scene(canvas=Canvas(width, height, width / 19.2), fps=fps, camera3d=camera)
    rect(s, 0, 0, 19.2, 10.8, BG, -100)
    label(s, "循环神经网络 · 08.4", -8.4, 4.6, 0.25, MUTED, left=True)
    label(s, "第一个输入，会留到第三步吗？", -8.4, 3.85, 0.47, INK, left=True)
    label(
        s,
        "观察状态沿时间传递；三处单元使用相同的权重。",
        -8.4,
        3.2,
        0.22,
        MUTED,
        left=True,
    )
    label(s, "RNN  /  MEMORY ACROSS TIME", 8.35, 4.55, 0.16, TEAL)
    frames = []
    step_labels = []
    paths = []
    for i, x in enumerate(XS, 1):
        frames.append(frame(s, x))
        step_labels.append(
            label(s, f"第 {i} 步", -4.95 + (i - 1) * 3.8, -3.65, 0.23, MUTED)
        )
    for a, b in zip(XS, XS[1:]):
        paths.append(
            strand(s, (a + 0.85, -1.65, 1.7), (b - 0.85, -1.65, 1.7), FRAME, 0.016)
        )
    label(s, "彩色球：当前输入", -8.4, -4.55, 0.21, INPUTS[0], left=True)
    label(s, "青色球：传下来的状态", -4.1, -4.55, 0.21, TEAL, left=True)
    label(s, "金色：原状态 · 灰色：对照", 0.8, -4.55, 0.21, GOLD, left=True)
    panel = []

    def story(head, note, color=INK):
        nonlocal panel
        if panel:
            with s.parallel():
                for item in panel:
                    item.fade_out(duration=0.22)
            remove(panel)
        panel = [
            label(s, head, -8.3, 2.48, 0.35, color, left=True),
            label(s, note, -8.3, 1.92, 0.21, MUTED, left=True),
        ]
        for item in panel:
            item.opacity(to=0, duration=0)
        with s.parallel():
            for item in panel:
                item.fade_in(duration=0.3)

    state = None
    anchors = []
    for t, token in enumerate(d.TOKENS, 1):
        center = point(t, actual[t])
        previous = point(t - 1, actual[t - 1]) if t > 1 else None
        story(
            f"输入 {t} 进入循环单元",
            f"当前输入 x{t} 与上一状态共同决定 h{t}",
            INPUTS[t - 1],
        )
        entry = np.array([XS[t - 1], -2.65, 2.35])
        newinput = orb(s, entry, 0.25, INPUTS[t - 1])
        if state is None:
            state = orb(s, (XS[t - 1] - 0.9, 0, -0.7), 0.21, TEAL)
        else:
            carry = np.array([XS[t - 1] - 0.65, previous[1], previous[2]])
            fly(s, state, previous, carry, 1.0, 0.24)
            paths.append(strand(s, previous, carry, TEAL, 0.032))
        target = np.array([XS[t - 1], 0, 0])
        with s.parallel():
            fly(s, newinput, entry, target, 1.05, 0.12)
            state.transform(
                to=Transform3D.translation(*map(float, target)), duration=1.05
            )
        newinput.fade_out(duration=0.27)
        newinput.remove()
        pulse(s, target, INPUTS[t - 1])
        story("状态已更新，继续传给下一步。", f"h{t} 将携带前面输入留下的信息", TEAL)
        state.transform(to=Transform3D.translation(*map(float, center)), duration=0.85)
        pulse(s, center, GOLD)
        anchors.append(orb(s, center, 0.14, GOLD))
        if previous is not None:
            paths.append(strand(s, previous, center, TEAL, 0.015))
        s.wait(0.7)
    # Replay with exactly the same weights and the other two inputs unchanged.
    story(
        "现在，只把第一个输入设为 0。",
        "第二、第三个输入，以及所有权重，都保持原样。",
        GOLD,
    )
    # Replay the same recurrence in the same three cells. Each gold segment
    # is the actual difference between the two hidden-state vectors.
    with s.parallel():
        for item in paths:
            item.fade_out(duration=0.45)
    remove(paths)
    state.transform(
        to=Transform3D.translation(*map(float, point(3, actual[3])))
        @ Transform3D.scaling(0.5),
        duration=0.35,
    )
    shadow = None
    for t in (1, 2, 3):
        target = point(t, ghost[t])
        if shadow is None:
            shadow = orb(s, (XS[0] - 0.85, 0, -0.65), 0.10, MUTED)
        shadow.transform(to=Transform3D.translation(*map(float, target)), duration=0.95)
        if t < 3:
            orb(s, target, 0.10, MUTED)
        original = point(t, actual[t])
        strand(s, original, target, GOLD, 0.027)
        if t > 1:
            strand(s, point(t - 1, actual[t - 1]), original, GOLD, 0.015)
            strand(s, point(t - 1, ghost[t - 1]), target, MUTED, 0.014)
        delta = float(np.linalg.norm(actual[t] - ghost[t]))
        story(
            f"第 {t} 步：两条状态路径开始分开。",
            f"原序列与 x1 = 0 的状态距离：{delta:.3f}",
            GOLD,
        )
        pulse(s, target, MUTED)
        s.wait(0.55)
    story(
        "第一个输入的影响，传到了第三步。",
        "后两次输入与权重完全相同；每一步的状态仍不同。",
        GOLD,
    )
    label(s, "亮色：原序列", -8.2, 0.7, 0.23, GOLD, left=True)
    label(s, "灰色：仅将 x₁ 设为 0", -8.2, 0.16, 0.22, MUTED, left=True)
    gaps = [np.linalg.norm(actual[t] - ghost[t]) for t in (1, 2, 3)]
    label(s, "状态距离  " + " → ".join(f"{gap:.3f}" for gap in gaps), -8.2, -0.52, 0.21, INK, left=True)
    label(s, "差异逐渐减弱，但没有消失。", -8.2, -0.98, 0.20, MUTED, left=True)
    s.wait(3)
    return s
