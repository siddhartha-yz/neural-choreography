"""One reusable block opens at three call sites and returns computed results.

All three calls share the chapter's frozen parameters. The moving module,
folding planes and routes depict invocation, not physical network geometry.
"""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Style, Transform2D, Easing
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.feature_suite import project
from zanim_scenes.conv_suite import point
from zanim_scenes.models import ep_05_1 as model

STARTS = (0.15, 2.8, 5.45)
ANCHORS = np.array([[-2.2, 1.35], [2.2, 1.35], [0, -1.35]])
CALLERS = np.array([[-6, 0], [6, 1.0], [0, -3.6]])
SECONDS = 8.4


def motion(t):
    call = max(0, min(2, int(np.searchsorted(STARTS, t, side="right") - 1)))
    local = max(0, t - STARTS[call])
    travel = ease(local / 0.45) * (1 - ease((local - 2.0) / 0.55))
    opened = ease((local - 0.42) / 0.4) * (1 - ease((local - 1.72) / 0.4))
    return call, local, ANCHORS[call] * travel, opened


def draw(b, incoming):
    from zanim_scenes.graph_suite import CALL_INPUTS, block_calls, projected

    pre, hidden, outputs = block_calls()
    colors = (b.palette[0], b.palette[3], b.palette[2])

    @lru_cache(maxsize=64)
    def frame(u):
        call, t, anchor, opened = motion(u * SECONDS)
        planes = []
        nodes = []
        for stage in range(3):
            cx = anchor[0] + (stage - 1) * (0.18 + 1.25 * opened)
            cy = anchor[1] + 0.16 * (stage - 1) * (1 - opened)
            z = (stage - 1) * 0.48 * (1 - opened)
            angle = 0.3 * (1 - opened) + 0.15 * math.sin(u * 2)

            def xyz(dx, dy):
                return [cx + dx * math.cos(angle), cy + dy, z - dx * math.sin(angle)]

            corners = np.array(
                [
                    xyz(x, y)
                    for x, y in (
                        (-0.53, -0.92),
                        (0.53, -0.92),
                        (0.53, 0.92),
                        (-0.53, 0.92),
                    )
                ]
            )
            planes.append(project(corners, u * 2))
            n = 3 if stage < 2 else 2
            nodes.append(
                project(np.array([xyz(0, y) for y in np.linspace(0.6, -0.6, n)]), u * 2)
            )
        return call, t, anchor, opened, planes, nodes

    # These same three objects remain present through all calls.
    for stage in range(3):
        for edge in range(4):
            item = b.line(b.palette[stage], 0.027)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.8, duration=0.35)
            item.transform_function(
                lambda u, stage=stage, edge=edge: pose(
                    frame(u)[4][stage][edge], frame(u)[4][stage][(edge + 1) % 4]
                ),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
        for j in range(3 if stage < 2 else 2):
            item = b.dot(b.palette[stage], 0.13)
            item.opacity(to=0, duration=0)
            item.opacity(to=1, duration=0.35)

            def activated(u, stage=stage, j=j):
                call, t, _, opened, _, nn = frame(u)
                value = (pre, hidden, outputs)[stage][call, j]
                pulse = ease((t - (0.65, 0.95, 1.25)[stage]) / 0.2)
                size = (0.2 + 0.8 * opened) * (0.23 + 0.7 * min(2, abs(value)) * pulse)
                return point(nn[stage][j], size)

            item.transform_function(activated, duration=SECONDS, easing=Easing.LINEAR)
    # Internal edges are revealed by unfolding, and carry the current call only.
    for stage, count in ((0, 3), (1, 2)):
        for src in range(3):
            for dst in range(count):
                if stage == 0 and src != dst:
                    continue
                item = b.line(b.palette[stage], 0.015)
                item.opacity(to=0.35, duration=0)

                def link(u, stage=stage, src=src, dst=dst):
                    _, _, _, opened, _, nn = frame(u)
                    return pose(
                        nn[stage][src], nn[stage + 1][dst]
                    ) @ Transform2D.scaling(1, max(0.001, opened))

                item.transform_function(link, duration=SECONDS, easing=Easing.LINEAR)
                for tail in range(7):
                    item = b.dot(b.palette[stage], 0.047)

                    def computation(u, stage=stage, src=src, dst=dst, tail=tail):
                        call, t, _, _, _, nn = frame(u)
                        value = (
                            pre[call, src]
                            if stage == 0
                            else hidden[call, src] * model.WEIGHTS_2[dst, src]
                        )
                        # Negative preactivations reach ReLU but do not pass it.
                        q = float(
                            np.clip(
                                (t - (0.8 if stage == 0 else 1.1) - tail * 0.018)
                                / 0.34,
                                0,
                                1,
                            )
                        )
                        a = nn[stage][src]
                        z = nn[stage + 1][dst]
                        gain = math.sin(math.pi * q) * min(1.6, abs(value))
                        return point(a + ease(q) * (z - a), gain)

                    item.transform_function(
                        computation, duration=SECONDS, easing=Easing.LINEAR
                    )
    parked = []
    for call, (start, color) in enumerate(zip(STARTS, colors)):
        center = CALLERS[call]
        # The first input exactly continues the previous chapter's two nodes.
        source = (
            np.array([center + [0, 1.1], center + [0, -1.1]])
            if call == 0
            else np.array([center + [-0.22, 0], center + [0.22, 0]])
        )
        target = np.array([center + [-0.23, 0], center + [0.23, 0]])
        parked.append(projected(target, 1))
        # A call-site ring stays at the caller, distinct from the shared module.
        item = b.add(
            Circle(
                0.38,
                style=Style.outline(color, 0.025),
                transform=Transform2D.translation(-30, 0),
                z_index=3,
            )
        )
        item.opacity(to=0, duration=0)
        item.opacity(to=0.7, duration=0.4, at=max(0, start - 0.1))
        item.transform_function(
            lambda u, center=center: point(projected(center[None, :], u)[0]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
        for j, value in enumerate(CALL_INPUTS[call]):
            item = b.dot(color, 0.1 + 0.06 * abs(value))
            item.opacity(to=1 if incoming and call == 0 else 0, duration=0)
            item.opacity(to=1, duration=0.25, at=max(0, start - 0.1))
            item.fade_out(duration=0.2, at=start + 0.9)

            def request(u, j=j, start=start, source=source, call=call):
                q = ease((u * SECONDS - start - 0.2) / 0.55)
                _, _, _, _, _, nn = frame(u)
                a = projected(source, u)[j]
                z = nn[0][min(j, 2)]
                return point(a + q * (z - a))

            item.transform_function(request, duration=SECONDS, easing=Easing.LINEAR)
        # Corresponding input contributions enter the first affine operation.
        for i in range(2):
            for j in range(3):
                for tail in range(4):
                    item = b.dot(color, 0.035)

                    def contribution(
                        u, i=i, j=j, tail=tail, start=start, source=source, call=call
                    ):
                        q = float(
                            np.clip(
                                (u * SECONDS - start - 0.4 - tail * 0.035) / 0.35, 0, 1
                            )
                        )
                        a = projected(source, u)[i]
                        z = frame(u)[5][0][j]
                        strength = min(
                            1.5, abs(CALL_INPUTS[call, i] * model.WEIGHTS_1[j, i])
                        )
                        return point(
                            a + ease(q) * (z - a), strength * math.sin(math.pi * q)
                        )

                    item.transform_function(
                        contribution, duration=SECONDS, easing=Easing.LINEAR
                    )
        for j, value in enumerate(outputs[call]):
            item = b.dot(color, 0.1 + 0.06 * abs(value))
            item.opacity(to=0, duration=0)
            item.opacity(to=1, duration=0.2, at=start + 1.5)

            def returned(u, j=j, start=start, target=target):
                q = ease((u * SECONDS - start - 1.5) / 0.5)
                a = frame(u)[5][2][j]
                z = projected(target, u)[j]
                return point(
                    a + q * (z - a) + np.array([0, 0.3 * math.sin(math.pi * q)])
                )

            item.transform_function(returned, duration=SECONDS, easing=Easing.LINEAR)
            for tail in range(6):
                item = b.dot(color, 0.035)

                def response(u, j=j, start=start, target=target, call=call, tail=tail):
                    q = float(
                        np.clip((u * SECONDS - start - 1.5 - tail * 0.025) / 0.5, 0, 1)
                    )
                    a = frame(u)[5][2][j]
                    z = projected(target, u)[j]
                    return point(
                        a
                        + ease(q) * (z - a)
                        + np.array([0, 0.3 * math.sin(math.pi * q)]),
                        math.sin(math.pi * q),
                    )

                item.transform_function(
                    response, duration=SECONDS, easing=Easing.LINEAR
                )
    b.note("同一个块", 0, 3.6)
    b.note("Linear → ReLU → Linear", 0, 2.85, start=0.45, end=2.3)
    return lambda: np.concatenate(parked)
