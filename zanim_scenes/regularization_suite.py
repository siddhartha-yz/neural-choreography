"""Fitting ribbons, ridge relaxation and masked routes from chapters 04.4–04.6.

Ribbons are geometric copies of a single numerical curve, not extra fits.
Dropout uses the original four-unit activations and seeded classic-dropout
masks; intermediate fades, depth and cross-chapter morphs are choreography.
"""

from functools import lru_cache
import math
import numpy as np
from zanim import Circle, Style, Transform2D, Easing
from zanim_scenes.conv_suite import Stage, point, GOLD
from zanim_scenes.feature_suite import project
from zanim_scenes.gated_art import ease, pose
from zanim_scenes.models import ep_04_4 as fit, ep_04_5 as ridge, ep_04_6 as drop

KINDS = ("capacity", "decay", "dropout")
PREFIX_SECONDS = 36.0
SECONDS = 8.4
SAMPLES = 65
LANES = 13
FIT_X = np.linspace(-2.25, 2.35, SAMPLES)
RIDGE_X = np.linspace(-2.25, 2.45, SAMPLES)
TITLES = ("欠拟合与过拟合 · 04.4", "权重衰减 · 04.5", "暂退法 · 04.6")
SUBTITLES = (
    "太僵硬，或追逐每一点起伏。",
    "约束权重，让曲线逐渐舒展。",
    "训练时随机切断，推断时全部参与。",
)


def ribbon_xyz(x, y, lane, row=0):
    """Common linear vertical scale; depth copies give the curve a ribbon."""
    depth = (lane - (LANES - 1) / 2) * 0.12
    return np.stack((x * 2.65, row + y * 1.02, np.full_like(x, depth)), -1)


def fit_values(q):
    truth = fit.true_function(FIT_X)
    return tuple(
        truth + ease(q) * (np.polyval(w, FIT_X) - truth)
        for w in (fit.LOW_COEFFS, fit.HIGH_COEFFS)
    )


def ridge_values(q):
    weights = ridge.ridge_weights(ridge.displayed_lambda(q))
    return weights, ridge.polynomial_values(RIDGE_X, weights)


def mask_values(t, row):
    """Exact masks at plateaus; continuous fades only illustrate mask changes."""
    step = max(0, t - 0.6) / 1.25
    k = min(3, int(step))
    q = ease((step - k - 0.75) / 0.25) if k < 3 else 0.0
    a = drop.MASKS[(k + row) % 4]
    z = drop.MASKS[(min(k + 1, 3) + row) % 4]
    masks = a + q * (z - a)
    inference = ease((t - 6.05) / 1.0)
    # Original model uses classic dropout, scaling all units at inference.
    return (1 - inference) * masks + inference * drop.KEEP_P


def dropout_nodes(stage, row, u=0):
    y = (1.5 - row) * 1.5
    if stage == 0:
        xyz = np.array(
            [[-6.4, y + offset, (row - 1.5) * 0.15] for offset in (0.25, -0.25)]
        )
    else:
        x = -0.7 if stage == 1 else 6.0
        xyz = np.array([[x, y + (1.5 - j) * 0.29, (j - 1.5) * 0.55] for j in range(4)])
    return project(xyz, u * 2)


def initial(kind, palette):
    if kind == "dropout":
        pp = np.concatenate([dropout_nodes(0, row) for row in range(4)])
        return (
            pp,
            np.tile([0.12, 0.095], 4),
            [palette[row] for row in range(4) for _ in range(2)],
        )
    x = FIT_X if kind == "capacity" else RIDGE_X
    ys = fit_values(0) if kind == "capacity" else (ridge_values(0)[1],)
    rows = (1.6, -1.2) if kind == "capacity" else (0.0,)
    pp = [
        project(ribbon_xyz(x, y, lane, row), 0)
        for y, row in zip(ys, rows)
        for lane in range(LANES)
    ]
    points = np.concatenate(pp)
    return points, np.full(len(points), 0.028), [palette[0]] * len(points)


def ribbons(b, state, colors, copies, incoming):
    for band in range(copies):
        color = colors[band]
        for lane in range(LANES):
            for j in range(SAMPLES - 1):
                item = b.line(color, 0.013 if lane != LANES // 2 else 0.028)
                item.opacity(
                    to=(0.9 if lane == LANES // 2 else 0.35) if incoming else 0,
                    duration=0,
                )
                item.opacity(to=0.35 if lane != LANES // 2 else 0.9, duration=0.6)
                item.transform_function(
                    lambda u, band=band, lane=lane, j=j: pose(
                        state(u)[band, lane, j], state(u)[band, lane, j + 1]
                    ),
                    duration=SECONDS,
                    easing=Easing.LINEAR,
                )
            # Several travelers keep the whole ribbon alive during deformation.
            for phase in range(2):
                for tail in range(4):
                    item = b.dot(color, 0.034)

                    def travel(u, band=band, lane=lane, phase=phase, tail=tail):
                        q = (u * 1.35 + phase / 2 - tail * 0.005 + lane * 0.017) % 1
                        location = q * (SAMPLES - 1)
                        j = min(int(location), SAMPLES - 2)
                        pp = state(u)[band, lane]
                        return point(
                            pp[j] + (location - j) * (pp[j + 1] - pp[j]),
                            0.3 + 0.7 * math.sin(math.pi * q),
                        )

                    item.transform_function(
                        travel, duration=SECONDS, easing=Easing.LINEAR
                    )


def capacity(b, incoming):
    @lru_cache(maxsize=64)
    def state(u):
        ys = fit_values(np.clip((u * SECONDS - 0.8) / 3.3, 0, 1))
        return np.array(
            [
                [
                    project(ribbon_xyz(FIT_X, y, lane, row), u * 3)
                    for lane in range(LANES)
                ]
                for y, row in zip(ys, (1.6, -1.2))
            ]
        )

    ribbons(b, state, (b.palette[0], b.palette[2]), 2, incoming)
    for band, row in enumerate((1.6, -1.2)):
        for j, (x, y) in enumerate(zip(fit.TRAIN_X, fit.TRAIN_Y)):
            xyz = np.array([[x * 2.65, row + y * 1.02, 0]])
            item = b.dot(GOLD, 0.072)
            item.opacity(to=0, duration=0)
            item.fade_in(duration=0.5, at=0.3 + j * 0.035)
            item.transform_function(
                lambda u, xyz=xyz: point(project(xyz, u * 3)[0]),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
            item = b.line(GOLD, 0.017)
            item.opacity(to=0, duration=0)
            item.opacity(to=0.65, duration=0.6, at=3.2)

            def residual(u, x=x, y=y, row=row, band=band):
                q = ease(np.clip((u * SECONDS - 0.8) / 3.3, 0, 1))
                pred = fit.true_function(x) + q * (
                    np.polyval((fit.LOW_COEFFS, fit.HIGH_COEFFS)[band], x)
                    - fit.true_function(x)
                )
                pp = project(
                    np.array(
                        [
                            [x * 2.65, row + y * 1.02, 0],
                            [x * 2.65, row + pred * 1.02, 0],
                        ]
                    ),
                    u * 3,
                )
                return pose(*pp)

            item.transform_function(residual, duration=SECONDS, easing=Easing.LINEAR)
        # Held-out location stays separate from the gold training anchors.
        item = b.add(
            Circle(
                0.14,
                style=Style.outline(b.palette[3], 0.025),
                transform=Transform2D.translation(-30, 0),
                z_index=6,
            )
        )
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.5, at=5.0)
        xyz = np.array([[fit.TRAVELER_X * 2.65, row + fit.TRAVELER_Y * 1.02, 0]])
        item.transform_function(
            lambda u, xyz=xyz: point(project(xyz, u * 3)[0]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
    b.note("欠拟合", -7.4, 2.0, color=b.palette[0])
    b.note("过拟合", -7.4, -1.5, color=b.palette[2])
    return lambda: state(1).reshape(-1, 2)


def decay(b, incoming):
    @lru_cache(maxsize=64)
    def numeric(u):
        return ridge_values(np.clip((u * SECONDS - 1.0) / 5.4, 0, 1))

    @lru_cache(maxsize=64)
    def state(u):
        _, y = numeric(u)
        return np.array(
            [[project(ribbon_xyz(RIDGE_X, y, lane), u * 3) for lane in range(LANES)]]
        )

    ribbons(b, state, (b.palette[0],), 1, incoming)
    # Each genuine penalized coefficient has a radial strand below the curve.
    for j in range(1, ridge.POLYNOMIAL_DEGREE + 1):
        theta = 2 * math.pi * (j - 1) / ridge.POLYNOMIAL_DEGREE
        base = np.array([0, -2.8])

        def tip(u, j=j, theta=theta):
            w = numeric(u)[0][j]
            length = abs(w) * 0.43
            return base + length * np.array([math.cos(theta), math.sin(theta)])

        color = b.palette[2] if ridge.WEIGHTS_UNPENALIZED[j] < 0 else GOLD
        item = b.line(color, 0.025)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.4, at=0.6)
        item.transform_function(
            lambda u, tip=tip: pose(base, tip(u)),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
        item = b.dot(color, 0.065)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.4, at=0.6)
        item.transform_function(
            lambda u, tip=tip: point(tip(u)), duration=SECONDS, easing=Easing.LINEAR
        )
    for x, y in zip(ridge.FEATURES, ridge.TARGETS):
        item = b.dot(GOLD, 0.06)
        item.opacity(to=0, duration=0)
        item.fade_in(duration=0.5)
        xyz = np.array([[x * 2.65, y * 1.02, 0]])
        item.transform_function(
            lambda u, xyz=xyz: point(project(xyz, u * 3)[0]),
            duration=SECONDS,
            easing=Easing.LINEAR,
        )
    return lambda: state(1).reshape(-1, 2)


def dropout(b, incoming):
    for row in range(4):
        color = b.palette[row]
        for i, value in enumerate(drop.INPUT_X.ravel()):
            item = b.dot(color, 0.12 if i == 0 else 0.095)
            item.opacity(to=1 if incoming else 0, duration=0)
            item.opacity(to=1, duration=0.4)
            item.transform_function(
                lambda u, row=row, i=i: point(dropout_nodes(0, row, u)[i]),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
        for j, value in enumerate(drop.HIDDEN.ravel()):
            for stage in (1, 2):
                item = b.dot(color, 0.12 + 0.045 * value)
                item.opacity(to=0, duration=0)
                item.fade_in(duration=0.45)
                item.transform_function(
                    lambda u, row=row, j=j, stage=stage: point(
                        dropout_nodes(stage, row, u)[j],
                        math.sqrt(max(0, mask_values(u * SECONDS, row)[j])),
                    ),
                    duration=SECONDS,
                    easing=Easing.LINEAR,
                )
            # Stationary hollow seats keep absent units easy to recognize.
            item = b.add(
                Circle(
                    0.22,
                    style=Style.outline(color, 0.018),
                    transform=Transform2D.translation(-30, 0),
                    z_index=3,
                )
            )
            item.opacity(to=0, duration=0)
            item.opacity(to=0.55, duration=0.45)
            item.transform_function(
                lambda u, row=row, j=j: point(dropout_nodes(1, row, u)[j]),
                duration=SECONDS,
                easing=Easing.LINEAR,
            )
            for i in range(3):
                # Two true affine contributions, then identity emission.
                a_stage = 0 if i < 2 else 1
                a_index = i if i < 2 else j
                z_stage = 1 if i < 2 else 2
                item = b.line(color, 0.013)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.25, duration=0.45)

                def route(
                    u, row=row, j=j, a_stage=a_stage, a_index=a_index, z_stage=z_stage
                ):
                    active = mask_values(u * SECONDS, row)[j]
                    return pose(
                        dropout_nodes(a_stage, row, u)[a_index],
                        dropout_nodes(z_stage, row, u)[j],
                    ) @ Transform2D.scaling(1, max(0.001, active))

                item.transform_function(route, duration=SECONDS, easing=Easing.LINEAR)
                for pulse in range(3):
                    for tail in range(5):
                        item = b.dot(color, 0.034)
                        item.opacity(to=0, duration=0)
                        item.fade_in(duration=0.45)

                        def flowing(
                            u,
                            row=row,
                            j=j,
                            a_stage=a_stage,
                            a_index=a_index,
                            z_stage=z_stage,
                            pulse=pulse,
                            tail=tail,
                        ):
                            q = (u * 3.0 + pulse / 3 - tail * 0.012 + row * 0.05) % 1
                            a = dropout_nodes(a_stage, row, u)[a_index]
                            z = dropout_nodes(z_stage, row, u)[j]
                            gain = math.sin(math.pi * q) * math.sqrt(
                                max(0, mask_values(u * SECONDS, row)[j])
                            )
                            return point(a + ease(q) * (z - a), gain)

                        item.transform_function(
                            flowing, duration=SECONDS, easing=Easing.LINEAR
                        )
    b.note("随机暂退", 0, 3.65, end=6.3)
    b.note("全部参与 · ×½", 0, 3.65, start=6.35)
    return lambda: np.concatenate([dropout_nodes(2, row, 1) for row in range(4)])


def append(s, chapter, palette, kind, incoming=()):
    i = KINDS.index(kind)
    chapter(TITLES[i], SUBTITLES[i])
    for item in incoming:
        item.remove()
    b = Stage(s, palette)
    end = (capacity, decay, dropout)[i](b, bool(incoming))
    b.play()
    source = end()
    if i == 2:
        from zanim_scenes.graph_suite import initial as graph_initial

        target, rr, colors = graph_initial("backprop", palette)
    else:
        target, rr, colors = initial(KINDS[i + 1], palette)
    old = list(b.objects)
    for item in old:
        item.fade_out(duration=0.8)
    carry = []
    if i == 0:
        # Preserve the ribbon's connected contour at the next scene boundary.
        for lane in range(LANES):
            for j in range(SAMPLES - 1):
                index = lane * SAMPLES + j
                item = b.line(palette[0], 0.028 if lane == LANES // 2 else 0.013)
                carry.append(item)
                item.opacity(to=0, duration=0)
                item.opacity(to=0.9 if lane == LANES // 2 else 0.35, duration=1)
                a, z = source[index : index + 2]
                aa, zz = target[index : index + 2]
                item.transform_function(
                    lambda u, a=a, z=z, aa=aa, zz=zz: pose(
                        a + ease(u) * (aa - a), z + ease(u) * (zz - z)
                    ),
                    duration=2.8,
                    easing=Easing.LINEAR,
                )
    else:
        for j, r in enumerate(rr):
            item = b.dot(colors[j], r)
            carry.append(item)
            item.opacity(to=0, duration=0)
            item.fade_in(duration=1)
            a = source[round(j * (len(source) - 1) / max(1, len(rr) - 1))]
            item.transform_function(
                lambda u, j=j, a=a: point(a + ease(u) * (target[j] - a)),
                duration=2.8,
                easing=Easing.LINEAR,
            )
    b.play()
    for item in old:
        item.remove()
    return carry
