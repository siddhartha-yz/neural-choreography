"""D2L 14.1 — skip-gram: v_c pulls window context u_o closer.

Every plotted number comes from the arrays below.  Four tokens start as
axis-aligned 2-D embeddings (one-hot spirit: v_c ⊥ u_o).  One skip-gram
pair (c, o) is trained with real softmax SGD.  u_o approaches v_c; u_n
and u_k stay far.  The halo traveler is the center vector v_c.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_D,
    BLUE_E,
    Circle,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
    Text,
    VGroup,
    ValueTracker,
    WHITE,
    YELLOW,
    YELLOW_A,
    config,
    linear,
    smooth,
    always_redraw,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth.  Rows of U are context embeddings u_i.
# v_c is the center embedding of token c.  Tokens sit on the axes at t=0
# so v_c · u_o = 0 (orthogonal one-hots in 2-D).  Only pair (c, o) is trained.
# ---------------------------------------------------------------------------
INITIAL_U = np.array(
    [
        [1.20, 0.00],
        [0.00, 1.20],
        [0.00, -1.20],
        [-1.20, 0.00],
    ],
    dtype=float,
)
INITIAL_V = np.array([1.20, 0.00], dtype=float)
CENTER = 0
CONTEXT = 1
FAR = 2
OTHER = 3
LEARNING_RATE = 0.22
SGD_STEPS = 6
TOKEN_NAMES = ("v_c", "u_o", "u_n", "u_k")
TOKEN_COLORS = (YELLOW, BLUE, BLUE_D, BLUE_A)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the same scores drawn on screen."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def skipgram_states() -> list[dict[str, np.ndarray]]:
    """SGD on one skip-gram pair: L = -log softmax(U v_c)[o]."""
    context = INITIAL_U.copy()
    center = INITIAL_V.copy()
    states: list[dict[str, np.ndarray]] = []

    def snapshot() -> None:
        scores = context @ center
        states.append(
            {
                "U": context.copy(),
                "v": center.copy(),
                "scores": scores.copy(),
                "p": softmax(scores).copy(),
            }
        )

    snapshot()
    for _ in range(SGD_STEPS):
        scores = context @ center
        probabilities = softmax(scores)
        grad_scores = probabilities.copy()
        grad_scores[CONTEXT] -= 1.0
        grad_context = grad_scores[:, None] * center[None, :]
        grad_center = context.T @ grad_scores
        context = context - LEARNING_RATE * grad_context
        center = center - LEARNING_RATE * grad_center
        snapshot()
    return states


STATES = skipgram_states()


def mix_state(progress: float) -> dict[str, np.ndarray]:
    """Linear blend between stored SGD snapshots. Numpy is the only arithmetic."""
    last = len(STATES) - 1
    clamped = min(max(progress, 0.0), float(last))
    index = int(np.floor(clamped))
    if index >= last:
        current = STATES[-1]
        return {
            "U": current["U"].copy(),
            "v": current["v"].copy(),
            "scores": current["scores"].copy(),
            "p": current["p"].copy(),
        }
    fraction = clamped - index
    context = (1.0 - fraction) * STATES[index]["U"] + fraction * STATES[index + 1]["U"]
    center = (1.0 - fraction) * STATES[index]["v"] + fraction * STATES[index + 1]["v"]
    scores = context @ center
    return {
        "U": context,
        "v": center,
        "scores": scores,
        "p": softmax(scores),
    }


class Episode141(Scene):
    """A ~32-second continuous, silent skip-gram visualization."""

    x_min, x_max = -2.05, 2.05
    y_min, y_max = -1.85, 2.05
    plot_left, plot_right = -4.55, 2.55
    plot_bottom, plot_top = -3.05, 2.70
    unit_dot_radius = 0.070

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line objects: never a filled VMobject mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-2.0, 2.01, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-1.5, 2.01, 0.5):
            if abs(y_value) > 1e-8:
                grid_lines.add(
                    Line(self.plot_point(self.x_min, y_value), self.plot_point(self.x_max, y_value)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        x_axis = Line(self.plot_point(self.x_min, 0.0), self.plot_point(self.x_max, 0.0)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        y_axis = Line(self.plot_point(0.0, self.y_min), self.plot_point(0.0, self.y_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        x_symbol = Text("e₁", font_size=24, color=BLUE_A).move_to(
            x_axis.get_end() + np.array([0.38, -0.32, 0.0])
        )
        y_symbol = Text("e₂", font_size=24, color=BLUE_A).move_to(
            y_axis.get_end() + np.array([0.38, 0.28, 0.0])
        )
        return grid_lines, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        final_state = STATES[-1]
        print("D2L 14.1 init U=\n{}".format(np.array2string(INITIAL_U, precision=2, separator=", ")))
        print("D2L 14.1 init v_c={}".format(np.array2string(INITIAL_V, precision=2, separator=", ")))
        print(
            "D2L 14.1 init scores={}, p={}".format(
                np.array2string(STATES[0]["scores"], precision=3, separator=", "),
                np.array2string(STATES[0]["p"], precision=3, separator=", "),
            )
        )
        for index, state in enumerate(STATES[1:], start=1):
            print(
                "D2L 14.1 step {} v_c={} u_o={} u_n={} u_k={} scores={} p={} dist_o={:.4f} dist_n={:.4f}".format(
                    index,
                    np.array2string(state["v"], precision=3, separator=", "),
                    np.array2string(state["U"][CONTEXT], precision=3, separator=", "),
                    np.array2string(state["U"][FAR], precision=3, separator=", "),
                    np.array2string(state["U"][OTHER], precision=3, separator=", "),
                    np.array2string(state["scores"], precision=3, separator=", "),
                    np.array2string(state["p"], precision=3, separator=", "),
                    float(np.linalg.norm(state["U"][CONTEXT] - state["v"])),
                    float(np.linalg.norm(state["U"][FAR] - state["v"])),
                )
            )
        print(
            "D2L 14.1 final v_c={} u_o={} s_o={:.6f} s_n={:.6f} p_o={:.6f} p_n={:.6f}".format(
                np.array2string(final_state["v"], precision=6, separator=", "),
                np.array2string(final_state["U"][CONTEXT], precision=6, separator=", "),
                float(final_state["U"][CONTEXT] @ final_state["v"]),
                float(final_state["U"][FAR] @ final_state["v"]),
                float(final_state["p"][CONTEXT]),
                float(final_state["p"][FAR]),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "14.1")

        grid, axes = self.make_grid_and_axes()
        origin = self.plot_point(0.0, 0.0)
        progress = ValueTracker(0.0)

        def current() -> dict[str, np.ndarray]:
            return mix_state(progress.get_value())

        def center_point() -> np.ndarray:
            vector = current()["v"]
            return self.plot_point(float(vector[0]), float(vector[1]))

        def token_point(index: int) -> np.ndarray:
            if index == CENTER:
                return center_point()
            vector = current()["U"][index]
            return self.plot_point(float(vector[0]), float(vector[1]))

        label_offsets = (
            np.array([0.52, 0.38, 0.0]),
            np.array([-0.42, 0.38, 0.0]),
            np.array([0.40, -0.38, 0.0]),
            np.array([-0.08, 0.40, 0.0]),
        )

        shafts = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(origin, token_point(index)).set_stroke(
                        TOKEN_COLORS[index], width=2.6, opacity=0.90
                    )
                )
                for index in range(4)
            ]
        )
        dots = VGroup(
            *[
                always_redraw(
                    lambda index=index: Dot(
                        token_point(index),
                        radius=self.unit_dot_radius,
                        color=TOKEN_COLORS[index],
                    ).set_fill(TOKEN_COLORS[index], opacity=0.96)
                    .set_stroke(TOKEN_COLORS[index], width=0.0, opacity=0.0)
                )
                for index in range(4)
            ]
        )
        name_labels = VGroup(
            *[
                always_redraw(
                    lambda index=index: Text(
                        TOKEN_NAMES[index],
                        font_size=24,
                        color=TOKEN_COLORS[index],
                    ).move_to(token_point(index) + label_offsets[index])
                )
                for index in range(4)
            ]
        )

        halo_outer = always_redraw(
            lambda: Circle(radius=0.265)
            .move_to(center_point())
            .set_stroke(BLUE_A, width=2.1, opacity=0.48)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=0.172)
            .move_to(center_point())
            .set_stroke(YELLOW, width=2.8, opacity=0.98)
        )

        window_chord = always_redraw(
            lambda: Line(center_point(), token_point(CONTEXT)).set_stroke(
                YELLOW, width=4.0, opacity=0.92
            )
        )

        s_o_prefix = Text("s_o = ", font_size=24, color=YELLOW)
        s_o_number = DecimalNumber(
            0.0,
            num_decimal_places=2,
            mob_class=Text,
            include_sign=True,
            color=WHITE,
            font_size=24,
        )
        s_o_row = VGroup(s_o_prefix, s_o_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
        s_o_row.move_to(np.array([-5.72, -2.18, 0.0]))
        s_n_prefix = Text("s_n = ", font_size=24, color=BLUE_D)
        s_n_number = DecimalNumber(
            0.0,
            num_decimal_places=2,
            mob_class=Text,
            include_sign=True,
            color=WHITE,
            font_size=24,
        )
        s_n_row = VGroup(s_n_prefix, s_n_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
        s_n_row.move_to(np.array([-5.72, -2.62, 0.0]))
        s_o_number.add_updater(
            lambda number: number.set_value(float(current()["U"][CONTEXT] @ current()["v"]))
        )
        s_n_number.add_updater(
            lambda number: number.set_value(float(current()["U"][FAR] @ current()["v"]))
        )

        formula = Text("P(w_o|w_c) ∝ exp(u_oᵀ v_c)", font_size=30, color=WHITE).move_to(
            np.array([2.55, 3.16, 0.0])
        )

        # 0.40–2.40 s: four orthogonal-at-start vectors. Short fade so t=1s is readable.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[Create(shaft) for shaft in shafts], lag_ratio=0.10),
            LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.10),
            LaggedStart(*[FadeIn(label) for label in name_labels], lag_ratio=0.10),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.45)

        # 2.40–4.40 s: halo traveler is the center word; pocket scores start at 0.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(s_o_row),
            FadeIn(s_n_row),
            Flash(center_point(), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(0.90)

        # 4.40–6.40 s: formula once.
        self.play(FadeIn(formula), run_time=1.20, rate_func=smooth)
        self.wait(0.80)

        # 6.40–8.40 s: the window pair. u_n, u_k get no yellow chord.
        self.play(Create(window_chord), run_time=1.20, rate_func=smooth)
        self.wait(0.80)

        # 8.40–20.40 s: six real skip-gram steps, ~2 s each.
        for step_index in range(1, SGD_STEPS + 1):
            self.play(
                progress.animate.set_value(float(step_index)),
                run_time=1.40,
                rate_func=smooth,
            )
            self.wait(0.60)

        # 20.40–22.40 s: the close pair, and the far token still far.
        self.play(
            Flash(center_point(), color=YELLOW, flash_radius=0.38, line_length=0.10),
            Flash(token_point(CONTEXT), color=YELLOW, flash_radius=0.32, line_length=0.09),
            run_time=1.00,
            rate_func=smooth,
        )
        self.wait(1.00)

        # 22.40–32.00 s: last frame keeps every vector.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=9.60, rate_func=linear)
