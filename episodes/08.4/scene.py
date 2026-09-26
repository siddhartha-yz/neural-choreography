"""D2L 8.4 — a hidden state carrying history across three RNN steps.

Every moving value comes from the tanh recurrence below.  Weights are fixed
and untrained.  The haloed traveler is the current token x_t; the yellow blob
is the same h, updated in place, with a faint trail of previous h left behind.
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
# One numerical source of truth: column-vector RNN
#   h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
# with a 2-D token and a 2-D hidden state.  Untrained; ϕ is tanh.
# ---------------------------------------------------------------------------
TOKENS = (
    np.array([[1.00], [0.50]], dtype=float),
    np.array([[-0.80], [1.10]], dtype=float),
    np.array([[0.60], [-0.90]], dtype=float),
)
WEIGHT_XH = np.array([[0.90, 0.20], [-0.25, 0.85]], dtype=float)
WEIGHT_HH = np.array([[0.60, -0.35], [0.40, 0.55]], dtype=float)
BIAS = np.array([[0.05], [-0.10]], dtype=float)
HIDDEN_START = np.zeros((2, 1), dtype=float)


def rnn_step(token: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """One real RNN update; ϕ = tanh."""
    pre_activation = WEIGHT_XH @ token + WEIGHT_HH @ hidden + BIAS
    return np.tanh(pre_activation)


def hidden_trajectory() -> list[np.ndarray]:
    hidden = HIDDEN_START.copy()
    states = [hidden.copy()]
    for token in TOKENS:
        hidden = rnn_step(token, hidden)
        states.append(hidden.copy())
    return states


HIDDEN_STATES = hidden_trajectory()
TOKEN_COLORS = (YELLOW, BLUE, BLUE_D)


class Episode084(Scene):
    """One continuous silent D2L 8.4 visualization, about 32 seconds."""

    # Left plot: 2-D token space. Right plot: 2-D hidden-state space.
    token_x_min, token_x_max = -1.60, 1.60
    token_y_min, token_y_max = -1.50, 1.50
    token_left, token_right = -5.85, -1.35
    token_bottom, token_top = -2.70, 2.55

    hidden_x_min, hidden_x_max = -1.20, 1.20
    hidden_y_min, hidden_y_max = -1.20, 1.20
    hidden_left, hidden_right = 0.55, 6.20
    hidden_bottom, hidden_top = -3.20, 2.55

    def token_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.token_left + (x_value - self.token_x_min) / (
            self.token_x_max - self.token_x_min
        ) * (self.token_right - self.token_left)
        scene_y = self.token_bottom + (y_value - self.token_y_min) / (
            self.token_y_max - self.token_y_min
        ) * (self.token_top - self.token_bottom)
        return np.array([scene_x, scene_y, 0.0])

    def hidden_point(self, first: float, second: float) -> np.ndarray:
        scene_x = self.hidden_left + (first - self.hidden_x_min) / (
            self.hidden_x_max - self.hidden_x_min
        ) * (self.hidden_right - self.hidden_left)
        scene_y = self.hidden_bottom + (second - self.hidden_y_min) / (
            self.hidden_y_max - self.hidden_y_min
        ) * (self.hidden_top - self.hidden_bottom)
        return np.array([scene_x, scene_y, 0.0])

    def make_token_grid(self) -> tuple[VGroup, VGroup]:
        """Stroke-only token-space grid; never a filled mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-1.5, 1.51, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(
                        self.token_point(x_value, self.token_y_min),
                        self.token_point(x_value, self.token_y_max),
                    ).set_stroke(BLUE_E, width=1.0, opacity=0.30)
                )
        for y_value in np.arange(-1.5, 1.51, 0.5):
            if abs(y_value) > 1e-8:
                grid_lines.add(
                    Line(
                        self.token_point(self.token_x_min, y_value),
                        self.token_point(self.token_x_max, y_value),
                    ).set_stroke(BLUE_E, width=1.0, opacity=0.30)
                )
        x_axis = Line(
            self.token_point(self.token_x_min, 0.0),
            self.token_point(self.token_x_max, 0.0),
        ).set_stroke(BLUE_D, width=2.0, opacity=0.88)
        y_axis = Line(
            self.token_point(0.0, self.token_y_min),
            self.token_point(0.0, self.token_y_max),
        ).set_stroke(BLUE_D, width=2.0, opacity=0.88)
        x_symbol = Text("x₁", font_size=24, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.10
        )
        y_symbol = Text("x₂", font_size=24, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.10
        )
        return grid_lines, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def make_hidden_grid(self) -> tuple[VGroup, VGroup]:
        """Stroke-only hidden-state grid; never a filled mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-1.0, 1.01, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(
                        self.hidden_point(x_value, self.hidden_y_min),
                        self.hidden_point(x_value, self.hidden_y_max),
                    ).set_stroke(BLUE_E, width=1.0, opacity=0.30)
                )
        for y_value in np.arange(-1.0, 1.01, 0.5):
            if abs(y_value) > 1e-8:
                grid_lines.add(
                    Line(
                        self.hidden_point(self.hidden_x_min, y_value),
                        self.hidden_point(self.hidden_x_max, y_value),
                    ).set_stroke(BLUE_E, width=1.0, opacity=0.30)
                )
        x_axis = Line(
            self.hidden_point(self.hidden_x_min, 0.0),
            self.hidden_point(self.hidden_x_max, 0.0),
        ).set_stroke(BLUE_D, width=2.0, opacity=0.88)
        y_axis = Line(
            self.hidden_point(0.0, self.hidden_y_min),
            self.hidden_point(0.0, self.hidden_y_max),
        ).set_stroke(BLUE_D, width=2.0, opacity=0.88)
        # No h₁/h₂ axis words: those names are the time-index blobs on this plane.
        return grid_lines, VGroup(x_axis, y_axis)

    def make_ghost(self, hidden: np.ndarray) -> Dot:
        """Previous h, same Dot radius as the live blob. Never scaled."""
        return (
            Dot(self.hidden_point(float(hidden[0, 0]), float(hidden[1, 0])), radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=0.26)
            .set_stroke(YELLOW, width=1.5, opacity=0.42)
        )

    def construct(self) -> None:
        for step_index, token in enumerate(TOKENS, start=1):
            hidden = HIDDEN_STATES[step_index]
            print(
                "D2L 8.4 t={} x=[{:.6f}, {:.6f}] h=[{:.6f}, {:.6f}]".format(
                    step_index,
                    float(token[0, 0]),
                    float(token[1, 0]),
                    float(hidden[0, 0]),
                    float(hidden[1, 0]),
                )
            )
        print(
            "D2L 8.4 h0=[{:.6f}, {:.6f}]".format(
                float(HIDDEN_STATES[0][0, 0]),
                float(HIDDEN_STATES[0][1, 0]),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "08.4")

        token_grid, token_axes = self.make_token_grid()
        hidden_grid, hidden_axes = self.make_hidden_grid()
        token_dots = [
            Dot(
                self.token_point(float(token[0, 0]), float(token[1, 0])),
                radius=0.070,
                color=color,
            )
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for token, color in zip(TOKENS, TOKEN_COLORS)
        ]

        hidden_first = ValueTracker(float(HIDDEN_STATES[0][0, 0]))
        hidden_second = ValueTracker(float(HIDDEN_STATES[0][1, 0]))
        token_first = ValueTracker(float(TOKENS[0][0, 0]))
        token_second = ValueTracker(float(TOKENS[0][1, 0]))

        def live_hidden_center() -> np.ndarray:
            return self.hidden_point(hidden_first.get_value(), hidden_second.get_value())

        def live_token_center() -> np.ndarray:
            return self.token_point(token_first.get_value(), token_second.get_value())

        hidden_blob = always_redraw(
            lambda: Dot(live_hidden_center(), radius=0.078, color=YELLOW).set_fill(YELLOW, opacity=1.0)
        )
        hidden_ring = always_redraw(
            lambda: Circle(radius=0.155).move_to(live_hidden_center()).set_stroke(YELLOW, width=2.2, opacity=0.90)
        )
        connector = always_redraw(
            lambda: Line(
                live_token_center() + np.array([0.28, 0.0, 0.0]),
                live_hidden_center() + np.array([-0.22, 0.0, 0.0]),
            ).set_stroke(BLUE_D, width=2.2, opacity=0.58)
        )

        halo_outer = always_redraw(
            lambda: Circle(radius=0.278).move_to(live_token_center()).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=0.178).move_to(live_token_center()).set_stroke(YELLOW, width=2.8, opacity=0.98)
        )

        # Time-index names sit on the hidden-state geometry, not in a HUD.
        hidden_names = [
            Text("h₀", font_size=22, color=YELLOW_A),
            Text("h₁", font_size=22, color=YELLOW_A),
            Text("h₂", font_size=22, color=YELLOW_A),
            Text("h₃", font_size=22, color=YELLOW_A),
        ]
        # Names live left/below the settled blob; live numbers take above/right.
        hidden_name_offsets = (
            np.array([-0.54, -0.48, 0.0]),
            np.array([0.02, -0.52, 0.0]),
            np.array([-0.58, 0.10, 0.0]),
            np.array([-0.54, -0.50, 0.0]),
        )
        for name, state, offset in zip(hidden_names, HIDDEN_STATES, hidden_name_offsets):
            name.move_to(
                self.hidden_point(float(state[0, 0]), float(state[1, 0])) + offset
            )

        formula = Text("hₜ = ϕ(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)", font_size=28, color=WHITE).move_to(
            np.array([3.40, 3.22, 0.0])
        )

        def make_component_row(label: str, x_position: float) -> tuple[VGroup, DecimalNumber]:
            prefix = Text(label, font_size=21, color=BLUE_A)
            number = DecimalNumber(
                0.0,
                num_decimal_places=2,
                mob_class=Text,
                include_sign=True,
                color=WHITE,
                font_size=21,
            )
            row = VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08).move_to(
                np.array([x_position, -3.30, 0.0])
            )
            return row, number

        token_row_1, token_number_1 = make_component_row("x₁ =", -4.90)
        token_row_2, token_number_2 = make_component_row("x₂ =", -2.35)
        token_number_1.set_value(token_first.get_value())
        token_number_2.set_value(token_second.get_value())
        token_number_1.add_updater(lambda number: number.set_value(token_first.get_value()))
        token_number_2.add_updater(lambda number: number.set_value(token_second.get_value()))
        token_readout = VGroup(token_row_1, token_row_2)

        def hidden_number_anchor() -> np.ndarray:
            """Park current h coordinates off the blob, axis, and time-index name."""
            center = live_hidden_center()
            first = hidden_first.get_value()
            second = hidden_second.get_value()
            # Default empty pocket: above-right. Names stay left/below.
            offset_x = 0.74
            offset_y = 0.40
            if first > 0.35:
                blend = min(1.0, (first - 0.35) / 0.45)
                offset_x = 0.74 * (1.0 - blend) + 0.00 * blend
                offset_y = 0.40 * (1.0 - blend) + 0.62 * blend
            if second > 0.40:
                blend = min(1.0, (second - 0.40) / 0.45)
                offset_x = offset_x * (1.0 - blend) + 0.88 * blend
                offset_y = offset_y * (1.0 - blend) + (-0.12) * blend
            return center + np.array([offset_x, offset_y, 0.0])

        hidden_num_1 = DecimalNumber(
            0.0,
            num_decimal_places=2,
            mob_class=Text,
            include_sign=True,
            color=YELLOW_A,
            font_size=20,
        )
        hidden_num_2 = DecimalNumber(
            0.0,
            num_decimal_places=2,
            mob_class=Text,
            include_sign=True,
            color=YELLOW_A,
            font_size=20,
        )
        hidden_num_1.set_value(hidden_first.get_value())
        hidden_num_2.set_value(hidden_second.get_value())
        hidden_readout = VGroup(hidden_num_1, hidden_num_2).arrange(
            np.array([0.0, -1.0, 0.0]), buff=0.08
        )
        hidden_num_1.add_updater(lambda number: number.set_value(hidden_first.get_value()))
        hidden_num_2.add_updater(lambda number: number.set_value(hidden_second.get_value()))
        hidden_readout.add_updater(lambda group: group.move_to(hidden_number_anchor()))


        # 0.40–2.80 s: both planes, then h₀ at the origin before any token.
        self.play(
            FadeIn(token_grid),
            FadeIn(token_axes),
            FadeIn(hidden_grid),
            FadeIn(hidden_axes),
            run_time=0.70,
            rate_func=linear,
        )
        self.add(hidden_blob, hidden_ring)
        self.play(FadeIn(hidden_names[0]), FadeIn(hidden_readout), run_time=0.45, rate_func=smooth)
        self.wait(1.25)

        # 2.80–4.20 s: current token x₁, formula once, connector into h-space.
        self.play(
            FadeIn(token_dots[0]),
            FadeIn(halo_outer),
            FadeIn(halo_inner),
            FadeIn(token_readout),
            Flash(self.token_point(1.00, 0.50), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.play(FadeIn(formula), FadeIn(connector), run_time=0.70, rate_func=smooth)

        # 4.20–8.00 s: first recurrence. Ghost h₀ stays; the same blob walks to h₁.
        ghost_zero = self.make_ghost(HIDDEN_STATES[0])
        trail_01 = Line(
            self.hidden_point(float(HIDDEN_STATES[0][0, 0]), float(HIDDEN_STATES[0][1, 0])),
            self.hidden_point(float(HIDDEN_STATES[1][0, 0]), float(HIDDEN_STATES[1][1, 0])),
        ).set_stroke(YELLOW, width=2.4, opacity=0.40)
        self.add(ghost_zero)
        self.play(
            hidden_first.animate.set_value(float(HIDDEN_STATES[1][0, 0])),
            hidden_second.animate.set_value(float(HIDDEN_STATES[1][1, 0])),
            Create(trail_01),
            run_time=3.40,
            rate_func=smooth,
        )
        self.play(FadeIn(hidden_names[1]), run_time=0.40, rate_func=smooth)
        self.wait(3.00)

        # 11.00–16.80 s: x₂ becomes current; h carries x₁ into h₂.
        self.play(
            FadeIn(token_dots[1]),
            token_first.animate.set_value(float(TOKENS[1][0, 0])),
            token_second.animate.set_value(float(TOKENS[1][1, 0])),
            Flash(self.token_point(-0.80, 1.10), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=2.00,
            rate_func=smooth,
        )
        ghost_one = self.make_ghost(HIDDEN_STATES[1])
        trail_12 = Line(
            self.hidden_point(float(HIDDEN_STATES[1][0, 0]), float(HIDDEN_STATES[1][1, 0])),
            self.hidden_point(float(HIDDEN_STATES[2][0, 0]), float(HIDDEN_STATES[2][1, 0])),
        ).set_stroke(YELLOW, width=2.4, opacity=0.40)
        self.add(ghost_one)
        self.play(
            hidden_first.animate.set_value(float(HIDDEN_STATES[2][0, 0])),
            hidden_second.animate.set_value(float(HIDDEN_STATES[2][1, 0])),
            Create(trail_12),
            run_time=3.40,
            rate_func=smooth,
        )
        self.play(FadeIn(hidden_names[2]), run_time=0.40, rate_func=smooth)
        self.wait(2.40)

        # 19.20–25.00 s: x₃; h₃ still depends on everything before.
        self.play(
            FadeIn(token_dots[2]),
            token_first.animate.set_value(float(TOKENS[2][0, 0])),
            token_second.animate.set_value(float(TOKENS[2][1, 0])),
            Flash(self.token_point(0.60, -0.90), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=2.00,
            rate_func=smooth,
        )
        ghost_two = self.make_ghost(HIDDEN_STATES[2])
        trail_23 = Line(
            self.hidden_point(float(HIDDEN_STATES[2][0, 0]), float(HIDDEN_STATES[2][1, 0])),
            self.hidden_point(float(HIDDEN_STATES[3][0, 0]), float(HIDDEN_STATES[3][1, 0])),
        ).set_stroke(YELLOW, width=2.4, opacity=0.40)
        self.add(ghost_two)
        self.play(
            hidden_first.animate.set_value(float(HIDDEN_STATES[3][0, 0])),
            hidden_second.animate.set_value(float(HIDDEN_STATES[3][1, 0])),
            Create(trail_23),
            run_time=3.40,
            rate_func=smooth,
        )
        self.play(FadeIn(hidden_names[3]), run_time=0.40, rate_func=smooth)

        # 25.00–32.00 s: keep the live blob and every ghost rendered through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.00, rate_func=linear)
