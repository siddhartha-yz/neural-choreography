"""D2L 9.4 — forward and backward hidden states concatenated at one t.

Every moving value comes from the tanh bidirectional recurrence below.
Weights are fixed and untrained. The haloed traveler is the concatenated
vector H_t = [→h_t, ←h_t] at one time step, not a token.
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
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    Scene,
    SurroundingRectangle,
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
# One numerical source of truth: scalar tokens, scalar hidden states.
#   →h_t = tanh(W_xh^(f) x_t + W_hh^(f) →h_{t-1} + b_h^(f))
#   ←h_t = tanh(W_xh^(b) x_t + W_hh^(b) ←h_{t+1} + b_h^(b))
#   H_t  = [→h_t, ←h_t]
# Untrained; ϕ is tanh. Ends are initialized at 0.
# ---------------------------------------------------------------------------
TOKENS = (
    np.array([[1.00]], dtype=float),
    np.array([[0.50]], dtype=float),
    np.array([[-0.80]], dtype=float),
    np.array([[0.60]], dtype=float),
)
WEIGHT_XH_FORWARD = np.array([[1.10]], dtype=float)
WEIGHT_HH_FORWARD = np.array([[0.40]], dtype=float)
BIAS_FORWARD = np.array([[-0.05]], dtype=float)
WEIGHT_XH_BACKWARD = np.array([[-0.90]], dtype=float)
WEIGHT_HH_BACKWARD = np.array([[0.55]], dtype=float)
BIAS_BACKWARD = np.array([[0.20]], dtype=float)
HIDDEN_START = np.zeros((1, 1), dtype=float)
CONCAT_INDEX = 2  # 1-based t = 3; opposite signs make the join readable
TOKEN_COLORS = (YELLOW, BLUE, YELLOW, BLUE)


def format_token_label(value: float) -> str:
    """Match the 3.1 minus glyph for negative features."""
    if value < 0:
        return f"−{abs(value):.2f}"
    return f"{value:.2f}"


def forward_hidden_states() -> np.ndarray:
    """Left-to-right pass; each column is one time step."""
    hidden = HIDDEN_START.copy()
    states = []
    for token in TOKENS:
        hidden = np.tanh(WEIGHT_XH_FORWARD @ token + WEIGHT_HH_FORWARD @ hidden + BIAS_FORWARD)
        states.append(hidden.copy())
    return np.hstack(states)


def backward_hidden_states() -> np.ndarray:
    """Right-to-left pass; each column is one time step."""
    hidden = HIDDEN_START.copy()
    states = [np.zeros((1, 1), dtype=float) for _ in TOKENS]
    for index in range(len(TOKENS) - 1, -1, -1):
        hidden = np.tanh(
            WEIGHT_XH_BACKWARD @ TOKENS[index] + WEIGHT_HH_BACKWARD @ hidden + BIAS_BACKWARD
        )
        states[index] = hidden.copy()
    return np.hstack(states)


FORWARD_H = forward_hidden_states()
BACKWARD_H = backward_hidden_states()
CONCAT_H = np.vstack((FORWARD_H[:, [CONCAT_INDEX]], BACKWARD_H[:, [CONCAT_INDEX]]))


class Episode094(Scene):
    """One continuous silent D2L 9.4 visualization, about 30 seconds."""

    bar_x = (-2.20, -0.10, 2.00, 4.10)
    token_y = 0.0
    forward_baseline = 1.70
    backward_baseline = -1.82
    needle_scale = 1.52
    axis_left = -3.05
    axis_right = 5.05

    def token_point(self, index: int) -> np.ndarray:
        return np.array([self.bar_x[index], self.token_y, 0.0])

    def forward_points(self, index: int, reveal: float) -> tuple[np.ndarray, np.ndarray]:
        x_position = self.bar_x[index]
        start = np.array([x_position, self.forward_baseline, 0.0])
        end = np.array(
            [
                x_position,
                self.forward_baseline + FORWARD_H[0, index] * self.needle_scale * reveal,
                0.0,
            ]
        )
        return start, end

    def backward_points(self, index: int, reveal: float) -> tuple[np.ndarray, np.ndarray]:
        x_position = self.bar_x[index]
        start = np.array([x_position, self.backward_baseline, 0.0])
        end = np.array(
            [
                x_position,
                self.backward_baseline + BACKWARD_H[0, index] * self.needle_scale * reveal,
                0.0,
            ]
        )
        return start, end

    def make_time_axes(self) -> tuple[VGroup, VGroup, VGroup]:
        """Stroke-only time axis and two signed zero axes. Never a filled mesh."""
        guides = VGroup()
        for x_position in self.bar_x:
            guides.add(
                Line(
                    np.array([x_position, self.backward_baseline, 0.0]),
                    np.array([x_position, self.forward_baseline, 0.0]),
                ).set_stroke(BLUE_E, width=1.0, opacity=0.30)
            )
        time_axis = Line(
            np.array([self.axis_left, self.token_y, 0.0]),
            np.array([self.axis_right, self.token_y, 0.0]),
        ).set_stroke(BLUE_D, width=2.0, opacity=0.88)
        forward_axis = Line(
            np.array([self.axis_left, self.forward_baseline, 0.0]),
            np.array([self.axis_right, self.forward_baseline, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        backward_axis = Line(
            np.array([self.axis_left, self.backward_baseline, 0.0]),
            np.array([self.axis_right, self.backward_baseline, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        time_symbol = Text("t", font_size=26, color=BLUE_A).next_to(
            time_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12
        )
        forward_mark = Text("→", font_size=30, color=YELLOW).next_to(
            forward_axis.get_start(), np.array([-1.0, 0.0, 0.0]), buff=0.14
        )
        backward_mark = Text("←", font_size=30, color=BLUE).next_to(
            backward_axis.get_start(), np.array([-1.0, 0.0, 0.0]), buff=0.14
        )
        forward_zero = Text("0", font_size=20, color=BLUE_A).next_to(
            forward_axis.get_start(), np.array([1.0, 1.0, 0.0]), buff=0.10
        )
        backward_zero = Text("0", font_size=20, color=BLUE_A).next_to(
            backward_axis.get_start(), np.array([1.0, -1.0, 0.0]), buff=0.10
        )
        axes = VGroup(time_axis, forward_axis, backward_axis, time_symbol)
        marks = VGroup(forward_mark, backward_mark, forward_zero, backward_zero)
        return guides, axes, marks

    def construct(self) -> None:
        token_values = np.array([float(token.item()) for token in TOKENS], dtype=float)
        print(
            "D2L 9.4 x={}".format(
                np.array2string(token_values, precision=3, separator=", ")
            )
        )
        print(
            "D2L 9.4 W_xh_f={}, W_hh_f={}, b_f={}".format(
                float(WEIGHT_XH_FORWARD.item()),
                float(WEIGHT_HH_FORWARD.item()),
                float(BIAS_FORWARD.item()),
            )
        )
        print(
            "D2L 9.4 W_xh_b={}, W_hh_b={}, b_b={}".format(
                float(WEIGHT_XH_BACKWARD.item()),
                float(WEIGHT_HH_BACKWARD.item()),
                float(BIAS_BACKWARD.item()),
            )
        )
        print(
            "D2L 9.4 forward h={}".format(
                np.array2string(FORWARD_H.ravel(), precision=6, separator=", ")
            )
        )
        print(
            "D2L 9.4 backward h={}".format(
                np.array2string(BACKWARD_H.ravel(), precision=6, separator=", ")
            )
        )
        print(
            "D2L 9.4 concat t={} H={}".format(
                CONCAT_INDEX + 1,
                np.array2string(CONCAT_H.ravel(), precision=6, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "09.4")

        guides, axes, marks = self.make_time_axes()
        token_dots = [
            Dot(self.token_point(index), radius=0.070, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for index, color in enumerate(TOKEN_COLORS)
        ]
        # Sit under the backward needles, never inside the concat box.
        token_labels = VGroup(
            *[
                Text(format_token_label(float(token.item())), font_size=20, color=BLUE_A).move_to(
                    np.array([self.bar_x[index], -3.22, 0.0])
                )
                for index, token in enumerate(TOKENS)
            ]
        )

        forward_reveal = [ValueTracker(0.0) for _ in TOKENS]
        backward_reveal = [ValueTracker(0.0) for _ in TOKENS]
        forward_needles = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(
                        *self.forward_points(index, forward_reveal[index].get_value())
                    ).set_stroke(YELLOW, width=14.0, opacity=0.98)
                )
                for index in range(len(TOKENS))
            ]
        )
        backward_needles = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(
                        *self.backward_points(index, backward_reveal[index].get_value())
                    ).set_stroke(BLUE, width=14.0, opacity=0.98)
                )
                for index in range(len(TOKENS))
            ]
        )

        formula = Text("Hₜ = [→hₜ, ←hₜ]", font_size=34, color=WHITE).move_to(
            np.array([4.55, 3.18, 0.0])
        )

        concat_forward_end = self.forward_points(CONCAT_INDEX, 1.0)[1]
        concat_backward_end = self.backward_points(CONCAT_INDEX, 1.0)[1]
        concat_span = Line(concat_forward_end, concat_backward_end).set_stroke(
            width=0.0, opacity=0.0
        )
        concat_box = SurroundingRectangle(
            concat_span, color=YELLOW, buff=0.32, corner_radius=0.08
        )
        concat_box.set_fill(opacity=0.0)
        concat_box.set_stroke(YELLOW, width=2.6, opacity=1.0)
        halo_outer = SurroundingRectangle(
            concat_span, color=BLUE_A, buff=0.46, corner_radius=0.10
        )
        halo_outer.set_fill(opacity=0.0)
        halo_outer.set_stroke(BLUE_A, width=2.1, opacity=0.50)

        def make_pocket_row(label: str, value: float, y_position: float, color, signed: bool) -> VGroup:
            prefix = Text(label, font_size=21, color=color)
            number = DecimalNumber(
                value,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=signed,
                color=WHITE,
                font_size=21,
            )
            return VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08).move_to(
                np.array([-5.95, y_position, 0.0])
            )

        forward_row = make_pocket_row("→h₃ =", float(FORWARD_H[0, CONCAT_INDEX]), 0.55, YELLOW, True)
        backward_row = make_pocket_row("←h₃ =", float(BACKWARD_H[0, CONCAT_INDEX]), 0.12, BLUE, True)
        concat_row = Text(
            "H₃ = [{:+.3f}, {:+.3f}]".format(
                float(FORWARD_H[0, CONCAT_INDEX]),
                float(BACKWARD_H[0, CONCAT_INDEX]),
            ),
            font_size=21,
            color=WHITE,
        ).move_to(np.array([-5.72, -0.32, 0.0]))
        traveler_pocket = VGroup(forward_row, backward_row, concat_row)

        # Names sit off the box stroke, in the empty gap toward t=4.
        forward_name = Text("→h₃", font_size=22, color=YELLOW_A).move_to(
            concat_forward_end + np.array([1.08, 0.02, 0.0])
        )
        backward_name = Text("←h₃", font_size=22, color=BLUE_A).move_to(
            concat_backward_end + np.array([1.08, -0.02, 0.0])
        )

        # 0.40–2.55 s: tiny sequence on a time axis, both direction axes present.
        self.play(
            FadeIn(guides),
            FadeIn(axes),
            FadeIn(marks),
            LaggedStart(*[FadeIn(dot) for dot in token_dots], lag_ratio=0.08),
            FadeIn(token_labels),
            run_time=0.80,
            rate_func=linear,
        )
        self.wait(1.35)

        # 2.55–4.50 s: formula once, before any hidden state grows.
        self.play(FadeIn(formula), run_time=0.70, rate_func=smooth)
        self.wait(1.25)

        # 4.50–12.50 s: left-to-right pass, ~2 s per time step. Dots are never scaled.
        self.add(forward_needles)
        for index in range(len(TOKENS)):
            color = TOKEN_COLORS[index]
            brightening = token_dots[index].animate.set_fill(color, opacity=1.0).set_stroke(
                color, width=3.2, opacity=1.0
            )
            dimming = token_dots[index].animate.set_fill(color, opacity=0.94).set_stroke(
                color, width=0.0, opacity=0.0
            )
            self.play(brightening, run_time=0.28, rate_func=smooth)
            self.play(
                forward_reveal[index].animate.set_value(1.0),
                run_time=1.40,
                rate_func=smooth,
            )
            self.play(dimming, run_time=0.24, rate_func=smooth)
            self.wait(0.08)

        # 12.50–20.50 s: right-to-left pass, same cadence.
        self.add(backward_needles)
        for index in range(len(TOKENS) - 1, -1, -1):
            color = TOKEN_COLORS[index]
            brightening = token_dots[index].animate.set_fill(color, opacity=1.0).set_stroke(
                color, width=3.2, opacity=1.0
            )
            dimming = token_dots[index].animate.set_fill(color, opacity=0.94).set_stroke(
                color, width=0.0, opacity=0.0
            )
            self.play(brightening, run_time=0.28, rate_func=smooth)
            self.play(
                backward_reveal[index].animate.set_value(1.0),
                run_time=1.40,
                rate_func=smooth,
            )
            self.play(dimming, run_time=0.24, rate_func=smooth)
            self.wait(0.08)

        # 20.50–23.20 s: concatenate at t=3. The halo sits on H_3, not on a token.
        concat_mid = 0.5 * (concat_forward_end + concat_backward_end)
        self.play(
            Create(halo_outer),
            Create(concat_box),
            FadeIn(forward_name),
            FadeIn(backward_name),
            Flash(concat_mid, color=YELLOW, flash_radius=0.46, line_length=0.11),
            run_time=0.80,
            rate_func=smooth,
        )
        self.play(FadeIn(traveler_pocket), run_time=0.55, rate_func=smooth)
        self.wait(1.35)

        # 23.20–30.20 s: keep both directions and the concat traveler through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.00, rate_func=linear)
