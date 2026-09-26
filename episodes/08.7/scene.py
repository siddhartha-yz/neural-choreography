"""D2L 8.7 — unroll one scalar RNN and walk a real gradient back in time.

The input sequence, recurrent state, terminal loss gradient, and truncated
window are all computed from the NumPy arrays below.  This is a fixed
forward/backward computation, not a training animation.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    AnimationGroup,
    Arrow,
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_D,
    BLUE_E,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    MoveAlongPath,
    Scene,
    SurroundingRectangle,
    Text,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    VGroup,
    ValueTracker,
    WHITE,
    YELLOW,
    YELLOW_A,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth.  A six-token scalar sequence enters a
# one-dimensional tanh RNN.  Gradients are BPTT derivatives of
# L = 1/2 (h_T - TARGET)^2, and the final visual window is truncated to 3.
# ---------------------------------------------------------------------------
INPUTS = np.array([[0.35], [-0.60], [0.75], [0.10], [-0.45], [0.65]], dtype=float)
W_XH = np.array([[0.74]], dtype=float)
W_HH = np.array([[0.70]], dtype=float)
BIAS = np.array([[-0.05]], dtype=float)
TARGET = np.array([[0.20]], dtype=float)
TRUNCATION_STEPS = 3


def forward_states(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return z_t and h_t for h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)."""
    preactivations = np.zeros_like(inputs)
    hidden = np.zeros((len(inputs) + 1, 1), dtype=float)
    for index, x_t in enumerate(inputs, start=1):
        preactivations[index - 1] = W_XH @ x_t + W_HH @ hidden[index - 1] + BIAS
        hidden[index] = np.tanh(preactivations[index - 1])
    return preactivations, hidden


def bptt_gradients(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dL/dh_t and dL/dz_t for the same unrolled scalar chain."""
    gradient_h = np.zeros_like(hidden)
    gradient_z = np.zeros_like(hidden)
    gradient_h[-1] = hidden[-1] - TARGET
    for time_index in range(len(hidden) - 1, 0, -1):
        gradient_z[time_index] = gradient_h[time_index] * (1.0 - hidden[time_index] ** 2)
        gradient_h[time_index - 1] += W_HH @ gradient_z[time_index]
    return gradient_h[1:], gradient_z[1:]


PREACTIVATIONS, HIDDEN = forward_states(INPUTS)
GRADIENT_H, GRADIENT_Z = bptt_gradients(HIDDEN)


class Episode087(Scene):
    """A 28-second continuous Cairo scene for D2L's BPTT mechanism."""

    hidden_y = 0.46
    input_y = -1.30
    hidden_radius = 0.36
    input_radius = 0.21
    chain_x = (-5.35, -3.21, -1.07, 1.07, 3.21, 5.35)

    def construct(self) -> None:
        print(
            "D2L 8.7 inputs={}, W_xh={}, W_hh={}, b={}, target={}, z={}, h={}, "
            "dL_dh={}, dL_dz={}, truncate={}".format(
                np.array2string(INPUTS.ravel(), precision=6, separator=", "),
                np.array2string(W_XH, precision=6, separator=", "),
                np.array2string(W_HH, precision=6, separator=", "),
                np.array2string(BIAS.ravel(), precision=6, separator=", "),
                np.array2string(TARGET.ravel(), precision=6, separator=", "),
                np.array2string(PREACTIVATIONS.ravel(), precision=6, separator=", "),
                np.array2string(HIDDEN[1:].ravel(), precision=6, separator=", "),
                np.array2string(GRADIENT_H.ravel(), precision=6, separator=", "),
                np.array2string(GRADIENT_Z.ravel(), precision=6, separator=", "),
                TRUNCATION_STEPS,
            )
        )

        # 0.00–0.50 s — chapter mark only.
        show_episode_intro(self, "08.7")

        hidden_centers = [np.array([x_position, self.hidden_y, 0.0]) for x_position in self.chain_x]
        input_centers = [np.array([x_position, self.input_y, 0.0]) for x_position in self.chain_x]
        hidden_nodes = [
            Circle(radius=self.hidden_radius)
            .move_to(center)
            .set_fill(BLACK, opacity=1.0)
            .set_stroke(BLUE_D, width=2.35, opacity=0.92)
            for center in hidden_centers
        ]
        hidden_symbols = [
            Text(f"h{'₁₂₃₄₅₆'[index]}", font_size=24, color=WHITE).move_to(hidden_nodes[index].get_center())
            for index in range(len(hidden_nodes))
        ]
        hidden_values = [
            Text(f"{HIDDEN[index + 1, 0]:+.2f}", font_size=20, color=BLUE_A).move_to(
                hidden_centers[index] + UP * 0.67
            )
            for index in range(len(hidden_nodes))
        ]

        input_nodes = [
            Circle(radius=self.input_radius)
            .move_to(center)
            .set_fill(BLACK, opacity=1.0)
            .set_stroke(BLUE_D, width=1.75, opacity=0.78)
            for center in input_centers
        ]
        input_symbols = [
            Text(f"x{'₁₂₃₄₅₆'[index]}", font_size=17, color=BLUE_A).move_to(input_nodes[index].get_center())
            for index in range(len(input_nodes))
        ]
        input_values = [
            Text(f"{INPUTS[index, 0]:+.2f}", font_size=18, color=YELLOW_A).move_to(
                input_centers[index] + DOWN * 0.48
            )
            for index in range(len(input_nodes))
        ]
        input_arrows = [
            Arrow(
                input_centers[index] + UP * self.input_radius,
                hidden_centers[index] + DOWN * self.hidden_radius,
                buff=0.03,
                stroke_width=1.85,
                max_tip_length_to_length_ratio=0.18,
                color=BLUE_D,
            ).set_opacity(0.65)
            for index in range(len(input_nodes))
        ]
        recurrent_edges = [
            Line(
                hidden_centers[index] + RIGHT * self.hidden_radius,
                hidden_centers[index + 1] + LEFT * self.hidden_radius,
            ).set_stroke(BLUE_D, width=2.45, opacity=0.74)
            for index in range(len(hidden_nodes) - 1)
        ]

        # The formula occupies the sole fixed upper pocket; all other labels
        # are attached to the states, inputs, or truncation geometry.
        recurrence_formula = Text(
            "hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)", font_size=31, color=WHITE
        ).move_to(np.array([0.0, 3.03, 0.0]))

        # 0.50–2.50 s — a compact recurrent loop, then its formula.
        loop_center = hidden_centers[0]
        loop_state = Circle(radius=self.hidden_radius).move_to(loop_center).set_fill(BLACK, opacity=1.0)
        loop_state.set_stroke(BLUE_D, width=2.35, opacity=0.92)
        loop_symbol = Text("hₜ", font_size=24, color=WHITE).move_to(loop_center)
        loop_input = Circle(radius=self.input_radius).move_to(input_centers[0]).set_fill(BLACK, opacity=1.0)
        loop_input.set_stroke(BLUE_D, width=1.75, opacity=0.78)
        loop_input_symbol = Text("xₜ", font_size=17, color=BLUE_A).move_to(input_centers[0])
        loop_input_arrow = Arrow(
            input_centers[0] + UP * self.input_radius,
            loop_center + DOWN * self.hidden_radius,
            buff=0.03,
            stroke_width=1.85,
            max_tip_length_to_length_ratio=0.18,
            color=BLUE_D,
        ).set_opacity(0.65)
        loop_feedback = CurvedArrow(
            loop_center + RIGHT * 0.28,
            loop_center + LEFT * 0.28,
            angle=-4.35,
            color=YELLOW,
            stroke_width=2.5,
            tip_length=0.18,
        ).shift(UP * 0.04)
        loop_group = VGroup(loop_state, loop_symbol, loop_input, loop_input_symbol, loop_input_arrow, loop_feedback)
        self.play(
            Create(loop_state),
            FadeIn(loop_symbol),
            Create(loop_input),
            FadeIn(loop_input_symbol),
            Create(loop_input_arrow),
            Create(loop_feedback),
            FadeIn(recurrence_formula),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(0.90)

        # 2.50–4.50 s — the feedback loop opens into the same six-state chain.
        chain_nodes = VGroup(*hidden_nodes, *hidden_symbols, *input_nodes, *input_symbols)
        chain_edges = VGroup(*recurrent_edges, *input_arrows)
        self.play(
            FadeOut(loop_group),
            FadeOut(recurrence_formula),
            LaggedStart(*[FadeIn(mobject) for mobject in chain_nodes], lag_ratio=0.035),
            LaggedStart(*[Create(edge) for edge in chain_edges], lag_ratio=0.055),
            run_time=2.00,
            rate_func=smooth,
        )

        def activation_color(value: float):
            return BLUE_A if value >= 0.0 else YELLOW_A

        def symbol_color(value: float):
            return BLACK if abs(value) >= 0.35 else WHITE

        def light_hidden(index: int):
            color = activation_color(HIDDEN[index + 1, 0])
            return AnimationGroup(
                hidden_nodes[index].animate.set_fill(color, opacity=0.92).set_stroke(color, width=3.45, opacity=1.0),
                hidden_symbols[index].animate.set_color(symbol_color(HIDDEN[index + 1, 0])),
            )

        halo_outer = Circle(radius=0.48).move_to(hidden_centers[0]).set_fill(opacity=0.0)
        halo_outer.set_stroke(BLUE_A, width=2.05, opacity=0.56)
        halo_inner = Circle(radius=0.415).move_to(hidden_centers[0]).set_fill(opacity=0.0)
        halo_inner.set_stroke(YELLOW, width=2.55, opacity=0.95)
        halo = VGroup(halo_outer, halo_inner)

        # 4.50–6.50 s — reveal the real sequence and h₁.
        self.play(
            FadeIn(VGroup(*input_values)),
            FadeIn(hidden_values[0]),
            Create(halo_outer),
            Create(halo_inner),
            light_hidden(0),
            Flash(hidden_centers[0], color=YELLOW, flash_radius=0.46, line_length=0.10),
            run_time=1.10,
            rate_func=smooth,
        )
        self.wait(0.90)

        def forward_step(index: int) -> None:
            """Move one fixed-size pulse and the same halo state to h_{t+1}."""
            edge = recurrent_edges[index]
            pulse = Dot(edge.get_start(), radius=0.064, color=BLUE_A).set_fill(BLUE_A, opacity=1.0)
            self.add(pulse)
            self.play(
                MoveAlongPath(pulse, edge),
                halo.animate.move_to(hidden_centers[index + 1]),
                light_hidden(index + 1),
                FadeIn(hidden_values[index + 1]),
                run_time=1.15,
                rate_func=smooth,
            )
            self.remove(pulse)
            self.wait(0.85)

        # 6.50–16.50 s — a visible forward pass through h₂…h₆.
        for index in range(len(recurrent_edges)):
            forward_step(index)

        terminal_gradient_label = Text(
            f"∂L/∂h₆ = {GRADIENT_H[-1, 0]:+.2f}", font_size=21, color=YELLOW
        ).move_to(hidden_centers[-1] + UP * 1.22 + LEFT * 0.02)

        # 16.50–18.50 s — establish the terminal derivative without training.
        self.play(
            FadeIn(terminal_gradient_label),
            Flash(hidden_centers[-1], color=YELLOW, flash_radius=0.46, line_length=0.10),
            run_time=0.80,
            rate_func=smooth,
        )
        self.wait(1.20)

        def backward_step(right_index: int, left_index: int) -> None:
            """Walk a fixed-size yellow gradient pulse backward on a drawn edge."""
            edge = recurrent_edges[left_index]
            reverse_path = Line(edge.get_end(), edge.get_start()).set_stroke(opacity=0.0)
            gradient_pulse = Dot(edge.get_end(), radius=0.064, color=YELLOW).set_fill(YELLOW, opacity=1.0)
            self.add(gradient_pulse)
            self.play(
                MoveAlongPath(gradient_pulse, reverse_path),
                edge.animate.set_stroke(YELLOW, width=4.0, opacity=1.0),
                hidden_nodes[left_index].animate.set_stroke(YELLOW, width=4.0, opacity=1.0),
                run_time=1.15,
                rate_func=smooth,
            )
            self.remove(gradient_pulse)
            self.wait(0.85)

        # 18.50–22.50 s — gradient returns on the exact recurrent edges h₆→h₅→h₄.
        backward_step(5, 4)
        backward_step(4, 3)

        # 22.50–24.50 s — draw the last-three-state truncated BPTT window.
        cut_x = (hidden_centers[2][0] + hidden_centers[3][0]) / 2
        cut_marker = DashedLine(
            np.array([cut_x, -0.43, 0.0]),
            np.array([cut_x, 1.48, 0.0]),
            dash_length=0.13,
        ).set_stroke(YELLOW, width=2.2, opacity=0.92)
        window = SurroundingRectangle(VGroup(*hidden_nodes[3:]), color=YELLOW, buff=0.66, corner_radius=0.11)
        window.set_fill(opacity=0.0)
        window.set_stroke(width=2.25, opacity=0.88)
        window_label = Text("τ = 3", font_size=22, color=YELLOW).move_to(
            np.array([hidden_centers[4][0], 1.80, 0.0])
        )
        self.play(
            recurrent_edges[2].animate.set_stroke(BLUE_D, width=1.8, opacity=0.20),
            Create(cut_marker),
            Create(window),
            FadeIn(window_label),
            run_time=1.00,
            rate_func=smooth,
        )
        self.wait(1.00)

        # 24.50–28.00 s — retain the entire unrolled chain in the final frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=3.50, rate_func=linear)
