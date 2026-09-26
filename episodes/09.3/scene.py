"""D2L 9.3 — a hidden state handed both to the next time and to the next layer.

Every moving value comes from the two-layer tanh recurrence below.  Weights
are fixed and untrained.  Each layer has one hidden unit.  The haloed traveler
is that unit in the upper layer, unrolled across three time steps.
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
config.renderer = "cairo"


# ---------------------------------------------------------------------------
# One numerical source of truth: scalar one-unit deep RNN
#   H_t^{(0)} = x_t
#   H_t^{(l)} = tanh(W_xh^{(l)} H_t^{(l-1)} + W_hh^{(l)} H_{t-1}^{(l)} + b^{(l)})
# Two layers, three tokens.  Untrained.
# ---------------------------------------------------------------------------
INPUTS = np.array([[1.00], [-0.80], [0.60]], dtype=float)
W_XH_1 = np.array([[0.85]], dtype=float)
W_HH_1 = np.array([[0.50]], dtype=float)
B_1 = np.array([[0.10]], dtype=float)
W_XH_2 = np.array([[0.90]], dtype=float)
W_HH_2 = np.array([[0.40]], dtype=float)
B_2 = np.array([[-0.08]], dtype=float)


def deep_rnn_layers() -> tuple[np.ndarray, np.ndarray]:
    """Return hidden trajectories including t=0 zeros. Shape (4, 1)."""
    hidden_1 = np.zeros((1, 1), dtype=float)
    hidden_2 = np.zeros((1, 1), dtype=float)
    layer_1 = [hidden_1.copy()]
    layer_2 = [hidden_2.copy()]
    for token in INPUTS:
        hidden_1 = np.tanh(W_XH_1 @ token + W_HH_1 @ hidden_1 + B_1)
        hidden_2 = np.tanh(W_XH_2 @ hidden_1 + W_HH_2 @ hidden_2 + B_2)
        layer_1.append(hidden_1.copy())
        layer_2.append(hidden_2.copy())
    return np.stack(layer_1), np.stack(layer_2)


LAYER_1, LAYER_2 = deep_rnn_layers()
TOKEN_COLORS = (YELLOW, BLUE, BLUE_D)


class Episode093(Scene):
    """One continuous silent D2L 9.3 visualization, about 32 seconds."""

    col_x = (-4.90, -1.80, 1.20, 4.20)
    y_layer_2 = 1.58
    y_layer_1 = -0.28
    y_input = -2.58
    node_radius = 0.078
    slot_radius = 0.168

    def node_center(self, time_index: int, layer: int) -> np.ndarray:
        y_value = self.y_layer_1 if layer == 1 else self.y_layer_2
        return np.array([self.col_x[time_index], y_value, 0.0])

    def input_center(self, time_index: int) -> np.ndarray:
        return np.array([self.col_x[time_index], self.y_input, 0.0])

    def edge_line(self, start: np.ndarray, end: np.ndarray, color, opacity: float, width: float = 2.2) -> Line:
        """Stroke-only connector that stops short of both node disks."""
        offset = end - start
        unit = offset / np.linalg.norm(offset)
        return Line(start + unit * 0.24, end - unit * 0.24).set_stroke(color, width=width, opacity=opacity)

    def make_slot(self, center: np.ndarray, color) -> Circle:
        return Circle(radius=self.slot_radius).move_to(center).set_fill(BLACK, opacity=0.0).set_stroke(
            color, width=1.8, opacity=0.42
        )

    def make_value_dot(self, center: np.ndarray, color, opacity: float = 1.0) -> Dot:
        """Fixed-radius disk. Never scaled."""
        return (
            Dot(center, radius=self.node_radius, color=color)
            .set_fill(color, opacity=opacity)
            .set_stroke(color, width=0.0, opacity=0.0)
        )

    def construct(self) -> None:
        print("D2L 9.3 X={}".format(np.array2string(INPUTS.ravel(), precision=6, separator=", ")))
        print(
            "D2L 9.3 W_xh1={} W_hh1={} b1={}".format(
                float(W_XH_1.item()),
                float(W_HH_1.item()),
                float(B_1.item()),
            )
        )
        print(
            "D2L 9.3 W_xh2={} W_hh2={} b2={}".format(
                float(W_XH_2.item()),
                float(W_HH_2.item()),
                float(B_2.item()),
            )
        )
        print(
            "D2L 9.3 H^(1)[t=0..3]={}".format(
                np.array2string(LAYER_1.reshape(-1), precision=6, separator=", ")
            )
        )
        print(
            "D2L 9.3 H^(2)[t=0..3]={}".format(
                np.array2string(LAYER_2.reshape(-1), precision=6, separator=", ")
            )
        )

        def signed_number(value: float, color, font_size: int = 21) -> DecimalNumber:
            number = DecimalNumber(
                value,
                num_decimal_places=2,
                mob_class=Text,
                include_sign=True,
                color=color,
                font_size=font_size,
            )
            return number

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "09.3")

        # Lattice geometry. Layer labels live on the rows; time labels on the
        # columns. Formula uses the empty band above the lattice, not a HUD.
        layer_2_name = Text("h⁽²⁾", font_size=26, color=YELLOW_A).move_to(np.array([-6.45, self.y_layer_2, 0.0]))
        layer_1_name = Text("h⁽¹⁾", font_size=26, color=BLUE_A).move_to(np.array([-6.45, self.y_layer_1, 0.0]))
        time_names = VGroup(
            *[
                Text(f"t={time_index}", font_size=22, color=BLUE_A).move_to(
                    np.array([self.col_x[time_index], 2.52, 0.0])
                )
                for time_index in (1, 2, 3)
            ]
        )
        h0_name = Text("h₀", font_size=22, color=BLUE_A).move_to(np.array([self.col_x[0], 2.52, 0.0]))

        formula = Text(
            "hₜ^(l) = ϕ(Wₓₕ hₜ^(l-1) + Wₕₕ hₜ₋₁^(l) + b)",
            font_size=28,
            color=WHITE,
        ).move_to(np.array([0.20, 3.30, 0.0]))

        slots_layer_1 = VGroup(*[self.make_slot(self.node_center(t, 1), BLUE_E) for t in (0, 1, 2, 3)])
        slots_layer_2 = VGroup(*[self.make_slot(self.node_center(t, 2), YELLOW) for t in (0, 1, 2, 3)])

        h0_dot_1 = self.make_value_dot(self.node_center(0, 1), BLUE, opacity=0.55)
        h0_dot_2 = self.make_value_dot(self.node_center(0, 2), YELLOW, opacity=0.55)
        h0_num_1 = signed_number(float(LAYER_1[0, 0, 0]), BLUE_A).move_to(
            self.node_center(0, 1) + np.array([0.0, -0.42, 0.0])
        )
        h0_num_2 = signed_number(float(LAYER_2[0, 0, 0]), YELLOW_A).move_to(
            self.node_center(0, 2) + np.array([0.0, 0.40, 0.0])
        )

        input_dots = [
            self.make_value_dot(self.input_center(time_index), color)
            for time_index, color in zip((1, 2, 3), TOKEN_COLORS)
        ]
        input_labels = VGroup(
            *[
                Text(
                    f"x{['₁', '₂', '₃'][time_index - 1]} = {float(INPUTS[time_index - 1, 0]):+.2f}",
                    font_size=20,
                    color=color,
                ).move_to(self.input_center(time_index) + np.array([0.0, -0.48, 0.0]))
                for time_index, color in zip((1, 2, 3), TOKEN_COLORS)
            ]
        )

        # Incoming edges: yellow = same layer, previous time; blue = layer below.
        horiz = {}
        vert = {}
        for time_index in (1, 2, 3):
            horiz[(time_index, 1)] = self.edge_line(
                self.node_center(time_index - 1, 1),
                self.node_center(time_index, 1),
                YELLOW,
                0.28,
            )
            horiz[(time_index, 2)] = self.edge_line(
                self.node_center(time_index - 1, 2),
                self.node_center(time_index, 2),
                YELLOW,
                0.28,
            )
            vert[(time_index, 1)] = self.edge_line(
                self.input_center(time_index),
                self.node_center(time_index, 1),
                BLUE,
                0.28,
            )
            vert[(time_index, 2)] = self.edge_line(
                self.node_center(time_index, 1),
                self.node_center(time_index, 2),
                BLUE,
                0.28,
            )
        all_edges = VGroup(*horiz.values(), *vert.values())

        hidden_dots: dict[tuple[int, int], Dot] = {}
        hidden_nums: dict[tuple[int, int], DecimalNumber] = {}
        for time_index in (1, 2, 3):
            hidden_dots[(time_index, 1)] = self.make_value_dot(self.node_center(time_index, 1), BLUE)
            hidden_dots[(time_index, 2)] = self.make_value_dot(self.node_center(time_index, 2), YELLOW)
            hidden_nums[(time_index, 1)] = signed_number(float(LAYER_1[time_index, 0, 0]), BLUE_A).move_to(
                self.node_center(time_index, 1) + np.array([0.72, -0.38, 0.0])
            )
            hidden_nums[(time_index, 2)] = signed_number(float(LAYER_2[time_index, 0, 0]), YELLOW_A).move_to(
                self.node_center(time_index, 2) + np.array([0.72, 0.40, 0.0])
            )

        halo_x = ValueTracker(self.col_x[1])
        halo_y = ValueTracker(self.y_layer_2)

        def live_halo_center() -> np.ndarray:
            return np.array([halo_x.get_value(), halo_y.get_value(), 0.0])

        halo_outer = always_redraw(
            lambda: Circle(radius=0.278).move_to(live_halo_center()).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=0.178).move_to(live_halo_center()).set_stroke(YELLOW, width=2.8, opacity=0.98)
        )

        # 0.40–2.40 s: both layers, three times, zeros at t=0, and the tokens.
        # One fade so the t=1s frame already has inputs and h₀, not dangling edges.
        self.play(
            FadeIn(layer_1_name),
            FadeIn(layer_2_name),
            FadeIn(time_names),
            FadeIn(h0_name),
            FadeIn(slots_layer_1),
            FadeIn(slots_layer_2),
            FadeIn(all_edges),
            FadeIn(h0_dot_1),
            FadeIn(h0_dot_2),
            FadeIn(h0_num_1),
            FadeIn(h0_num_2),
            FadeIn(VGroup(*input_dots)),
            FadeIn(input_labels),
            run_time=0.55,
            rate_func=linear,
        )
        self.wait(1.45)

        # 2.40–4.50 s: formula once, in the empty band above the lattice.
        self.play(FadeIn(formula), run_time=0.70, rate_func=smooth)
        self.wait(1.40)

        def settle_edges(time_index: int, layer: int):
            return [
                horiz[(time_index, layer)].animate.set_stroke(YELLOW, width=2.4, opacity=0.52),
                vert[(time_index, layer)].animate.set_stroke(BLUE, width=2.4, opacity=0.52),
            ]

        for time_index in (1, 2, 3):
            token_center = self.input_center(time_index)
            # Layer 1 eats x_t from below and h_{t-1}^{(1)} from the left.
            self.play(
                horiz[(time_index, 1)].animate.set_stroke(YELLOW, width=4.4, opacity=0.98),
                vert[(time_index, 1)].animate.set_stroke(BLUE, width=4.4, opacity=0.98),
                Flash(token_center, color=TOKEN_COLORS[time_index - 1], flash_radius=0.36, line_length=0.09),
                run_time=0.45,
                rate_func=smooth,
            )
            self.play(
                FadeIn(hidden_dots[(time_index, 1)]),
                FadeIn(hidden_nums[(time_index, 1)]),
                run_time=1.25,
                rate_func=smooth,
            )
            self.play(*settle_edges(time_index, 1), run_time=0.35, rate_func=linear)

            # Layer 2 eats h_t^{(1)} from below and h_{t-1}^{(2)} from the left.
            self.play(
                horiz[(time_index, 2)].animate.set_stroke(YELLOW, width=4.4, opacity=0.98),
                vert[(time_index, 2)].animate.set_stroke(BLUE, width=4.4, opacity=0.98),
                run_time=0.45,
                rate_func=smooth,
            )
            if time_index == 1:
                self.add(halo_outer, halo_inner)
                self.play(
                    FadeIn(hidden_dots[(time_index, 2)]),
                    FadeIn(hidden_nums[(time_index, 2)]),
                    Flash(self.node_center(1, 2), color=YELLOW, flash_radius=0.42, line_length=0.11),
                    run_time=1.45,
                    rate_func=smooth,
                )
            else:
                self.play(
                    halo_x.animate.set_value(self.col_x[time_index]),
                    FadeIn(hidden_dots[(time_index, 2)]),
                    FadeIn(hidden_nums[(time_index, 2)]),
                    run_time=1.45,
                    rate_func=smooth,
                )
            self.play(*settle_edges(time_index, 2), run_time=0.35, rate_func=linear)

        # Keep both layers, every node, and the upper-layer halo through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=14.60, rate_func=linear)
