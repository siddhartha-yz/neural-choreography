"""D2L 4.8 — sigmoid saturation turning backward needles into stubs.

Every moving value comes from the shared affine+sigmoid chain below. The
network is never trained; the two inits are two fixed parameter sets, and
``network_at`` is the only numerical source for activations and |∂ℓ/∂h|.
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


# Same halo traveler as 3.4. Four real affine+sigmoid layers: 2 → 1 → 1 → 1 → 1.
INPUT_X = np.array([[1.00], [0.50]], dtype=float)

# Saturated init: large scale pushes every pre-activation onto the sigmoid flat.
SAT_W1 = np.array([[2.40, 1.80]], dtype=float)
SAT_B1 = np.array([[1.10]], dtype=float)
SAT_W = np.array([3.20, 3.40, 2.80], dtype=float)
SAT_B = np.array([0.80, -0.40, 0.60], dtype=float)

# Better-scaled init of the same topology: pre-activations sit where σ' is not tiny.
GOOD_W1 = np.array([[0.80, 0.50]], dtype=float)
GOOD_B1 = np.array([[0.00]], dtype=float)
GOOD_W = np.array([1.00, 0.90, 1.00], dtype=float)
GOOD_B = np.array([0.00, 0.00, 0.00], dtype=float)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Elementwise logistic, the same σ drawn on screen."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_prime(values: np.ndarray) -> np.ndarray:
    """σ'(z) = σ(z) (1 − σ(z)), the local factor in every backward needle."""
    activated = sigmoid(values)
    return activated * (1.0 - activated)


def network_at(mix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward and backward pass at a convex combination of the two inits.

    Incoming sensitivity is ∂ℓ/∂h₄ = 1. Nothing is trained; mix only blends
    the two frozen parameter sets so the optional linear-region beat stays
    on the same numpy chain.
    """
    mix = float(mix)
    weight_1 = (1.0 - mix) * SAT_W1 + mix * GOOD_W1
    bias_1 = (1.0 - mix) * SAT_B1 + mix * GOOD_B1
    weights = (1.0 - mix) * SAT_W + mix * GOOD_W
    biases = (1.0 - mix) * SAT_B + mix * GOOD_B

    preactivations = []
    activations = []
    hidden = None
    for layer_index in range(4):
        if layer_index == 0:
            preactivation = weight_1 @ INPUT_X + bias_1
        else:
            preactivation = np.array(
                [[weights[layer_index - 1] * float(hidden.item()) + biases[layer_index - 1]]],
                dtype=float,
            )
        hidden = sigmoid(preactivation)
        preactivations.append(preactivation)
        activations.append(hidden)

    # Unit incoming gradient at the last hidden unit: isolate the chain rule.
    grad_h = [np.array([[0.0]], dtype=float) for _ in range(4)]
    grad_h[3] = np.array([[1.0]], dtype=float)
    for layer_index in reversed(range(4)):
        local = sigmoid_prime(preactivations[layer_index])
        grad_z = grad_h[layer_index] * local
        if layer_index == 0:
            grad_x = weight_1.T @ grad_z
        else:
            grad_h[layer_index - 1] = grad_z * weights[layer_index - 1]

    zs = np.array([float(item.item()) for item in preactivations], dtype=float)
    hs = np.array([float(item.item()) for item in activations], dtype=float)
    grad_hs = np.array([abs(float(item.item())) for item in grad_h], dtype=float)
    return zs, hs, grad_hs, np.abs(grad_x.reshape(-1))


SAT_Z, SAT_H, SAT_GRAD_H, SAT_GRAD_X = network_at(0.0)
GOOD_Z, GOOD_H, GOOD_GRAD_H, GOOD_GRAD_X = network_at(1.0)


class Episode048(Scene):
    """A ~32-second silent D2L 4.8 visualization: saturate, then vanish."""

    x_min, x_max = -2.8, 2.8
    y_min, y_max = -2.5, 2.5
    # Dedicated left pocket for the traveler coordinates, outside the cloud.
    plot_left, plot_right = -5.15, -0.72
    plot_bottom, plot_top = -3.00, 2.58
    activation_baseline_y = 0.38
    activation_scale = 2.02
    gradient_baseline_y = -3.02
    gradient_scale = 1.68
    bar_x = (0.92, 2.62, 4.32, 6.02)

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line grid; never a filled VMobject mesh."""
        grid = VGroup()
        for x_value in np.arange(-2.5, 2.51, 0.5):
            if abs(x_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-2.0, 2.01, 0.5):
            if abs(y_value) > 1e-8:
                grid.add(
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
        x_symbol = Text("x₁", font_size=26, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        y_symbol = Text("x₂", font_size=26, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        print(
            "D2L 4.8 SAT z={}, h={}, h_range=[{:.4f},{:.4f}], "
            "|dℓ/dh|={}, |dℓ/dx|={}".format(
                np.array2string(SAT_Z, precision=4, separator=", "),
                np.array2string(SAT_H, precision=4, separator=", "),
                float(SAT_H.min()),
                float(SAT_H.max()),
                np.array2string(SAT_GRAD_H, precision=6, separator=", "),
                np.array2string(SAT_GRAD_X, precision=6, separator=", "),
            )
        )
        print(
            "D2L 4.8 GOOD z={}, h={}, h_range=[{:.4f},{:.4f}], "
            "|dℓ/dh|={}, |dℓ/dx|={}".format(
                np.array2string(GOOD_Z, precision=4, separator=", "),
                np.array2string(GOOD_H, precision=4, separator=", "),
                float(GOOD_H.min()),
                float(GOOD_H.max()),
                np.array2string(GOOD_GRAD_H, precision=6, separator=", "),
                np.array2string(GOOD_GRAD_X, precision=6, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "04.8")

        grid, axes = self.make_grid_and_axes()
        class_points = (
            (BLUE, [(1.00, 0.50), (0.72, 0.82), (1.35, 0.84), (1.52, 0.24), (0.62, 0.20)]),
            (YELLOW, [(-1.72, 1.04), (-1.38, 0.62), (-1.08, 1.34), (-1.95, 0.42), (-0.88, 0.82)]),
            (BLUE_D, [(-0.30, -1.52), (0.10, -1.78), (-0.70, -1.25), (0.42, -1.22), (-0.10, -2.02)]),
        )
        point_cloud = VGroup(
            *[
                Dot(self.plot_point(x_value, y_value), radius=0.070, color=color)
                .set_fill(color, opacity=0.94)
                .set_stroke(color, width=0.0, opacity=0.0)
                for color, points in class_points
                for x_value, y_value in points
            ]
        )

        traveler_center = self.plot_point(INPUT_X[0, 0], INPUT_X[1, 0])
        halo_outer = Circle(radius=0.278).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_inner = Circle(radius=0.178).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(
            np.array([-6.28, -1.68, 0.0])
        )
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(
            np.array([-6.28, -2.10, 0.0])
        )

        sigmoid_formula = Text("h = σ(z)", font_size=34, color=WHITE).move_to(np.array([3.48, 3.28, 0.0]))
        derivative_formula = Text("σ′(z) = σ(z)(1 − σ(z))", font_size=30, color=WHITE).move_to(
            np.array([3.48, 3.28, 0.0])
        )

        mix = ValueTracker(0.0)
        activation_reveal = [ValueTracker(0.0) for _ in range(4)]
        gradient_reveal = [ValueTracker(0.0) for _ in range(4)]

        def current_hidden() -> np.ndarray:
            _, hidden, _, _ = network_at(mix.get_value())
            return hidden

        def current_grads() -> np.ndarray:
            _, _, grads, _ = network_at(mix.get_value())
            return grads

        one_line = Line(
            np.array([0.48, self.activation_baseline_y + self.activation_scale, 0.0]),
            np.array([6.48, self.activation_baseline_y + self.activation_scale, 0.0]),
        ).set_stroke(BLUE_E, width=1.4, opacity=0.55)
        zero_line = Line(
            np.array([0.48, self.activation_baseline_y, 0.0]),
            np.array([6.48, self.activation_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        gradient_zero_line = Line(
            np.array([0.48, self.gradient_baseline_y, 0.0]),
            np.array([6.48, self.gradient_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)

        tracks = VGroup(
            *[
                Line(
                    np.array([x_position, self.activation_baseline_y, 0.0]),
                    np.array([x_position, self.activation_baseline_y + self.activation_scale, 0.0]),
                ).set_stroke(BLUE_E, width=1.2, opacity=0.40)
                for x_position in self.bar_x
            ]
        )
        one_tick = Text("1", font_size=20, color=BLUE_A).move_to(
            np.array([0.48, self.activation_baseline_y + self.activation_scale, 0.0])
            + np.array([-0.28, 0.0, 0.0])
        )
        zero_tick = Text("0", font_size=20, color=BLUE_A).move_to(
            np.array([0.48, self.activation_baseline_y, 0.0]) + np.array([-0.28, 0.0, 0.0])
        )
        gradient_zero_tick = Text("0", font_size=20, color=BLUE_A).move_to(
            np.array([0.48, self.gradient_baseline_y, 0.0]) + np.array([-0.28, -0.02, 0.0])
        )
        gradient_tracks = VGroup(
            *[
                Line(
                    np.array([x_position, self.gradient_baseline_y, 0.0]),
                    np.array(
                        [x_position, self.gradient_baseline_y + self.gradient_scale, 0.0]
                    ),
                ).set_stroke(BLUE_E, width=1.2, opacity=0.40)
                for x_position in self.bar_x
            ]
        )

        column_tags = VGroup(
            *[
                Text(f"h{'₁₂₃₄'[index]}", font_size=24, color=WHITE).move_to(
                    np.array([self.bar_x[index], -0.52, 0.0])
                )
                for index in range(4)
            ]
        )
        gradient_axis_name = Text("|∂ℓ/∂h|", font_size=24, color=YELLOW_A).move_to(
            np.array([3.48, -3.74, 0.0])
        )

        forward_path = Line(
            traveler_center + np.array([0.32, 0.0, 0.0]),
            np.array([0.48, self.activation_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=2.2, opacity=0.62)

        def activation_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            hidden = current_hidden()[index] * activation_reveal[index].get_value()
            start = np.array([self.bar_x[index], self.activation_baseline_y, 0.0])
            end = np.array(
                [self.bar_x[index], self.activation_baseline_y + hidden * self.activation_scale, 0.0]
            )
            return start, end

        def gradient_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            magnitude = current_grads()[index] * gradient_reveal[index].get_value()
            start = np.array([self.bar_x[index], self.gradient_baseline_y, 0.0])
            end = np.array(
                [self.bar_x[index], self.gradient_baseline_y + magnitude * self.gradient_scale, 0.0]
            )
            return start, end

        activation_needles = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*activation_points(index)).set_stroke(
                        BLUE, width=14.0, opacity=0.98
                    )
                )
                for index in range(4)
            ]
        )
        gradient_needles = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*gradient_points(index)).set_stroke(
                        YELLOW, width=12.0, opacity=0.98
                    )
                )
                for index in range(4)
            ]
        )

        def make_tip_number(color) -> DecimalNumber:
            number = DecimalNumber(
                0.0,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=color,
                font_size=21,
            )
            return number

        activation_labels = VGroup(*[make_tip_number(WHITE) for _ in range(4)])
        gradient_labels = VGroup(*[make_tip_number(YELLOW_A) for _ in range(4)])

        def activation_label_position(index: int) -> np.ndarray:
            _, end = activation_points(index)
            # Right-and-above pocket: off the needle and off the 1-line.
            return end + np.array([0.62, 0.22, 0.0])

        def gradient_label_position(index: int) -> np.ndarray:
            _, end = gradient_points(index)
            # Stubs park just above the zero axis; taller needles keep a tip pocket.
            pocket_y = self.gradient_baseline_y + 0.36
            return np.array([self.bar_x[index] + 0.64, max(end[1] + 0.16, pocket_y), 0.0])

        for index, label in enumerate(activation_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.set_value(current_hidden()[index]).move_to(
                    activation_label_position(index)
                )
            )
        for index, label in enumerate(gradient_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.set_value(current_grads()[index]).move_to(
                    gradient_label_position(index)
                )
            )

        # 0.40–3.50 s: cloud and traveler before any layer needle appears.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.025),
            run_time=0.80,
            rate_func=linear,
        )
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_x1_label),
            FadeIn(traveler_x2_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.60,
            rate_func=smooth,
        )
        self.wait(1.70)

        # 3.50–4.45 s: send the traveler into the sigmoid chain.
        self.play(
            Create(forward_path),
            FadeIn(sigmoid_formula),
            FadeIn(zero_line),
            FadeIn(one_line),
            FadeIn(tracks),
            FadeIn(one_tick),
            FadeIn(zero_tick),
            FadeIn(column_tags),
            run_time=0.95,
            rate_func=smooth,
        )

        # 4.45–11.77 s: four hidden activations pile against the 1-line.
        self.add(activation_needles)
        for index in range(4):
            self.play(
                activation_reveal[index].animate.set_value(1.0),
                FadeIn(activation_labels[index]),
                run_time=1.15,
                rate_func=smooth,
            )
            self.wait(0.38)
        self.wait(1.20)

        # Swap the one on-screen formula; then grow backward needles right → left.
        self.play(FadeOut(sigmoid_formula), run_time=0.28, rate_func=linear)
        self.play(FadeIn(derivative_formula), run_time=0.40, rate_func=smooth)
        self.play(
            FadeIn(gradient_zero_line),
            FadeIn(gradient_tracks),
            FadeIn(gradient_zero_tick),
            FadeIn(gradient_axis_name),
            run_time=0.45,
            rate_func=smooth,
        )

        self.add(gradient_needles)
        for index in reversed(range(4)):
            self.play(
                gradient_reveal[index].animate.set_value(1.0),
                FadeIn(gradient_labels[index]),
                run_time=1.18,
                rate_func=smooth,
            )
            self.wait(0.32)
        self.wait(1.80)

        # Optional readable beat: same chain, better-scaled init, linear region.
        self.play(mix.animate.set_value(1.0), run_time=4.40, rate_func=smooth)

        # Keep live updaters through the last encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.20, rate_func=linear)
