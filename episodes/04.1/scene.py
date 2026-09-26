"""D2L 4.1 — one hidden layer, forward only, with a visible ReLU clip.

The arrays below are the only numerical source of truth for the traveler,
affine pre-activations, ReLU hidden units, and output. This is forward
computation only: no parameters are trained or updated.
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
    config,
    linear,
    smooth,
    always_redraw,
)


config.background_color = BLACK


# A real 2-D column vector, a 2 × 2 hidden map, and a 2 × 2 output map.
# Rows of WEIGHTS_1 are the two hidden units: Z = W1 x + b1, H = relu(Z),
# O = W2 H + b2. Unit 2 is deliberately negative before ReLU.
INPUT_X = np.array([[0.80], [0.60]], dtype=float)
WEIGHTS_1 = np.array(
    [
        [1.00, 0.50],
        [-0.50, -0.50],
    ],
    dtype=float,
)
BIAS_1 = np.array([[0.10], [-0.10]], dtype=float)
PREACTIVATIONS = WEIGHTS_1 @ INPUT_X + BIAS_1


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


HIDDEN = relu(PREACTIVATIONS)
WEIGHTS_2 = np.array(
    [
        [1.00, 0.50],
        [-0.75, 0.80],
    ],
    dtype=float,
)
BIAS_2 = np.array([[0.10], [0.20]], dtype=float)
OUTPUTS = WEIGHTS_2 @ HIDDEN + BIAS_2


class Episode041(Scene):
    """A ~29-second continuous, silent MLP forward-pass visualization."""

    x_min, x_max = -2.8, 2.8
    y_min, y_max = -2.5, 2.5
    # The left-side label pocket is deliberately separate from the cloud.
    plot_left, plot_right = -4.85, 0.30
    plot_bottom, plot_top = -3.20, 2.70
    hidden_baseline_y = 0.15
    output_baseline_y = -2.15
    needle_scale = 0.95
    bar_x = (2.45, 4.85)

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Use only stroked Line mobjects; never a filled grid mesh."""
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
            "D2L 4.1 x={}, W1={}, b1={}, z={}, h={}, W2={}, b2={}, o={}".format(
                np.array2string(INPUT_X.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHTS_1, precision=3, separator=", "),
                np.array2string(BIAS_1.ravel(), precision=3, separator=", "),
                np.array2string(PREACTIVATIONS.ravel(), precision=3, separator=", "),
                np.array2string(HIDDEN.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHTS_2, precision=3, separator=", "),
                np.array2string(BIAS_2.ravel(), precision=3, separator=", "),
                np.array2string(OUTPUTS.ravel(), precision=3, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "04.1")

        grid, axes = self.make_grid_and_axes()
        # Original groups only; no point color represents a prediction.
        class_points = (
            (
                BLUE,
                [
                    (0.80, 0.60),
                    (0.48, 0.98),
                    (1.22, 0.88),
                    (1.38, 0.28),
                    (0.42, 0.22),
                    (1.08, 1.18),
                    (0.28, 0.52),
                ],
            ),
            (
                YELLOW,
                [
                    (-1.48, -0.82),
                    (-1.12, -1.28),
                    (-1.88, -0.48),
                    (-0.78, -0.72),
                    (-1.62, -1.42),
                    (-2.08, -0.92),
                    (-0.68, -1.18),
                ],
            ),
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
        # These fixed-pocket coordinates stay entirely outside the plotted data.
        traveler_x1_label = Text("x₁ = 0.80", font_size=21, color=BLUE_A).move_to(
            np.array([-6.02, -1.66, 0.0])
        )
        traveler_x2_label = Text("x₂ = 0.60", font_size=21, color=BLUE_A).move_to(
            np.array([-6.02, -2.08, 0.0])
        )

        affine_formula = Text("z = W₁x + b₁", font_size=34, color=WHITE).move_to(
            np.array([3.55, 3.12, 0.0])
        )
        relu_formula = Text("h = relu(z)", font_size=34, color=WHITE).move_to(
            np.array([3.55, 3.12, 0.0])
        )
        output_formula = Text("o = W₂h + b₂", font_size=34, color=WHITE).move_to(
            np.array([3.55, 2.58, 0.0])
        )
        hidden_axis = Line(
            np.array([1.55, self.hidden_baseline_y, 0.0]),
            np.array([5.75, self.hidden_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        output_axis = Line(
            np.array([1.55, self.output_baseline_y, 0.0]),
            np.array([5.75, self.output_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        hidden_zero_label = Text("0", font_size=20, color=BLUE_A).next_to(
            hidden_axis.get_start(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        output_zero_label = Text("0", font_size=20, color=BLUE_A).next_to(
            output_axis.get_start(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )

        # The connector is the sole forward-motion cue: no second traveler dot.
        forward_path = Line(
            traveler_center + np.array([0.32, 0.0, 0.0]),
            np.array([1.55, self.hidden_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=2.2, opacity=0.62)

        z_reveal = [ValueTracker(0.0) for _ in range(2)]
        relu_mix = ValueTracker(0.0)
        o_reveal = [ValueTracker(0.0) for _ in range(2)]

        def hidden_value(index: int) -> float:
            preactivation = PREACTIVATIONS[index, 0]
            hidden = HIDDEN[index, 0]
            mix = relu_mix.get_value()
            return z_reveal[index].get_value() * ((1.0 - mix) * preactivation + mix * hidden)

        def hidden_bar_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            x_position = self.bar_x[index]
            start = np.array([x_position, self.hidden_baseline_y, 0.0])
            end = np.array(
                [
                    x_position,
                    self.hidden_baseline_y + hidden_value(index) * self.needle_scale,
                    0.0,
                ]
            )
            return start, end

        def output_bar_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            x_position = self.bar_x[index]
            start = np.array([x_position, self.output_baseline_y, 0.0])
            end = np.array(
                [
                    x_position,
                    self.output_baseline_y
                    + OUTPUTS[index, 0] * self.needle_scale * o_reveal[index].get_value(),
                    0.0,
                ]
            )
            return start, end

        # Hidden and output needles inherit the two original point-cloud colors.
        # Yellow remains a group color here, as in 3.4; the halo is still yellow.
        bar_colors = (BLUE, YELLOW)
        hidden_bars = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*hidden_bar_points(index)).set_stroke(
                        bar_colors[index], width=16.0, opacity=0.98
                    )
                )
                for index in range(2)
            ]
        )
        output_bars = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*output_bar_points(index)).set_stroke(
                        bar_colors[index], width=16.0, opacity=0.98
                    )
                )
                for index in range(2)
            ]
        )

        def make_bar_label(name: str, value: float, color, signed: bool) -> VGroup:
            prefix = Text(f"{name} = ", font_size=23, color=color)
            number = DecimalNumber(
                value,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=signed,
                color=WHITE,
                font_size=23,
            )
            return VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.07)

        z_labels = VGroup(
            *[
                make_bar_label(f"z{'₁₂'[index]}", PREACTIVATIONS[index, 0], bar_colors[index], True)
                for index in range(2)
            ]
        )
        h_labels = VGroup(
            *[
                make_bar_label(
                    f"h{'₁₂'[index]}",
                    HIDDEN[index, 0],
                    bar_colors[index],
                    HIDDEN[index, 0] < 0,
                )
                for index in range(2)
            ]
        )
        o_labels = VGroup(
            *[make_bar_label(f"o{'₁₂'[index]}", OUTPUTS[index, 0], bar_colors[index], True) for index in range(2)]
        )

        def hidden_label_position(index: int, value: float) -> np.ndarray:
            _, end = hidden_bar_points(index)
            if abs(value) <= 0.08:
                # Zero-length needle: sit above that column, not on the axis stroke.
                return end + np.array([0.0, 0.42, 0.0])
            direction = 1.0 if value >= 0 else -1.0
            return end + np.array([0.0, direction * 0.38, 0.0])

        def output_label_position(index: int) -> np.ndarray:
            _, end = output_bar_points(index)
            direction = 1.0 if OUTPUTS[index, 0] >= 0 else -1.0
            return end + np.array([0.0, direction * 0.38, 0.0])

        for index, label in enumerate(z_labels):
            number = label[1]
            number.add_updater(lambda mob, index=index: mob.set_value(hidden_value(index)))
            label.add_updater(
                lambda mobject, index=index: mobject.move_to(hidden_label_position(index, hidden_value(index)))
            )
        for index, label in enumerate(h_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.move_to(
                    hidden_label_position(index, HIDDEN[index, 0])
                )
            )
        for index, label in enumerate(o_labels):
            label.add_updater(lambda mobject, index=index: mobject.move_to(output_label_position(index)))

        # 0.40–3.50 s: establish the 2-D class cloud and traveler before forward motion.
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

        # 3.50–4.55 s: send the traveler through the first affine map.
        self.play(
            Create(forward_path),
            FadeIn(affine_formula),
            FadeIn(hidden_axis),
            FadeIn(hidden_zero_label),
            run_time=1.05,
            rate_func=smooth,
        )

        # 4.55–10.05 s: let each signed pre-activation needle stand on its own.
        self.add(hidden_bars)
        for index in range(2):
            self.play(
                z_reveal[index].animate.set_value(1.0),
                FadeIn(z_labels[index]),
                run_time=1.50,
                rate_func=smooth,
            )
            self.wait(0.40)
        self.wait(2.00)

        # Formula swap is sequential so the two titles never occupy the same slot.
        self.play(FadeOut(affine_formula), run_time=0.22, rate_func=linear)
        self.wait(0.10)
        self.play(FadeIn(relu_formula), run_time=0.48, rate_func=smooth)

        # The negative needle clips to the zero axis; that is ReLU.
        self.play(relu_mix.animate.set_value(1.0), run_time=2.50, rate_func=smooth)
        self.wait(0.25)
        self.play(FadeOut(z_labels), run_time=0.35, rate_func=smooth)
        self.play(FadeIn(h_labels), run_time=0.50, rate_func=smooth)
        self.wait(2.20)

        # Hidden units stay; a second affine map reads them as output needles.
        self.play(
            FadeIn(output_formula),
            FadeIn(output_axis),
            FadeIn(output_zero_label),
            run_time=0.90,
            rate_func=smooth,
        )
        self.add(output_bars)
        for index in range(2):
            self.play(
                o_reveal[index].animate.set_value(1.0),
                FadeIn(o_labels[index]),
                run_time=1.50,
                rate_func=smooth,
            )
            self.wait(0.45)
        self.wait(1.20)

        # Hold the cloud, clipped hidden needles, and outputs through the end.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.00, rate_func=linear)
