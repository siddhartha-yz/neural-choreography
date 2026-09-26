"""D2L 3.4 — one 2-D sample moving from logits to softmax probabilities.

The arrays below are the only numerical source of truth for the traveler,
three-class affine map, logits, and softmax probabilities. This is forward
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


# A real 2-D column vector and a 3 × 2 linear map. The rows of WEIGHTS
# correspond to the three output logits, so LOGITS = WEIGHTS @ INPUT_X + BIAS.
INPUT_X = np.array([[1.00], [0.50]], dtype=float)
WEIGHTS = np.array(
    [
        [1.10, 0.40],
        [-0.80, 0.65],
        [0.25, 1.05],
    ],
    dtype=float,
)
BIAS = np.array([[0.36], [-0.295], [-0.075]], dtype=float)
LOGITS = WEIGHTS @ INPUT_X + BIAS


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the same values drawn on screen."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


PROBABILITIES = softmax(LOGITS)
MAX_CLASS = int(np.argmax(PROBABILITIES))


class Episode034(Scene):
    """A 25-second continuous, silent softmax forward-pass visualization."""

    x_min, x_max = -2.8, 2.8
    y_min, y_max = -2.5, 2.5
    # The left-side label pocket is deliberately separate from the cloud.
    plot_left, plot_right = -4.85, 0.30
    plot_bottom, plot_top = -3.20, 2.70
    logit_baseline_y = 0.05
    probability_baseline_y = -2.35
    logit_scale = 0.88
    probability_scale = 4.50
    bar_x = (1.45, 3.35, 5.25)

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
            "D2L 3.4 x={}, W={}, b={}, o={}, p={}, sum(p)={:.6f}".format(
                np.array2string(INPUT_X.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHTS, precision=3, separator=", "),
                np.array2string(BIAS.ravel(), precision=3, separator=", "),
                np.array2string(LOGITS.ravel(), precision=3, separator=", "),
                np.array2string(PROBABILITIES.ravel(), precision=6, separator=", "),
                float(np.sum(PROBABILITIES)),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "03.4")

        grid, axes = self.make_grid_and_axes()
        # Original groups only; no point color represents a prediction.
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
        # These fixed-pocket coordinates mirror the scatter axes while staying
        # entirely outside the plotted data rectangle.
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -1.66, 0.0]))
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -2.08, 0.0]))

        logit_formula = Text("o = Wx + b", font_size=34, color=WHITE).move_to(np.array([3.38, 3.10, 0.0]))
        softmax_formula = Text("p = softmax(o)", font_size=34, color=WHITE).move_to(np.array([3.38, 3.10, 0.0]))
        logit_axis = Line(np.array([0.82, self.logit_baseline_y, 0.0]), np.array([5.92, self.logit_baseline_y, 0.0])).set_stroke(
            BLUE_D, width=1.7, opacity=0.82
        )
        probability_axis = Line(
            np.array([0.82, self.probability_baseline_y, 0.0]),
            np.array([5.92, self.probability_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        logit_zero_label = Text("0", font_size=20, color=BLUE_A).next_to(
            logit_axis.get_start(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        probability_zero_label = Text("0", font_size=20, color=BLUE_A).next_to(
            probability_axis.get_start(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )

        # The connector is the sole forward-motion cue: no second traveler dot.
        forward_path = Line(traveler_center + np.array([0.32, 0.0, 0.0]), np.array([0.82, self.logit_baseline_y, 0.0])).set_stroke(
            BLUE_D, width=2.2, opacity=0.62
        )

        reveal = [ValueTracker(0.0) for _ in range(3)]
        logit_visibility = ValueTracker(1.0)
        bar_stage = ValueTracker(0.0)
        probability_reveal = ValueTracker(0.0)

        def bar_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            x_position = self.bar_x[index]
            logit_start = np.array([x_position, self.logit_baseline_y, 0.0])
            logit_end = np.array(
                [
                    x_position,
                    self.logit_baseline_y
                    + LOGITS[index, 0] * self.logit_scale * reveal[index].get_value() * logit_visibility.get_value(),
                    0.0,
                ]
            )
            probability_start = np.array([x_position, self.probability_baseline_y, 0.0])
            probability_end = np.array(
                [
                    x_position,
                    self.probability_baseline_y
                    + PROBABILITIES[index, 0] * self.probability_scale * probability_reveal.get_value(),
                    0.0,
                ]
            )
            if bar_stage.get_value() < 0.5:
                return logit_start, logit_end
            return probability_start, probability_end

        # Output bars inherit the three original point-cloud colors. Yellow
        # accents remain reserved for the traveler halo and winning outline.
        bar_colors = (BLUE, YELLOW, BLUE_D)
        bars = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*bar_points(index)).set_stroke(
                        bar_colors[index], width=16.0, opacity=0.98
                    )
                )
                for index in range(3)
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

        logit_labels = VGroup(
            *[make_bar_label(f"o{'₁₂₃'[index]}", LOGITS[index, 0], bar_colors[index], True) for index in range(3)]
        )
        probability_labels = VGroup(
            *[
                make_bar_label(f"p{'₁₂₃'[index]}", PROBABILITIES[index, 0], bar_colors[index], False)
                for index in range(3)
            ]
        )

        def logit_label_position(index: int) -> np.ndarray:
            _, end = bar_points(index)
            direction = 1.0 if LOGITS[index, 0] >= 0 else -1.0
            return end + np.array([0.0, direction * 0.36, 0.0])

        def probability_label_position(index: int) -> np.ndarray:
            _, end = bar_points(index)
            # The smallest probability is a short stub, so its label uses an
            # empty side pocket rather than sitting directly on the stroke.
            if PROBABILITIES[index, 0] <= 0.10:
                return end + np.array([1.10, 0.02, 0.0])
            return end + np.array([0.0, 0.34, 0.0])

        for index, label in enumerate(logit_labels):
            label.add_updater(lambda mobject, index=index: mobject.move_to(logit_label_position(index)))
        for index, label in enumerate(probability_labels):
            label.add_updater(lambda mobject, index=index: mobject.move_to(probability_label_position(index)))

        probability_sum = Text("Σpᵢ = 1.000", font_size=24, color=WHITE).move_to(np.array([3.35, -3.00, 0.0]))

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

        # 3.50–4.55 s: send the traveler through the affine map.
        self.play(
            Create(forward_path),
            FadeIn(logit_formula),
            FadeIn(logit_axis),
            FadeIn(logit_zero_label),
            run_time=0.85,
            rate_func=smooth,
        )

        # 4.35–9.85 s: let each signed logit needle stand on its own.
        self.add(bars)
        for index in range(3):
            self.play(
                reveal[index].animate.set_value(1.0),
                FadeIn(logit_labels[index]),
                run_time=1.10,
                rate_func=smooth,
            )
            self.wait(0.40)
        self.wait(1.80)

        # Remove o labels before the bars start their softmax transition.
        self.play(FadeOut(logit_formula), run_time=0.22, rate_func=linear)
        self.wait(0.10)
        self.play(FadeOut(logit_labels), run_time=0.28, rate_func=smooth)
        # Signed needles retract to their own zero axis before the probability
        # baseline appears; the negative logit is never left floating.
        self.play(logit_visibility.animate.set_value(0.0), run_time=0.55, rate_func=smooth)
        self.play(FadeOut(logit_axis), FadeOut(logit_zero_label), run_time=0.18, rate_func=linear)
        self.play(
            FadeIn(softmax_formula),
            FadeIn(probability_axis),
            FadeIn(probability_zero_label),
            run_time=0.35,
            rate_func=smooth,
        )
        # Both old needles now have zero length, so changing the bar stage is
        # visually continuous; probabilities grow from their own zero axis.
        bar_stage.set_value(1.0)
        self.play(probability_reveal.animate.set_value(1.0), run_time=2.00, rate_func=smooth)
        self.play(FadeIn(probability_labels), run_time=0.45, rate_func=smooth)
        self.wait(1.35)
        self.play(FadeIn(probability_sum), run_time=0.40, rate_func=smooth)
        self.wait(1.35)

        max_box = SurroundingRectangle(probability_labels[MAX_CLASS], color=YELLOW, buff=0.14, corner_radius=0.08)
        max_box.set_fill(opacity=0.0)
        max_box.set_stroke(width=2.6, opacity=1.0)
        self.play(
            Create(max_box),
            Flash(bar_points(MAX_CLASS)[1], color=YELLOW, flash_radius=0.34, line_length=0.09),
            run_time=0.60,
            rate_func=smooth,
        )

        # Hold all classes, all bars, and the selected maximum through the end.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.00, rate_func=linear)
