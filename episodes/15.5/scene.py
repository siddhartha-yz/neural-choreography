"""D2L 15.5 — one premise token queries the hypothesis keys, then a 3-way score.

Premise rows ``A`` and hypothesis rows ``B`` are the only numerical source:
scores are A Bᵀ / √d, each α row sums to 1, β = α B, and the tiny untrained
inference is softmax of a 3-vector built from a·β and ||a−β||. Nothing is trained.
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


# Three premise tokens and three hypothesis tokens in R².
# Token a₂ (0-based index 1) is the halo traveler at the 3.4 coordinates.
PREMISE = np.array(
    [
        [0.20, 1.20],
        [1.00, 0.50],
        [-0.50, 0.90],
    ],
    dtype=float,
)
HYPOTHESIS = np.array(
    [
        [1.80, 0.20],
        [1.65, 1.15],
        [-1.10, -0.95],
    ],
    dtype=float,
)
TRAVELER = 1
FEATURE_DIM = PREMISE.shape[1]
SCALE = 1.0 / np.sqrt(FEATURE_DIM)


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, the same α drawn on screen."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the 3-way score."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


SCORES = PREMISE @ HYPOTHESIS.T * SCALE
WEIGHTS = softmax_rows(SCORES)
ALIGNED = WEIGHTS @ HYPOTHESIS
TRAVELER_WEIGHTS = WEIGHTS[TRAVELER]
TRAVELER_BETA = ALIGNED[TRAVELER]
TRAVELER_A = PREMISE[TRAVELER]
ALIGN_DOT = float(TRAVELER_A @ TRAVELER_BETA)
ALIGN_GAP = float(np.linalg.norm(TRAVELER_A - TRAVELER_BETA))
# Tiny untrained 3-way head: entailment ~ overlap, contradiction ~ anti-overlap,
# neutral ~ leftover gap. Coefficients stay in numpy; they are not dumped on screen.
LOGITS = np.array([0.70 * ALIGN_DOT, -0.35 * ALIGN_DOT, 0.80 * ALIGN_GAP], dtype=float)
PROBABILITIES = softmax(LOGITS)
MAX_CLASS = int(np.argmax(PROBABILITIES))
CLASS_NAMES = ("e", "c", "n")


class Episode155(Scene):
    """A ~31-second silent NLI-attention visualization."""

    x_min, x_max = -1.65, 2.35
    y_min, y_max = -1.45, 1.65
    plot_left, plot_right = -5.35, 0.35
    plot_bottom, plot_top = -3.20, 2.70
    probability_baseline_y = -2.30
    probability_scale = 4.40
    bar_x = (2.10, 4.10, 6.10)

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
        for x_value in np.arange(-1.5, 2.01, 0.5):
            if abs(x_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-1.5, 1.51, 0.5):
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
        # Axis names sit on the plane, like 10.5 — never in the e/c/n bar column.
        x_symbol = Text("x₁", font_size=26, color=BLUE_A).next_to(
            self.plot_point(1.20, 0.0), np.array([0.0, -1.0, 0.0]), buff=0.16
        )
        y_symbol = Text("x₂", font_size=26, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, 0.15, 0.0]), buff=0.10
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        print("D2L 15.5 A={}".format(np.array2string(PREMISE, precision=3, separator=", ")))
        print("D2L 15.5 B={}".format(np.array2string(HYPOTHESIS, precision=3, separator=", ")))
        print(
            "D2L 15.5 scores={}".format(np.array2string(SCORES, precision=4, separator=", "))
        )
        print(
            "D2L 15.5 alpha={} row_sums={}".format(
                np.array2string(WEIGHTS, precision=6, separator=", "),
                np.array2string(WEIGHTS.sum(axis=1), precision=6, separator=", "),
            )
        )
        print(
            "D2L 15.5 beta={}".format(np.array2string(ALIGNED, precision=4, separator=", "))
        )
        print(
            "D2L 15.5 traveler_alpha={} beta={} d={} scale={:.6f}".format(
                np.array2string(TRAVELER_WEIGHTS, precision=6, separator=", "),
                np.array2string(TRAVELER_BETA, precision=4, separator=", "),
                FEATURE_DIM,
                SCALE,
            )
        )
        print(
            "D2L 15.5 dot={:.6f} gap={:.6f} o={} p={} sum(p)={:.6f}".format(
                ALIGN_DOT,
                ALIGN_GAP,
                np.array2string(LOGITS, precision=4, separator=", "),
                np.array2string(PROBABILITIES, precision=6, separator=", "),
                float(np.sum(PROBABILITIES)),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "15.5")

        grid, axes = self.make_grid_and_axes()
        premise_dots = [
            Dot(self.plot_point(token[0], token[1]), radius=0.070, color=BLUE)
            .set_fill(BLUE, opacity=0.94)
            .set_stroke(BLUE, width=0.0, opacity=0.0)
            for token in PREMISE
        ]
        hypothesis_dots = [
            Dot(self.plot_point(token[0], token[1]), radius=0.070, color=YELLOW)
            .set_fill(YELLOW, opacity=0.94)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
            for token in HYPOTHESIS
        ]
        point_cloud = VGroup(*premise_dots, *hypothesis_dots)

        premise_tag_offsets = (
            np.array([-0.44, 0.34, 0.0]),
            np.array([-0.60, -0.50, 0.0]),
            np.array([-0.48, 0.32, 0.0]),
        )
        hypothesis_tag_offsets = (
            np.array([0.58, 0.20, 0.0]),
            np.array([0.48, 0.36, 0.0]),
            np.array([-0.50, -0.38, 0.0]),
        )
        premise_tags = VGroup(
            *[
                Text(f"a{['₁', '₂', '₃'][index]}", font_size=22, color=WHITE).move_to(
                    self.plot_point(token[0], token[1]) + offset
                )
                for index, (token, offset) in enumerate(zip(PREMISE, premise_tag_offsets))
            ]
        )
        hypothesis_tags = VGroup(
            *[
                Text(f"b{['₁', '₂', '₃'][index]}", font_size=22, color=WHITE).move_to(
                    self.plot_point(token[0], token[1]) + offset
                )
                for index, (token, offset) in enumerate(zip(HYPOTHESIS, hypothesis_tag_offsets))
            ]
        )

        traveler_center = self.plot_point(PREMISE[TRAVELER, 0], PREMISE[TRAVELER, 1])
        halo_outer = Circle(radius=0.278).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_inner = Circle(radius=0.178).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        # One empty pocket, top-left, away from b₃ in the lower-left of the plot.
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(
            np.array([-6.32, 2.85, 0.0])
        )
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(
            np.array([-6.32, 2.43, 0.0])
        )
        sum_label = Text("Σαᵢ = 1.000", font_size=22, color=WHITE).move_to(
            np.array([-6.32, 2.01, 0.0])
        )

        formula = Text("β = softmax(a Bᵀ / √d) B", font_size=30, color=WHITE).move_to(
            np.array([3.85, 3.24, 0.0])
        )

        ray_reveal = ValueTracker(0.0)
        beta_reveal = ValueTracker(0.0)

        def ray_end(index: int) -> np.ndarray:
            destination = self.plot_point(HYPOTHESIS[index, 0], HYPOTHESIS[index, 1])
            direction = destination - traveler_center
            length = np.linalg.norm(direction)
            if length < 1e-6:
                return traveler_center
            return traveler_center + direction * ((length - 0.16) / length)

        def make_ray(index: int):
            def draw() -> Line:
                alpha = float(TRAVELER_WEIGHTS[index]) * ray_reveal.get_value()
                opacity = 0.0 if alpha < 0.012 else min(0.22 + 1.7 * alpha, 0.96)
                width = 2.0 + 8.0 * alpha
                return Line(traveler_center, ray_end(index)).set_stroke(
                    YELLOW, width=width, opacity=opacity
                )

            return always_redraw(draw)

        rays = VGroup(*[make_ray(index) for index in range(3)])

        alpha_offsets = (
            np.array([1.18, 0.20, 0.0]),
            np.array([0.48, 0.74, 0.0]),
            np.array([-0.50, -0.78, 0.0]),
        )
        alpha_labels = VGroup()
        for index in range(3):
            number = DecimalNumber(
                float(TRAVELER_WEIGHTS[index]),
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=YELLOW_A,
                font_size=20,
            )
            number.move_to(self.plot_point(HYPOTHESIS[index, 0], HYPOTHESIS[index, 1]) + alpha_offsets[index])
            alpha_labels.add(number)

        beta_center = self.plot_point(float(TRAVELER_BETA[0]), float(TRAVELER_BETA[1]))
        beta_dot = always_redraw(
            lambda: Dot(beta_center, radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=0.98 * beta_reveal.get_value())
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        beta_locator = always_redraw(
            lambda: Line(traveler_center, beta_center).set_stroke(
                YELLOW_A, width=2.4, opacity=0.90 * beta_reveal.get_value()
            )
        )
        beta_label = Text("β", font_size=24, color=YELLOW).move_to(
            beta_center + np.array([0.62, 0.40, 0.0])
        )

        probability_axis = Line(
            np.array([1.15, self.probability_baseline_y, 0.0]),
            np.array([6.95, self.probability_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        probability_zero_label = Text("0", font_size=20, color=BLUE_A).move_to(
            np.array([0.88, self.probability_baseline_y - 0.28, 0.0])
        )

        probability_reveal = ValueTracker(0.0)
        bar_colors = (BLUE, YELLOW, BLUE_D)

        def bar_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            x_position = self.bar_x[index]
            start = np.array([x_position, self.probability_baseline_y, 0.0])
            end = np.array(
                [
                    x_position,
                    self.probability_baseline_y
                    + PROBABILITIES[index] * self.probability_scale * probability_reveal.get_value(),
                    0.0,
                ]
            )
            return start, end

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

        def make_bar_label(name: str, value: float, color) -> VGroup:
            prefix = Text(f"{name} = ", font_size=23, color=color)
            number = DecimalNumber(
                value,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=WHITE,
                font_size=23,
            )
            return VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.07)

        probability_labels = VGroup(
            *[
                make_bar_label(CLASS_NAMES[index], float(PROBABILITIES[index]), bar_colors[index])
                for index in range(3)
            ]
        )

        def probability_label_position(index: int) -> np.ndarray:
            _, end = bar_points(index)
            # The contradiction stub is too short to wear its label on the stroke.
            if PROBABILITIES[index] <= 0.10:
                return end + np.array([1.12, 0.06, 0.0])
            return end + np.array([0.0, 0.36, 0.0])

        for index, label in enumerate(probability_labels):
            label.add_updater(lambda mobject, index=index: mobject.move_to(probability_label_position(index)))

        probability_sum = Text("Σpᵢ = 1.000", font_size=24, color=WHITE).move_to(
            np.array([4.10, -3.05, 0.0])
        )

        # 0.40–3.30 s: two tiny sequences before any query ray.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.08),
            FadeIn(premise_tags),
            FadeIn(hypothesis_tags),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(2.00)

        # 3.30–5.50 s: halo the premise traveler; pocket holds its two coordinates.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_x1_label),
            FadeIn(traveler_x2_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.65,
            rate_func=smooth,
        )
        self.wait(1.55)

        # 5.50–7.60 s: one formula.
        self.play(FadeIn(formula), run_time=0.90, rate_func=smooth)
        self.wait(1.20)

        # 7.60–12.10 s: this premise token attends over the hypothesis keys.
        self.add(rays)
        self.play(ray_reveal.animate.set_value(1.0), run_time=2.40, rate_func=smooth)
        self.play(FadeIn(alpha_labels), FadeIn(sum_label), run_time=0.70, rate_func=smooth)
        self.wait(1.40)

        # 12.10–15.40 s: β is the aligned mix of the three hypothesis values.
        self.add(beta_locator, beta_dot)
        self.play(
            beta_reveal.animate.set_value(1.0),
            FadeIn(beta_label),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(2.10)

        # 15.40–21.00 s: tiny 3-way score. Alignment rays stay.
        self.play(
            FadeIn(probability_axis),
            FadeIn(probability_zero_label),
            run_time=0.70,
            rate_func=smooth,
        )
        self.add(bars)
        self.play(probability_reveal.animate.set_value(1.0), run_time=2.00, rate_func=smooth)
        self.play(FadeIn(probability_labels), run_time=0.45, rate_func=smooth)
        self.wait(0.40)
        self.play(FadeIn(probability_sum), run_time=0.40, rate_func=smooth)
        max_box = SurroundingRectangle(
            probability_labels[MAX_CLASS], color=YELLOW, buff=0.14, corner_radius=0.08
        )
        max_box.set_fill(opacity=0.0)
        max_box.set_stroke(width=2.6, opacity=1.0)
        self.play(
            Create(max_box),
            Flash(bar_points(MAX_CLASS)[1], color=YELLOW, flash_radius=0.34, line_length=0.09),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.10)

        # Hold the alignment (rays + β) and the 3-way score through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=10.20, rate_func=linear)
