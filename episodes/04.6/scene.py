"""D2L 4.6 — dropout on the traveler’s hidden units.

Every moving value comes from the arrays below. Hidden activations are a real
ReLU MLP forward pass; masks are real Bernoulli draws. Weights are never
trained. Training uses classic dropout (mask zeros some hᵢ); inference keeps
every unit and scales by the keep probability 1−p.
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


# Same traveler as 3.4, now read through a 2 → 4 ReLU hidden layer.
INPUT_X = np.array([[1.00], [0.50]], dtype=float)
WEIGHTS = np.array(
    [
        [1.20, 0.80],
        [0.40, 0.20],
        [0.80, 0.40],
        [0.50, 0.60],
    ],
    dtype=float,
)
BIAS = np.array([[0.20], [0.10], [0.20], [0.10]], dtype=float)
PRE_ACTIVATION = WEIGHTS @ INPUT_X + BIAS
HIDDEN = np.maximum(PRE_ACTIVATION, 0.0)

# D2L’s p is the drop probability. Classic dropout: train with a 0/1 mask,
# infer with a deterministic keep-probability scale.
DROP_P = 0.50
KEEP_P = 1.0 - DROP_P
MASK_SEED = 47
N_MASKS = 4


def sample_masks() -> np.ndarray:
    """Four distinct Bernoulli masks from one seeded generator."""
    rng = np.random.default_rng(MASK_SEED)
    return np.stack([(rng.random(HIDDEN.size) > DROP_P).astype(float) for _ in range(N_MASKS)])


MASKS = sample_masks()
H_TRAIN = MASKS * HIDDEN.ravel()
H_INFER = KEEP_P * HIDDEN.ravel()


class Episode046(Scene):
    """A ~32-second continuous, silent dropout visualization."""

    x_min, x_max = -2.8, 2.8
    y_min, y_max = -2.5, 2.5
    # Left-side label pocket stays outside the plotted data rectangle.
    plot_left, plot_right = -4.85, 0.22
    plot_bottom, plot_top = -3.20, 2.70
    hidden_baseline_y = -0.22
    hidden_scale = 1.42
    bar_x = (1.52, 3.08, 4.64, 6.20)
    label_row_y = -0.92
    unit_dot_radius = 0.105

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
            "D2L 4.6 p={:.2f}, keep={:.2f}, x={}, W={}, b={}, h={}, mask={}, h_train={}, h_infer={}".format(
                DROP_P,
                KEEP_P,
                np.array2string(INPUT_X.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHTS, precision=3, separator=", "),
                np.array2string(BIAS.ravel(), precision=3, separator=", "),
                np.array2string(HIDDEN.ravel(), precision=3, separator=", "),
                np.array2string(MASKS, precision=0, separator=", "),
                np.array2string(H_TRAIN, precision=3, separator=", "),
                np.array2string(H_INFER, precision=3, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "04.6")

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
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -1.66, 0.0]))
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(np.array([-6.02, -2.08, 0.0]))

        hidden_axis = Line(
            np.array([0.78, self.hidden_baseline_y, 0.0]),
            np.array([6.88, self.hidden_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        hidden_zero_label = Text("0", font_size=20, color=BLUE_A).next_to(
            hidden_axis.get_start(), np.array([0.0, -1.0, 0.0]), buff=0.14
        )
        forward_path = Line(
            traveler_center + np.array([0.32, 0.0, 0.0]),
            np.array([0.78, self.hidden_baseline_y, 0.0]),
        ).set_stroke(BLUE_D, width=2.2, opacity=0.62)

        train_formula = Text("h′ = m*h", font_size=34, color=WHITE).move_to(np.array([3.12, 3.12, 0.0]))
        infer_formula = Text("h′ = (1−p)h", font_size=34, color=WHITE).move_to(np.array([3.12, 3.12, 0.0]))
        # p is written once and stays; it is the drop probability from D2L 4.6.
        drop_p_label = Text("p = 0.50", font_size=28, color=YELLOW_A).move_to(np.array([6.08, 3.12, 0.0]))

        unit_colors = (BLUE, YELLOW, BLUE_D, YELLOW_A)
        display = [ValueTracker(0.0) for _ in range(4)]

        def bar_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            x_position = self.bar_x[index]
            start = np.array([x_position, self.hidden_baseline_y, 0.0])
            height = display[index].get_value() * self.hidden_scale
            end = np.array([x_position, self.hidden_baseline_y + height, 0.0])
            return start, end

        bars = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*bar_points(index)).set_stroke(
                        unit_colors[index], width=16.0, opacity=0.98
                    )
                )
                for index in range(4)
            ]
        )

        def make_unit_dot(index: int):
            def draw_dot() -> Dot:
                active = display[index].get_value() > 1e-6
                return (
                    Dot(
                        np.array([self.bar_x[index], self.hidden_baseline_y, 0.0]),
                        radius=self.unit_dot_radius,
                        color=unit_colors[index],
                    )
                    .set_fill(unit_colors[index], opacity=0.94 if active else 0.0)
                    .set_stroke(unit_colors[index], width=2.2, opacity=0.96 if active else 0.40)
                )

            return always_redraw(draw_dot)

        unit_dots = VGroup(*[make_unit_dot(index) for index in range(4)])

        def make_unit_label(index: int) -> tuple[VGroup, DecimalNumber]:
            prefix = Text(f"h{'₁₂₃₄'[index]} = ", font_size=20, color=unit_colors[index])
            number = DecimalNumber(
                0.0,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=WHITE,
                font_size=20,
            )
            row = VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06)
            row.move_to(np.array([self.bar_x[index], self.label_row_y, 0.0]))
            number.add_updater(lambda mob, index=index: mob.set_value(display[index].get_value()))
            return row, number

        unit_labels = VGroup(*[make_unit_label(index)[0] for index in range(4)])

        # 0.40–3.50 s: establish the cloud and traveler before the hidden layer.
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

        # 3.50–6.00 s: the traveler’s four hidden units at full h.
        self.play(
            Create(forward_path),
            FadeIn(hidden_axis),
            FadeIn(hidden_zero_label),
            run_time=0.85,
            rate_func=smooth,
        )
        self.add(bars, unit_dots)
        self.play(
            *[display[index].animate.set_value(float(HIDDEN[index, 0])) for index in range(4)],
            FadeIn(unit_labels),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(0.45)

        # 6.00–7.80 s: write p once, then the train-time mask formula.
        self.play(FadeIn(drop_p_label), run_time=0.40, rate_func=smooth)
        self.play(FadeIn(train_formula), run_time=0.45, rate_func=smooth)
        self.wait(0.50)

        # 7.80–20.20 s: four distinct real masks, slowly emptying the same units.
        for mask_values in H_TRAIN:
            self.play(
                *[display[index].animate.set_value(float(mask_values[index])) for index in range(4)],
                run_time=1.20,
                rate_func=smooth,
            )
            self.wait(1.90)

        # 20.20–23.70 s: infer — every unit on, values scaled by 1−p.
        # Formula leaves first so (1−p)h never crossfades into m*h.
        self.play(FadeOut(train_formula), run_time=0.22, rate_func=linear)
        self.wait(0.10)
        self.play(
            FadeIn(infer_formula),
            *[display[index].animate.set_value(float(H_INFER[index])) for index in range(4)],
            run_time=1.90,
            rate_func=smooth,
        )
        self.wait(1.08)

        # Hold the scaled hidden layer through the final encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.26, rate_func=linear)
