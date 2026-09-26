"""D2L 10.6 — one query attending over the same four tokens that are also keys and values.

The 4 × 2 array ``TOKENS`` is the only numerical source: Q = K = V = X,
scores are X Xᵀ / √d, rows of α sum to 1, and ŷ is α V. Nothing is trained.
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
    Square,
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


# Four tokens in R². Token 1 (0-based) is the halo traveler at the 3.4 coordinates.
# Projections are identity: the same vectors are q, k, and v.
TOKENS = np.array(
    [
        [0.20, 1.20],
        [1.00, 0.50],
        [1.45, 0.85],
        [-1.00, -0.90],
    ],
    dtype=float,
)
TRAVELER = 1
FEATURE_DIM = TOKENS.shape[1]
SCALE = 1.0 / np.sqrt(FEATURE_DIM)


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, the same α drawn on screen."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


SCORES = TOKENS @ TOKENS.T * SCALE
WEIGHTS = softmax_rows(SCORES)
OUTPUTS = WEIGHTS @ TOKENS
TRAVELER_WEIGHTS = WEIGHTS[TRAVELER]
TRAVELER_YHAT = OUTPUTS[TRAVELER]


class Episode106(Scene):
    """A ~32-second silent self-attention visualization."""

    x_min, x_max = -1.85, 2.05
    y_min, y_max = -1.55, 1.75
    plot_left, plot_right = -5.20, 0.10
    plot_bottom, plot_top = -3.10, 2.62
    cell_size = 0.68
    cell_step = 0.78
    heat_left = 1.62
    heat_top = 1.92

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def cell_center(self, row: int, column: int) -> np.ndarray:
        return np.array(
            [
                self.heat_left + column * self.cell_step,
                self.heat_top - row * self.cell_step,
                0.0,
            ]
        )

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
        x_symbol = Text("x₁", font_size=26, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        y_symbol = Text("x₂", font_size=26, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        print(
            "D2L 10.6 X={}".format(np.array2string(TOKENS, precision=3, separator=", "))
        )
        print(
            "D2L 10.6 scores={}".format(np.array2string(SCORES, precision=4, separator=", "))
        )
        print(
            "D2L 10.6 alpha={} row_sums={}".format(
                np.array2string(WEIGHTS, precision=6, separator=", "),
                np.array2string(WEIGHTS.sum(axis=1), precision=6, separator=", "),
            )
        )
        print(
            "D2L 10.6 Y={} yhat_traveler={} d={} scale={:.6f}".format(
                np.array2string(OUTPUTS, precision=4, separator=", "),
                np.array2string(TRAVELER_YHAT, precision=4, separator=", "),
                FEATURE_DIM,
                SCALE,
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "10.6")

        grid, axes = self.make_grid_and_axes()
        palette = (YELLOW, BLUE, YELLOW, BLUE_D)
        token_dots = [
            Dot(self.plot_point(token[0], token[1]), radius=0.070, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for token, color in zip(TOKENS, palette)
        ]
        point_cloud = VGroup(*token_dots)

        # Tiny index tags sit beside the dots, off the fill.
        tag_offsets = (
            np.array([0.36, 0.30, 0.0]),
            np.array([0.58, -0.48, 0.0]),
            np.array([0.40, 0.30, 0.0]),
            np.array([-0.40, -0.32, 0.0]),
        )
        token_tags = VGroup(
            *[
                Text(f"{index + 1}", font_size=22, color=WHITE).move_to(
                    self.plot_point(token[0], token[1]) + offset
                )
                for index, (token, offset) in enumerate(zip(TOKENS, tag_offsets))
            ]
        )

        traveler_center = self.plot_point(TOKENS[TRAVELER, 0], TOKENS[TRAVELER, 1])
        halo_outer = Circle(radius=0.278).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_inner = Circle(radius=0.178).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(
            np.array([-6.28, -1.70, 0.0])
        )
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(
            np.array([-6.28, -2.12, 0.0])
        )

        formula = Text("α = softmax(q kᵀ / √d)", font_size=32, color=WHITE).move_to(
            np.array([3.55, 3.24, 0.0])
        )

        ray_reveal = ValueTracker(0.0)
        yhat_reveal = ValueTracker(0.0)

        def ray_end(index: int) -> np.ndarray:
            destination = self.plot_point(TOKENS[index, 0], TOKENS[index, 1])
            direction = destination - traveler_center
            length = np.linalg.norm(direction)
            if length < 1e-6:
                return traveler_center
            return traveler_center + direction * ((length - 0.16) / length)

        def make_ray(index: int):
            def draw() -> Line:
                alpha = float(TRAVELER_WEIGHTS[index]) * ray_reveal.get_value()
                start = traveler_center
                end = ray_end(index)
                opacity = 0.0 if alpha < 0.012 else min(0.22 + 1.7 * alpha, 0.96)
                width = 2.0 + 8.0 * alpha
                return Line(start, end).set_stroke(YELLOW, width=width, opacity=opacity)

            return always_redraw(draw)

        other_rays = VGroup(*[make_ray(index) for index in range(4) if index != TRAVELER])
        self_ring = always_redraw(
            lambda: Circle(radius=0.305)
            .move_to(traveler_center)
            .set_stroke(
                YELLOW,
                width=2.8,
                opacity=min(0.18 + 1.8 * float(TRAVELER_WEIGHTS[TRAVELER]) * ray_reveal.get_value(), 0.90),
            )
            .set_fill(BLACK, opacity=0.0)
        )

        yhat_center = self.plot_point(float(TRAVELER_YHAT[0]), float(TRAVELER_YHAT[1]))
        yhat_dot = always_redraw(
            lambda: Dot(yhat_center, radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=0.98 * yhat_reveal.get_value())
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        yhat_locator = always_redraw(
            lambda: Line(traveler_center, yhat_center).set_stroke(
                YELLOW_A, width=2.4, opacity=0.90 * yhat_reveal.get_value()
            )
        )
        yhat_label = Text("ŷ", font_size=24, color=YELLOW).move_to(
            yhat_center + np.array([0.46, 0.30, 0.0])
        )
        sum_label = Text("Σαᵢ = 1.000", font_size=22, color=WHITE).move_to(
            np.array([-6.28, -2.54, 0.0])
        )

        # Heatmap: stroke cells, fill opacity = α. Not a number spreadsheet.
        heat_cells = []
        for row in range(4):
            for column in range(4):
                cell = Square(side_length=self.cell_size).move_to(self.cell_center(row, column))
                cell.set_stroke(BLUE_E, width=1.5, opacity=0.90)
                cell.set_fill(YELLOW if row == TRAVELER else BLUE, opacity=0.0)
                heat_cells.append(cell)
        heat_group = VGroup(*heat_cells)
        row_tags = VGroup(
            *[
                Text(f"q{['₁', '₂', '₃', '₄'][row]}", font_size=20, color=WHITE).move_to(
                    self.cell_center(row, 0) + np.array([-0.78, 0.0, 0.0])
                )
                for row in range(4)
            ]
        )
        col_tags = VGroup(
            *[
                Text(f"k{['₁', '₂', '₃', '₄'][column]}", font_size=20, color=WHITE).move_to(
                    self.cell_center(3, column) + np.array([0.0, -0.52, 0.0])
                )
                for column in range(4)
            ]
        )
        # Traveler-row numbers sit under the key tags, on the heatmap geometry.
        alpha_labels = VGroup()
        for column in range(4):
            number = DecimalNumber(
                float(TRAVELER_WEIGHTS[column]),
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=YELLOW_A,
                font_size=20,
            )
            number.move_to(self.cell_center(3, column) + np.array([0.0, -0.96, 0.0]))
            alpha_labels.add(number)
        query_box = SurroundingRectangle(
            VGroup(*[heat_cells[TRAVELER * 4 + column] for column in range(4)]),
            color=YELLOW,
            buff=0.08,
            corner_radius=0.06,
        )
        query_box.set_fill(opacity=0.0)
        query_box.set_stroke(width=2.6, opacity=1.0)

        # 0.40–3.20 s: four tokens before any attention ray.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.08),
            FadeIn(token_tags),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(1.90)

        # 3.20–5.35 s: halo the traveler; pocket holds its two coordinates.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_x1_label),
            FadeIn(traveler_x2_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.65,
            rate_func=smooth,
        )
        self.wait(1.50)

        # 5.35–6.25 s: one formula.
        self.play(FadeIn(formula), run_time=0.90, rate_func=smooth)

        # 6.25–7.15 s: empty 4 × 4 so later fills have a place to sit.
        self.play(
            FadeIn(heat_group),
            FadeIn(row_tags),
            FadeIn(col_tags),
            run_time=0.90,
            rate_func=smooth,
        )

        # 7.15–12.20 s: this query attends to every key; the boxed row is the same α.
        self.add(other_rays, self_ring)
        traveler_fills = [
            heat_cells[TRAVELER * 4 + column].animate.set_fill(
                YELLOW, opacity=min(0.18 + 1.35 * float(TRAVELER_WEIGHTS[column]), 0.92)
            )
            for column in range(4)
        ]
        self.play(
            ray_reveal.animate.set_value(1.0),
            *traveler_fills,
            run_time=2.40,
            rate_func=smooth,
        )
        self.play(
            Create(query_box),
            FadeIn(alpha_labels),
            FadeIn(sum_label),
            run_time=0.75,
            rate_func=smooth,
        )
        self.wait(1.90)

        # 12.20–16.00 s: ŷ is the weighted mix of the four values.
        self.add(yhat_locator, yhat_dot)
        self.play(yhat_reveal.animate.set_value(1.0), FadeIn(yhat_label), run_time=1.30, rate_func=smooth)
        self.wait(2.50)

        # 16.00–21.30 s: remaining queries fill the grid. No extra numbers.
        other_fills = []
        for row in range(4):
            if row == TRAVELER:
                continue
            for column in range(4):
                other_fills.append(
                    heat_cells[row * 4 + column].animate.set_fill(
                        BLUE, opacity=min(0.12 + 1.20 * float(WEIGHTS[row, column]), 0.88)
                    )
                )
        self.play(*other_fills, run_time=2.30, rate_func=smooth)
        self.wait(2.00)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=9.80, rate_func=linear)
