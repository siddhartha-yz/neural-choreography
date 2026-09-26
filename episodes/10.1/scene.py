"""D2L 10.1 — a query against keys; the weights light up the matching values.

Every heatmap cell, key glow, and ŷ comes from the same numpy arrays below.
Nothing is trained. The on-screen formula is only the weighted sum of values.
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
config.renderer = "cairo"


# ---------------------------------------------------------------------------
# One numerical source of truth. Queries and keys live in R²; values are
# scalars paired with those keys. α = softmax(Q Kᵀ) along each query row.
# The halo traveler is query row 1 (0-based), the series point (1.00, 0.50).
# ---------------------------------------------------------------------------
QUERIES = np.array(
    [
        [0.20, 1.20],
        [1.00, 0.50],
        [-1.00, -0.90],
    ],
    dtype=float,
)
KEYS = np.array(
    [
        [0.20, 2.00],
        [1.60, 0.20],
        [-1.50, -1.20],
    ],
    dtype=float,
)
VALUES = np.array([[0.50], [2.00], [1.20]], dtype=float)
TRAVELER = 1


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, the same α drawn on the heatmap."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


SCORES = QUERIES @ KEYS.T
WEIGHTS = softmax_rows(SCORES)
OUTPUTS = WEIGHTS @ VALUES
TRAVELER_WEIGHTS = WEIGHTS[TRAVELER]
TRAVELER_YHAT = float(OUTPUTS[TRAVELER, 0])
WINNER = int(np.argmax(TRAVELER_WEIGHTS))


class Episode101(Scene):
    """A ~32-second continuous, silent D2L 10.1 visualization."""

    x_min, x_max = -2.10, 2.20
    y_min, y_max = -1.80, 2.40
    plot_left, plot_right = -5.40, 0.35
    plot_bottom, plot_top = -3.15, 2.70
    cell_size = 0.72
    cell_step = 0.84
    heat_left = 2.28
    heat_top = 1.68
    needle_scale = 1.15
    yhat_scale = 1.55
    yhat_x = 5.42
    yhat_base = -2.40

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
        """Stroke-only Line mobjects; never a filled grid mesh."""
        grid = VGroup()
        for x_value in np.arange(-2.0, 2.01, 0.5):
            if abs(x_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-1.5, 2.01, 0.5):
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
        # Park axis names past the tips, off the key/value pockets.
        x_symbol = Text("x₁", font_size=26, color=BLUE_A).move_to(
            self.plot_point(self.x_max, 0.0) + np.array([0.40, -0.40, 0.0])
        )
        y_symbol = Text("x₂", font_size=26, color=BLUE_A).move_to(
            self.plot_point(0.0, self.y_max) + np.array([0.42, 0.24, 0.0])
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        print("D2L 10.1 Q={}".format(np.array2string(QUERIES, precision=3, separator=", ")))
        print("D2L 10.1 K={}".format(np.array2string(KEYS, precision=3, separator=", ")))
        print("D2L 10.1 V={}".format(np.array2string(VALUES.ravel(), precision=3, separator=", ")))
        print("D2L 10.1 scores={}".format(np.array2string(SCORES, precision=4, separator=", ")))
        print(
            "D2L 10.1 alpha={} row_sums={}".format(
                np.array2string(WEIGHTS, precision=6, separator=", "),
                np.array2string(WEIGHTS.sum(axis=1), precision=6, separator=", "),
            )
        )
        print(
            "D2L 10.1 yhat={} traveler_yhat={:.6f} traveler_alpha={}".format(
                np.array2string(OUTPUTS.ravel(), precision=6, separator=", "),
                TRAVELER_YHAT,
                np.array2string(TRAVELER_WEIGHTS, precision=6, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "10.1")

        grid, axes = self.make_grid_and_axes()
        palette = (YELLOW, BLUE, YELLOW)
        key_dots = [
            Dot(self.plot_point(key[0], key[1]), radius=0.068, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for key, color in zip(KEYS, palette)
        ]
        point_cloud = VGroup(*key_dots)

        # Value labels sit in empty pockets beside the keys, never on the fill.
        value_offsets = (
            np.array([-0.62, 0.38, 0.0]),
            np.array([0.08, -0.55, 0.0]),
            np.array([-0.22, -0.42, 0.0]),
        )
        value_labels = VGroup(
            *[
                Text(
                    f"v{['₁', '₂', '₃'][index]} = {VALUES[index, 0]:.2f}",
                    font_size=20,
                    color=WHITE,
                ).move_to(self.plot_point(key[0], key[1]) + offset)
                for index, (key, offset) in enumerate(zip(KEYS, value_offsets))
            ]
        )

        traveler_center = self.plot_point(QUERIES[TRAVELER, 0], QUERIES[TRAVELER, 1])
        traveler_dot = (
            Dot(traveler_center, radius=0.068, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        halo_outer = Circle(radius=0.278).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        halo_outer.set_fill(BLACK, opacity=0.0)
        halo_inner = Circle(radius=0.178).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        halo_inner.set_fill(BLACK, opacity=0.0)
        # Pocket holds this query's coordinates, not heatmap row indices.
        traveler_x1_label = Text("x₁ = 1.00", font_size=21, color=BLUE_A).move_to(
            np.array([-6.28, -0.72, 0.0])
        )
        traveler_x2_label = Text("x₂ = 0.50", font_size=21, color=BLUE_A).move_to(
            np.array([-6.28, -1.14, 0.0])
        )

        needle_reveal = ValueTracker(0.0)
        yhat_reveal = ValueTracker(0.0)

        def style_key(index: int, color) -> None:
            def update(mobject: Dot) -> None:
                alpha = float(TRAVELER_WEIGHTS[index]) * needle_reveal.get_value()
                glow = min(max((alpha - 0.02) / 0.20, 0.0), 1.0)
                mobject.set_fill(color, opacity=0.94)
                mobject.set_stroke(color, width=3.2 * glow, opacity=glow)
                # Radius stays 0.068; never scale the Dot.

            key_dots[index].add_updater(update)
            update(key_dots[index])

        for index, color in enumerate(palette):
            style_key(index, color)

        def make_needle(index: int):
            def draw() -> Line:
                alpha = float(TRAVELER_WEIGHTS[index]) * needle_reveal.get_value()
                start = self.plot_point(float(KEYS[index, 0]), float(KEYS[index, 1]))
                length = max(alpha * self.needle_scale, 0.001)
                end = start + np.array([length, 0.0, 0.0])
                opacity = 0.0 if alpha < 0.018 else min(0.30 + 1.8 * alpha, 0.98)
                return Line(start, end).set_stroke(YELLOW, width=4.0, opacity=opacity)

            return always_redraw(draw)

        needles = VGroup(*[make_needle(index) for index in range(len(KEYS))])

        def winner_caption() -> VGroup:
            alpha = float(TRAVELER_WEIGHTS[WINNER]) * needle_reveal.get_value()
            start = self.plot_point(float(KEYS[WINNER, 0]), float(KEYS[WINNER, 1]))
            title = Text("α", font_size=20, color=YELLOW_A)
            number = Text(f"= {alpha:.3f}", font_size=20, color=YELLOW)
            caption = VGroup(title, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
            # Above the matching key, in the gap north of the needle, never on q₃.
            caption.move_to(start + np.array([0.18, 0.50, 0.0]))
            caption.set_opacity(0.0 if alpha < 0.12 else 1.0)
            return caption

        alpha_label = always_redraw(winner_caption)

        formula = Text("ŷ = Σ αᵢ vᵢ", font_size=34, color=WHITE).move_to(np.array([3.90, 3.18, 0.0]))

        yhat_bar = always_redraw(
            lambda: Line(
                np.array([self.yhat_x, self.yhat_base, 0.0]),
                np.array(
                    [
                        self.yhat_x,
                        self.yhat_base + TRAVELER_YHAT * self.yhat_scale * yhat_reveal.get_value(),
                        0.0,
                    ]
                ),
            ).set_stroke(YELLOW, width=16.0, opacity=0.98 * max(yhat_reveal.get_value(), 0.001))
        )
        yhat_axis = Line(
            np.array([self.yhat_x - 0.38, self.yhat_base, 0.0]),
            np.array([self.yhat_x + 0.38, self.yhat_base, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        yhat_zero = Text("0", font_size=20, color=BLUE_A).next_to(
            yhat_axis.get_start(), np.array([-1.0, -1.0, 0.0]), buff=0.10
        )
        yhat_prefix = Text("ŷ = ", font_size=22, color=YELLOW)
        yhat_number = DecimalNumber(
            TRAVELER_YHAT,
            num_decimal_places=3,
            mob_class=Text,
            include_sign=False,
            color=YELLOW_A,
            font_size=22,
        )
        yhat_label = VGroup(yhat_prefix, yhat_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
        yhat_label.move_to(
            np.array(
                [
                    self.yhat_x + 0.08,
                    self.yhat_base + TRAVELER_YHAT * self.yhat_scale + 0.38,
                    0.0,
                ]
            )
        )

        sum_label = Text("Σαᵢ = 1.000", font_size=22, color=WHITE).move_to(np.array([3.12, -1.48, 0.0]))

        heat_cells = []
        for row in range(3):
            for column in range(3):
                cell = Square(side_length=self.cell_size).move_to(self.cell_center(row, column))
                cell.set_stroke(BLUE_E, width=1.5, opacity=0.90)
                cell.set_fill(BLACK, opacity=0.0)
                heat_cells.append(cell)
        heat_group = VGroup(*heat_cells)
        row_tags = VGroup(
            *[
                Text(
                    f"q{['₁', '₂', '₃'][row]}",
                    font_size=20,
                    color=YELLOW if row == TRAVELER else WHITE,
                ).move_to(self.cell_center(row, 0) + np.array([-0.78, 0.0, 0.0]))
                for row in range(3)
            ]
        )
        col_tags = VGroup(
            *[
                Text(f"k{['₁', '₂', '₃'][column]}", font_size=20, color=WHITE).move_to(
                    self.cell_center(2, column) + np.array([0.0, -0.58, 0.0])
                )
                for column in range(3)
            ]
        )
        query_box = SurroundingRectangle(
            VGroup(*[heat_cells[TRAVELER * 3 + column] for column in range(3)]),
            color=YELLOW,
            buff=0.08,
            corner_radius=0.06,
        )
        query_box.set_fill(opacity=0.0)
        query_box.set_stroke(width=2.6, opacity=1.0)

        # 0.40–3.20 s: key–value pairs before any query.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.08),
            FadeIn(value_labels),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(1.90)

        # 3.20–5.40 s: halo the traveler query; pocket holds its two coordinates.
        self.play(
            FadeIn(traveler_dot),
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(traveler_x1_label),
            FadeIn(traveler_x2_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.65,
            rate_func=smooth,
        )
        self.wait(1.55)

        # 5.40–8.00 s: empty heatmap, rows = queries, columns = keys.
        self.play(
            FadeIn(heat_group),
            FadeIn(row_tags),
            FadeIn(col_tags),
            run_time=0.90,
            rate_func=smooth,
        )
        self.wait(1.70)

        # 8.00–10.20 s: this query row lights; the matching key column is brightest.
        traveler_fills = [
            heat_cells[TRAVELER * 3 + column].animate.set_fill(
                YELLOW, opacity=min(0.18 + 1.35 * float(TRAVELER_WEIGHTS[column]), 0.92)
            )
            for column in range(3)
        ]
        self.play(*traveler_fills, run_time=1.65, rate_func=smooth)
        self.play(Create(query_box), run_time=0.55, rate_func=smooth)

        # 10.20–13.00 s: the same weights light the matching values.
        self.add(needles, alpha_label)
        self.play(needle_reveal.animate.set_value(1.0), run_time=2.20, rate_func=smooth)
        self.wait(0.60)

        # 13.00–16.00 s: ŷ is the weighted sum of those values. Formula once.
        self.add(yhat_bar)
        self.play(FadeIn(formula), FadeIn(yhat_axis), FadeIn(yhat_zero), run_time=0.80, rate_func=smooth)
        self.play(yhat_reveal.animate.set_value(1.0), FadeIn(yhat_label), FadeIn(sum_label), run_time=1.50, rate_func=smooth)
        self.wait(0.70)

        # 16.00–20.50 s: the rest of the heatmap, still the same numpy α.
        other_fills = []
        for row in range(3):
            if row == TRAVELER:
                continue
            for column in range(3):
                other_fills.append(
                    heat_cells[row * 3 + column].animate.set_fill(
                        BLUE, opacity=min(0.12 + 1.20 * float(WEIGHTS[row, column]), 0.88)
                    )
                )
        self.play(*other_fills, run_time=2.50, rate_func=smooth)
        self.wait(2.00)

        # Hold live updaters and the heatmap through the final encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=11.50, rate_func=linear)
