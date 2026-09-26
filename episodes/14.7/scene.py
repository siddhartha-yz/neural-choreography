"""D2L 14.7 — word analogy as a parallelogram in 2-D.

Every on-screen coordinate comes from the toy vectors below.
``composed = king − man + woman`` is numpy; embeddings are not trained.
The haloed traveler is that composed point, landing near queen.
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
    DashedVMobject,
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
)


config.background_color = BLACK


# Toy 2-D tokens. Identity colors, not classes. Untrained.
MAN = np.array([0.30, 0.35], dtype=float)
WOMAN = np.array([1.80, 0.48], dtype=float)
KING = np.array([0.52, 1.62], dtype=float)
QUEEN = np.array([1.84, 1.52], dtype=float)
COMPOSED = KING - MAN + WOMAN
RESIDUAL = COMPOSED - QUEEN

TOKENS = (MAN, WOMAN, KING, QUEEN)
TOKEN_NAMES = ("man", "woman", "king", "queen")
TOKEN_COLORS = (BLUE, BLUE_A, BLUE_D, WHITE)


def fmt_pair(value: float) -> str:
    return f"{value:.2f}"


class Episode147(Scene):
    """A ~32-second silent D2L 14.7 parallelogram visualization."""

    x_min, x_max = -0.20, 2.50
    y_min, y_max = -0.10, 2.10
    plot_left, plot_right = -4.20, 5.45
    plot_bottom, plot_top = -3.20, 2.48

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
        for x_value in np.arange(0.0, 2.51, 0.5):
            if abs(x_value) > 1e-8:
                grid.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(0.0, 2.11, 0.5):
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
            y_axis.get_end(), np.array([-1.0, 0.15, 0.0]), buff=0.10
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def stroke_edge(self, start: np.ndarray, end: np.ndarray, color, width: float = 3.2) -> Line:
        edge = Line(start, end).set_stroke(color, width=width, opacity=0.94)
        edge.set_fill(opacity=0.0)
        return edge

    def make_halo(self, center: np.ndarray) -> tuple[Circle, Circle]:
        """10.5 rings around a Dot. Never a fat scaled Dot."""
        outer = Circle(radius=0.278).move_to(center).set_stroke(BLUE_A, width=2.1, opacity=0.50)
        inner = Circle(radius=0.178).move_to(center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        outer.set_fill(opacity=0.0)
        inner.set_fill(opacity=0.0)
        return outer, inner

    def construct(self) -> None:
        assert COMPOSED.shape == (2,)
        nearest = min(np.linalg.norm(COMPOSED - token) for token in TOKENS)
        assert abs(nearest - float(np.linalg.norm(RESIDUAL))) < 1e-12
        print(
            "D2L 14.7 man={} woman={} king={} queen={}\n"
            "composed=king-man+woman={}\nresidual=composed-queen={} dist={:.6f}".format(
                np.array2string(MAN, precision=2, separator=", "),
                np.array2string(WOMAN, precision=2, separator=", "),
                np.array2string(KING, precision=2, separator=", "),
                np.array2string(QUEEN, precision=2, separator=", "),
                np.array2string(COMPOSED, precision=2, separator=", "),
                np.array2string(RESIDUAL, precision=2, separator=", "),
                float(np.linalg.norm(RESIDUAL)),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "14.7")

        grid, axes = self.make_grid_and_axes()
        centers = [self.plot_point(token[0], token[1]) for token in TOKENS]
        man_c, woman_c, king_c, queen_c = centers
        composed_c = self.plot_point(float(COMPOSED[0]), float(COMPOSED[1]))

        token_dots = [
            Dot(center, radius=0.070, color=color).set_fill(color, opacity=0.94).set_stroke(color, width=0.0, opacity=0.0)
            for center, color in zip(centers, TOKEN_COLORS)
        ]
        tag_offsets = (
            np.array([-0.58, -0.34, 0.0]),
            np.array([0.68, -0.34, 0.0]),
            np.array([-0.52, 0.36, 0.0]),
            np.array([0.18, -0.42, 0.0]),
        )
        token_tags = [
            Text(name, font_size=22, color=color).move_to(center + offset)
            for name, color, center, offset in zip(TOKEN_NAMES, TOKEN_COLORS, centers, tag_offsets)
        ]

        formula = Text("king − man + woman ≈ queen", font_size=30, color=WHITE).move_to(
            np.array([0.55, 3.20, 0.0])
        )

        edge_man_woman = self.stroke_edge(man_c, woman_c, BLUE_A, 3.0)
        edge_man_king = self.stroke_edge(man_c, king_c, BLUE_D, 3.0)
        edge_king_yhat = self.stroke_edge(king_c, composed_c, YELLOW, 3.4)
        edge_woman_yhat = self.stroke_edge(woman_c, composed_c, YELLOW, 3.4)

        composed_dot = (
            Dot(king_c, radius=0.078, color=YELLOW).set_fill(YELLOW, opacity=0.98).set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        yhat_tag = Text("ŷ", font_size=24, color=YELLOW).move_to(composed_c + np.array([0.40, 0.32, 0.0]))
        halo_outer, halo_inner = self.make_halo(composed_c)

        # One fixed empty pocket, left of the plane — traveler coords only, not a vector table.
        pocket_x1 = Text(f"ŷ₁ = {fmt_pair(float(COMPOSED[0]))}", font_size=21, color=BLUE_A).move_to(
            np.array([-6.18, -1.72, 0.0])
        )
        pocket_x2 = Text(f"ŷ₂ = {fmt_pair(float(COMPOSED[1]))}", font_size=21, color=BLUE_A).move_to(
            np.array([-6.18, -2.14, 0.0])
        )

        residual_line = Line(composed_c, queen_c).set_stroke(YELLOW_A, width=2.2, opacity=0.90)
        residual_line.set_fill(opacity=0.0)
        residual_dash = DashedVMobject(residual_line, num_dashes=8)

        # 0.40–2.50 s: stroke lattice first, so later points have a plane.
        self.play(FadeIn(grid), FadeIn(axes), run_time=0.90, rate_func=linear)
        self.wait(1.20)

        # 2.50–4.60 s: four word tokens. Colors are identity, not two classes.
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in token_dots], lag_ratio=0.12),
            LaggedStart(*[FadeIn(tag) for tag in token_tags], lag_ratio=0.12),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(1.20)

        # 4.60–6.70 s: formula once.
        self.play(FadeIn(formula), run_time=0.80, rate_func=smooth)
        self.wait(1.30)

        # 6.70–8.80 s: woman − man, the first side of the parallelogram.
        self.play(Create(edge_man_woman), run_time=0.90, rate_func=smooth)
        self.wait(1.20)

        # 8.80–10.90 s: king − man, the second side.
        self.play(Create(edge_man_king), run_time=0.90, rate_func=smooth)
        self.wait(1.20)

        # 10.90–13.20 s: ŷ copies that offset, traveling from king toward queen.
        self.play(FadeIn(composed_dot), run_time=0.30, rate_func=linear)
        self.play(
            composed_dot.animate.move_to(composed_c),
            Create(edge_king_yhat),
            run_time=1.50,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 13.20–15.40 s: halo on the composed landing; pocket holds its two coords.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(yhat_tag),
            FadeIn(pocket_x1),
            FadeIn(pocket_x2),
            Flash(composed_c, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(1.35)

        # 15.40–17.50 s: last side closes the parallelogram.
        self.play(Create(edge_woman_yhat), run_time=0.90, rate_func=smooth)
        self.wait(1.20)

        # 17.50–19.60 s: dashed residual — ŷ ≈ queen, not equal.
        self.play(Create(residual_dash), run_time=0.90, rate_func=smooth)
        self.wait(1.20)

        # Hold traveler, four words, parallelogram, residual, and pocket numbers.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.00, rate_func=linear)
