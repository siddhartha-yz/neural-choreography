"""D2L 11.3 — gradient descent on a 2-D elliptical bowl.

Every moving value comes from the numpy path below. The traveler is the
parameter point w. Contours are stroke-only level sets of
f(w) = w₁² + 2 w₂². Updates are exactly w ← w − η ∇f.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    BLACK,
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
    VMobject,
    ValueTracker,
    WHITE,
    YELLOW,
    always_redraw,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# D2L 11.3.2 bowl: f(x) = x₁² + 2 x₂², start [-5, -2].
# Moderate η = 0.1 walks in. η = 0.4 crosses the valley in w₂.
W_START = np.array([-5.0, -2.0], dtype=float)
ETA_GOOD = 0.10
ETA_LARGE = 0.40
STEPS_GOOD = 8
STEPS_LARGE = 5
CONTOUR_LEVELS = (2.0, 6.0, 12.0, 22.0, 33.0)


def objective(weights: np.ndarray) -> float:
    """f(w) = w₁² + 2 w₂²."""
    return float(weights[0] ** 2 + 2.0 * weights[1] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (2 w₁, 4 w₂)."""
    return np.array([2.0 * weights[0], 4.0 * weights[1]], dtype=float)


def gradient_descent(start: np.ndarray, eta: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps+1) parameter points and the matching f values."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    path[0] = weights
    values[0] = objective(weights)
    for index in range(steps):
        weights = weights - eta * gradient(weights)
        path[index + 1] = weights
        values[index + 1] = objective(weights)
    return path, values


PATH_GOOD, F_GOOD = gradient_descent(W_START, ETA_GOOD, STEPS_GOOD)
PATH_LARGE, F_LARGE = gradient_descent(W_START, ETA_LARGE, STEPS_LARGE)


class Episode113(Scene):
    """A ~30-second silent 2-D gradient-descent visualization."""

    w1_min, w1_max = -6.05, 1.85
    w2_min, w2_max = -3.15, 2.25
    plot_left, plot_right = -3.45, 5.55
    plot_bottom, plot_top = -3.20, 2.80

    def plot_point(self, w1: float, w2: float) -> np.ndarray:
        scene_x = self.plot_left + (w1 - self.w1_min) / (self.w1_max - self.w1_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (w2 - self.w2_min) / (self.w2_max - self.w2_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def inside_plot(self, w1: float, w2: float) -> bool:
        return self.w1_min <= w1 <= self.w1_max and self.w2_min <= w2 <= self.w2_max

    def make_contour(self, level: float, opacity: float) -> VGroup:
        """Stroke-only level set, clipped to the plot rectangle. Never filled."""
        samples = 220
        theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
        weights_1 = np.sqrt(level) * np.cos(theta)
        weights_2 = np.sqrt(level / 2.0) * np.sin(theta)
        chunks: list[list[np.ndarray]] = []
        current: list[np.ndarray] = []
        for w1, w2 in zip(weights_1, weights_2):
            if self.inside_plot(float(w1), float(w2)):
                current.append(self.plot_point(float(w1), float(w2)))
            elif current:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        group = VGroup()
        for points in chunks:
            if len(points) < 2:
                continue
            stroke = VMobject()
            stroke.set_fill(opacity=0.0)
            stroke.set_stroke(BLUE_E, width=1.5, opacity=opacity)
            stroke.set_points_as_corners(points)
            group.add(stroke)
        return group

    def make_axes(self) -> VGroup:
        x_axis = Line(self.plot_point(self.w1_min, 0.0), self.plot_point(self.w1_max, 0.0)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        y_axis = Line(self.plot_point(0.0, self.w2_min), self.plot_point(0.0, self.w2_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        w1_symbol = Text("w₁", font_size=26, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-0.15, -1.0, 0.0]), buff=0.14
        )
        w2_symbol = Text("w₂", font_size=26, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -0.2, 0.0]), buff=0.12
        )
        plus_h = Line(self.plot_point(-0.12, 0.0), self.plot_point(0.12, 0.0)).set_stroke(
            WHITE, width=2.0, opacity=0.90
        )
        plus_v = Line(self.plot_point(0.0, -0.12), self.plot_point(0.0, 0.12)).set_stroke(
            WHITE, width=2.0, opacity=0.90
        )
        return VGroup(x_axis, y_axis, w1_symbol, w2_symbol, plus_h, plus_v)

    def construct(self) -> None:
        print("D2L 11.3  f(w)=w1**2 + 2 w2**2")
        print("start w =", W_START, "  f =", objective(W_START))
        print("η = {:.2f}  path:".format(ETA_GOOD))
        for index, (weights, value) in enumerate(zip(PATH_GOOD, F_GOOD)):
            print(
                "  k={:d}  w=[{:+.5f}, {:+.5f}]  f={:.4f}".format(
                    index, weights[0], weights[1], value
                )
            )
        print("η = {:.2f}  path:".format(ETA_LARGE))
        for index, (weights, value) in enumerate(zip(PATH_LARGE, F_LARGE)):
            print(
                "  k={:d}  w=[{:+.5f}, {:+.5f}]  f={:.4f}".format(
                    index, weights[0], weights[1], value
                )
            )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "11.3")

        opacities = (0.78, 0.62, 0.48, 0.36, 0.28)
        contours = VGroup(
            *[self.make_contour(level, opacity) for level, opacity in zip(CONTOUR_LEVELS, opacities)]
        )
        axes = self.make_axes()
        # Level labels sit on the right-hand side of each ellipse, inside the plot.
        contour_labels = VGroup()
        # Upper-left of each ellipse, where the arcs actually sit in the plot.
        for level in (6.0, 12.0, 33.0):
            angle = 2.70
            w1 = float(np.sqrt(level) * np.cos(angle))
            w2 = float(np.sqrt(level / 2.0) * np.sin(angle))
            if self.inside_plot(w1, w2):
                label = Text(f"{level:.0f}", font_size=18, color=BLUE_A).move_to(
                    self.plot_point(w1, w2) + np.array([-0.22, 0.16, 0.0])
                )
                contour_labels.add(label)

        self.play(
            LaggedStart(*[Create(stroke) for stroke in contours], lag_ratio=0.08),
            FadeIn(axes),
            run_time=1.00,
            rate_func=linear,
        )
        self.play(FadeIn(contour_labels), run_time=0.28, rate_func=linear)

        w1_tracker = ValueTracker(W_START[0])
        w2_tracker = ValueTracker(W_START[1])
        eta_tracker = ValueTracker(ETA_GOOD)

        def traveler_center() -> np.ndarray:
            return self.plot_point(w1_tracker.get_value(), w2_tracker.get_value())

        def draw_traveler() -> VGroup:
            center = traveler_center()
            dot = (
                Dot(center, radius=0.068, color=YELLOW)
                .set_fill(YELLOW, opacity=0.94)
                .set_stroke(YELLOW, width=0.0, opacity=0.0)
            )
            outer = Circle(radius=0.278).move_to(center).set_fill(opacity=0.0).set_stroke(
                BLUE_A, width=2.1, opacity=0.50
            )
            inner = Circle(radius=0.178).move_to(center).set_fill(opacity=0.0).set_stroke(
                YELLOW, width=2.8, opacity=0.98
            )
            return VGroup(dot, outer, inner)

        traveler = always_redraw(draw_traveler)
        w_glyph = Text("w", font_size=24, color=YELLOW)

        def w_glyph_position() -> np.ndarray:
            return traveler_center() + np.array([0.52, 0.38, 0.0])

        w_glyph.add_updater(lambda mob: mob.move_to(w_glyph_position()))

        formula = Text("w ← w − η∇f", font_size=34, color=WHITE).move_to(np.array([1.15, 3.22, 0.0]))

        def make_pocket_row(label: str, y_position: float, decimals: int) -> tuple[VGroup, DecimalNumber]:
            prefix = Text(label, font_size=22, color=WHITE)
            number = DecimalNumber(
                0.0,
                num_decimal_places=decimals,
                mob_class=Text,
                include_sign=False,
                color=YELLOW,
                font_size=22,
            )
            row = VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.10).move_to(
                np.array([-5.95, y_position, 0.0])
            )
            return row, number

        eta_row, eta_number = make_pocket_row("η =", 1.35, 2)
        f_row, f_number = make_pocket_row("f(w) =", 0.88, 2)
        eta_number.set_value(ETA_GOOD)
        f_number.set_value(objective(W_START))
        eta_number.add_updater(lambda mob: mob.set_value(eta_tracker.get_value()))
        f_number.add_updater(
            lambda mob: mob.set_value(objective(np.array([w1_tracker.get_value(), w2_tracker.get_value()])))
        )
        pocket = VGroup(eta_row, f_row)

        good_segments = [
            Line(self.plot_point(*PATH_GOOD[index]), self.plot_point(*PATH_GOOD[index + 1])).set_stroke(
                YELLOW, width=3.4, opacity=0.95
            )
            for index in range(STEPS_GOOD)
        ]
        large_segments = [
            Line(self.plot_point(*PATH_LARGE[index]), self.plot_point(*PATH_LARGE[index + 1])).set_stroke(
                YELLOW, width=3.4, opacity=0.95
            )
            for index in range(STEPS_LARGE)
        ]

        self.add(traveler, w_glyph)
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.42, line_length=0.11),
            FadeIn(formula),
            FadeIn(pocket),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(0.70)

        # Moderate η: each step is a real numpy update along −η∇f.
        for index, segment in enumerate(good_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_GOOD[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_GOOD[index + 1, 1])),
                run_time=0.92,
                rate_func=smooth,
            )

        self.wait(1.00)

        # Same start, larger η: the steep axis overshoots the valley.
        self.play(
            *[segment.animate.set_stroke(YELLOW, width=2.2, opacity=0.28) for segment in good_segments],
            run_time=0.40,
            rate_func=linear,
        )
        self.play(
            w1_tracker.animate.set_value(float(W_START[0])),
            w2_tracker.animate.set_value(float(W_START[1])),
            eta_tracker.animate.set_value(ETA_LARGE),
            run_time=0.75,
            rate_func=smooth,
        )
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.36, line_length=0.10),
            run_time=0.35,
            rate_func=smooth,
        )

        for index, segment in enumerate(large_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_LARGE[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_LARGE[index + 1, 1])),
                run_time=1.10,
                rate_func=smooth,
            )

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=11.80, rate_func=linear)
