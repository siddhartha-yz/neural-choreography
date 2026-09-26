"""D2L 4.4 — capacity vs fit, from two real polynomial least-squares curves.

Every plotted number comes from the arrays below.  A quadratic generates the
noisy train cloud; the traveler is a held-out x with its true y.  The two
curves are ``numpy.polyfit`` of degree 1 and degree 6 — no training loop,
no closed-form overlay besides those two fits.
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
    VMobject,
    ValueTracker,
    WHITE,
    YELLOW,
    YELLOW_A,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth.  Hidden process: y = 1.40 − 0.45 x² + noise.
# The traveler is not in TRAIN_X; its y is the noiseless true value.
# ---------------------------------------------------------------------------
TRAIN_X = np.array([-2.25, -1.50, -0.80, -0.10, 0.55, 1.90, 2.35], dtype=float)
TRUE_INTERCEPT = 1.40
TRUE_QUAD = -0.45
NOISE = np.array([0.18, -0.22, 0.30, -0.20, 0.40, -0.40, 0.20], dtype=float)


def true_function(x_values: np.ndarray | float) -> np.ndarray | float:
    """Noiseless quadratic that generated the cloud and the traveler."""
    return TRUE_INTERCEPT + TRUE_QUAD * np.asarray(x_values, dtype=float) ** 2


TRAIN_Y = np.asarray(true_function(TRAIN_X), dtype=float) + NOISE
TRAVELER_X = 1.20
TRAVELER_Y = float(true_function(TRAVELER_X))

LOW_DEGREE = 1
HIGH_DEGREE = len(TRAIN_X) - 1
LOW_COEFFS = np.polyfit(TRAIN_X, TRAIN_Y, LOW_DEGREE)
HIGH_COEFFS = np.polyfit(TRAIN_X, TRAIN_Y, HIGH_DEGREE)


def mean_squared_error(coeffs: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> float:
    """MSE of a polynomial against the same points drawn on screen."""
    residuals = np.polyval(coeffs, x_values) - y_values
    return float(np.mean(residuals**2))


LOW_TRAIN_LOSS = mean_squared_error(LOW_COEFFS, TRAIN_X, TRAIN_Y)
HIGH_TRAIN_LOSS = mean_squared_error(HIGH_COEFFS, TRAIN_X, TRAIN_Y)
LOW_VAL_LOSS = mean_squared_error(LOW_COEFFS, np.array([TRAVELER_X]), np.array([TRAVELER_Y]))
HIGH_VAL_LOSS = mean_squared_error(HIGH_COEFFS, np.array([TRAVELER_X]), np.array([TRAVELER_Y]))
LOW_PREDICTION = float(np.polyval(LOW_COEFFS, TRAVELER_X))
HIGH_PREDICTION = float(np.polyval(HIGH_COEFFS, TRAVELER_X))
HIGH_RESIDUAL = HIGH_PREDICTION - TRAVELER_Y


def signed_glyphs(value: float, digits: int) -> str:
    """Match 3.1: a true minus glyph, never a hyphen-minus."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"


class Episode044(Scene):
    """A ~28-second continuous, silent D2L 4.4 visualization."""

    x_min, x_max = -3.0, 3.0
    y_min, y_max = -2.4, 2.6
    # Same left-shifted plot language as 3.1: labels live in empty pockets.
    plot_left, plot_right = -5.35, 2.05
    plot_bottom, plot_top = -3.35, 2.75

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def clipped_line_points(self, weight: float, bias: float) -> tuple[np.ndarray, np.ndarray]:
        """Intersections of y = weight·x + bias with the visible plot rectangle."""
        candidates: list[tuple[float, float]] = []
        for x_value in (self.x_min, self.x_max):
            y_value = weight * x_value + bias
            if self.y_min <= y_value <= self.y_max:
                candidates.append((x_value, y_value))
        if abs(weight) > 1e-8:
            for y_value in (self.y_min, self.y_max):
                x_value = (y_value - bias) / weight
                if self.x_min <= x_value <= self.x_max:
                    candidates.append((x_value, y_value))
        candidates.sort()
        first, last = candidates[0], candidates[-1]
        return self.plot_point(*first), self.plot_point(*last)

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line objects: never a filled VMobject mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-3.0, 3.01, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-2.0, 2.51, 0.5):
            if abs(y_value) > 1e-8:
                grid_lines.add(
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
        x_symbol = Text("x", font_size=27, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        y_symbol = Text("y", font_size=27, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12
        )
        return grid_lines, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def polynomial_curve(self, coeffs: np.ndarray, color, width: float, opacity: float) -> VMobject:
        """Visible polyline of a polynomial, clipped to the plot rectangle."""
        samples = np.linspace(self.x_min, self.x_max, 900)
        values = np.polyval(coeffs, samples)
        inside = (values >= self.y_min) & (values <= self.y_max)
        points: list[np.ndarray] = []
        segments: list[list[np.ndarray]] = []
        for x_value, y_value, is_inside in zip(samples, values, inside):
            if is_inside:
                points.append(self.plot_point(float(x_value), float(y_value)))
            elif points:
                segments.append(points)
                points = []
        if points:
            segments.append(points)

        longest = max(segments, key=len)
        curve = VMobject(fill_opacity=0.0, stroke_opacity=opacity)
        curve.set_points_as_corners(longest)
        curve.set_fill(BLACK, opacity=0.0)
        curve.set_stroke(color, width=width, opacity=opacity)
        return curve

    def construct(self) -> None:
        print(
            "D2L 4.4 underfit d={} L_train={:.6f} L_val={:.6f} yhat={:.6f}".format(
                LOW_DEGREE, LOW_TRAIN_LOSS, LOW_VAL_LOSS, LOW_PREDICTION
            )
        )
        print(
            "D2L 4.4 overfit d={} L_train={:.6e} L_val={:.6f} yhat={:.6f} residual={:+.6f}".format(
                HIGH_DEGREE, HIGH_TRAIN_LOSS, HIGH_VAL_LOSS, HIGH_PREDICTION, HIGH_RESIDUAL
            )
        )
        print(
            "D2L 4.4 traveler x={:.6f} y={:.6f} train_x={} train_y={}".format(
                TRAVELER_X,
                TRAVELER_Y,
                np.array2string(TRAIN_X, precision=3, separator=", "),
                np.array2string(TRAIN_Y, precision=3, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "04.4")

        grid, axes = self.make_grid_and_axes()
        palette = (YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW)
        data_dots = [
            Dot(self.plot_point(x_value, y_value), radius=0.068, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for x_value, y_value, color in zip(TRAIN_X, TRAIN_Y, palette)
        ]
        point_cloud = VGroup(*data_dots)

        traveler_center = self.plot_point(TRAVELER_X, TRAVELER_Y)
        traveler_dot = (
            Dot(traveler_center, radius=0.068, color=BLUE)
            .set_fill(BLUE, opacity=0.94)
            .set_stroke(BLUE, width=0.0, opacity=0.0)
        )
        halo_outer = Circle(radius=0.265).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.48)
        halo_inner = Circle(radius=0.172).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        sample_x_label = Text(f"x = {signed_glyphs(TRAVELER_X, 2)}", font_size=24, color=BLUE_A).move_to(
            np.array([-6.18, -1.18, 0.0])
        )
        sample_y_label = Text(f"y = {signed_glyphs(TRAVELER_Y, 3)}", font_size=24, color=BLUE_A).move_to(
            np.array([-6.18, -1.63, 0.0])
        )

        formula = Text("ŷ = Σ wᵢ xⁱ", font_size=35, color=WHITE).move_to(np.array([4.72, 3.08, 0.0]))

        low_weight, low_bias = float(LOW_COEFFS[0]), float(LOW_COEFFS[1])
        low_curve = Line(*self.clipped_line_points(low_weight, low_bias)).set_stroke(BLUE, width=5.0, opacity=0.96)
        high_curve = self.polynomial_curve(HIGH_COEFFS, YELLOW, width=5.0, opacity=0.96)

        d1_anchor = self.plot_point(-2.35, float(np.polyval(LOW_COEFFS, -2.35)))
        d1_label = Text("d = 1", font_size=24, color=BLUE).move_to(d1_anchor + np.array([0.28, 0.55, 0.0]))
        # Park d = 6 in the empty cell left of the left-hand valley, not on the stroke.
        d6_label = Text("d = 6", font_size=24, color=YELLOW).move_to(self.plot_point(-2.62, -1.72))

        train_tracker = ValueTracker(LOW_TRAIN_LOSS)
        val_tracker = ValueTracker(LOW_VAL_LOSS)

        def make_loss_row(label: str, y_position: float, tracker: ValueTracker) -> VGroup:
            prefix = Text(label, font_size=21, color=WHITE)
            number = DecimalNumber(
                tracker.get_value(),
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=WHITE,
                font_size=22,
            )
            number.add_updater(lambda mob, tracked=tracker: mob.set_value(tracked.get_value()))
            return VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.10).move_to(
                np.array([4.78, y_position, 0.0])
            )

        train_row = make_loss_row("L_tr =", 2.43, train_tracker)
        val_row = make_loss_row("L_val =", 2.04, val_tracker)
        loss_pocket = VGroup(train_row, val_row)

        prediction_dot = (
            Dot(self.plot_point(TRAVELER_X, HIGH_PREDICTION), radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        residual_segment = Line(
            self.plot_point(TRAVELER_X, TRAVELER_Y),
            self.plot_point(TRAVELER_X, HIGH_PREDICTION),
        ).set_stroke(YELLOW_A, width=3.6, opacity=0.98)

        prediction_label = Text(
            f"ŷ = {signed_glyphs(HIGH_PREDICTION, 3)}",
            font_size=24,
            color=YELLOW,
        ).move_to(self.plot_point(TRAVELER_X, HIGH_PREDICTION) + np.array([1.02, 0.36, 0.0]))
        residual_mid = (
            self.plot_point(TRAVELER_X, TRAVELER_Y) + self.plot_point(TRAVELER_X, HIGH_PREDICTION)
        ) / 2
        residual_label = Text("r", font_size=24, color=YELLOW_A).move_to(
            residual_mid + np.array([-0.38, 0.02, 0.0])
        )

        # 0.40–1.20 s: plot and the smaller train cloud.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.04),
            run_time=0.80,
            rate_func=linear,
        )

        # 1.20–1.80 s: held-out traveler, same halo language as 3.1 / 3.4.
        self.play(
            FadeIn(traveler_dot),
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(sample_x_label),
            FadeIn(sample_y_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.60,
            rate_func=smooth,
        )
        self.wait(2.20)

        # 4.00–7.40 s: the degree-1 fit misses the curved cloud.
        self.play(
            FadeIn(low_curve),
            FadeIn(formula),
            FadeIn(d1_label),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(2.20)

        # 7.40–10.80 s: two pocket numbers, matching this line.
        self.play(FadeIn(loss_pocket), run_time=1.20, rate_func=smooth)
        self.wait(2.20)

        # 10.80–15.40 s: the interpolating curve threads every train point.
        self.play(
            low_curve.animate.set_stroke(BLUE, width=5.0, opacity=0.34),
            d1_label.animate.set_opacity(0.40),
            run_time=0.40,
            rate_func=smooth,
        )
        self.play(Create(high_curve), FadeIn(d6_label), run_time=2.00, rate_func=smooth)
        # Pocket numbers switch to the interpolating curve before t = 15 s.
        self.play(
            train_tracker.animate.set_value(HIGH_TRAIN_LOSS),
            val_tracker.animate.set_value(HIGH_VAL_LOSS),
            run_time=1.60,
            rate_func=smooth,
        )
        self.wait(0.60)

        # 15.40–18.40 s: held-out residual to the wild curve stays through the end.
        self.play(
            FadeIn(residual_segment),
            FadeIn(prediction_dot),
            FadeIn(prediction_label),
            FadeIn(residual_label),
            run_time=1.40,
            rate_func=smooth,
        )

        # 16.80–28.00 s: keep the residual and both curves through the last frame.
        self.wait(2.60)
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.60, rate_func=linear)
