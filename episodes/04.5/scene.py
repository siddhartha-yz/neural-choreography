"""D2L 4.5 — L2 weight decay pulling a high-capacity polynomial smoother.

Every moving value comes from the ridge / penalized-polynomial fit below.
The haloed traveler is held out: it never enters the design matrix.
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
    always_redraw,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth: a degree-9 polynomial in the raw feature x,
# fit by ridge regression.  The intercept is not penalized (D2L 4.5).  The
# traveler (X_HOLD, Y_HOLD) is absent from FEATURES / TARGETS.
# ---------------------------------------------------------------------------
FEATURES = np.array(
    [-2.25, -1.80, -1.35, -0.90, -0.45, -0.05, 1.35, 1.75, 2.15, 2.45],
    dtype=float,
)
TRUE_COEFF_1 = 0.55
TRUE_COEFF_3 = -0.085
NOISE = np.array(
    [0.22, -0.18, 0.20, -0.16, 0.18, -0.19, 0.17, -0.21, 0.15, 0.18],
    dtype=float,
)
TARGETS = TRUE_COEFF_1 * FEATURES + TRUE_COEFF_3 * FEATURES**3 + NOISE

X_HOLD = 0.55
Y_HOLD = 0.290
POLYNOMIAL_DEGREE = 9
LAMBDA_FINAL = 0.08
DESIGN = np.vander(FEATURES, N=POLYNOMIAL_DEGREE + 1, increasing=True)


def ridge_weights(penalty: float) -> np.ndarray:
    """min ||Xw − y||² + λ||w_{1:}||², intercept unpenalized."""
    if penalty <= 1e-12:
        weights, *_ = np.linalg.lstsq(DESIGN, TARGETS, rcond=None)
        return weights
    regularizer = np.eye(POLYNOMIAL_DEGREE + 1)
    regularizer[0, 0] = 0.0
    gram = DESIGN.T @ DESIGN + penalty * regularizer
    return np.linalg.solve(gram, DESIGN.T @ TARGETS)


def polynomial_values(x_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vandermonde = np.vander(np.asarray(x_values, dtype=float).ravel(), N=len(weights), increasing=True)
    return vandermonde @ weights


def polynomial_value(x_value: float, weights: np.ndarray) -> float:
    return float(polynomial_values(np.array([x_value], dtype=float), weights)[0])


def weight_norm(weights: np.ndarray) -> float:
    """L2 norm of the penalized coefficients, matching the ridge term."""
    return float(np.linalg.norm(weights[1:]))


def held_out_residual(weights: np.ndarray) -> float:
    return polynomial_value(X_HOLD, weights) - Y_HOLD


def displayed_lambda(unit: float) -> float:
    """Map a 0–1 tracker onto λ so the wiggle dies across the move, not in a jump."""
    if unit <= 1e-8:
        return 0.0
    return LAMBDA_FINAL * (10.0 ** (1.7 * unit) - 1.0) / (10.0**1.7 - 1.0)


WEIGHTS_UNPENALIZED = ridge_weights(0.0)
WEIGHTS_DECAYED = ridge_weights(LAMBDA_FINAL)


class Episode045(Scene):
    """One continuous silent D2L 4.5 visualization, about 28 seconds."""

    x_min, x_max = -2.32, 2.52
    y_min, y_max = -2.40, 3.00
    # Leave a dedicated label pocket to the left of the point cloud.
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

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Create only stroke Line objects: never a filled VMobject mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-2.0, 2.51, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-2.0, 3.01, 0.5):
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

    def clipped_curve_points(self, weights: np.ndarray) -> list[np.ndarray]:
        """Sample the polynomial and keep only the visible plot rectangle."""
        x_samples = np.linspace(self.x_min, self.x_max, 360)
        y_samples = polynomial_values(x_samples, weights)
        points: list[np.ndarray] = []
        for x_value, y_value in zip(x_samples, y_samples):
            if self.y_min <= y_value <= self.y_max:
                points.append(self.plot_point(float(x_value), float(y_value)))
            elif points:
                clipped_y = float(np.clip(y_value, self.y_min, self.y_max))
                points.append(self.plot_point(float(x_value), clipped_y))
        return points

    def make_prediction_curve(self, weights: np.ndarray) -> VMobject:
        """Stroke-only polyline. Fill stays off; never call set_opacity on it."""
        path = VMobject()
        corners = self.clipped_curve_points(weights)
        if len(corners) >= 2:
            path.set_points_as_corners(corners)
        path.set_fill(BLACK, opacity=0.0)
        path.set_stroke(YELLOW, width=5.0, opacity=0.96)
        return path

    def construct(self) -> None:
        residual_unpenalized = held_out_residual(WEIGHTS_UNPENALIZED)
        residual_decayed = held_out_residual(WEIGHTS_DECAYED)
        print(
            "D2L 4.5 lambda=0 ||w||={:.6f} held-out yhat={:.6f} residual={:+.6f}; "
            "lambda={:.6f} ||w||={:.6f} held-out yhat={:.6f} residual={:+.6f}".format(
                weight_norm(WEIGHTS_UNPENALIZED),
                polynomial_value(X_HOLD, WEIGHTS_UNPENALIZED),
                residual_unpenalized,
                LAMBDA_FINAL,
                weight_norm(WEIGHTS_DECAYED),
                polynomial_value(X_HOLD, WEIGHTS_DECAYED),
                residual_decayed,
            )
        )

        # 0.00–0.40 s: a short card, with no overlap into the scatter reveal.
        show_episode_intro(self, "04.5")

        grid, axes = self.make_grid_and_axes()
        palette = (YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE)
        data_dots = [
            Dot(self.plot_point(feature, target), radius=0.068, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for feature, target, color in zip(FEATURES, TARGETS, palette)
        ]
        point_cloud = VGroup(*data_dots)

        lambda_unit = ValueTracker(0.0)

        def current_lambda() -> float:
            return displayed_lambda(lambda_unit.get_value())

        def current_weights() -> np.ndarray:
            return ridge_weights(current_lambda())

        def current_prediction() -> float:
            return polynomial_value(X_HOLD, current_weights())

        prediction_curve = always_redraw(lambda: self.make_prediction_curve(current_weights()))
        prediction_dot = always_redraw(
            lambda: Dot(
                self.plot_point(X_HOLD, current_prediction()),
                radius=0.078,
                color=YELLOW,
            ).set_fill(YELLOW, opacity=1.0)
        )
        residual_segment = always_redraw(
            lambda: Line(
                self.plot_point(X_HOLD, Y_HOLD),
                self.plot_point(X_HOLD, current_prediction()),
            ).set_stroke(YELLOW_A, width=3.6, opacity=0.98)
        )

        halo_center = self.plot_point(X_HOLD, Y_HOLD)
        halo_outer = Circle(radius=0.265).move_to(halo_center).set_stroke(BLUE_A, width=2.1, opacity=0.48)
        halo_inner = Circle(radius=0.172).move_to(halo_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        traveler_dot = (
            Dot(halo_center, radius=0.068, color=BLUE)
            .set_fill(BLUE, opacity=0.94)
            .set_stroke(BLUE, width=0.0, opacity=0.0)
        )
        # Fixed left-pocket labels cannot collide with the traveler, residual,
        # yellow curve, or a neighboring training point.
        sample_x_label = Text("x = 0.55", font_size=24, color=BLUE_A).move_to(np.array([-6.18, -1.18, 0.0]))
        sample_y_label = Text("y = 0.290", font_size=24, color=BLUE_A).move_to(np.array([-6.18, -1.63, 0.0]))

        formula = Text("λ‖w‖²", font_size=35, color=WHITE).move_to(np.array([4.72, 3.08, 0.0]))

        def make_number_row(label: str, y_position: float, color) -> tuple[VGroup, DecimalNumber]:
            prefix = Text(label, font_size=21, color=WHITE)
            number = DecimalNumber(
                0.0,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=False,
                color=color,
                font_size=22,
            )
            row = VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.10).move_to(
                np.array([4.78, y_position, 0.0])
            )
            return row, number

        lambda_row, lambda_number = make_number_row("λ =", 2.43, BLUE_A)
        norm_row, norm_number = make_number_row("‖w‖ =", 2.04, YELLOW_A)
        lambda_number.set_value(current_lambda())
        norm_number.set_value(weight_norm(current_weights()))
        lambda_number.add_updater(lambda number: number.set_value(current_lambda()))
        norm_number.add_updater(lambda number: number.set_value(weight_norm(current_weights())))
        secondary_readout = VGroup(lambda_row, norm_row)

        prediction_prefix = Text("ŷ = ", font_size=24, color=YELLOW)
        prediction_number = DecimalNumber(
            current_prediction(),
            num_decimal_places=3,
            mob_class=Text,
            include_sign=True,
            color=YELLOW_A,
            font_size=24,
        )
        prediction_label = VGroup(prediction_prefix, prediction_number).arrange(
            np.array([1.0, 0.0, 0.0]), buff=0.08
        )
        prediction_number.add_updater(lambda number: number.set_value(current_prediction()))

        def prediction_label_position() -> np.ndarray:
            """Keep ŷ off the stroke, off the halo, and off the residual."""
            prediction_position = self.plot_point(X_HOLD, current_prediction())
            target_position = self.plot_point(X_HOLD, Y_HOLD)
            residual_length = np.linalg.norm(prediction_position - target_position)
            if residual_length >= 0.58:
                # Long miss: empty cell below-right of the trough, not on the rising arm.
                return prediction_position + np.array([1.02, -0.52, 0.0])
            # Near the traveler: the cell above the halo stays empty after the wiggle dies.
            return target_position + np.array([0.18, 0.70, 0.0])

        prediction_label.add_updater(lambda label: label.move_to(prediction_label_position()))
        residual_label = Text("r", font_size=24, color=YELLOW_A)

        def residual_label_position() -> np.ndarray:
            prediction_position = self.plot_point(X_HOLD, current_prediction())
            target_position = self.plot_point(X_HOLD, Y_HOLD)
            if np.linalg.norm(prediction_position - target_position) >= 0.58:
                return (prediction_position + target_position) / 2 + np.array([0.38, 0.0, 0.0])
            return target_position + np.array([0.66, -0.48, 0.0])

        residual_label.add_updater(lambda label: label.move_to(residual_label_position()))

        # 0.40–2.70 s: plot and label the held-out traveler before any curve appears.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.025),
            run_time=0.70,
            rate_func=linear,
        )
        self.play(
            FadeIn(traveler_dot),
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(sample_x_label),
            FadeIn(sample_y_label),
            Flash(halo_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.55,
            rate_func=smooth,
        )
        self.wait(1.05)

        # 2.70–3.90 s: the unconstrained high-capacity curve and its long residual.
        # Residual finishes before t=4 so the first overfit QA frame is opaque.
        self.play(
            FadeIn(prediction_curve),
            FadeIn(formula),
            FadeIn(secondary_readout),
            run_time=0.70,
            rate_func=smooth,
        )
        self.play(
            FadeIn(residual_segment),
            FadeIn(prediction_dot),
            FadeIn(prediction_label),
            FadeIn(residual_label),
            run_time=0.50,
            rate_func=smooth,
        )

        # 3.90–11.50 s: hold the wild interpolant so the miss is readable.
        overfit_hold = ValueTracker(0.0)
        self.play(overfit_hold.animate.set_value(1.0), run_time=7.60, rate_func=linear)

        # 11.50–19.50 s: raise λ; the same capacity is pulled smoother.
        self.play(lambda_unit.animate.set_value(1.0), run_time=8.00, rate_func=smooth)

        # 19.50–28.20 s: a timed no-op keeps every live updater rendered through
        # the final encoded frame (a terminal wait can freeze them unevenly).
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.70, rate_func=linear)
