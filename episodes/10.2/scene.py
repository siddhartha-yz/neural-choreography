"""D2L 10.2 — Nadaraya–Watson: closer key, larger weight on its value.

Every needle, the yellow ŷ, and the curve come from the same numpy
softmax kernel below. This *is* the estimator: nothing is trained.
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
# One numerical source of truth. Keys are 1-D train x, values are train y.
# Query q is the halo traveler. Weights are softmax(−½ (q − kᵢ)²).
# ---------------------------------------------------------------------------
KEYS = np.array([0.40, 0.90, 1.40, 1.90, 2.40, 2.90, 3.40, 3.90, 4.40], dtype=float)
NOISE = np.array([0.18, -0.22, 0.12, -0.18, 0.20, -0.15, 0.10, -0.20, 0.16], dtype=float)


def latent_function(x_values: np.ndarray | float) -> np.ndarray | float:
    """D2L 10.2 generator: 2 sin(x) + x^0.8, without the noise term."""
    x_array = np.asarray(x_values, dtype=float)
    return 2.0 * np.sin(x_array) + np.power(np.maximum(x_array, 0.0), 0.8)


VALUES = np.asarray(latent_function(KEYS), dtype=float) + NOISE
QUERY_A = 1.00
QUERY_B = 3.60


def attention_weights(query: float) -> np.ndarray:
    """Gaussian-kernel softmax over the same keys drawn on screen."""
    scores = -0.5 * (query - KEYS) ** 2
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def nw_predict(query: float) -> float:
    """Nadaraya–Watson ŷ = Σ αᵢ vᵢ for this query."""
    return float(np.dot(attention_weights(query), VALUES))


WEIGHTS_A = attention_weights(QUERY_A)
WEIGHTS_B = attention_weights(QUERY_B)
YHAT_A = nw_predict(QUERY_A)
YHAT_B = nw_predict(QUERY_B)


def signed_glyphs(value: float, digits: int) -> str:
    """Match 3.1: a true minus glyph, never a hyphen-minus."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"


class Episode102(Scene):
    """A ~32-second continuous, silent D2L 10.2 visualization."""

    x_min, x_max = 0.0, 5.0
    y_min, y_max = -0.45, 4.55
    plot_left, plot_right = -5.20, 2.35
    plot_bottom, plot_top = -3.20, 2.70
    needle_scale = 3.65

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line mobjects; never a filled grid mesh."""
        grid = VGroup()
        for x_value in np.arange(0.5, 5.01, 0.5):
            grid.add(
                Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                    BLUE_E, width=1.0, opacity=0.30
                )
            )
        for y_value in np.arange(0.5, 4.51, 0.5):
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
        x_symbol = Text("x", font_size=27, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.12
        )
        y_symbol = Text("y", font_size=27, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.12
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def nw_curve(self) -> VMobject:
        """Stroke-only Nadaraya–Watson estimator, same kernel as the needles."""
        samples = np.linspace(self.x_min + 0.05, self.x_max - 0.05, 240)
        points = [self.plot_point(float(x_value), nw_predict(float(x_value))) for x_value in samples]
        curve = VMobject(fill_opacity=0.0, stroke_opacity=0.96)
        curve.set_points_as_corners(points)
        curve.set_fill(BLACK, opacity=0.0)
        curve.set_stroke(YELLOW, width=5.0, opacity=0.96)
        return curve

    def construct(self) -> None:
        print(
            "D2L 10.2 q1={:.6f} yhat1={:.6f} weights1={} sum1={:.6f}".format(
                QUERY_A,
                YHAT_A,
                np.array2string(WEIGHTS_A, precision=6, separator=", "),
                float(np.sum(WEIGHTS_A)),
            )
        )
        print(
            "D2L 10.2 q2={:.6f} yhat2={:.6f} weights2={} sum2={:.6f}".format(
                QUERY_B,
                YHAT_B,
                np.array2string(WEIGHTS_B, precision=6, separator=", "),
                float(np.sum(WEIGHTS_B)),
            )
        )
        print(
            "D2L 10.2 keys={} values={}".format(
                np.array2string(KEYS, precision=3, separator=", "),
                np.array2string(VALUES, precision=3, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "10.2")

        grid, axes = self.make_grid_and_axes()
        palette = (YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW, BLUE, YELLOW)
        data_dots = [
            Dot(self.plot_point(key, value), radius=0.068, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for key, value, color in zip(KEYS, VALUES, palette)
        ]
        point_cloud = VGroup(*data_dots)

        query_tracker = ValueTracker(QUERY_A)
        needle_reveal = ValueTracker(0.0)
        yhat_reveal = ValueTracker(0.0)

        def current_query() -> float:
            return float(query_tracker.get_value())

        def current_weights() -> np.ndarray:
            return attention_weights(current_query())

        def current_yhat() -> float:
            return nw_predict(current_query())

        def query_axis_point() -> np.ndarray:
            return self.plot_point(current_query(), 0.0)

        def yhat_point() -> np.ndarray:
            return self.plot_point(current_query(), current_yhat())

        halo_outer = always_redraw(
            lambda: Circle(radius=0.265)
            .move_to(query_axis_point())
            .set_stroke(BLUE_A, width=2.1, opacity=0.50)
            .set_fill(BLACK, opacity=0.0)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=0.172)
            .move_to(query_axis_point())
            .set_stroke(YELLOW, width=2.8, opacity=0.98)
            .set_fill(BLACK, opacity=0.0)
        )
        query_tick = always_redraw(
            lambda: Dot(query_axis_point(), radius=0.068, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )

        def style_training_dot(index: int, color) -> None:
            def update(mobject: Dot) -> None:
                alpha = float(current_weights()[index]) * needle_reveal.get_value()
                glow = min(max((alpha - 0.04) / 0.20, 0.0), 1.0)
                mobject.set_fill(color, opacity=0.94)
                mobject.set_stroke(color, width=3.2 * glow, opacity=glow)
                # Radius stays 0.068; never scale the Dot.

            data_dots[index].add_updater(update)
            update(data_dots[index])

        for index, color in enumerate(palette):
            style_training_dot(index, color)

        def make_needle(index: int):
            def draw() -> Line:
                alpha = float(current_weights()[index]) * needle_reveal.get_value()
                start = self.plot_point(float(KEYS[index]), float(VALUES[index]))
                length = max(alpha * self.needle_scale, 0.001)
                end = start + np.array([0.0, length, 0.0])
                opacity = 0.0 if alpha < 0.018 else min(0.35 + 2.2 * alpha, 0.98)
                return Line(start, end).set_stroke(YELLOW, width=4.0, opacity=opacity)

            return always_redraw(draw)

        needles = VGroup(*[make_needle(index) for index in range(len(KEYS))])

        locator = always_redraw(
            lambda: Line(query_axis_point(), yhat_point()).set_stroke(
                YELLOW_A, width=3.2, opacity=0.92 * yhat_reveal.get_value()
            )
        )
        prediction_dot = always_redraw(
            lambda: Dot(yhat_point(), radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=yhat_reveal.get_value())
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )

        def winner_caption() -> VGroup:
            weights = current_weights()
            index = int(np.argmax(weights))
            alpha = float(weights[index]) * needle_reveal.get_value()
            start = self.plot_point(float(KEYS[index]), float(VALUES[index]))
            tip = start + np.array([0.0, max(alpha * self.needle_scale, 0.001), 0.0])
            title = Text("α", font_size=20, color=YELLOW_A)
            number = Text(f"= {alpha:.3f}", font_size=20, color=YELLOW)
            caption = VGroup(title, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.08)
            # Sit above the winning needle. A left offset landed on the neighbor
            # needle and on the yellow curve (the 3.1 rule: never on the stroke).
            sky = 0.50
            if tip[1] + sky > self.plot_top - 0.16:
                sky = max(0.30, self.plot_top - 0.16 - tip[1])
            caption.move_to(tip + np.array([0.0, sky, 0.0]))
            caption.set_opacity(0.0 if alpha < 0.12 else 1.0)
            return caption

        alpha_label = always_redraw(winner_caption)

        prediction_prefix = Text("ŷ = ", font_size=20, color=YELLOW)
        prediction_number = DecimalNumber(
            YHAT_A,
            num_decimal_places=3,
            mob_class=Text,
            include_sign=True,
            color=YELLOW_A,
            font_size=20,
        )
        prediction_label = VGroup(prediction_prefix, prediction_number).arrange(
            np.array([1.0, 0.0, 0.0]), buff=0.08
        )
        prediction_number.add_updater(lambda number: number.set_value(current_yhat()))

        def prediction_label_position() -> np.ndarray:
            """Park ŷ on the locator stick, off the yellow curve (3.1 residual analog)."""
            axis_point = query_axis_point()
            prediction = yhat_point()
            along = axis_point + 0.34 * (prediction - axis_point)
            right = along + np.array([0.82, 0.0, 0.0])
            if right[0] > self.plot_right - 0.70:
                return along + np.array([-0.82, 0.0, 0.0])
            return right

        prediction_label.add_updater(
            lambda label: label.move_to(prediction_label_position()).set_opacity(yhat_reveal.get_value())
        )

        query_row_prefix = Text("q = ", font_size=22, color=BLUE_A)
        query_row_number = DecimalNumber(
            QUERY_A,
            num_decimal_places=2,
            mob_class=Text,
            include_sign=False,
            color=BLUE_A,
            font_size=22,
        )
        query_row = VGroup(query_row_prefix, query_row_number).arrange(
            np.array([1.0, 0.0, 0.0]), buff=0.08
        ).move_to(np.array([-6.18, -1.18, 0.0]))
        query_row_number.add_updater(lambda number: number.set_value(current_query()))

        yhat_row_prefix = Text("ŷ = ", font_size=22, color=YELLOW)
        yhat_row_number = DecimalNumber(
            YHAT_A,
            num_decimal_places=3,
            mob_class=Text,
            include_sign=True,
            color=YELLOW_A,
            font_size=22,
        )
        yhat_row = VGroup(yhat_row_prefix, yhat_row_number).arrange(
            np.array([1.0, 0.0, 0.0]), buff=0.08
        ).move_to(np.array([-6.18, -1.63, 0.0]))
        yhat_row_number.add_updater(lambda number: number.set_value(current_yhat()))

        sum_label = Text("Σαᵢ = 1.000", font_size=22, color=WHITE).move_to(np.array([-6.18, -2.08, 0.0]))
        formula = Text("ŷ = Σ softmax(−½‖q−kᵢ‖²) vᵢ", font_size=24, color=WHITE).move_to(
            np.array([3.55, 3.12, 0.0])
        )
        estimator_curve = self.nw_curve()

        # 0.40–3.40 s: establish the train (k, v) cloud before any query.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.04),
            run_time=0.85,
            rate_func=linear,
        )
        self.wait(2.15)

        # 3.40–5.55 s: the traveler sits as query q on the x-axis.
        self.add(query_tick, halo_outer, halo_inner)
        self.play(
            FadeIn(query_row),
            Flash(query_axis_point(), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.65,
            rate_func=smooth,
        )
        self.wait(1.50)

        # 5.55–9.20 s: attention needles grow on the keys. Closer keys light more.
        self.add(needles, alpha_label)
        self.play(needle_reveal.animate.set_value(1.0), run_time=2.40, rate_func=smooth)
        self.wait(1.25)

        # 9.20–13.10 s: ŷ is the weighted sum, sitting on the NW curve.
        self.add(locator, prediction_dot, prediction_label)
        self.play(
            FadeIn(estimator_curve),
            yhat_reveal.animate.set_value(1.0),
            FadeIn(yhat_row),
            FadeIn(sum_label),
            run_time=1.35,
            rate_func=smooth,
        )
        self.play(FadeIn(formula), run_time=0.55, rate_func=smooth)
        self.wait(2.00)

        # 13.10–21.10 s: second query. Weight mass moves with q; ŷ slides on the curve.
        self.play(query_tracker.animate.set_value(QUERY_B), run_time=6.80, rate_func=smooth)
        self.wait(1.20)

        # Hold live updaters through the final encoded frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=6.20, rate_func=linear)
