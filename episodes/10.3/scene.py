"""D2L 10.3 — two scoring functions a(q, k) on the same q, k, v.

Additive and scaled-dot scores, softmax weights, and ŷ all come from the
numpy arrays below. Nothing is trained. Values equal keys so the weighted
sum sits in the same plane as the query.
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
    config,
    linear,
    smooth,
    always_redraw,
)


config.background_color = BLACK
config.renderer = "cairo"


# ---------------------------------------------------------------------------
# One numerical source of truth. Query is the halo traveler. Three key/value
# pairs in R². Additive a = w_vᵀ tanh(W_q q + W_k k). Scaled-dot a = qᵀk / √d.
# ---------------------------------------------------------------------------
QUERY = np.array([1.00, 0.50], dtype=float)
KEYS = np.array(
    [
        [0.30, 1.95],
        [-1.50, 1.05],
        [0.50, -1.55],
    ],
    dtype=float,
)
VALUES = KEYS.copy()
W_Q = np.array(
    [
        [-0.09, 0.92],
        [0.32, -0.91],
    ],
    dtype=float,
)
W_K = np.array(
    [
        [-0.06, 0.93],
        [-0.83, -0.02],
    ],
    dtype=float,
)
W_V = np.array([0.05, 1.00], dtype=float)
FEATURE_DIM = QUERY.shape[0]
SCALE = 1.0 / np.sqrt(FEATURE_DIM)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for the same scores drawn on screen."""
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def additive_score(key: np.ndarray) -> float:
    """D2L (10.3.3): a(q, k) = w_vᵀ tanh(W_q q + W_k k)."""
    return float(W_V @ np.tanh(W_Q @ QUERY + W_K @ key))


ADDITIVE_SCORES = np.array([additive_score(key) for key in KEYS], dtype=float)
DOT_SCORES = KEYS @ QUERY * SCALE
ADDITIVE_WEIGHTS = softmax(ADDITIVE_SCORES)
DOT_WEIGHTS = softmax(DOT_SCORES)
ADDITIVE_YHAT = ADDITIVE_WEIGHTS @ VALUES
DOT_YHAT = DOT_WEIGHTS @ VALUES
ADDITIVE_WINNER = int(np.argmax(ADDITIVE_WEIGHTS))
DOT_WINNER = int(np.argmax(DOT_WEIGHTS))
KEY_COLORS = (BLUE, YELLOW, BLUE_D)


class Episode103(Scene):
    """A ~34-second silent additive vs scaled-dot scoring visualization."""

    x_min, x_max = -2.05, 2.15
    y_min, y_max = -2.05, 2.35
    plot_left, plot_right = -5.42, -0.28
    plot_bottom, plot_top = -3.18, 2.58
    additive_x = (0.95, 1.88, 2.81)
    dot_x = (3.88, 4.81, 5.74)
    score_baseline = 1.08
    score_scale = 1.12
    alpha_baseline = -2.48
    alpha_scale = 2.02

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
        for x_value in np.arange(-2.0, 2.01, 0.5):
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
            y_axis.get_end(), np.array([-1.0, 1.0, 0.0]), buff=0.14
        )
        return grid, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def make_score_axis(self, xs: tuple[float, float, float]) -> tuple[Line, Text]:
        axis = Line(
            np.array([xs[0] - 0.42, self.score_baseline, 0.0]),
            np.array([xs[2] + 0.42, self.score_baseline, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)
        zero = Text("0", font_size=20, color=BLUE_A).next_to(
            axis.get_start(), np.array([-1.0, -1.0, 0.0]), buff=0.10
        )
        return axis, zero

    def make_alpha_axis(self, xs: tuple[float, float, float]) -> Line:
        return Line(
            np.array([xs[0] - 0.42, self.alpha_baseline, 0.0]),
            np.array([xs[2] + 0.42, self.alpha_baseline, 0.0]),
        ).set_stroke(BLUE_D, width=1.7, opacity=0.82)

    def needle_points(
        self,
        x_position: float,
        score: float,
        weight: float,
        score_reveal: ValueTracker,
        alpha_reveal: ValueTracker,
        kind: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if kind == "score":
            start = np.array([x_position, self.score_baseline, 0.0])
            length = score * self.score_scale * score_reveal.get_value()
            end = np.array([x_position, self.score_baseline + length, 0.0])
            return start, end
        start = np.array([x_position, self.alpha_baseline, 0.0])
        length = max(weight * self.alpha_scale * alpha_reveal.get_value(), 0.001)
        end = np.array([x_position, self.alpha_baseline + length, 0.0])
        return start, end

    def make_signed_label(self, name: str, value: float, color) -> VGroup:
        prefix = Text(f"{name} = ", font_size=20, color=color)
        number = DecimalNumber(
            value,
            num_decimal_places=3,
            mob_class=Text,
            include_sign=True,
            color=WHITE,
            font_size=20,
        )
        return VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06)

    def make_alpha_label(self, name: str, value: float, color) -> VGroup:
        prefix = Text(f"{name} = ", font_size=20, color=color)
        number = DecimalNumber(
            value,
            num_decimal_places=3,
            mob_class=Text,
            include_sign=False,
            color=WHITE,
            font_size=20,
        )
        return VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06)

    def construct(self) -> None:
        print(
            "D2L 10.3 q={} K={} V={}".format(
                np.array2string(QUERY, precision=3, separator=", "),
                np.array2string(KEYS, precision=3, separator=", "),
                np.array2string(VALUES, precision=3, separator=", "),
            )
        )
        print(
            "D2L 10.3 W_q={} W_k={} w_v={} d={} scale={:.6f}".format(
                np.array2string(W_Q, precision=3, separator=", "),
                np.array2string(W_K, precision=3, separator=", "),
                np.array2string(W_V, precision=3, separator=", "),
                FEATURE_DIM,
                SCALE,
            )
        )
        print(
            "D2L 10.3 additive a={} alpha={} sum={:.6f} yhat={}".format(
                np.array2string(ADDITIVE_SCORES, precision=6, separator=", "),
                np.array2string(ADDITIVE_WEIGHTS, precision=6, separator=", "),
                float(np.sum(ADDITIVE_WEIGHTS)),
                np.array2string(ADDITIVE_YHAT, precision=6, separator=", "),
            )
        )
        print(
            "D2L 10.3 scaled-dot a={} alpha={} sum={:.6f} yhat={}".format(
                np.array2string(DOT_SCORES, precision=6, separator=", "),
                np.array2string(DOT_WEIGHTS, precision=6, separator=", "),
                float(np.sum(DOT_WEIGHTS)),
                np.array2string(DOT_YHAT, precision=6, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "10.3")

        grid, axes = self.make_grid_and_axes()
        key_dots = [
            Dot(self.plot_point(key[0], key[1]), radius=0.070, color=color)
            .set_fill(color, opacity=0.94)
            .set_stroke(color, width=0.0, opacity=0.0)
            for key, color in zip(KEYS, KEY_COLORS)
        ]
        point_cloud = VGroup(*key_dots)
        tag_offsets = (
            np.array([0.52, -0.24, 0.0]),
            np.array([-0.50, 0.30, 0.0]),
            np.array([0.48, -0.32, 0.0]),
        )
        key_tags = VGroup(
            *[
                Text(f"k{['₁', '₂', '₃'][index]}", font_size=22, color=WHITE).move_to(
                    self.plot_point(key[0], key[1]) + offset
                )
                for index, (key, offset) in enumerate(zip(KEYS, tag_offsets))
            ]
        )

        traveler_center = self.plot_point(QUERY[0], QUERY[1])
        halo_outer = (
            Circle(radius=0.278)
            .move_to(traveler_center)
            .set_stroke(BLUE_A, width=2.1, opacity=0.50)
            .set_fill(BLACK, opacity=0.0)
        )
        halo_inner = (
            Circle(radius=0.178)
            .move_to(traveler_center)
            .set_stroke(YELLOW, width=2.8, opacity=0.98)
            .set_fill(BLACK, opacity=0.0)
        )
        traveler_dot = (
            Dot(traveler_center, radius=0.070, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        query_x1 = Text("q₁ = 1.00", font_size=21, color=BLUE_A).move_to(np.array([-6.22, -1.70, 0.0]))
        query_x2 = Text("q₂ = 0.50", font_size=21, color=BLUE_A).move_to(np.array([-6.22, -2.12, 0.0]))

        formula = Text("ŷ = Σ softmax(a(q, k)) v", font_size=30, color=WHITE).move_to(
            np.array([3.34, 3.52, 0.0])
        )
        additive_title = Text("wᵀ tanh(Wq q + Wk k)", font_size=22, color=WHITE).move_to(
            np.array([1.88, 3.08, 0.0])
        )
        dot_title = Text("qᵀk / √d", font_size=22, color=WHITE).move_to(np.array([4.81, 3.08, 0.0]))

        additive_score_axis, additive_zero = self.make_score_axis(self.additive_x)
        dot_score_axis, dot_zero = self.make_score_axis(self.dot_x)
        additive_alpha_axis = self.make_alpha_axis(self.additive_x)
        dot_alpha_axis = self.make_alpha_axis(self.dot_x)

        additive_score_reveal = [ValueTracker(0.0) for _ in range(3)]
        additive_alpha_reveal = ValueTracker(0.0)
        dot_score_reveal = [ValueTracker(0.0) for _ in range(3)]
        dot_alpha_reveal = ValueTracker(0.0)
        additive_yhat_reveal = ValueTracker(0.0)
        dot_yhat_reveal = ValueTracker(0.0)

        def make_needle(xs, scores, weights, score_reveal, alpha_reveal, kind: str):
            bars = VGroup()
            for index in range(3):
                def draw(
                    index=index,
                    xs=xs,
                    scores=scores,
                    weights=weights,
                    score_reveal=score_reveal,
                    alpha_reveal=alpha_reveal,
                    kind=kind,
                ):
                    start, end = self.needle_points(
                        xs[index],
                        float(scores[index]),
                        float(weights[index]),
                        score_reveal[index] if kind == "score" else score_reveal[0],
                        alpha_reveal,
                        kind,
                    )
                    reveal = (
                        score_reveal[index].get_value() if kind == "score" else alpha_reveal.get_value()
                    )
                    opacity = 0.0 if reveal < 0.02 else 0.98
                    width = 13.0 if kind == "score" else 15.0
                    return Line(start, end).set_stroke(KEY_COLORS[index], width=width, opacity=opacity)

                bars.add(always_redraw(draw))
            return bars

        additive_score_bars = make_needle(
            self.additive_x,
            ADDITIVE_SCORES,
            ADDITIVE_WEIGHTS,
            additive_score_reveal,
            additive_alpha_reveal,
            "score",
        )
        additive_alpha_bars = make_needle(
            self.additive_x,
            ADDITIVE_SCORES,
            ADDITIVE_WEIGHTS,
            additive_score_reveal,
            additive_alpha_reveal,
            "alpha",
        )
        dot_score_bars = make_needle(
            self.dot_x,
            DOT_SCORES,
            DOT_WEIGHTS,
            dot_score_reveal,
            dot_alpha_reveal,
            "score",
        )
        dot_alpha_bars = make_needle(
            self.dot_x,
            DOT_SCORES,
            DOT_WEIGHTS,
            dot_score_reveal,
            dot_alpha_reveal,
            "alpha",
        )

        additive_score_labels = VGroup(
            *[
                self.make_signed_label(f"a{['₁', '₂', '₃'][index]}", float(ADDITIVE_SCORES[index]), KEY_COLORS[index])
                for index in range(3)
            ]
        )
        dot_score_labels = VGroup(
            *[
                self.make_signed_label(f"a{['₁', '₂', '₃'][index]}", float(DOT_SCORES[index]), KEY_COLORS[index])
                for index in range(3)
            ]
        )
        additive_alpha_labels = VGroup(
            *[
                self.make_alpha_label(f"α{['₁', '₂', '₃'][index]}", float(ADDITIVE_WEIGHTS[index]), KEY_COLORS[index])
                for index in range(3)
            ]
        )
        dot_alpha_labels = VGroup(
            *[
                self.make_alpha_label(f"α{['₁', '₂', '₃'][index]}", float(DOT_WEIGHTS[index]), KEY_COLORS[index])
                for index in range(3)
            ]
        )

        def score_label_position(xs, scores, score_reveal, index: int) -> np.ndarray:
            _, end = self.needle_points(
                xs[index],
                float(scores[index]),
                0.0,
                score_reveal[index],
                ValueTracker(0.0),
                "score",
            )
            direction = 1.0 if scores[index] >= 0 else -1.0
            return end + np.array([0.0, direction * 0.34, 0.0])

        def alpha_label_position(xs, weights, alpha_reveal, index: int, side: float) -> np.ndarray:
            _, end = self.needle_points(
                xs[index],
                0.0,
                float(weights[index]),
                [ValueTracker(0.0)] * 3,
                alpha_reveal,
                "alpha",
            )
            if side < 0:
                return end + np.array([-0.78, 0.04, 0.0])
            if side > 0:
                return end + np.array([0.78, 0.04, 0.0])
            return end + np.array([0.0, 0.34, 0.0])

        for index, label in enumerate(additive_score_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.move_to(
                    score_label_position(self.additive_x, ADDITIVE_SCORES, additive_score_reveal, index)
                )
            )
        for index, label in enumerate(dot_score_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.move_to(
                    score_label_position(self.dot_x, DOT_SCORES, dot_score_reveal, index)
                )
            )
        # Short α bars keep their numbers beside the stroke, never sliding
        # into the neighbouring column.
        additive_alpha_sides = (0.0, 0.0, 0.0)
        dot_alpha_sides = (0.0, -1.0, 1.0)
        for index, label in enumerate(additive_alpha_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.move_to(
                    alpha_label_position(
                        self.additive_x,
                        ADDITIVE_WEIGHTS,
                        additive_alpha_reveal,
                        index,
                        additive_alpha_sides[index],
                    )
                )
            )
        for index, label in enumerate(dot_alpha_labels):
            label.add_updater(
                lambda mobject, index=index: mobject.move_to(
                    alpha_label_position(
                        self.dot_x,
                        DOT_WEIGHTS,
                        dot_alpha_reveal,
                        index,
                        dot_alpha_sides[index],
                    )
                )
            )

        additive_sum = Text("Σαᵢ = 1.000", font_size=22, color=WHITE).move_to(np.array([1.88, -3.18, 0.0]))
        dot_sum = Text("Σαᵢ = 1.000", font_size=22, color=WHITE).move_to(np.array([4.81, -3.18, 0.0]))

        additive_yhat_center = self.plot_point(float(ADDITIVE_YHAT[0]), float(ADDITIVE_YHAT[1]))
        dot_yhat_center = self.plot_point(float(DOT_YHAT[0]), float(DOT_YHAT[1]))
        additive_yhat_dot = always_redraw(
            lambda: Dot(additive_yhat_center, radius=0.078, color=YELLOW)
            .set_fill(YELLOW, opacity=0.98 * additive_yhat_reveal.get_value())
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        dot_yhat_dot = always_redraw(
            lambda: Dot(dot_yhat_center, radius=0.078, color=BLUE)
            .set_fill(BLUE, opacity=0.98 * dot_yhat_reveal.get_value())
            .set_stroke(BLUE, width=0.0, opacity=0.0)
        )
        additive_yhat_ring = always_redraw(
            lambda: Circle(radius=0.155)
            .move_to(additive_yhat_center)
            .set_stroke(YELLOW, width=2.0, opacity=0.95 * additive_yhat_reveal.get_value())
            .set_fill(BLACK, opacity=0.0)
        )
        dot_yhat_ring = always_redraw(
            lambda: Circle(radius=0.155)
            .move_to(dot_yhat_center)
            .set_stroke(BLUE, width=2.0, opacity=0.95 * dot_yhat_reveal.get_value())
            .set_fill(BLACK, opacity=0.0)
        )
        additive_yhat_label = Text("ŷ", font_size=24, color=YELLOW).move_to(
            additive_yhat_center + np.array([0.36, -0.34, 0.0])
        )
        dot_yhat_label = Text("ŷ", font_size=24, color=BLUE).move_to(
            dot_yhat_center + np.array([0.38, -0.34, 0.0])
        )

        # 0.40–3.30 s: three keys before any scoring.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.10),
            FadeIn(key_tags),
            run_time=0.90,
            rate_func=linear,
        )
        self.wait(2.00)

        # 3.30–5.50 s: halo the traveler; pocket holds q.
        self.play(
            FadeIn(traveler_dot),
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(query_x1),
            FadeIn(query_x2),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(1.50)

        # 5.50–7.70 s: pooling formula once.
        self.play(FadeIn(formula), run_time=0.80, rate_func=smooth)
        self.wait(1.40)

        # 7.70–13.20 s: additive scores. Needles stay through the end.
        self.add(additive_score_bars)
        self.play(
            FadeIn(additive_title),
            FadeIn(additive_score_axis),
            FadeIn(additive_zero),
            run_time=0.80,
            rate_func=smooth,
        )
        for index in range(3):
            self.play(
                additive_score_reveal[index].animate.set_value(1.0),
                FadeIn(additive_score_labels[index]),
                run_time=1.10,
                rate_func=smooth,
            )
            self.wait(0.35)
        self.wait(1.15)

        # 13.20–17.80 s: additive softmax and ŷ. Scores remain.
        self.add(additive_alpha_bars)
        self.play(FadeIn(additive_alpha_axis), run_time=0.35, rate_func=linear)
        self.play(additive_alpha_reveal.animate.set_value(1.0), run_time=1.80, rate_func=smooth)
        self.play(FadeIn(additive_alpha_labels), FadeIn(additive_sum), run_time=0.55, rate_func=smooth)
        self.add(additive_yhat_dot, additive_yhat_ring)
        self.play(
            additive_yhat_reveal.animate.set_value(1.0),
            FadeIn(additive_yhat_label),
            run_time=0.80,
            rate_func=smooth,
        )
        additive_box = SurroundingRectangle(
            additive_alpha_labels[ADDITIVE_WINNER], color=YELLOW, buff=0.12, corner_radius=0.08
        )
        additive_box.set_fill(opacity=0.0)
        additive_box.set_stroke(width=2.6, opacity=1.0)
        self.play(Create(additive_box), run_time=0.50, rate_func=smooth)
        self.wait(1.60)

        # 17.80–23.30 s: scaled-dot scores on the same q, k, v.
        self.add(dot_score_bars)
        self.play(
            FadeIn(dot_title),
            FadeIn(dot_score_axis),
            FadeIn(dot_zero),
            run_time=0.80,
            rate_func=smooth,
        )
        for index in range(3):
            self.play(
                dot_score_reveal[index].animate.set_value(1.0),
                FadeIn(dot_score_labels[index]),
                run_time=1.10,
                rate_func=smooth,
            )
            self.wait(0.35)
        self.wait(1.15)

        # 23.30–27.90 s: scaled-dot softmax and the other ŷ.
        self.add(dot_alpha_bars)
        self.play(FadeIn(dot_alpha_axis), run_time=0.35, rate_func=linear)
        self.play(dot_alpha_reveal.animate.set_value(1.0), run_time=1.80, rate_func=smooth)
        self.play(FadeIn(dot_alpha_labels), FadeIn(dot_sum), run_time=0.55, rate_func=smooth)
        self.add(dot_yhat_dot, dot_yhat_ring)
        self.play(
            dot_yhat_reveal.animate.set_value(1.0),
            FadeIn(dot_yhat_label),
            run_time=0.80,
            rate_func=smooth,
        )
        dot_box = SurroundingRectangle(
            dot_alpha_labels[DOT_WINNER], color=YELLOW, buff=0.12, corner_radius=0.08
        )
        dot_box.set_fill(opacity=0.0)
        dot_box.set_stroke(width=2.6, opacity=1.0)
        self.play(Create(dot_box), run_time=0.50, rate_func=smooth)
        self.wait(1.60)

        # Hold both scoring functions, both weight vectors, both ŷ.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=6.50, rate_func=linear)
