"""D2L 11.10 — Adam versus SGD on the 11.6 narrow valley.

The valley is f(w) = 0.1 w₁² + 2 w₂² + const, the empirical mean of
per-sample losses fᵢ(w) = 0.1(w₁−Aᵢ)² + 2(w₂−Bᵢ)². Offsets are
mean-centered, so E[∇fᵢ] = ∇f = (0.2 w₁, 4 w₂). SGD takes w ← w − η∇fᵢ.
Adam keeps a first moment v and a second moment s, bias-corrects them,
then steps. Numpy is the only arithmetic.
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
    Dot,
    FadeIn,
    FadeOut,
    Flash,
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


W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.40
BETA1 = 0.90
BETA2 = 0.999
EPS = 1e-6
STEPS_SGD = 5
STEPS_ADAM = 6
CONTOUR_LEVELS = (0.4, 1.0, 2.5, 5.0, 10.5)

# Five mean-centered samples. E[∇fᵢ] = ∇f of the 11.6 valley.
SAMPLES = np.array(
    [
        [-0.400000, 0.100000],
        [0.250000, -0.080000],
        [0.150000, 0.120000],
        [-0.200000, -0.060000],
        [0.200000, -0.080000],
    ],
    dtype=float,
)
SGD_INDEX = np.array([0, 1, 2, 3, 4, 0], dtype=int)


def valley(weights: np.ndarray) -> float:
    """The shared 11.6 valley, ignoring the sample-variance constant."""
    return float(0.1 * weights[0] ** 2 + 2.0 * weights[1] ** 2)


def sample_gradient(weights: np.ndarray, index: int) -> np.ndarray:
    """∇fᵢ(w) = (0.2(w₁−Aᵢ), 4(w₂−Bᵢ))."""
    offset = SAMPLES[index]
    return np.array(
        [0.2 * (weights[0] - offset[0]), 4.0 * (weights[1] - offset[1])],
        dtype=float,
    )


def run_sgd(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    grads = np.zeros((steps, 2), dtype=float)
    path[0] = weights
    values[0] = valley(weights)
    for index in range(steps):
        grad = sample_gradient(weights, int(SGD_INDEX[index]))
        grads[index] = grad
        weights = weights - ETA * grad
        path[index + 1] = weights
        values[index + 1] = valley(weights)
    return path, values, grads


def run_adam(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    first_moment = np.zeros(2, dtype=float)
    second_moment = np.zeros(2, dtype=float)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    grads = np.zeros((steps, 2), dtype=float)
    path[0] = weights
    values[0] = valley(weights)
    for index in range(steps):
        time_step = index + 1
        grad = sample_gradient(weights, int(SGD_INDEX[index]))
        grads[index] = grad
        first_moment = BETA1 * first_moment + (1.0 - BETA1) * grad
        second_moment = BETA2 * second_moment + (1.0 - BETA2) * (grad * grad)
        first_hat = first_moment / (1.0 - BETA1**time_step)
        second_hat = second_moment / (1.0 - BETA2**time_step)
        weights = weights - ETA * first_hat / (np.sqrt(second_hat) + EPS)
        path[index + 1] = weights
        values[index + 1] = valley(weights)
    return path, values, grads


PATH_SGD, F_SGD, GRADS_SGD = run_sgd(STEPS_SGD)
PATH_ADAM, F_ADAM, GRADS_ADAM = run_adam(STEPS_ADAM)


class Episode1110(Scene):
    """A ~32-second silent SGD-versus-Adam visualization on the 11.6 valley."""

    w1_min, w1_max = -6.35, 0.55
    w2_min, w2_max = -2.55, 2.15
    plot_left, plot_right = -3.55, 5.65
    plot_bottom, plot_top = -3.15, 2.85
    step_time = 2.00

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
        """Stroke-only elongated ellipse, clipped to the plot. Never filled."""
        theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=True)
        weights_1 = np.sqrt(level / 0.1) * np.cos(theta)
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
            stroke.set_points_as_corners(points)
            stroke.set_fill(BLACK, opacity=0.0)
            stroke.set_stroke(BLUE_E, width=1.6, opacity=opacity)
            group.add(stroke)
        return group

    def contour_label_point(self, level: float, prefer: str) -> np.ndarray | None:
        """Park the level number on the curve, clear of the top formula strip."""
        theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        best = None
        best_score = -1e9
        for angle in theta:
            w1 = float(np.sqrt(level / 0.1) * np.cos(angle))
            w2 = float(np.sqrt(level / 2.0) * np.sin(angle))
            if not self.inside_plot(w1, w2):
                continue
            if prefer == "inner":
                if w2 < 0.18:
                    continue
                score = w1 + 0.20 * w2
                offset = np.array([0.26, 0.12, 0.0])
            else:
                if w2 > -0.20:
                    continue
                score = -w1 - 0.35 * w2
                offset = np.array([-0.28, -0.16, 0.0])
            if score > best_score:
                best_score = score
                best = (w1, w2, offset)
        if best is None:
            return None
        return self.plot_point(best[0], best[1]) + best[2]

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
        print("D2L 11.10  same valley f(w)=0.1 w1**2 + 2 w2**2 + const")
        print("start w =", W_START, "  valley =", valley(W_START), "  η =", ETA)
        print("β1 =", BETA1, "  β2 =", BETA2, "  ε =", EPS)
        print("samples (mean-centered offsets) =\n", SAMPLES)
        print("mean(samples) =", SAMPLES.mean(axis=0))
        print("SGD path:")
        for index, (weights, value) in enumerate(zip(PATH_SGD, F_SGD)):
            if index == 0:
                print(
                    "  k=0  w=[{:+.5f}, {:+.5f}]  f={:.4f}".format(weights[0], weights[1], value)
                )
                continue
            sample = int(SGD_INDEX[index - 1])
            print(
                "  k={:d}  i={:d}  ∇fᵢ=[{:+.4f}, {:+.4f}]  w=[{:+.5f}, {:+.5f}]  f={:.4f}".format(
                    index,
                    sample,
                    GRADS_SGD[index - 1, 0],
                    GRADS_SGD[index - 1, 1],
                    weights[0],
                    weights[1],
                    value,
                )
            )
        print("Adam path:")
        for index, (weights, value) in enumerate(zip(PATH_ADAM, F_ADAM)):
            if index == 0:
                print(
                    "  k=0  w=[{:+.5f}, {:+.5f}]  f={:.4f}".format(weights[0], weights[1], value)
                )
                continue
            sample = int(SGD_INDEX[index - 1])
            print(
                "  k={:d}  i={:d}  ∇fᵢ=[{:+.4f}, {:+.4f}]  w=[{:+.5f}, {:+.5f}]  f={:.4f}".format(
                    index,
                    sample,
                    GRADS_ADAM[index - 1, 0],
                    GRADS_ADAM[index - 1, 1],
                    weights[0],
                    weights[1],
                    value,
                )
            )
        print(
            "D2L 11.10 last SGD  w=[{:+.5f}, {:+.5f}]  f={:.6f}".format(
                PATH_SGD[-1, 0], PATH_SGD[-1, 1], float(F_SGD[-1])
            )
        )
        print(
            "D2L 11.10 last Adam w=[{:+.5f}, {:+.5f}]  f={:.6f}".format(
                PATH_ADAM[-1, 0], PATH_ADAM[-1, 1], float(F_ADAM[-1])
            )
        )

        show_episode_intro(self, "11.10")

        opacities = (0.82, 0.68, 0.52, 0.38, 0.28)
        contours = VGroup(
            *[self.make_contour(level, opacity) for level, opacity in zip(CONTOUR_LEVELS, opacities)]
        )
        axes = self.make_axes()
        contour_labels = VGroup()
        for level, prefer in ((1.0, "inner"), (2.5, "inner"), (10.5, "outer")):
            position = self.contour_label_point(level, prefer)
            if position is None:
                continue
            label = Text(f"{level:g}", font_size=16, color=BLUE_A).move_to(position)
            contour_labels.add(label)

        self.play(
            FadeIn(contours),
            FadeIn(axes),
            run_time=1.10,
            rate_func=linear,
        )
        self.play(FadeIn(contour_labels), run_time=0.30, rate_func=linear)

        w1_tracker = ValueTracker(W_START[0])
        w2_tracker = ValueTracker(W_START[1])

        def traveler_center() -> np.ndarray:
            return self.plot_point(w1_tracker.get_value(), w2_tracker.get_value())

        def draw_traveler() -> VGroup:
            center = traveler_center()
            dot = (
                Dot(center, radius=0.068, color=YELLOW)
                .set_fill(YELLOW, opacity=0.94)
                .set_stroke(YELLOW, width=0.0, opacity=0.0)
            )
            outer = (
                Circle(radius=0.278)
                .move_to(center)
                .set_fill(BLACK, opacity=0.0)
                .set_stroke(BLUE_A, width=2.1, opacity=0.50)
            )
            inner = (
                Circle(radius=0.178)
                .move_to(center)
                .set_fill(BLACK, opacity=0.0)
                .set_stroke(YELLOW, width=2.8, opacity=0.98)
            )
            return VGroup(dot, outer, inner)

        traveler = always_redraw(draw_traveler)
        w_glyph = Text("w", font_size=24, color=YELLOW)

        def w_glyph_position() -> np.ndarray:
            return traveler_center() + np.array([0.40, 0.34, 0.0])

        w_glyph.add_updater(lambda mob: mob.move_to(w_glyph_position()))

        sgd_segments = [
            Line(self.plot_point(*PATH_SGD[index]), self.plot_point(*PATH_SGD[index + 1])).set_stroke(
                BLUE_A, width=3.6, opacity=0.96
            )
            for index in range(STEPS_SGD)
        ]
        adam_segments = [
            Line(self.plot_point(*PATH_ADAM[index]), self.plot_point(*PATH_ADAM[index + 1])).set_stroke(
                YELLOW, width=3.6, opacity=0.98
            )
            for index in range(STEPS_ADAM)
        ]
        # First SGD peak, left of the bounce, under contour 2.5.
        sgd_tag = Text("SGD", font_size=26, color=BLUE_A).move_to(
            self.plot_point(*PATH_SGD[1]) + np.array([-0.62, 0.22, 0.0])
        )
        # Below-right of the smooth curve, off the stroke, left of the traveler's halt.
        adam_tag = Text("Adam", font_size=26, color=YELLOW).move_to(
            self.plot_point(*PATH_ADAM[3]) + np.array([0.62, -0.78, 0.0])
        )
        formula = Text("w ← w − η v̂/(√ŝ+ε)", font_size=34, color=WHITE).move_to(
            np.array([1.85, 3.22, 0.0])
        )
        eta_mark = Text("η = 0.40", font_size=22, color=YELLOW).next_to(
            formula, np.array([-1.0, 0.0, 0.0]), buff=0.38
        )

        self.add(traveler, w_glyph)
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(0.20)

        for index, segment in enumerate(sgd_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_SGD[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_SGD[index + 1, 1])),
                run_time=self.step_time,
                rate_func=smooth,
            )
        self.play(FadeIn(sgd_tag), run_time=0.35, rate_func=smooth)
        self.wait(0.20)

        self.play(
            *[segment.animate.set_stroke(BLUE_A, width=2.4, opacity=0.32) for segment in sgd_segments],
            sgd_tag.animate.set_opacity(0.40),
            run_time=0.40,
            rate_func=linear,
        )
        self.play(
            w1_tracker.animate.set_value(float(W_START[0])),
            w2_tracker.animate.set_value(float(W_START[1])),
            run_time=0.55,
            rate_func=smooth,
        )
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.36, line_length=0.10),
            FadeIn(formula),
            FadeIn(eta_mark),
            run_time=0.45,
            rate_func=smooth,
        )

        for index, segment in enumerate(adam_segments):
            animations = [
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_ADAM[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_ADAM[index + 1, 1])),
            ]
            if index == 2:
                animations.append(FadeIn(adam_tag))
            self.play(*animations, run_time=self.step_time, rate_func=smooth)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=3.60, rate_func=linear)
