"""D2L 11.4 — full gradient vs one-sample SGD on the same 2-D bowl.

The bowl is the 11.3 family: the empirical mean of the per-sample losses is
f(w) = w₁² + 2 w₂² + const, so ∇f = (2 w₁, 4 w₂). Each SGD step uses one
sample’s gradient ∇fᵢ, not the mean. Numpy is the only arithmetic.
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


W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.10
STEPS_FULL = 5
STEPS_SGD = 8
CONTOUR_LEVELS = (2.0, 6.0, 12.0, 22.0, 33.0)

# Five mean-centered samples. E[∇fᵢ] = ∇f of the 11.3 bowl.
SAMPLES = np.array(
    [
        [-1.331203, 1.583793],
        [0.205971, -0.940458],
        [-1.303238, 1.177641],
        [0.224042, -0.179648],
        [2.204428, -1.641329],
    ],
    dtype=float,
)
SGD_INDEX = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=int)


def bowl(weights: np.ndarray) -> float:
    """The shared 11.3 bowl, ignoring the sample-variance constant."""
    return float(weights[0] ** 2 + 2.0 * weights[1] ** 2)


def full_gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (2 w₁, 4 w₂), equal to the mean of the five ∇fᵢ."""
    return np.array([2.0 * weights[0], 4.0 * weights[1]], dtype=float)


def sample_gradient(weights: np.ndarray, index: int) -> np.ndarray:
    """∇fᵢ(w) for one sample: 2(w − offset) with the 2× scale on w₂."""
    offset = SAMPLES[index]
    return np.array(
        [2.0 * (weights[0] - offset[0]), 4.0 * (weights[1] - offset[1])],
        dtype=float,
    )


def run_full(steps: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    path[0] = weights
    values[0] = bowl(weights)
    for index in range(steps):
        weights = weights - ETA * full_gradient(weights)
        path[index + 1] = weights
        values[index + 1] = bowl(weights)
    return path, values


def run_sgd(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array(W_START, dtype=float, copy=True)
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    grads = np.zeros((steps, 2), dtype=float)
    path[0] = weights
    values[0] = bowl(weights)
    for index in range(steps):
        sample = int(SGD_INDEX[index])
        grad = sample_gradient(weights, sample)
        grads[index] = grad
        weights = weights - ETA * grad
        path[index + 1] = weights
        values[index + 1] = bowl(weights)
    return path, values, grads


PATH_FULL, F_FULL = run_full(STEPS_FULL)
PATH_SGD, F_SGD, GRADS_SGD = run_sgd(STEPS_SGD)


class Episode114(Scene):
    """A ~35-second silent full-gradient vs one-sample SGD visualization."""

    w1_min, w1_max = -6.05, 1.85
    w2_min, w2_max = -3.15, 2.25
    plot_left, plot_right = -3.45, 5.55
    plot_bottom, plot_top = -3.20, 2.80
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
        print("D2L 11.4  same bowl f(w)=w1**2 + 2 w2**2 + const")
        print("start w =", W_START, "  bowl =", bowl(W_START), "  η =", ETA)
        print("samples (mean-centered offsets) =\n", SAMPLES)
        print("mean(samples) =", SAMPLES.mean(axis=0))
        print("full ∇f path:")
        for index, (weights, value) in enumerate(zip(PATH_FULL, F_FULL)):
            print(
                "  k={:d}  w=[{:+.5f}, {:+.5f}]  f={:.4f}  ∇f={}".format(
                    index,
                    weights[0],
                    weights[1],
                    value,
                    full_gradient(weights) if index < STEPS_FULL else full_gradient(weights),
                )
            )
        print("one-sample ∇fᵢ path:")
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

        show_episode_intro(self, "11.4")

        opacities = (0.78, 0.62, 0.48, 0.36, 0.28)
        contours = VGroup(
            *[self.make_contour(level, opacity) for level, opacity in zip(CONTOUR_LEVELS, opacities)]
        )
        axes = self.make_axes()
        contour_labels = VGroup()
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

        full_segments = [
            Line(self.plot_point(*PATH_FULL[index]), self.plot_point(*PATH_FULL[index + 1])).set_stroke(
                BLUE_A, width=3.6, opacity=0.96
            )
            for index in range(STEPS_FULL)
        ]
        sgd_segments = [
            Line(self.plot_point(*PATH_SGD[index]), self.plot_point(*PATH_SGD[index + 1])).set_stroke(
                YELLOW, width=3.6, opacity=0.96
            )
            for index in range(STEPS_SGD)
        ]
        full_tag = Text("∇f", font_size=26, color=BLUE_A).move_to(
            self.plot_point(*PATH_FULL[2]) + np.array([0.12, -0.42, 0.0])
        )
        # Sit left of the first w₂ sign-cross, in the empty wedge under contour 12.
        sgd_tag = Text("∇fᵢ", font_size=26, color=YELLOW).move_to(
            self.plot_point(*PATH_SGD[3]) + np.array([-0.70, 0.42, 0.0])
        )
        formula = Text("w ← w − η∇fᵢ", font_size=34, color=WHITE).move_to(np.array([1.15, 3.22, 0.0]))
        eta_mark = Text("η = 0.10", font_size=22, color=YELLOW).move_to(np.array([4.55, 3.22, 0.0]))

        self.add(traveler, w_glyph)
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(0.20)

        for index, segment in enumerate(full_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_FULL[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_FULL[index + 1, 1])),
                run_time=self.step_time,
                rate_func=smooth,
            )
        self.play(FadeIn(full_tag), run_time=0.35, rate_func=smooth)
        self.wait(0.20)

        self.play(
            *[segment.animate.set_stroke(BLUE_A, width=2.4, opacity=0.38) for segment in full_segments],
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

        for index, segment in enumerate(sgd_segments):
            animations = [
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_SGD[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_SGD[index + 1, 1])),
            ]
            if index == 3:
                animations.append(FadeIn(sgd_tag))
            self.play(*animations, run_time=self.step_time, rate_func=smooth)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=3.60, rate_func=linear)
