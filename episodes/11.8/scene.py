"""D2L 11.8 — RMSProp keeps walking where AdaGrad's sum stalls.

Both trajectories use the same fixed narrow quadratic valley and start point.
Every displayed metric, segment, and traveler position is derived from the
NumPy update paths below; neither optimizer is trained in the animation.
"""

from __future__ import annotations

import numpy as np
from scripts.episode_intro import show_episode_intro
from manim import (
    AnimationGroup,
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
    Line,
    RIGHT,
    Scene,
    Text,
    VGroup,
    VMobject,
    ValueTracker,
    WHITE,
    YELLOW,
    YELLOW_A,
    always_redraw,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# ---------------------------------------------------------------------------
# One numerical source of truth.  Both optimizers run on D2L's narrow-valley
# family f(w) = 0.1 w_1^2 + 2 w_2^2 from the same start and base learning rate.
# AdaGrad accumulates all squared gradients; RMSProp uses their EMA instead.
# ---------------------------------------------------------------------------
W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.30
GAMMA = 0.90
EPSILON = 1e-8
ADAGRAD_STEPS = 12
RMSPROP_STEPS = 12
CONTOUR_LEVELS = (0.4, 1.0, 2.5, 5.0, 10.5)


def objective(weights: np.ndarray) -> float:
    """f(w) = 0.1 w_1^2 + 2 w_2^2."""
    return float(0.1 * weights[0] ** 2 + 2.0 * weights[1] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (0.2 w_1, 4 w_2)."""
    return np.array([0.2 * weights[0], 4.0 * weights[1]], dtype=float)


def adagrad_descent(
    start: np.ndarray, eta: float, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, f values, cumulative squares, and x-axis step scales."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    accumulator_history = np.zeros((steps + 1, 2), dtype=float)
    x_step_scales = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    accumulator = np.zeros(2, dtype=float)
    path[0], values[0] = weights, objective(weights)
    for index in range(steps):
        current_gradient = gradient(weights)
        accumulator += current_gradient**2
        weights = weights - eta * current_gradient / (np.sqrt(accumulator) + EPSILON)
        path[index + 1], values[index + 1] = weights, objective(weights)
        accumulator_history[index + 1] = accumulator
        x_step_scales[index + 1] = eta / np.sqrt(accumulator[0])
    return path, values, accumulator_history, x_step_scales


def rmsprop_descent(
    start: np.ndarray, eta: float, gamma: float, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, f values, EMA squares, and x-axis step scales."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    square_history = np.zeros((steps + 1, 2), dtype=float)
    x_step_scales = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    square_average = np.zeros(2, dtype=float)
    path[0], values[0] = weights, objective(weights)
    for index in range(steps):
        current_gradient = gradient(weights)
        square_average = gamma * square_average + (1.0 - gamma) * current_gradient**2
        weights = weights - eta * current_gradient / (np.sqrt(square_average) + EPSILON)
        path[index + 1], values[index + 1] = weights, objective(weights)
        square_history[index + 1] = square_average
        x_step_scales[index + 1] = eta / np.sqrt(square_average[0])
    return path, values, square_history, x_step_scales


PATH_ADA, F_ADA, S_ADA, ALPHA_ADA = adagrad_descent(W_START, ETA, ADAGRAD_STEPS)
PATH_RMS, F_RMS, S_RMS, ALPHA_RMS = rmsprop_descent(W_START, ETA, GAMMA, RMSPROP_STEPS)


class Episode118(Scene):
    """A continuous 28-second Cairo comparison of AdaGrad and RMSProp."""

    w1_min, w1_max = -5.90, 0.70
    w2_min, w2_max = -2.55, 2.10
    plot_left, plot_right = -6.10, 1.25
    plot_bottom, plot_top = -3.05, 2.62

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
        """Make stroke-only, clipped contours; no VMobject is allowed a fill."""
        theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=True)
        values_1 = np.sqrt(level / 0.1) * np.cos(theta)
        values_2 = np.sqrt(level / 2.0) * np.sin(theta)
        chunks: list[list[np.ndarray]] = []
        current: list[np.ndarray] = []
        for value_1, value_2 in zip(values_1, values_2):
            if self.inside_plot(float(value_1), float(value_2)):
                current.append(self.plot_point(float(value_1), float(value_2)))
            elif current:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        contours = VGroup()
        for points in chunks:
            if len(points) < 2:
                continue
            contour = VMobject()
            contour.set_points_as_corners(points)
            contour.set_fill(BLACK, opacity=0.0)
            contour.set_stroke(BLUE_E, width=1.55, opacity=opacity)
            contours.add(contour)
        return contours

    def make_axes(self) -> VGroup:
        x_axis = Line(self.plot_point(self.w1_min, 0.0), self.plot_point(self.w1_max, 0.0)).set_stroke(
            BLUE_D, width=1.9, opacity=0.84
        )
        y_axis = Line(self.plot_point(0.0, self.w2_min), self.plot_point(0.0, self.w2_max)).set_stroke(
            BLUE_D, width=1.9, opacity=0.84
        )
        w1_symbol = Text("w₁", font_size=25, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-0.15, -1.0, 0.0]), buff=0.13
        )
        w2_symbol = Text("w₂", font_size=25, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -0.2, 0.0]), buff=0.12
        )
        origin_h = Line(self.plot_point(-0.11, 0.0), self.plot_point(0.11, 0.0)).set_stroke(
            WHITE, width=1.9, opacity=0.88
        )
        origin_v = Line(self.plot_point(0.0, -0.11), self.plot_point(0.0, 0.11)).set_stroke(
            WHITE, width=1.9, opacity=0.88
        )
        return VGroup(x_axis, y_axis, w1_symbol, w2_symbol, origin_h, origin_v)

    def construct(self) -> None:
        print("D2L 11.8 f(w)=0.1 w1**2 + 2 w2**2")
        print("start w =", W_START, " f =", objective(W_START))
        print("AdaGrad last w =", PATH_ADA[-1], " f =", float(F_ADA[-1]))
        print("RMSProp last w =", PATH_RMS[-1], " f =", float(F_RMS[-1]))
        print("AdaGrad cumulative squares =", S_ADA[-1], " x scale =", float(ALPHA_ADA[-1]))
        print("RMSProp EMA squares =", S_RMS[-1], " x scale =", float(ALPHA_RMS[-1]))

        # 0.00–0.50 s — chapter mark only.
        show_episode_intro(self, "11.8")

        # 0.50–2.50 s — same narrow valley used for both update rules.
        opacities = (0.82, 0.67, 0.51, 0.38, 0.27)
        contours = VGroup(
            *[self.make_contour(level, opacity) for level, opacity in zip(CONTOUR_LEVELS, opacities)]
        )
        axes = self.make_axes()
        self.play(FadeIn(contours), FadeIn(axes), run_time=1.10, rate_func=linear)
        self.wait(0.90)

        w1_tracker = ValueTracker(W_START[0])
        w2_tracker = ValueTracker(W_START[1])

        def traveler_center() -> np.ndarray:
            return self.plot_point(w1_tracker.get_value(), w2_tracker.get_value())

        def make_traveler() -> VGroup:
            """A constant-radius Dot plus halo rings; no Dot is ever scaled."""
            center = traveler_center()
            dot = Dot(center, radius=0.068, color=YELLOW).set_fill(YELLOW, opacity=0.96)
            dot.set_stroke(YELLOW, width=0.0, opacity=0.0)
            outer = Circle(radius=0.276).move_to(center).set_fill(BLACK, opacity=0.0)
            outer.set_stroke(BLUE_A, width=2.05, opacity=0.52)
            inner = Circle(radius=0.178).move_to(center).set_fill(BLACK, opacity=0.0)
            inner.set_stroke(YELLOW, width=2.75, opacity=0.98)
            return VGroup(dot, outer, inner)

        traveler = always_redraw(make_traveler)
        traveler_symbol = Text("w", font_size=23, color=YELLOW)
        traveler_symbol.add_updater(lambda label: label.move_to(traveler_center() + np.array([0.44, 0.34, 0.0])))
        self.add(traveler, traveler_symbol)

        # The right-side empty pocket is the only fixed readout area.
        ada_title = Text("AdaGrad", font_size=29, color=YELLOW).move_to(np.array([4.55, 2.65, 0.0]))
        ada_sum_prefix = Text("Σg²ₜ,₁ =", font_size=21, color=WHITE)
        ada_sum_number = DecimalNumber(
            0.0, num_decimal_places=2, mob_class=Text, include_sign=False, color=YELLOW_A, font_size=22
        )
        ada_sum_row = VGroup(ada_sum_prefix, ada_sum_number).arrange(RIGHT, buff=0.10).move_to(
            np.array([4.62, 1.83, 0.0])
        )
        ada_alpha_prefix = Text("η/√Σg²ₜ,₁ =", font_size=21, color=WHITE)
        ada_alpha_number = DecimalNumber(
            0.0, num_decimal_places=3, mob_class=Text, include_sign=False, color=YELLOW_A, font_size=22
        )
        ada_alpha_row = VGroup(ada_alpha_prefix, ada_alpha_number).arrange(RIGHT, buff=0.10).move_to(
            np.array([4.62, 1.31, 0.0])
        )
        ada_index = ValueTracker(0.0)
        ada_sum_number.add_updater(
            lambda number: number.set_value(np.interp(ada_index.get_value(), np.arange(ADAGRAD_STEPS + 1), S_ADA[:, 0]))
        )
        ada_alpha_number.add_updater(
            lambda number: number.set_value(
                np.interp(ada_index.get_value(), np.arange(ADAGRAD_STEPS + 1), ALPHA_ADA)
            )
        )
        ada_readout = VGroup(ada_title, ada_sum_row, ada_alpha_row)

        # 2.50–4.50 s — establish the start and cumulative-square readout.
        self.play(
            FadeIn(ada_readout),
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.85,
            rate_func=smooth,
        )
        self.wait(1.15)

        ada_segments = [
            Line(self.plot_point(*PATH_ADA[index]), self.plot_point(*PATH_ADA[index + 1])).set_stroke(
                YELLOW, width=3.35, opacity=0.96
            )
            for index in range(ADAGRAD_STEPS)
        ]

        # 4.50–11.70 s — cumulative AdaGrad steps get steadily shorter.
        for index, segment in enumerate(ada_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_ADA[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_ADA[index + 1, 1])),
                ada_index.animate.set_value(float(index + 1)),
                run_time=0.48,
                rate_func=smooth,
            )
            self.wait(0.12)

        ada_tag = Text("AdaGrad", font_size=23, color=YELLOW).move_to(
            self.plot_point(*PATH_ADA[-1]) + np.array([0.20, 0.62, 0.0])
        )
        ada_stall_ring = Circle(radius=0.255).move_to(self.plot_point(*PATH_ADA[-1]))
        ada_stall_ring.set_fill(BLACK, opacity=0.0)
        ada_stall_ring.set_stroke(YELLOW, width=2.55, opacity=0.92)
        # 11.70–13.70 s — leave the visibly stalled trajectory in the valley.
        self.play(
            FadeIn(ada_tag),
            Create(ada_stall_ring),
            Flash(self.plot_point(*PATH_ADA[-1]), color=YELLOW, flash_radius=0.34, line_length=0.09),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(1.30)

        # RMSProp's two-line formula appears once in the same fixed pocket.
        rms_formula_s = Text("sₜ = γsₜ₋₁ + (1−γ)gₜ²", font_size=25, color=WHITE).move_to(
            np.array([4.48, 2.68, 0.0])
        )
        rms_formula_w = Text("wₜ₊₁ = wₜ − ηgₜ/(√sₜ + ε)", font_size=25, color=WHITE).move_to(
            np.array([4.48, 2.19, 0.0])
        )
        gamma_eta = Text("γ = 0.90    η = 0.30", font_size=20, color=BLUE_A).move_to(np.array([4.48, 1.72, 0.0]))
        rms_s_prefix = Text("sₜ,₁ =", font_size=21, color=WHITE)
        rms_s_number = DecimalNumber(
            0.0, num_decimal_places=2, mob_class=Text, include_sign=False, color=BLUE_A, font_size=22
        )
        rms_s_row = VGroup(rms_s_prefix, rms_s_number).arrange(RIGHT, buff=0.10).move_to(np.array([4.48, 1.18, 0.0]))
        rms_alpha_prefix = Text("η/√sₜ,₁ =", font_size=21, color=WHITE)
        rms_alpha_number = DecimalNumber(
            0.0, num_decimal_places=3, mob_class=Text, include_sign=False, color=BLUE_A, font_size=22
        )
        rms_alpha_row = VGroup(rms_alpha_prefix, rms_alpha_number).arrange(RIGHT, buff=0.10).move_to(
            np.array([4.48, 0.68, 0.0])
        )
        rms_index = ValueTracker(0.0)
        rms_s_number.add_updater(
            lambda number: number.set_value(np.interp(rms_index.get_value(), np.arange(RMSPROP_STEPS + 1), S_RMS[:, 0]))
        )
        rms_alpha_number.add_updater(
            lambda number: number.set_value(
                np.interp(rms_index.get_value(), np.arange(RMSPROP_STEPS + 1), ALPHA_RMS)
            )
        )
        rms_readout = VGroup(rms_formula_s, rms_formula_w, gamma_eta, rms_s_row, rms_alpha_row)

        # 13.70–15.70 s — reset only the traveler; preserve AdaGrad's history.
        self.play(
            *[segment.animate.set_stroke(YELLOW, width=2.45, opacity=0.62) for segment in ada_segments],
            FadeOut(ada_readout),
            w1_tracker.animate.set_value(float(W_START[0])),
            w2_tracker.animate.set_value(float(W_START[1])),
            run_time=0.90,
            rate_func=smooth,
        )
        self.play(FadeIn(rms_readout), run_time=0.75, rate_func=smooth)
        self.play(
            Flash(self.plot_point(*W_START), color=BLUE_A, flash_radius=0.36, line_length=0.09),
            run_time=0.35,
            rate_func=smooth,
        )

        rms_segments = [
            Line(self.plot_point(*PATH_RMS[index]), self.plot_point(*PATH_RMS[index + 1])).set_stroke(
                BLUE_A, width=3.55, opacity=0.98
            )
            for index in range(RMSPROP_STEPS)
        ]

        # 15.70–22.90 s — EMA-normalized steps continue along the same valley.
        for index, segment in enumerate(rms_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_RMS[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_RMS[index + 1, 1])),
                rms_index.animate.set_value(float(index + 1)),
                run_time=0.48,
                rate_func=smooth,
            )
            self.wait(0.12)

        rms_tag = Text("RMSProp", font_size=23, color=BLUE_A).move_to(
            self.plot_point(*PATH_RMS[-1]) + np.array([0.74, -0.56, 0.0])
        )
        # 22.90–24.90 s — both real trajectories remain visible for comparison.
        self.play(
            FadeIn(rms_tag),
            Flash(self.plot_point(*PATH_RMS[-1]), color=BLUE_A, flash_radius=0.34, line_length=0.09),
            run_time=0.70,
            rate_func=smooth,
        )
        self.wait(1.30)

        # 24.90–28.00 s — retain contours, both paths, and the parameter traveler.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=3.10, rate_func=linear)
