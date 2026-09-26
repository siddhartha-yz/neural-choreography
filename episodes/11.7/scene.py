"""D2L 11.7 — AdaGrad versus vanilla GD on a narrow quadratic valley.

Every moving value comes from the numpy paths below. The traveler is the
parameter point w on f(w) = 0.1 w₁² + 2 w₂². Contours are stroke-only
level sets. Beat 1 is w ← w − η∇f. Beat 2 is s ← s + (∇f)²,
w ← w − η ∇f / √s. Same η; the steep axis is the one 1/√s damps.
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


# D2L 11.6 / 11.7 valley: f(x) = 0.1 x₁² + 2 x₂², start [-5, -2].
# Same η for both beats so the new writing is s and 1/√s.
W_START = np.array([-5.0, -2.0], dtype=float)
ETA = 0.40
EPS = 1e-6
STEPS_GD = 10
STEPS_ADA = 12
CONTOUR_LEVELS = (0.4, 1.0, 2.5, 5.0, 10.5)


def objective(weights: np.ndarray) -> float:
    """f(w) = 0.1 w₁² + 2 w₂²."""
    return float(0.1 * weights[0] ** 2 + 2.0 * weights[1] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (0.2 w₁, 4 w₂)."""
    return np.array([0.2 * weights[0], 4.0 * weights[1]], dtype=float)


def gradient_descent(start: np.ndarray, eta: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
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


def adagrad_descent(
    start: np.ndarray, eta: float, steps: int, eps: float = EPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """AdaGrad: s ← s + g⊙g, w ← w − η g / √(s+ε). Also return s and 1/√s."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    accum = np.zeros((steps + 1, 2), dtype=float)
    inv_sqrt = np.zeros((steps + 1, 2), dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    state = np.zeros(2, dtype=float)
    path[0] = weights
    values[0] = objective(weights)
    for index in range(steps):
        grad = gradient(weights)
        state = state + grad * grad
        scale = 1.0 / np.sqrt(state + eps)
        weights = weights - eta * scale * grad
        path[index + 1] = weights
        values[index + 1] = objective(weights)
        accum[index + 1] = state
        inv_sqrt[index + 1] = scale
    return path, values, accum, inv_sqrt


PATH_GD, F_GD = gradient_descent(W_START, ETA, STEPS_GD)
PATH_ADA, F_ADA, S_ADA, INV_SQRT_S = adagrad_descent(W_START, ETA, STEPS_ADA)


class Episode117(Scene):
    """A ~32-second silent AdaGrad-versus-GD visualization."""

    w1_min, w1_max = -6.35, 0.55
    w2_min, w2_max = -2.55, 2.15
    plot_left, plot_right = -3.55, 5.65
    plot_bottom, plot_top = -3.15, 2.85

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
        print("D2L 11.7  f(w)=0.1 w1**2 + 2 w2**2")
        print("start w =", W_START, "  f =", objective(W_START))
        print("η = {:.2f}  GD path:".format(ETA))
        for index, (weights, value) in enumerate(zip(PATH_GD, F_GD)):
            print(
                "  k={:d}  w=[{:+.5f}, {:+.5f}]  f={:.6f}".format(
                    index, weights[0], weights[1], value
                )
            )
        print("η = {:.2f}  AdaGrad path:".format(ETA))
        for index, (weights, value) in enumerate(zip(PATH_ADA, F_ADA)):
            extra = ""
            if index > 0:
                extra = "  s=[{:.3f}, {:.3f}]  1/√s=[{:.4f}, {:.4f}]".format(
                    S_ADA[index, 0],
                    S_ADA[index, 1],
                    INV_SQRT_S[index, 0],
                    INV_SQRT_S[index, 1],
                )
            print(
                "  k={:d}  w=[{:+.5f}, {:+.5f}]  f={:.6f}{}".format(
                    index, weights[0], weights[1], value, extra
                )
            )
        print(
            "D2L 11.7 last GD w=[{:+.6f}, {:+.6f}]  f={:.6f}".format(
                PATH_GD[-1, 0], PATH_GD[-1, 1], float(F_GD[-1])
            )
        )
        print(
            "D2L 11.7 last AdaGrad w=[{:+.6f}, {:+.6f}]  f={:.6f}".format(
                PATH_ADA[-1, 0], PATH_ADA[-1, 1], float(F_ADA[-1])
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "11.7")

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
            label = Text(
                f"{level:g}",
                font_size=16,
                color=BLUE_A,
            ).move_to(position)
            contour_labels.add(label)

        self.play(
            FadeIn(contours),
            FadeIn(axes),
            run_time=1.10,
            rate_func=linear,
        )
        self.play(FadeIn(contour_labels), run_time=0.25, rate_func=linear)

        w1_tracker = ValueTracker(W_START[0])
        w2_tracker = ValueTracker(W_START[1])
        beat_tracker = ValueTracker(0.0)
        inv_s1_tracker = ValueTracker(float(INV_SQRT_S[1, 0]))
        inv_s2_tracker = ValueTracker(float(INV_SQRT_S[1, 1]))

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
            if beat_tracker.get_value() > 0.5:
                if w1_tracker.get_value() > -3.35:
                    return traveler_center() + np.array([0.54, 0.12, 0.0])
                return traveler_center() + np.array([-0.46, 0.36, 0.0])
            return traveler_center() + np.array([0.40, 0.34, 0.0])

        w_glyph.add_updater(lambda mob: mob.move_to(w_glyph_position()))

        f_prefix = Text("f(w) =", font_size=20, color=WHITE)
        f_number = DecimalNumber(
            objective(W_START),
            num_decimal_places=2,
            mob_class=Text,
            include_sign=False,
            color=YELLOW,
            font_size=20,
        )
        f_number.add_updater(
            lambda mob: mob.set_value(objective(np.array([w1_tracker.get_value(), w2_tracker.get_value()])))
        )

        def f_prefix_anchor() -> np.ndarray:
            """GD: ahead of the zigzag. AdaGrad: above the trail while far left, then below-right."""
            center = traveler_center()
            w1 = w1_tracker.get_value()
            if beat_tracker.get_value() > 0.5:
                if w1 > -3.35:
                    return center + np.array([0.78, -0.56, 0.0])
                return center + np.array([0.72, 0.60, 0.0])
            if w1 > -1.15:
                return center + np.array([-1.72, -0.58, 0.0])
            return center + np.array([0.78, -0.52, 0.0])

        f_prefix.add_updater(lambda mob: mob.move_to(f_prefix_anchor()))
        f_number.add_updater(
            lambda mob: mob.next_to(f_prefix, np.array([1.0, 0.0, 0.0]), buff=0.10)
        )

        gd_formula = Text("w ← w − η∇f", font_size=34, color=WHITE).move_to(np.array([1.55, 3.22, 0.0]))
        eta_label = Text("η = 0.4", font_size=24, color=YELLOW).next_to(
            gd_formula, np.array([-1.0, 0.0, 0.0]), buff=0.45
        )
        ada_formula_s = Text("s ← s + (∇f)²", font_size=32, color=WHITE).move_to(np.array([1.70, 3.38, 0.0]))
        ada_formula_w = Text("w ← w − η ∇f / √s", font_size=32, color=WHITE).move_to(
            np.array([1.70, 2.88, 0.0])
        )

        # Per-coordinate 1/√s parked on the axis-end glyphs, clear of paths and contour numbers.
        inv_s1_prefix = Text("1/√s₁", font_size=20, color=YELLOW).move_to(
            self.plot_point(-0.72, 0.0) + np.array([0.0, -0.52, 0.0])
        )
        inv_s1_number = DecimalNumber(
            float(INV_SQRT_S[1, 0]),
            num_decimal_places=2,
            mob_class=Text,
            include_sign=False,
            color=YELLOW,
            font_size=20,
        )
        inv_s1_number.add_updater(lambda mob: mob.set_value(inv_s1_tracker.get_value()))
        inv_s1_number.add_updater(
            lambda mob: mob.next_to(inv_s1_prefix, np.array([1.0, 0.0, 0.0]), buff=0.10)
        )
        inv_s2_prefix = Text("1/√s₂", font_size=20, color=YELLOW).move_to(
            self.plot_point(0.0, 1.78) + np.array([1.02, 0.18, 0.0])
        )
        inv_s2_number = DecimalNumber(
            float(INV_SQRT_S[1, 1]),
            num_decimal_places=2,
            mob_class=Text,
            include_sign=False,
            color=YELLOW,
            font_size=20,
        )
        inv_s2_number.add_updater(lambda mob: mob.set_value(inv_s2_tracker.get_value()))
        inv_s2_number.add_updater(
            lambda mob: mob.next_to(inv_s2_prefix, np.array([0.0, -1.0, 0.0]), buff=0.08)
        )

        gd_segments = [
            Line(self.plot_point(*PATH_GD[index]), self.plot_point(*PATH_GD[index + 1])).set_stroke(
                YELLOW, width=3.4, opacity=0.95
            )
            for index in range(STEPS_GD)
        ]
        ada_segments = [
            Line(self.plot_point(*PATH_ADA[index]), self.plot_point(*PATH_ADA[index + 1])).set_stroke(
                YELLOW, width=3.6, opacity=0.98
            )
            for index in range(STEPS_ADA)
        ]

        self.add(traveler, w_glyph, f_prefix, f_number)
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.42, line_length=0.11),
            FadeIn(gd_formula),
            FadeIn(eta_label),
            run_time=0.80,
            rate_func=smooth,
        )
        self.wait(0.55)

        # Beat 1: vanilla GD zigzags across the steep axis.
        for index, segment in enumerate(gd_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_GD[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_GD[index + 1, 1])),
                run_time=0.78,
                rate_func=smooth,
            )
        self.wait(0.60)

        self.play(
            *[segment.animate.set_stroke(YELLOW, width=2.2, opacity=0.30) for segment in gd_segments],
            run_time=0.35,
            rate_func=linear,
        )
        self.play(FadeOut(gd_formula), run_time=0.28, rate_func=linear)
        self.play(
            w1_tracker.animate.set_value(float(W_START[0])),
            w2_tracker.animate.set_value(float(W_START[1])),
            run_time=0.70,
            rate_func=smooth,
        )
        self.play(
            FadeIn(ada_formula_s),
            FadeIn(ada_formula_w),
            FadeIn(inv_s1_prefix),
            FadeIn(inv_s1_number),
            FadeIn(inv_s2_prefix),
            FadeIn(inv_s2_number),
            beat_tracker.animate.set_value(1.0),
            run_time=0.75,
            rate_func=smooth,
        )
        self.play(
            Flash(self.plot_point(*W_START), color=YELLOW, flash_radius=0.36, line_length=0.10),
            run_time=0.32,
            rate_func=smooth,
        )

        # Beat 2: AdaGrad damps the axis that has already walked (w₂).
        for index, segment in enumerate(ada_segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH_ADA[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH_ADA[index + 1, 1])),
                inv_s1_tracker.animate.set_value(float(INV_SQRT_S[index + 1, 0])),
                inv_s2_tracker.animate.set_value(float(INV_SQRT_S[index + 1, 1])),
                run_time=0.68,
                rate_func=smooth,
            )
        self.wait(0.70)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=8.60, rate_func=linear)
