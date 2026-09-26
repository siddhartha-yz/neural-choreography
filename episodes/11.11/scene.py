"""D2L 11.11 — one descent path whose step size follows a schedule.

Every moving value comes from the numpy arrays below. The traveler is the
parameter point w on the same bowl as 11.3: f(w) = w₁² + 2 w₂². Updates are
exactly w ← w − η(t) ∇f. η(t) is cosine decay after a linear warmup — one
schedule, not an inventory. Contours are stroke-only. Kernels are not trained.
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


# Same 11.3 bowl. Column vectors. One traveler, one schedule.
W_START = np.array([[-5.0], [-2.0]], dtype=float)
MAX_UPDATE = 10
WARMUP_STEPS = 3
BASE_LR = 0.18
FINAL_LR = 0.02
WARMUP_BEGIN_LR = 0.03
STEPS = 10
CONTOUR_LEVELS = (2.0, 6.0, 12.0, 22.0, 33.0)


def objective(weights: np.ndarray) -> float:
    """f(w) = w₁² + 2 w₂². ``weights`` is a 2 × 1 column."""
    return float(weights[0, 0] ** 2 + 2.0 * weights[1, 0] ** 2)


def gradient(weights: np.ndarray) -> np.ndarray:
    """∇f(w) = (2 w₁, 4 w₂) as a 2 × 1 column."""
    return np.array([[2.0 * weights[0, 0]], [4.0 * weights[1, 0]]], dtype=float)


def eta_of(step: float) -> float:
    """D2L CosineScheduler: linear warmup, then cosine to η_T."""
    time = float(step)
    if time < WARMUP_STEPS:
        return WARMUP_BEGIN_LR + (BASE_LR - WARMUP_BEGIN_LR) * time / WARMUP_STEPS
    if time <= MAX_UPDATE:
        cosine = np.cos(np.pi * (time - WARMUP_STEPS) / (MAX_UPDATE - WARMUP_STEPS))
        return FINAL_LR + (BASE_LR - FINAL_LR) * (1.0 + float(cosine)) / 2.0
    return FINAL_LR


def scheduled_descent(start: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w ← w − η(t) ∇f. Returns path (steps+1, 2), f values, and η at each index."""
    path = np.zeros((steps + 1, 2), dtype=float)
    values = np.zeros(steps + 1, dtype=float)
    rates = np.zeros(steps + 1, dtype=float)
    weights = np.array(start, dtype=float, copy=True)
    path[0] = weights[:, 0]
    values[0] = objective(weights)
    rates[0] = eta_of(0.0)
    for index in range(steps):
        weights = weights - eta_of(index) * gradient(weights)
        path[index + 1] = weights[:, 0]
        values[index + 1] = objective(weights)
        rates[index + 1] = eta_of(index + 1)
    return path, values, rates


PATH, F_PATH, ETAS = scheduled_descent(W_START, STEPS)


class Episode1111(Scene):
    """A ~33-second silent warmup-then-cosine descent."""

    w1_min, w1_max = -6.05, 1.85
    w2_min, w2_max = -3.15, 2.25
    plot_left, plot_right = -2.45, 6.25
    plot_bottom, plot_top = -3.20, 2.72
    sched_left, sched_right = -6.52, -3.22
    sched_bottom, sched_top = -2.35, 1.48
    eta_y_max = BASE_LR * 1.12

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

    def sched_point(self, time: float, rate: float) -> np.ndarray:
        scene_x = self.sched_left + time / MAX_UPDATE * (self.sched_right - self.sched_left)
        scene_y = self.sched_bottom + rate / self.eta_y_max * (self.sched_top - self.sched_bottom)
        return np.array([scene_x, scene_y, 0.0])

    def make_contour(self, level: float, opacity: float) -> VGroup:
        """Stroke-only level set. Never filled."""
        theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=True)
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
            stroke.set_points_as_corners(points)
            stroke.set_fill(BLACK, opacity=0.0)
            stroke.set_stroke(BLUE_E, width=1.5, opacity=opacity)
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

    def make_schedule(self) -> VGroup:
        """Stroke-only η(t): warmup ramp, then cosine. Labels sit on the axes."""
        times = np.linspace(0.0, float(MAX_UPDATE), 240)
        points = [self.sched_point(time, eta_of(time)) for time in times]
        curve = VMobject()
        curve.set_points_as_corners(points)
        curve.set_fill(BLACK, opacity=0.0)
        curve.set_stroke(BLUE_A, width=2.6, opacity=0.92)

        t_axis = Line(
            self.sched_point(0.0, 0.0),
            self.sched_point(float(MAX_UPDATE), 0.0),
        ).set_stroke(BLUE_D, width=1.8, opacity=0.88)
        eta_axis = Line(
            self.sched_point(0.0, 0.0),
            self.sched_point(0.0, self.eta_y_max),
        ).set_stroke(BLUE_D, width=1.8, opacity=0.88)
        t_symbol = Text("t", font_size=22, color=BLUE_A).next_to(
            t_axis.get_end(), np.array([1.0, 0.0, 0.0]), buff=0.12
        )
        eta_symbol = Text("η", font_size=22, color=BLUE_A).next_to(
            eta_axis.get_end(), np.array([0.0, 1.0, 0.0]), buff=0.10
        )

        warmup_x = self.sched_point(float(WARMUP_STEPS), 0.0)
        warmup_tick = Line(
            warmup_x + np.array([0.0, -0.08, 0.0]),
            warmup_x + np.array([0.0, 0.10, 0.0]),
        ).set_stroke(YELLOW, width=1.8, opacity=0.90)
        warmup_tick.set_fill(opacity=0.0)
        return VGroup(t_axis, eta_axis, curve, t_symbol, eta_symbol, warmup_tick)

    def construct(self) -> None:
        print("D2L 11.11  f(w)=w1**2 + 2 w2**2")
        print("start w =\n{}".format(np.array2string(W_START, precision=1, separator=", ")))
        print(
            "schedule warmup={} T={} η0={:.2f} ηT={:.2f} ηw={:.2f}".format(
                WARMUP_STEPS, MAX_UPDATE, BASE_LR, FINAL_LR, WARMUP_BEGIN_LR
            )
        )
        for index, (weights, value, rate) in enumerate(zip(PATH, F_PATH, ETAS)):
            column = np.array([[weights[0]], [weights[1]]], dtype=float)
            print(
                "  t={:d}  η={:.4f}  w=\n{}  f={:.4f}".format(
                    index,
                    rate,
                    np.array2string(column, precision=5, separator=", "),
                    value,
                )
            )
        print("D2L 11.11 ETAS={}".format(np.array2string(ETAS, precision=4, separator=", ")))
        print("D2L 11.11 last w={} f={:.4f}".format(PATH[-1], float(F_PATH[-1])))

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "11.11")

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

        # Finish the bowl before t = 1 so the first QA frame is whole ellipses.
        self.play(FadeIn(contours), FadeIn(axes), run_time=0.50, rate_func=linear)
        self.play(FadeIn(contour_labels), run_time=0.25, rate_func=linear)

        t_tracker = ValueTracker(0.0)
        w1_tracker = ValueTracker(float(W_START[0, 0]))
        w2_tracker = ValueTracker(float(W_START[1, 0]))

        def traveler_center() -> np.ndarray:
            return self.plot_point(w1_tracker.get_value(), w2_tracker.get_value())

        def draw_traveler() -> VGroup:
            """7.4 / 11.3 halo: concentric stroke rings. Dot radius never changes."""
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
        w_glyph.add_updater(lambda mob: mob.move_to(traveler_center() + np.array([0.42, 0.34, 0.0])))

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
            lambda mob: mob.set_value(
                objective(np.array([[w1_tracker.get_value()], [w2_tracker.get_value()]]))
            )
        )

        def f_prefix_anchor() -> np.ndarray:
            """Keep f(w) off the incoming trail. The path always arrives from −w₁."""
            center = traveler_center()
            if w1_tracker.get_value() > -1.20:
                return center + np.array([0.12, -0.80, 0.0])
            return center + np.array([0.92, -0.56, 0.0])

        f_prefix.add_updater(lambda mob: mob.move_to(f_prefix_anchor()))
        f_number.add_updater(lambda mob: mob.next_to(f_prefix, np.array([1.0, 0.0, 0.0]), buff=0.10))

        formula = Text("w ← w − η(t) ∇f", font_size=34, color=WHITE).move_to(
            np.array([1.55, 3.24, 0.0])
        )

        eta_prefix = Text("η =", font_size=24, color=WHITE)
        eta_number = DecimalNumber(
            eta_of(0.0),
            num_decimal_places=2,
            mob_class=Text,
            include_sign=False,
            color=YELLOW,
            font_size=24,
        )
        eta_number.add_updater(lambda mob: mob.set_value(eta_of(t_tracker.get_value())))
        eta_row = VGroup(eta_prefix, eta_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.12)
        eta_row.move_to(
            np.array(
                [0.5 * (self.sched_left + self.sched_right), self.sched_top + 0.72, 0.0]
            )
        )

        schedule = self.make_schedule()

        def draw_eta_mark() -> Circle:
            time = t_tracker.get_value()
            mark = Circle(radius=0.11).move_to(self.sched_point(time, eta_of(time)))
            mark.set_fill(BLACK, opacity=0.0)
            mark.set_stroke(YELLOW, width=2.6, opacity=0.96)
            return mark

        eta_mark = always_redraw(draw_eta_mark)

        segments = [
            Line(self.plot_point(*PATH[index]), self.plot_point(*PATH[index + 1])).set_stroke(
                YELLOW, width=3.4, opacity=0.95
            )
            for index in range(STEPS)
        ]

        self.add(traveler, w_glyph, f_prefix, f_number)
        self.play(
            Flash(
                self.plot_point(float(W_START[0, 0]), float(W_START[1, 0])),
                color=YELLOW,
                flash_radius=0.42,
                line_length=0.11,
            ),
            FadeIn(formula),
            FadeIn(eta_row),
            FadeIn(schedule),
            run_time=1.00,
            rate_func=smooth,
        )
        self.add(eta_mark)
        self.wait(2.30)

        # Ten scheduled steps. η in the pocket tracks t; step k uses η(k).
        for index, segment in enumerate(segments):
            self.play(
                Create(segment),
                w1_tracker.animate.set_value(float(PATH[index + 1, 0])),
                w2_tracker.animate.set_value(float(PATH[index + 1, 1])),
                run_time=1.35,
                rate_func=smooth,
            )
            self.play(
                t_tracker.animate.set_value(float(index + 1)),
                run_time=0.25,
                rate_func=linear,
            )

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.70, rate_func=linear)
