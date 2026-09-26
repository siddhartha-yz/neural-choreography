"""D2L 11.2 — convex chord above the surface vs a non-convex extra pit.

Every moving value comes from the numpy samples below. Left bowl is
f(x) = 0.5 x²; right bowl is g(x) = cos(π x). The traveler sits on the
chord, never on the surface: z = λx + (1−λ)x′.
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
    always_redraw,
    config,
    linear,
    smooth,
)


config.background_color = BLACK


# D2L 11.2.1.2: f is convex, g is not. Same segment [-1.5, 1] as the book.
def convex_f(x: np.ndarray | float) -> np.ndarray | float:
    """f(x) = 0.5 x²."""
    return 0.5 * np.asarray(x, dtype=float) ** 2


def nonconvex_g(x: np.ndarray | float) -> np.ndarray | float:
    """g(x) = cos(π x)."""
    return np.cos(np.pi * np.asarray(x, dtype=float))


XA = -1.50
XB = 1.00
LAM = 0.40
Z = float(LAM * XA + (1.0 - LAM) * XB)
X_GRID = np.linspace(-2.00, 2.00, 241)
F_GRID = np.asarray(convex_f(X_GRID), dtype=float)
G_GRID = np.asarray(nonconvex_g(X_GRID), dtype=float)
F_A = float(convex_f(XA))
F_B = float(convex_f(XB))
F_Z = float(convex_f(Z))
F_CHORD = float(LAM * F_A + (1.0 - LAM) * F_B)
G_A = float(nonconvex_g(XA))
G_B = float(nonconvex_g(XB))
G_Z = float(nonconvex_g(Z))
G_CHORD = float(LAM * G_A + (1.0 - LAM) * G_B)
PIT_X = -1.00
G_PIT = float(nonconvex_g(PIT_X))
SAMPLE_X = np.array([XA, PIT_X, Z, XB], dtype=float)


def chord_x(lam: float) -> float:
    return float(lam * XA + (1.0 - lam) * XB)


def chord_f(lam: float) -> float:
    return float(lam * F_A + (1.0 - lam) * F_B)


class Episode112(Scene):
    """A ~32-second silent convex-vs-nonconvex visualization."""

    x_min, x_max = -2.05, 2.05
    left_y_min, left_y_max = -0.22, 2.18
    right_y_min, right_y_max = -1.32, 1.32
    left_l, left_r, left_b, left_t = -6.52, -0.72, -3.12, 2.38
    right_l, right_r, right_b, right_t = 0.72, 6.52, -3.12, 2.38

    def map_left(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.left_l + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.left_r - self.left_l
        )
        scene_y = self.left_b + (y_value - self.left_y_min) / (self.left_y_max - self.left_y_min) * (
            self.left_t - self.left_b
        )
        return np.array([scene_x, scene_y, 0.0])

    def map_right(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.right_l + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.right_r - self.right_l
        )
        scene_y = self.right_b + (y_value - self.right_y_min) / (self.right_y_max - self.right_y_min) * (
            self.right_t - self.right_b
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_stroke_curve(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        mapper,
        color,
        width: float,
        opacity: float,
    ) -> VMobject:
        """Stroke-only polyline. Never filled, never set_points_smoothly."""
        points = [mapper(float(x_value), float(y_value)) for x_value, y_value in zip(xs, ys)]
        stroke = VMobject()
        stroke.set_fill(opacity=0.0)
        stroke.set_stroke(color, width=width, opacity=opacity)
        stroke.set_points_as_corners(points)
        return stroke

    def make_axes(self, mapper, y_min: float, y_max: float) -> VGroup:
        x_axis = Line(mapper(self.x_min, 0.0), mapper(self.x_max, 0.0)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        y_axis = Line(mapper(0.0, y_min), mapper(0.0, y_max)).set_stroke(
            BLUE_D, width=2.0, opacity=0.88
        )
        x_symbol = Text("x", font_size=26, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([1.0, -0.15, 0.0]), buff=0.12
        )
        ticks = VGroup()
        labels = VGroup()
        for tick_x, name in ((-1.50, "−1.5"), (1.00, "1")):
            ticks.add(
                Line(mapper(tick_x, -0.06), mapper(tick_x, 0.06)).set_stroke(
                    BLUE_E, width=1.5, opacity=0.70
                )
            )
            labels.add(
                Text(name, font_size=18, color=BLUE_A).move_to(mapper(tick_x, 0.0) + np.array([0.0, -0.28, 0.0]))
            )
        origin = Text("0", font_size=18, color=BLUE_A).move_to(mapper(0.0, 0.0) + np.array([-0.22, -0.28, 0.0]))
        return VGroup(x_axis, y_axis, x_symbol, ticks, labels, origin)

    def endpoint_dot(self, point: np.ndarray) -> Dot:
        return (
            Dot(point, radius=0.055, color=BLUE)
            .set_fill(BLUE, opacity=0.94)
            .set_stroke(BLUE, width=0.0, opacity=0.0)
        )

    def construct(self) -> None:
        print("D2L 11.2  f(x)=0.5*x**2  g(x)=cos(pi*x)")
        print("segment x, x' =", XA, XB)
        print("lambda = {:.2f}  z = {:.6f}".format(LAM, Z))
        print(
            "f samples x={}  f={}  f(z)={:.6f}  chord={:.6f}".format(
                np.array2string(SAMPLE_X, precision=4, separator=", "),
                np.array2string(np.asarray(convex_f(SAMPLE_X), dtype=float), precision=6, separator=", "),
                F_Z,
                F_CHORD,
            )
        )
        print(
            "g samples x={}  g={}  g(z)={:.6f}  chord={:.6f}".format(
                np.array2string(SAMPLE_X, precision=4, separator=", "),
                np.array2string(np.asarray(nonconvex_g(SAMPLE_X), dtype=float), precision=6, separator=", "),
                G_Z,
                G_CHORD,
            )
        )
        print("g extra pit x={:.2f}  g={:.6f}".format(PIT_X, G_PIT))

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "11.2")

        left_curve = self.make_stroke_curve(X_GRID, F_GRID, self.map_left, BLUE, 3.2, 0.96)
        right_curve = self.make_stroke_curve(X_GRID, G_GRID, self.map_right, BLUE, 3.2, 0.96)
        left_axes = self.make_axes(self.map_left, self.left_y_min, self.left_y_max)
        right_axes = self.make_axes(self.map_right, self.right_y_min, self.right_y_max)
        f_label = Text("f = 0.5 x²", font_size=24, color=BLUE_A).move_to(
            self.map_left(-1.55, 2.08) + np.array([0.10, 0.16, 0.0])
        )
        g_label = Text("g = cos(πx)", font_size=24, color=BLUE_A).move_to(
            self.map_right(0.55, 1.12) + np.array([0.78, 0.10, 0.0])
        )

        # 0.40–2.40 s: both bowls, stroke-only.
        self.play(
            FadeIn(left_axes),
            FadeIn(right_axes),
            LaggedStart(Create(left_curve), Create(right_curve), lag_ratio=0.12),
            FadeIn(f_label),
            FadeIn(g_label),
            run_time=2.00,
            rate_func=linear,
        )

        left_x_dot = self.endpoint_dot(self.map_left(XA, F_A))
        left_xp_dot = self.endpoint_dot(self.map_left(XB, F_B))
        left_x_name = Text("x", font_size=22, color=WHITE).move_to(
            self.map_left(XA, F_A) + np.array([-0.46, -0.30, 0.0])
        )
        left_xp_name = Text("x′", font_size=22, color=WHITE).move_to(
            self.map_left(XB, F_B) + np.array([0.42, -0.30, 0.0])
        )
        left_chord = Line(self.map_left(XA, F_A), self.map_left(XB, F_B)).set_stroke(
            YELLOW, width=3.4, opacity=0.95
        )

        # 2.40–4.40 s: the two points on the convex bowl.
        self.play(
            FadeIn(left_x_dot),
            FadeIn(left_xp_dot),
            FadeIn(left_x_name),
            FadeIn(left_xp_name),
            run_time=2.00,
            rate_func=smooth,
        )

        # 4.40–6.40 s: chord stays above f.
        self.play(Create(left_chord), run_time=2.00, rate_func=linear)

        lam_tracker = ValueTracker(0.0)

        def traveler_center() -> np.ndarray:
            lam = lam_tracker.get_value()
            return self.map_left(chord_x(lam), chord_f(lam))

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
        z_glyph = Text("z", font_size=24, color=YELLOW)

        def z_glyph_position() -> np.ndarray:
            x_now = chord_x(lam_tracker.get_value())
            if x_now >= 0.45:
                return traveler_center() + np.array([0.0, 0.52, 0.0])
            if x_now <= -0.85:
                return traveler_center() + np.array([0.50, 0.40, 0.0])
            return traveler_center() + np.array([0.50, 0.38, 0.0])

        z_glyph.add_updater(lambda mob: mob.move_to(z_glyph_position()))

        formula = Text("f(z) ≤ λf(x)+(1−λ)f(x′)", font_size=32, color=WHITE).move_to(
            np.array([1.55, 3.20, 0.0])
        )
        lam_prefix = Text("λ =", font_size=22, color=WHITE)
        lam_number = DecimalNumber(
            0.0,
            num_decimal_places=2,
            mob_class=Text,
            include_sign=False,
            color=YELLOW,
            font_size=22,
        )
        pocket = VGroup(lam_prefix, lam_number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.10).move_to(
            np.array([-5.35, 3.20, 0.0])
        )
        lam_number.add_updater(lambda mob: mob.set_value(lam_tracker.get_value()))

        # 6.40–8.40 s: traveler on the chord; formula once; one pocket.
        self.add(traveler, z_glyph)
        self.play(
            Flash(self.map_left(XB, F_B), color=YELLOW, flash_radius=0.42, line_length=0.11),
            FadeIn(formula),
            FadeIn(pocket),
            run_time=2.00,
            rate_func=smooth,
        )

        # 8.40–10.40 s: walk the whole convex chord.
        self.play(lam_tracker.animate.set_value(1.0), run_time=2.00, rate_func=smooth)

        # 10.40–12.40 s: settle at the book’s λ = 0.40 (z = 0).
        self.play(lam_tracker.animate.set_value(LAM), run_time=2.00, rate_func=smooth)

        left_gap = Line(self.map_left(Z, F_Z), self.map_left(Z, F_CHORD)).set_stroke(
            YELLOW, width=4.0, opacity=0.98
        )

        # 12.40–14.40 s: Jensen gap — chord above the bowl.
        self.play(Create(left_gap), run_time=2.00, rate_func=linear)

        right_x_dot = self.endpoint_dot(self.map_right(XA, G_A))
        right_xp_dot = self.endpoint_dot(self.map_right(XB, G_B))
        right_x_name = Text("x", font_size=22, color=WHITE).move_to(
            self.map_right(XA, G_A) + np.array([-0.40, 0.34, 0.0])
        )
        right_xp_name = Text("x′", font_size=22, color=WHITE).move_to(
            self.map_right(XB, G_B) + np.array([0.42, 0.24, 0.0])
        )
        right_chord = Line(self.map_right(XA, G_A), self.map_right(XB, G_B)).set_stroke(
            YELLOW, width=3.4, opacity=0.95
        )

        # 14.40–16.40 s: same segment on the non-convex bowl.
        self.play(
            FadeIn(right_x_dot),
            FadeIn(right_xp_dot),
            Create(right_chord),
            run_time=2.00,
            rate_func=smooth,
        )

        pit_center = self.map_right(PIT_X, G_PIT)
        pit_ring = Circle(radius=0.16).move_to(pit_center).set_fill(opacity=0.0).set_stroke(
            WHITE, width=2.0, opacity=0.90
        )
        pit_plus_h = Line(
            pit_center + np.array([-0.11, 0.0, 0.0]),
            pit_center + np.array([0.11, 0.0, 0.0]),
        ).set_stroke(WHITE, width=2.0, opacity=0.90)
        pit_plus_v = Line(
            pit_center + np.array([0.0, -0.11, 0.0]),
            pit_center + np.array([0.0, 0.11, 0.0]),
        ).set_stroke(WHITE, width=2.0, opacity=0.90)
        pit_mark = VGroup(pit_ring, pit_plus_h, pit_plus_v)

        # 16.40–18.40 s: extra pit between the same two points.
        self.play(
            FadeIn(right_x_name),
            FadeIn(right_xp_name),
            FadeIn(pit_mark),
            Flash(pit_center, color=WHITE, flash_radius=0.36, line_length=0.10),
            run_time=2.00,
            rate_func=smooth,
        )

        right_gap = Line(self.map_right(Z, G_CHORD), self.map_right(Z, G_Z)).set_stroke(
            YELLOW, width=4.0, opacity=0.98
        )

        # 18.40–20.40 s: surface sits above the chord — convexity fails.
        self.play(Create(right_gap), run_time=2.00, rate_func=linear)

        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.00, rate_func=linear)
