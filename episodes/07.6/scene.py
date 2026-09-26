"""D2L 7.6 — a residual block: F(x) on the main path, x on a bypass.

Every plotted number comes from the arrays below.  F is two tiny affine
stages with a ReLU in between, untrained.  The shortcut copies x and never
enters F.  y = relu(F(x) + x).  When F(x) is small, y stays next to x.
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
    Rectangle,
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
# One numerical source of truth.  Traveler x is the same visual identity as
# 3.4 (a 2-vector).  F is two affine stages; weights are small on purpose so
# F(x) ≈ 0 and y ≈ x is readable on the halo.
# ---------------------------------------------------------------------------
INPUT_X = np.array([[1.00], [0.50]], dtype=float)
WEIGHTS_1 = np.array(
    [
        [0.140, 0.030],
        [-0.110, -0.090],
    ],
    dtype=float,
)
BIAS_1 = np.array([[-0.010], [-0.025]], dtype=float)
WEIGHTS_2 = np.array(
    [
        [0.200, 0.070],
        [-0.160, 0.140],
    ],
    dtype=float,
)
BIAS_2 = np.array([[0.040], [-0.045]], dtype=float)


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


def residual_block(x: np.ndarray) -> dict[str, np.ndarray]:
    """Untrained residual block: F = W₂ relu(W₁x + b₁) + b₂, y = relu(F + x)."""
    hidden_pre = WEIGHTS_1 @ x + BIAS_1
    hidden = relu(hidden_pre)
    residual = WEIGHTS_2 @ hidden + BIAS_2
    summed = residual + x
    output = relu(summed)
    return {
        "z": hidden_pre,
        "h": hidden,
        "F": residual,
        "s": summed,
        "y": output,
    }


BLOCK = residual_block(INPUT_X)
RESIDUAL_F = np.asarray(BLOCK["F"], dtype=float)
SUMMED = np.asarray(BLOCK["s"], dtype=float)
OUTPUT_Y = np.asarray(BLOCK["y"], dtype=float)


def signed_glyphs(value: float, digits: int = 3) -> str:
    """Match 3.1: a true minus glyph, never a hyphen-minus."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value):.{digits}f}"


class Episode076(Scene):
    """A ~30-second continuous, silent residual-block visualization."""

    x_min, x_max = -2.2, 2.2
    y_min, y_max = -2.0, 2.0
    # Narrow left plot: the residual block needs the whole right half.
    plot_left, plot_right = -5.90, -1.55
    plot_bottom, plot_top = -2.70, 2.55
    f_scale = 5.6

    def plot_point(self, x_value: float, y_value: float) -> np.ndarray:
        scene_x = self.plot_left + (x_value - self.x_min) / (self.x_max - self.x_min) * (
            self.plot_right - self.plot_left
        )
        scene_y = self.plot_bottom + (y_value - self.y_min) / (self.y_max - self.y_min) * (
            self.plot_top - self.plot_bottom
        )
        return np.array([scene_x, scene_y, 0.0])

    def make_grid_and_axes(self) -> tuple[VGroup, VGroup]:
        """Stroke-only Line objects: never a filled VMobject mesh."""
        grid_lines = VGroup()
        for x_value in np.arange(-2.0, 2.01, 0.5):
            if abs(x_value) > 1e-8:
                grid_lines.add(
                    Line(self.plot_point(x_value, self.y_min), self.plot_point(x_value, self.y_max)).set_stroke(
                        BLUE_E, width=1.0, opacity=0.30
                    )
                )
        for y_value in np.arange(-2.0, 2.01, 0.5):
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
        x_symbol = Text("x₁", font_size=24, color=BLUE_A).next_to(
            x_axis.get_end(), np.array([-1.0, -1.0, 0.0]), buff=0.10
        )
        y_symbol = Text("x₂", font_size=24, color=BLUE_A).next_to(
            y_axis.get_end(), np.array([1.0, -1.0, 0.0]), buff=0.10
        )
        return grid_lines, VGroup(x_axis, y_axis, x_symbol, y_symbol)

    def construct(self) -> None:
        x_flat = INPUT_X.ravel()
        f_flat = RESIDUAL_F.ravel()
        s_flat = SUMMED.ravel()
        y_flat = OUTPUT_Y.ravel()
        print(
            "D2L 7.6 x={}, F(x)={}, x+F(x)={}, y={}".format(
                np.array2string(x_flat, precision=6, separator=", "),
                np.array2string(f_flat, precision=6, separator=", "),
                np.array2string(s_flat, precision=6, separator=", "),
                np.array2string(y_flat, precision=6, separator=", "),
            )
        )
        print(
            "D2L 7.6 z={}, h={}, |F|={:.6f}".format(
                np.array2string(np.asarray(BLOCK["z"]).ravel(), precision=6, separator=", "),
                np.array2string(np.asarray(BLOCK["h"]).ravel(), precision=6, separator=", "),
                float(np.linalg.norm(f_flat)),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "07.6")

        grid, axes = self.make_grid_and_axes()
        class_points = (
            (BLUE, [(1.00, 0.50), (0.72, 0.82), (1.35, 0.84), (1.52, 0.24), (0.62, 0.20)]),
            (YELLOW, [(-1.48, 0.72), (-1.12, 1.08), (-1.78, 0.38), (-0.88, 0.52)]),
        )
        point_cloud = VGroup(
            *[
                Dot(self.plot_point(x_value, y_value), radius=0.068, color=color)
                .set_fill(color, opacity=0.94)
                .set_stroke(color, width=0.0, opacity=0.0)
                for color, points in class_points
                for x_value, y_value in points
            ]
        )

        traveler_center = self.plot_point(x_flat[0], x_flat[1])
        halo_outer = Circle(radius=0.265).move_to(traveler_center).set_stroke(BLUE_A, width=2.1, opacity=0.48)
        halo_inner = Circle(radius=0.172).move_to(traveler_center).set_stroke(YELLOW, width=2.8, opacity=0.98)
        sample_x1_label = Text("x₁ = 1.00", font_size=22, color=BLUE_A).move_to(np.array([-5.00, -3.10, 0.0]))
        sample_x2_label = Text("x₂ = 0.50", font_size=22, color=BLUE_A).move_to(np.array([-5.00, -3.55, 0.0]))
        # Two columns below the plot keep signed readouts inside the frame.
        # y ≈ x is the residual punchline.
        y1_on_traveler = Text(
            f"y₁ = {signed_glyphs(y_flat[0])}",
            font_size=22,
            color=YELLOW,
        ).move_to(np.array([-2.70, -3.10, 0.0]))
        y2_on_traveler = Text(
            f"y₂ = {signed_glyphs(y_flat[1])}",
            font_size=22,
            color=YELLOW,
        ).move_to(np.array([-2.70, -3.55, 0.0]))

        formula = Text("y = relu(F(x) + x)", font_size=34, color=WHITE).move_to(np.array([3.55, 3.14, 0.0]))

        fork = np.array([0.08, 0.18, 0.0])
        f_box_center = np.array([2.42, 1.48, 0.0])
        f_box = Rectangle(width=3.20, height=1.92).move_to(f_box_center)
        f_box.set_fill(BLACK, opacity=0.0)
        f_box.set_stroke(BLUE_D, width=2.2, opacity=0.92)
        f_name = Text("F", font_size=26, color=BLUE_A).move_to(f_box_center + np.array([-1.28, 0.72, 0.0]))
        w1_mark = Text("W₁", font_size=24, color=BLUE_A).move_to(f_box_center + np.array([-0.92, 0.0, 0.0]))
        inner_relu = Text("relu", font_size=22, color=WHITE).move_to(f_box_center)
        w2_mark = Text("W₂", font_size=24, color=BLUE_A).move_to(f_box_center + np.array([0.92, 0.0, 0.0]))
        inner_left = Line(f_box_center + np.array([-0.58, 0.0, 0.0]), f_box_center + np.array([-0.28, 0.0, 0.0])).set_stroke(
            BLUE_D, width=2.0, opacity=0.80
        )
        inner_right = Line(f_box_center + np.array([0.28, 0.0, 0.0]), f_box_center + np.array([0.58, 0.0, 0.0])).set_stroke(
            BLUE_D, width=2.0, opacity=0.80
        )
        inner_stages = VGroup(w1_mark, inner_left, inner_relu, inner_right, w2_mark)

        f_left = np.array([f_box_center[0] - 1.60, f_box_center[1], 0.0])
        f_right = np.array([f_box_center[0] + 1.60, f_box_center[1], 0.0])
        add_center = np.array([4.72, 0.18, 0.0])
        relu_center = np.array([5.78, 0.18, 0.0])
        y_center = np.array([6.62, 0.18, 0.0])
        shortcut_low = -1.72

        into_f = Line(fork, f_left).set_stroke(BLUE_D, width=2.4, opacity=0.78)
        out_of_f = Line(f_right, np.array([add_center[0], f_box_center[1], 0.0])).set_stroke(
            BLUE_D, width=2.4, opacity=0.78
        )
        down_to_add = Line(np.array([add_center[0], f_box_center[1], 0.0]), add_center).set_stroke(
            BLUE_D, width=2.4, opacity=0.78
        )
        main_path = VGroup(into_f, out_of_f, down_to_add)

        shortcut = VMobject(fill_opacity=0.0, stroke_opacity=0.96)
        shortcut.set_points_as_corners(
            [
                fork,
                np.array([fork[0], shortcut_low, 0.0]),
                np.array([add_center[0], shortcut_low, 0.0]),
                add_center,
            ]
        )
        shortcut.set_fill(BLACK, opacity=0.0)
        shortcut.set_stroke(YELLOW, width=4.4, opacity=0.96)
        shortcut_x_label = Text("x", font_size=26, color=YELLOW).move_to(
            np.array([(fork[0] + add_center[0]) / 2.0, shortcut_low - 0.38, 0.0])
        )

        add_ring = Circle(radius=0.22).move_to(add_center).set_stroke(WHITE, width=2.2, opacity=0.95)
        add_ring.set_fill(BLACK, opacity=0.0)
        add_sign = Text("+", font_size=28, color=WHITE).move_to(add_center)
        relu_box = Rectangle(width=0.92, height=0.58).move_to(relu_center)
        relu_box.set_fill(BLACK, opacity=0.0)
        relu_box.set_stroke(BLUE_D, width=2.0, opacity=0.90)
        relu_word = Text("relu", font_size=20, color=WHITE).move_to(relu_center)
        into_relu = Line(add_center + np.array([0.22, 0.0, 0.0]), relu_center + np.array([-0.46, 0.0, 0.0])).set_stroke(
            BLUE_D, width=2.2, opacity=0.78
        )
        into_y = Line(relu_center + np.array([0.46, 0.0, 0.0]), y_center + np.array([-0.22, 0.0, 0.0])).set_stroke(
            BLUE_D, width=2.2, opacity=0.78
        )
        y_ring = Circle(radius=0.22).move_to(y_center).set_stroke(YELLOW, width=2.4, opacity=0.95)
        y_ring.set_fill(BLACK, opacity=0.0)
        y_sign = Text("y", font_size=24, color=YELLOW).move_to(y_center)

        connector = Line(traveler_center + np.array([0.30, 0.0, 0.0]), fork).set_stroke(
            BLUE_D, width=2.2, opacity=0.62
        )

        f_origin = np.array([4.18, f_box_center[1], 0.0])
        f_bar_x = (f_origin[0] - 0.20, f_origin[0] + 0.20)
        f_reveal = ValueTracker(0.0)
        f_colors = (BLUE, YELLOW)

        def f_bar_points(index: int) -> tuple[np.ndarray, np.ndarray]:
            start = np.array([f_bar_x[index], f_origin[1], 0.0])
            end = np.array(
                [
                    f_bar_x[index],
                    f_origin[1] + f_flat[index] * self.f_scale * f_reveal.get_value(),
                    0.0,
                ]
            )
            return start, end

        f_bars = VGroup(
            *[
                always_redraw(
                    lambda index=index: Line(*f_bar_points(index)).set_stroke(
                        f_colors[index], width=10.0, opacity=0.96
                    )
                )
                for index in range(2)
            ]
        )
        f_zero = Line(
            np.array([f_bar_x[0] - 0.18, f_origin[1], 0.0]),
            np.array([f_bar_x[1] + 0.18, f_origin[1], 0.0]),
        ).set_stroke(BLUE_D, width=1.6, opacity=0.80)

        def make_pair_label(name: str, value: float, color, position: np.ndarray) -> VGroup:
            prefix = Text(f"{name} = ", font_size=20, color=color)
            number = DecimalNumber(
                value,
                num_decimal_places=3,
                mob_class=Text,
                include_sign=True,
                color=WHITE,
                font_size=20,
            )
            row = VGroup(prefix, number).arrange(np.array([1.0, 0.0, 0.0]), buff=0.06)
            row.move_to(position)
            return row

        f1_label = make_pair_label("F₁", f_flat[0], BLUE, f_origin + np.array([1.05, 0.42, 0.0]))
        f2_label = make_pair_label("F₂", f_flat[1], YELLOW, f_origin + np.array([1.05, -0.42, 0.0]))
        s1_label = make_pair_label("s₁", s_flat[0], WHITE, np.array([3.78, -0.18, 0.0]))
        s2_label = make_pair_label("s₂", s_flat[1], WHITE, np.array([3.78, -0.60, 0.0]))

        # 0.40–1.20 s: the 2-D cloud.
        self.play(
            FadeIn(grid),
            FadeIn(axes),
            LaggedStart(*[FadeIn(point) for point in point_cloud], lag_ratio=0.04),
            run_time=0.80,
            rate_func=linear,
        )
        # 1.20–1.80 s: halo traveler, same language as 3.4 / 4.1.
        self.play(
            Create(halo_outer),
            Create(halo_inner),
            FadeIn(sample_x1_label),
            FadeIn(sample_x2_label),
            Flash(traveler_center, color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=0.60,
            rate_func=smooth,
        )
        self.wait(2.20)

        # 4.00–7.20 s: formula once, then the F box with two inner stages.
        self.play(FadeIn(formula), Create(connector), run_time=1.20, rate_func=smooth)
        self.play(Create(f_box), FadeIn(f_name), run_time=1.00, rate_func=smooth)
        self.play(FadeIn(inner_stages), run_time=1.00, rate_func=smooth)
        self.wait(0.40)

        # 7.60–11.40 s: main path through F; F(x) needles are the residual.
        self.play(Create(into_f), run_time=0.90, rate_func=smooth)
        self.play(Create(out_of_f), FadeIn(f_zero), run_time=0.80, rate_func=smooth)
        self.add(f_bars)
        self.play(f_reveal.animate.set_value(1.0), FadeIn(f1_label), FadeIn(f2_label), run_time=1.50, rate_func=smooth)
        self.play(Create(down_to_add), run_time=0.60, rate_func=smooth)
        self.wait(1.80)

        # 11.40–15.40 s: shortcut copies x around F, never through the box.
        self.play(Create(shortcut), FadeIn(shortcut_x_label), run_time=2.00, rate_func=smooth)
        self.wait(2.00)

        # 15.40–19.40 s: add, then ReLU. Both s components stay positive.
        self.play(Create(add_ring), FadeIn(add_sign), run_time=0.80, rate_func=smooth)
        self.play(FadeIn(s1_label), FadeIn(s2_label), run_time=1.20, rate_func=smooth)
        self.wait(0.80)
        self.play(
            Create(into_relu),
            Create(relu_box),
            FadeIn(relu_word),
            Create(into_y),
            Create(y_ring),
            FadeIn(y_sign),
            run_time=1.40,
            rate_func=smooth,
        )
        self.wait(0.80)

        # 19.40–22.20 s: y on the traveler — next to x, so y ≈ x is readable.
        self.play(
            FadeIn(y1_on_traveler),
            FadeIn(y2_on_traveler),
            Flash(traveler_center, color=YELLOW, flash_radius=0.38, line_length=0.10),
            run_time=1.20,
            rate_func=smooth,
        )
        self.wait(1.60)

        # 22.20–30.00 s: keep shortcut, F, sum, and y ≈ x through the last frame.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=7.80, rate_func=linear)
