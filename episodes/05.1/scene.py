"""D2L 5.1 — layers nested in a Sequential block, forward only.

Every moving value comes from the column-vector computation below.
The traveler is rewritten as it crosses each inner box:

    h = relu(W1 x + b1)
    o = W2 h + b2

Weights are never trained. Geometry is nested stroke rectangles, not a
class diagram and not a weight spreadsheet.
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
    ValueTracker,
    WHITE,
    YELLOW,
    config,
    interpolate_color,
    linear,
    smooth,
    always_redraw,
)


config.background_color = BLACK


# Same 2-D traveler as 3.4 / 4.6. First affine map widens to 3, ReLU
# clips the negative hidden unit, second affine map returns 2 outputs.
INPUT_X = np.array([[1.00], [0.50]], dtype=float)
WEIGHTS_1 = np.array(
    [
        [1.20, 0.40],
        [-0.80, 0.20],
        [0.40, 0.80],
    ],
    dtype=float,
)
BIAS_1 = np.array([[0.10], [-0.30], [0.05]], dtype=float)
PREACTIVATIONS = WEIGHTS_1 @ INPUT_X + BIAS_1


def relu(values: np.ndarray) -> np.ndarray:
    """Component-wise max(z, 0) for the same values drawn on screen."""
    return np.maximum(values, 0.0)


HIDDEN = relu(PREACTIVATIONS)
WEIGHTS_2 = np.array(
    [
        [0.60, 0.20, 0.40],
        [-0.80, 0.40, 0.20],
    ],
    dtype=float,
)
BIAS_2 = np.array([[0.10], [0.25]], dtype=float)
OUTPUTS = WEIGHTS_2 @ HIDDEN + BIAS_2


class Episode051(Scene):
    """A ~32-second continuous, silent nested-block visualization."""

    path_y = 0.0
    needle_scale = 0.55
    needle_baseline_y = 1.06
    traveler_radius = 0.078
    halo_inner_r = 0.178
    halo_outer_r = 0.278

    # Nested rectangles. Inner boxes sit fully inside Sequential, with a
    # dedicated gap after relu so the hidden vector never sits on W₂.
    outer_c = np.array([0.28, 0.20, 0.0])
    outer_w, outer_h = 9.85, 4.72
    w1_c = np.array([-3.18, 0.04, 0.0])
    w1_w, w1_h = 2.28, 3.48
    relu_c = np.array([-0.72, 0.04, 0.0])
    relu_w, relu_h = 1.92, 3.48
    w2_c = np.array([2.88, 0.04, 0.0])
    w2_w, w2_h = 2.28, 3.48

    # Path stations: x is outside the block, h is the gap after relu,
    # o is just outside Sequential. Labels live in the pocket below.
    x_station = -5.72
    h_station = 0.96
    o_station = 6.12
    pocket_y = (-2.86, -3.26, -3.66)

    def stroke_rect(self, width: float, height: float, center: np.ndarray, color, stroke_width: float) -> Rectangle:
        """Stroke-only rectangle; fill stays fully transparent."""
        rect = Rectangle(width=width, height=height)
        rect.move_to(center)
        rect.set_fill(BLACK, opacity=0.0)
        rect.set_stroke(color, width=stroke_width, opacity=0.94)
        return rect

    def attach_highlight(self, rect: Rectangle, tracker: ValueTracker, cool, hot) -> None:
        cool_width, hot_width = 2.4, 4.4

        def updater(mobject: Rectangle) -> None:
            mix = tracker.get_value()
            mobject.set_stroke(
                interpolate_color(cool, hot, mix),
                width=cool_width + (hot_width - cool_width) * mix,
                opacity=0.88 + 0.12 * mix,
            )

        rect.add_updater(updater)

    def needle_offsets(self, count: int) -> tuple[float, ...]:
        if count == 2:
            return (-0.20, 0.20)
        return (-0.34, 0.00, 0.34)

    def station_axis(self, station_x: float, count: int) -> Line:
        """A short zero-axis under the needles; signed values go below it."""
        offsets = self.needle_offsets(count)
        start = np.array([station_x + offsets[0] - 0.18, self.needle_baseline_y, 0.0])
        end = np.array([station_x + offsets[-1] + 0.18, self.needle_baseline_y, 0.0])
        return Line(start, end).set_stroke(BLUE_D, width=1.7, opacity=0.84)

    def make_needles(
        self,
        station_x: float,
        trackers: list[ValueTracker],
        colors: tuple,
    ) -> VGroup:
        offsets = self.needle_offsets(len(trackers))

        def draw_needle(index: int):
            def redraw() -> Line:
                x_position = station_x + offsets[index]
                start = np.array([x_position, self.needle_baseline_y, 0.0])
                end = np.array(
                    [
                        x_position,
                        self.needle_baseline_y + trackers[index].get_value() * self.needle_scale,
                        0.0,
                    ]
                )
                return Line(start, end).set_stroke(colors[index], width=8.0, opacity=0.98)

            return always_redraw(redraw)

        def draw_foot(index: int):
            def redraw() -> Dot:
                value = abs(trackers[index].get_value())
                opacity = 0.94 if value > 1e-6 else 0.50
                return (
                    Dot(
                        np.array([station_x + offsets[index], self.needle_baseline_y, 0.0]),
                        radius=0.055,
                        color=colors[index],
                    )
                    .set_fill(colors[index], opacity=opacity)
                    .set_stroke(colors[index], width=0.0, opacity=0.0)
                )

            return always_redraw(redraw)

        return VGroup(
            self.station_axis(station_x, len(trackers)),
            *[draw_foot(index) for index in range(len(trackers))],
            *[draw_needle(index) for index in range(len(trackers))],
        )

    def make_pocket_column(self, names: tuple[str, ...], values: np.ndarray, colors: tuple, signed: tuple[bool, ...]) -> VGroup:
        rows = []
        for name, value, color, use_sign in zip(names, values, colors, signed):
            if use_sign and value < 0:
                text = f"{name} = −{abs(value):.2f}"
            else:
                text = f"{name} = {value:.2f}"
            rows.append(Text(text, font_size=20, color=color))
        column = VGroup(*rows).arrange(np.array([0.0, -1.0, 0.0]), buff=0.10)
        return column

    def construct(self) -> None:
        print(
            "D2L 5.1 x={}, h={}, o={}, z={}, W1={}, b1={}, W2={}, b2={}".format(
                np.array2string(INPUT_X.ravel(), precision=3, separator=", "),
                np.array2string(HIDDEN.ravel(), precision=3, separator=", "),
                np.array2string(OUTPUTS.ravel(), precision=3, separator=", "),
                np.array2string(PREACTIVATIONS.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHTS_1, precision=3, separator=", "),
                np.array2string(BIAS_1.ravel(), precision=3, separator=", "),
                np.array2string(WEIGHTS_2, precision=3, separator=", "),
                np.array2string(BIAS_2.ravel(), precision=3, separator=", "),
            )
        )

        # 4-second Chinese title and explanation; later timing comments refer to the original cut.
        show_episode_intro(self, "05.1")

        outer = self.stroke_rect(self.outer_w, self.outer_h, self.outer_c, BLUE_A, 3.2)
        w1_box = self.stroke_rect(self.w1_w, self.w1_h, self.w1_c, BLUE_D, 2.4)
        relu_box = self.stroke_rect(self.relu_w, self.relu_h, self.relu_c, BLUE_D, 2.4)
        w2_box = self.stroke_rect(self.w2_w, self.w2_h, self.w2_c, BLUE_D, 2.4)

        sequential_label = Text("Sequential", font_size=28, color=BLUE_A).move_to(
            np.array([self.outer_c[0], self.outer_c[1] + self.outer_h / 2 - 0.24, 0.0])
        )
        w1_formula = Text("W₁x + b₁", font_size=24, color=WHITE).move_to(
            np.array([self.w1_c[0], self.w1_c[1] + self.w1_h / 2 - 0.36, 0.0])
        )
        relu_formula = Text("relu", font_size=24, color=WHITE).move_to(
            np.array([self.relu_c[0], self.relu_c[1] + self.relu_h / 2 - 0.36, 0.0])
        )
        w2_formula = Text("W₂h + b₂", font_size=24, color=WHITE).move_to(
            np.array([self.w2_c[0], self.w2_c[1] + self.w2_h / 2 - 0.36, 0.0])
        )

        path = Line(
            np.array([self.x_station - 0.55, self.path_y, 0.0]),
            np.array([self.o_station + 0.42, self.path_y, 0.0]),
        ).set_stroke(BLUE_E, width=2.0, opacity=0.55)

        traveler_x = ValueTracker(self.x_station)

        def traveler_pos() -> np.ndarray:
            return np.array([traveler_x.get_value(), self.path_y, 0.0])

        traveler_dot = always_redraw(
            lambda: Dot(traveler_pos(), radius=self.traveler_radius, color=YELLOW)
            .set_fill(YELLOW, opacity=1.0)
            .set_stroke(YELLOW, width=0.0, opacity=0.0)
        )
        halo_inner = always_redraw(
            lambda: Circle(radius=self.halo_inner_r)
            .move_to(traveler_pos())
            .set_stroke(YELLOW, width=2.8, opacity=0.98)
        )
        halo_outer = always_redraw(
            lambda: Circle(radius=self.halo_outer_r)
            .move_to(traveler_pos())
            .set_stroke(BLUE_A, width=2.1, opacity=0.50)
        )

        x_colors = (BLUE, YELLOW)
        h_colors = (BLUE, YELLOW, BLUE_A)
        o_colors = (BLUE, YELLOW)
        x_trackers = [ValueTracker(0.0) for _ in range(2)]
        h_trackers = [ValueTracker(0.0) for _ in range(3)]
        o_trackers = [ValueTracker(0.0) for _ in range(2)]
        w1_hot = ValueTracker(0.0)
        relu_hot = ValueTracker(0.0)
        w2_hot = ValueTracker(0.0)

        x_needles = self.make_needles(self.x_station, x_trackers, x_colors)
        h_needles = self.make_needles(self.h_station, h_trackers, h_colors)
        o_needles = self.make_needles(self.o_station, o_trackers, o_colors)

        x_column = self.make_pocket_column(
            ("x₁", "x₂"),
            INPUT_X.ravel(),
            x_colors,
            (False, False),
        ).move_to(np.array([self.x_station, (self.pocket_y[0] + self.pocket_y[1]) / 2, 0.0]))
        h_column = self.make_pocket_column(
            ("h₁", "h₂", "h₃"),
            HIDDEN.ravel(),
            h_colors,
            (False, False, False),
        ).move_to(np.array([self.h_station, (self.pocket_y[0] + self.pocket_y[2]) / 2, 0.0]))
        o_column = self.make_pocket_column(
            ("o₁", "o₂"),
            OUTPUTS.ravel(),
            o_colors,
            (False, True),
        ).move_to(np.array([5.92, (self.pocket_y[0] + self.pocket_y[1]) / 2, 0.0]))

        path_x_mark = Text("x", font_size=30, color=WHITE).move_to(
            np.array([self.x_station, -2.02, 0.0])
        )
        path_h_mark = Text("h", font_size=30, color=WHITE).move_to(
            np.array([self.h_station, -2.02, 0.0])
        )
        path_o_mark = Text("o", font_size=30, color=WHITE).move_to(
            np.array([self.o_station, -2.02, 0.0])
        )

        # 0.40–1.55 s: the outer block exists before any layer is drawn.
        self.play(Create(outer), FadeIn(sequential_label), run_time=1.15, rate_func=smooth)

        # 1.55–3.45 s: layers nest inside the block.
        self.play(
            LaggedStart(Create(w1_box), Create(relu_box), Create(w2_box), lag_ratio=0.22),
            LaggedStart(FadeIn(w1_formula), FadeIn(relu_formula), FadeIn(w2_formula), lag_ratio=0.22),
            run_time=1.90,
            rate_func=smooth,
        )
        self.attach_highlight(w1_box, w1_hot, BLUE_D, YELLOW)
        self.attach_highlight(relu_box, relu_hot, BLUE_D, YELLOW)
        self.attach_highlight(w2_box, w2_hot, BLUE_D, YELLOW)

        # 3.45–6.85 s: traveler, input vector, and path. x is written once.
        self.play(Create(path), run_time=0.55, rate_func=smooth)
        self.add(traveler_dot, halo_inner, halo_outer)
        self.add(x_needles)
        self.play(
            x_trackers[0].animate.set_value(float(INPUT_X[0, 0])),
            x_trackers[1].animate.set_value(float(INPUT_X[1, 0])),
            FadeIn(path_x_mark),
            FadeIn(x_column),
            Flash(np.array([self.x_station, self.path_y, 0.0]), color=YELLOW, flash_radius=0.42, line_length=0.11),
            run_time=1.05,
            rate_func=smooth,
        )
        self.wait(1.70)

        # 6.85–9.55 s: enter Sequential; first inner box rewrites x into z.
        self.add(h_needles)
        self.play(
            traveler_x.animate.set_value(self.w1_c[0]),
            w1_hot.animate.set_value(1.0),
            h_trackers[0].animate.set_value(float(PREACTIVATIONS[0, 0])),
            h_trackers[1].animate.set_value(float(PREACTIVATIONS[1, 0])),
            h_trackers[2].animate.set_value(float(PREACTIVATIONS[2, 0])),
            run_time=2.20,
            rate_func=smooth,
        )
        self.wait(0.50)

        # 9.55–13.35 s: relu box clips the negative hidden unit; stamp h.
        self.play(
            traveler_x.animate.set_value(self.relu_c[0]),
            w1_hot.animate.set_value(0.0),
            relu_hot.animate.set_value(1.0),
            run_time=1.35,
            rate_func=smooth,
        )
        self.play(
            h_trackers[0].animate.set_value(float(HIDDEN[0, 0])),
            h_trackers[1].animate.set_value(float(HIDDEN[1, 0])),
            h_trackers[2].animate.set_value(float(HIDDEN[2, 0])),
            run_time=1.70,
            rate_func=smooth,
        )
        self.play(
            traveler_x.animate.set_value(self.h_station),
            FadeIn(path_h_mark),
            FadeIn(h_column),
            run_time=0.75,
            rate_func=smooth,
        )
        self.wait(1.00)

        # 14.10–18.20 s: second affine box rewrites h into o; traveler leaves.
        self.add(o_needles)
        self.play(
            traveler_x.animate.set_value(self.w2_c[0]),
            relu_hot.animate.set_value(0.0),
            w2_hot.animate.set_value(1.0),
            o_trackers[0].animate.set_value(float(OUTPUTS[0, 0])),
            o_trackers[1].animate.set_value(float(OUTPUTS[1, 0])),
            run_time=2.10,
            rate_func=smooth,
        )
        self.play(
            traveler_x.animate.set_value(self.o_station),
            w2_hot.animate.set_value(0.0),
            FadeIn(path_o_mark),
            FadeIn(o_column),
            run_time=1.40,
            rate_func=smooth,
        )
        self.wait(0.80)

        # Hold nested boxes, x/h/o on the path, and the exited traveler.
        final_hold = ValueTracker(0.0)
        self.play(final_hold.animate.set_value(1.0), run_time=12.05, rate_func=linear)
